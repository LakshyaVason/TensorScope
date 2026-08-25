"""TensorScope — inspect the exact intermediate tensors of a real local LLM forward pass.

Run the GUI with:  python TensorScope.py
Run non-GUI checks with: python TensorScope.py --self-test

TensorScope loads a local Hugging Face model in-process and captures its real
intermediate tensors with PyTorch forward hooks plus a registered attention
implementation.  There is no separate server and no build step.

The design invariant: no tensor is ever inferred, simulated, or reconstructed.
Every displayed value is a tensor the model actually computed.  Two mechanisms
are needed, because one is not sufficient:

  * Module forward hooks supply embedding, normalized_input, q/k/v,
    attention_output and layer_output.
  * `CaptureAttention` supplies attention_scores, attention_weights and the
    post-RoPE q/k/v.  These are unreachable by hooks: the pre-softmax score
    tensor is a local variable inside the attention function, and transformers
    5.x removed `output_attentions` from the model forward signature.
    `CaptureAttention` mirrors upstream's `eager_attention_forward` line for
    line and *proves* it by re-running upstream's own function on its first
    call and requiring bitwise-equal outputs.  `RunCapture.validate()` rejects
    any capture that did not pass that check.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import sqlite3
import sys
import traceback
import zlib
from contextlib import closing
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

# Keep matplotlib's cache beside the application when a user profile is locked down.
APP_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(APP_DIR / ".matplotlib"))

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)


DB_PATH = APP_DIR / "tensorscope_runs.sqlite3"
SCHEMA_VERSION = 2
CAPTURE_SOURCE = "pytorch-forward-hooks"

# A dense instruct model.  Dense matters: a Mixture-of-Experts model routes through
# a router and many experts between attention_output and layer_output, and none of
# that appears in the tensor set below -- the recap would silently hide the most
# interesting part of the computation.  See CLAUDE.md "Model choice".
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B"

REQUIRED_LAYER_TENSORS = {
    "normalized_input", "q", "k", "v",
    # Post-RoPE (and post QK-norm where the architecture has it), KV heads expanded.
    # These are the tensors that actually produce the scores, so a reader can verify
    # scores == q_attended @ k_attended^T * scaling + mask.  The plain q/k/v above are
    # the raw projection outputs, before rotary embedding.
    "q_attended", "k_attended", "v_attended",
    "attention_scores", "attention_weights", "attention_output", "layer_output",
}

# Attention score tensors are heads x N x N, so capture cost grows with the square
# of the prompt length.  TensorScope's scope is basic prompts; refuse rather than
# quietly produce a multi-gigabyte run.
MAX_CAPTURE_TOKENS = 256


class CaptureProtocolError(RuntimeError):
    """Raised when a capture cannot prove it contains real, complete tensors."""


def array_to_blob(array: np.ndarray) -> bytes:
    """Serialize an exact captured array without converting its numeric type."""
    stream = io.BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return zlib.compress(stream.getvalue(), level=6)


def blob_to_array(blob: bytes) -> np.ndarray:
    """Restore an array previously written by array_to_blob."""
    return np.load(io.BytesIO(zlib.decompress(blob)), allow_pickle=False)


def tensor_to_numpy(tensor) -> np.ndarray:
    """Move a captured device tensor to host, preserving every bit of its value.

    numpy has no bfloat16 dtype, so bf16 captures are *widened* to float32.  Every
    bf16 value is exactly representable in float32, so this is lossless and
    reversible.  It is not the downcast this project forbids: nothing is narrowed
    and no value changes.
    """
    import torch

    host = tensor.detach().to("cpu")
    if host.dtype == torch.bfloat16:
        host = host.to(torch.float32)
    return host.numpy()


def numeric_sample(array: np.ndarray, size: int = 5) -> np.ndarray:
    """Return the leading 5×5 (or smaller) real values for compact text display."""
    matrix = np.asarray(array)
    while matrix.ndim > 2:
        matrix = matrix[0]
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    if matrix.ndim == 0:
        matrix = matrix.reshape(1, 1)
    return matrix[:size, :size]


def display_matrix(array: np.ndarray, maximum: int = 96) -> np.ndarray:
    """Stride-sample an existing tensor only for display; it never changes persisted data."""
    matrix = np.asarray(array)
    while matrix.ndim > 2:
        matrix = matrix[0]
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    row_step = max(1, int(np.ceil(matrix.shape[0] / maximum)))
    col_step = max(1, int(np.ceil(matrix.shape[1] / maximum)))
    return matrix[::row_step, ::col_step]


def sample_text(array: np.ndarray) -> str:
    return np.array2string(numeric_sample(array), precision=5, suppress_small=False, max_line_width=95)


@dataclass
class LayerCapture:
    """Exact tensors received for one model transformer layer."""
    index: int
    tensors: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class RunCapture:
    """An in-memory representation of a validated capture."""
    prompt: str
    response: str = ""
    token_ids: list[int] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    prompt_token_ids: list[int] = field(default_factory=list)
    prompt_tokens: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None
    layers: dict[int, LayerCapture] = field(default_factory=dict)
    generated_embedding: np.ndarray | None = None
    generated_layers: dict[int, LayerCapture] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    completed_at: str | None = None

    def validate(self) -> None:
        """Reject incomplete captures so the UI never presents invented computations."""
        if not self.response:
            raise CaptureProtocolError("Capture ended without a generated response.")
        if self.embedding is None:
            raise CaptureProtocolError("Capture did not include the real embedding output.")
        if self.generated_embedding is None:
            raise CaptureProtocolError("Capture did not include the first generated-token forward pass embedding.")
        if not self.prompt_token_ids or len(self.prompt_token_ids) != len(self.prompt_tokens):
            raise CaptureProtocolError("Capture did not include consistent prompt tokenization.")
        if not self.token_ids or len(self.token_ids) != len(self.tokens):
            raise CaptureProtocolError("Capture token IDs and token text are missing or inconsistent.")
        if not self.layers:
            raise CaptureProtocolError("Capture did not include any transformer layers.")
        for index, layer in self.layers.items():
            missing = REQUIRED_LAYER_TENSORS - set(layer.tensors)
            if missing:
                raise CaptureProtocolError(f"Layer {index} is missing: {', '.join(sorted(missing))}.")
        for index, layer in self.generated_layers.items():
            missing = REQUIRED_LAYER_TENSORS - set(layer.tensors)
            if missing:
                raise CaptureProtocolError(f"Generated-token layer {index} is missing: {', '.join(sorted(missing))}.")
        if set(self.layers) != set(self.generated_layers):
            raise CaptureProtocolError("Prefill and generated-token captures do not cover the same layers.")
        if self.metadata.get("capture_source") != CAPTURE_SOURCE:
            raise CaptureProtocolError(f"Capture did not attest capture_source={CAPTURE_SOURCE}.")
        if self.metadata.get("faithful_to_upstream_eager") is not True:
            raise CaptureProtocolError(
                "Capture did not verify its attention implementation against upstream "
                "eager_attention_forward; the values cannot be trusted as exact."
            )
        if self.metadata.get("logits_match_stock_eager") is not True:
            raise CaptureProtocolError(
                "Capture did not prove its logits match stock eager attention, so the "
                "capture machinery may have changed what the model computed."
            )


class CaptureAttention:
    """An attention implementation that mirrors upstream's eager path and captures
    the pre-softmax scores in flight.

    This is not a reconstruction: `scores` is the exact tensor handed to softmax by
    the code that actually ran.  Faithfulness is verified on the first call by
    re-running upstream's own `eager_attention_forward` with identical inputs and
    requiring bitwise-equal outputs.  If a transformers upgrade changes the eager
    path, `faithful` goes False and validate() refuses the capture.
    """

    def __init__(self, module_path: str) -> None:
        source = import_module(module_path)
        for symbol in ("repeat_kv", "eager_attention_forward"):
            if not hasattr(source, symbol):
                raise CaptureProtocolError(
                    f"{module_path} has no {symbol}; this architecture cannot be captured."
                )
        self.repeat_kv = source.repeat_kv
        self.reference = source.eager_attention_forward
        self.sink: dict | None = None
        self.faithful: bool | None = None
        self.calls = 0
        self.verified_calls = 0

    def __call__(self, module, query, key, value, attention_mask, scaling,
                 dropout=0.0, **kwargs):
        import torch

        # Mirrors upstream eager_attention_forward line for line.
        key_states = self.repeat_kv(key, module.num_key_value_groups)
        value_states = self.repeat_kv(value, module.num_key_value_groups)

        scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
        attn_output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()

        self.calls += 1
        # Verify against upstream's own implementation on every call that is actually
        # being captured -- i.e. every layer of both captured phases -- not just the
        # first one. Checking a single call once passed while later layers diverged.
        if self.sink is not None or self.faithful is None:
            reference_output, reference_weights = self.reference(
                module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
            matches = bool(torch.equal(attn_output, reference_output)
                           and torch.equal(weights, reference_weights))
            self.faithful = matches if self.faithful is None else (self.faithful and matches)
            self.verified_calls += 1

        if self.sink is not None:
            self.sink["attention_scores"] = scores.detach()
            self.sink["attention_weights"] = weights.detach()
            self.sink["q_attended"] = query.detach()
            self.sink["k_attended"] = key_states.detach()
            self.sink["v_attended"] = value_states.detach()
        return attn_output, weights


class ModelCapture:
    """Loads a local Hugging Face model and captures its exact intermediate tensors.

    Capture scope is fixed: prompt prefill plus the first generated token.  Any
    further tokens are generated with capture switched off, purely to show the
    reader a complete answer.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.attention: CaptureAttention | None = None
        self._scratch: dict[tuple[int, str], Any] = {}
        self._embedding_scratch: dict[str, Any] = {}
        self._per_layer: dict[int, dict] = {}
        self._capturing = False

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def describe(self) -> str:
        if not self.loaded:
            return "no model loaded"
        config = self.model.config
        return (f"{self.model_id} — {len(self._layers())} layers, "
                f"hidden {config.hidden_size}, {config.num_attention_heads} heads, "
                f"on {self.device}")

    def _layers(self):
        return self.model.model.layers

    def load(self, progress: Callable[[str], None] | None = None) -> None:
        """Import torch, fetch the model, and install the capture machinery."""
        def announce(message: str) -> None:
            if progress:
                progress(message)

        announce("Importing torch...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        announce(f"Loading {self.model_id} on {self.device} (first run downloads weights)...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        weight_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        # attn_implementation="eager" is mandatory: fused/flash attention kernels never
        # materialize the full attention matrix, so it could not be captured.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=weight_dtype, attn_implementation="eager")
        except TypeError:  # older transformers spell this torch_dtype
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=weight_dtype, attn_implementation="eager")
        self.model = model.to(self.device).eval()

        announce("Installing capture hooks...")
        self._install()
        announce(f"Ready — {self.describe()}")

    @staticmethod
    def _registry(module_path: str, name: str):
        try:
            return getattr(import_module(module_path), name)
        except (ImportError, AttributeError) as exc:
            raise CaptureProtocolError(
                f"Could not find {name} in this transformers version; "
                "pre-softmax attention scores cannot be captured."
            ) from exc

    def _register_implementation(self, name: str) -> None:
        """Register the capture implementation as a first-class attention backend.

        Two subtleties, both of which silently corrupt the run if you get them wrong:

        1. Registering the attention function alone is NOT enough. transformers builds
           the *causal mask* through a separate registry keyed by the same
           implementation name, so the "eager" mask builder must be registered under
           this name too.
        2. It must be registered with `.register()`, not `registry[name] = ...`.
           `__setitem__` writes to a per-instance `_local_mapping`, but
           `masking_utils._preprocess_mask_arguments` tests membership against the
           class-wide `_global_mapping` and treats a miss as "custom backend that needs
           no mask" -- returning `None`. A `None` mask makes the eager path skip masking
           entirely, so attention becomes bidirectional: every tensor still looks real
           and every value is genuinely computed, but the model is reading the future.
        """
        attentions = self._registry("transformers.modeling_utils", "ALL_ATTENTION_FUNCTIONS")
        attentions.register(name, self.attention)

        masks = self._registry("transformers.masking_utils", "ALL_MASK_ATTENTION_FUNCTIONS")
        if "eager" not in masks:
            raise CaptureProtocolError(
                "transformers has no 'eager' mask builder to borrow; cannot capture safely.")
        masks.register(name, masks["eager"])

        # Belt and braces: if a future version drops `_global_mapping`, the mask would
        # go quietly missing again. Fail loudly instead.
        global_mapping = getattr(type(masks), "_global_mapping", None)
        if global_mapping is None or name not in global_mapping:
            raise CaptureProtocolError(
                f"{name} did not reach the global mask registry; refusing to capture, "
                "because the causal mask would silently be dropped.")

    def _install(self) -> None:
        layers = self._layers()
        module_path = type(layers[0].self_attn).__module__
        self.attention = CaptureAttention(module_path)
        self._register_implementation("tensorscope_capture")

        self.model.config._attn_implementation = "tensorscope_capture"
        for layer in layers:
            layer.self_attn.config._attn_implementation = "tensorscope_capture"

        def first_tensor(value):
            return value[0] if isinstance(value, tuple) else value

        def record(layer_index: int, name: str):
            def hook(_module, _inputs, output):
                if self._capturing:
                    self._scratch[(layer_index, name)] = first_tensor(output).detach()
            return hook

        def embed_hook(_module, _inputs, output):
            if self._capturing:
                self._embedding_scratch["embedding"] = first_tensor(output).detach()

        self.model.model.embed_tokens.register_forward_hook(embed_hook)

        for index, layer in enumerate(layers):
            self._per_layer[index] = {}
            layer.input_layernorm.register_forward_hook(record(index, "normalized_input"))
            layer.self_attn.q_proj.register_forward_hook(record(index, "q"))
            layer.self_attn.k_proj.register_forward_hook(record(index, "k"))
            layer.self_attn.v_proj.register_forward_hook(record(index, "v"))
            layer.self_attn.o_proj.register_forward_hook(record(index, "attention_output"))
            layer.register_forward_hook(record(index, "layer_output"))

            def bind(layer_index):
                def pre_hook(_module, _args, _kwargs):
                    # Point the attention implementation at this layer's slot.
                    self.attention.sink = self._per_layer[layer_index] if self._capturing else None
                    return None
                return pre_hook
            layer.self_attn.register_forward_pre_hook(bind(index), with_kwargs=True)

    def _drain(self) -> tuple[dict[int, LayerCapture], np.ndarray]:
        """Convert one forward pass worth of captures to host arrays and reset."""
        layers: dict[int, LayerCapture] = {}
        for index in range(len(self._layers())):
            tensors = {name: tensor_to_numpy(tensor)
                       for (layer_i, name), tensor in self._scratch.items() if layer_i == index}
            tensors.update({name: tensor_to_numpy(tensor)
                            for name, tensor in self._per_layer[index].items()})
            layers[index] = LayerCapture(index, tensors)
        if "embedding" not in self._embedding_scratch:
            raise CaptureProtocolError("The embedding hook did not fire; capture is incomplete.")
        embedding = tensor_to_numpy(self._embedding_scratch["embedding"])
        self._scratch.clear()
        self._embedding_scratch.clear()
        for slot in self._per_layer.values():
            slot.clear()
        return layers, embedding

    def _set_implementation(self, name: str) -> None:
        self.model.config._attn_implementation = name
        for layer in self._layers():
            layer.self_attn.config._attn_implementation = name

    def _verify_undisturbed(self, input_ids, captured_logits) -> float:
        """Re-run the same prefill with stock eager attention and require identical logits.

        This is the check that matters. Verifying the capture implementation in isolation
        is not enough: a capture backend can be a perfect mirror of the eager math and
        still be handed a differently-built causal mask, yielding plausible-but-wrong
        values that all look real. Comparing end-to-end logits catches that.
        """
        import torch

        was_capturing = self._capturing
        self._capturing = False
        self.attention.sink = None
        self._set_implementation("eager")
        try:
            with torch.no_grad():
                stock = self.model(input_ids=input_ids, use_cache=False).logits
        finally:
            self._set_implementation("tensorscope_capture")
            self._capturing = was_capturing

        delta = float((stock.to(torch.float32) - captured_logits.to(torch.float32)).abs().max())
        if delta != 0.0:
            raise CaptureProtocolError(
                f"Capture perturbed the computation: prefill logits differ from stock "
                f"eager attention by {delta:.3e}. The captured tensors would not be the "
                "ones this model computes normally, so the run was discarded."
            )
        return delta

    def _templated(self, prompt: str) -> str:
        """Format the prompt the way the model expects, when it defines a template."""
        if not getattr(self.tokenizer, "chat_template", None):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        for extra in ({"enable_thinking": False}, {}):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **extra)
            except TypeError:
                continue
        return prompt

    def _eos_ids(self) -> set[int]:
        ids: set[int] = set()
        for source in (getattr(self.tokenizer, "eos_token_id", None),
                       getattr(self.model.generation_config, "eos_token_id", None)):
            if isinstance(source, int):
                ids.add(source)
            elif isinstance(source, (list, tuple)):
                ids.update(int(i) for i in source)
        return ids

    def generate(self, prompt: str, max_new_tokens: int = 128,
                 on_text: Callable[[str], None] | None = None) -> RunCapture:
        import torch

        if not self.loaded:
            raise CaptureProtocolError("No model is loaded.")

        encoded = self.tokenizer(self._templated(prompt), return_tensors="pt").to(self.device)
        input_ids = encoded["input_ids"]
        if input_ids.shape[1] > MAX_CAPTURE_TOKENS:
            raise CaptureProtocolError(
                f"Prompt is {input_ids.shape[1]} tokens; TensorScope captures at most "
                f"{MAX_CAPTURE_TOKENS}. Attention tensors grow with the square of the "
                "prompt length, so longer prompts would produce enormous runs."
            )

        capture = RunCapture(prompt=prompt)
        capture.prompt_token_ids = [int(i) for i in input_ids[0]]
        capture.prompt_tokens = [str(t) for t in
                                 self.tokenizer.convert_ids_to_tokens(input_ids[0])]

        def emit(token_id: int) -> None:
            text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            if not text:  # keep the response non-empty even if the model stops at once
                text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            capture.token_ids.append(token_id)
            capture.tokens.append(text)
            capture.response += text
            if on_text and text:
                on_text(text)

        eos_ids = self._eos_ids()
        with torch.no_grad():
            # --- captured: prompt prefill ---
            self._capturing = True
            prefill = self.model(input_ids=input_ids, use_cache=True)
            capture.layers, capture.embedding = self._drain()

            # Prove the capture machinery did not change what the model computes.
            logits_delta = self._verify_undisturbed(input_ids, prefill.logits)

            first_id = int(prefill.logits[:, -1, :].argmax(dim=-1)[0])

            # --- captured: first generated token ---
            step = self.model(
                input_ids=torch.tensor([[first_id]], device=self.device),
                past_key_values=prefill.past_key_values, use_cache=True)
            capture.generated_layers, capture.generated_embedding = self._drain()
            self._capturing = False
            self.attention.sink = None

            emit(first_id)

            # --- uncaptured: finish the answer so the reader sees a whole response ---
            past = step.past_key_values
            logits = step.logits[:, -1, :]
            for _ in range(max(0, max_new_tokens - 1)):
                token_id = int(logits.argmax(dim=-1)[0])
                if token_id in eos_ids:
                    break
                emit(token_id)
                step = self.model(
                    input_ids=torch.tensor([[token_id]], device=self.device),
                    past_key_values=past, use_cache=True)
                past = step.past_key_values
                logits = step.logits[:, -1, :]

        capture.metadata = {
            "capture_source": CAPTURE_SOURCE,
            "faithful_to_upstream_eager": self.attention.faithful,
            "logits_match_stock_eager": True,  # _verify_undisturbed raises otherwise
            "logits_max_abs_diff_vs_stock_eager": logits_delta,
            "schema_version": SCHEMA_VERSION,
            "model": self.model_id,
            "backend": (torch.cuda.get_device_name(0) if self.device == "cuda"
                        else "cpu"),
            "torch": torch.__version__,
            "weight_dtype": str(next(self.model.parameters()).dtype),
            "attention_calls": self.attention.calls,
            "attention_calls_verified": self.attention.verified_calls,
        }
        capture.completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
        capture.validate()
        return capture


class RunDatabase:
    """SQLite persistence for exact capture blobs and searchable run metadata."""

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = path
        self._setup()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        return connection

    def _setup(self) -> None:
        with closing(self.connect()) as db:
            db.executescript("""
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    token_ids_json TEXT NOT NULL,
                    tokens_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_blob BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tensors (
                    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    layer_index INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    shape_json TEXT NOT NULL,
                    dtype TEXT NOT NULL,
                    data_blob BLOB NOT NULL,
                    PRIMARY KEY (run_id, layer_index, name)
                );
            """)
            db.commit()

    def save(self, capture: RunCapture) -> int:
        capture.validate()
        with closing(self.connect()) as db:
            cursor = db.execute(
                """INSERT INTO runs (created_at, completed_at, prompt, response, token_ids_json,
                   tokens_json, metadata_json, embedding_blob) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (capture.started_at, capture.completed_at, capture.prompt, capture.response,
                 json.dumps(capture.token_ids), json.dumps(capture.tokens),
                 json.dumps({**capture.metadata, "prompt_token_ids": capture.prompt_token_ids,
                             "prompt_tokens": capture.prompt_tokens}), array_to_blob(capture.embedding)),
            )
            run_id = int(cursor.lastrowid)
            for index, layer in capture.layers.items():
                for name, tensor in layer.tensors.items():
                    db.execute(
                        "INSERT INTO tensors VALUES (?, ?, ?, ?, ?, ?)",
                        (run_id, index, name, json.dumps(list(tensor.shape)), str(tensor.dtype), array_to_blob(tensor)),
                    )
            db.execute(
                "INSERT INTO tensors VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, -1, "first_generated_token:embedding",
                 json.dumps(list(capture.generated_embedding.shape)), str(capture.generated_embedding.dtype),
                 array_to_blob(capture.generated_embedding)),
            )
            for index, layer in capture.generated_layers.items():
                for name, tensor in layer.tensors.items():
                    db.execute(
                        "INSERT INTO tensors VALUES (?, ?, ?, ?, ?, ?)",
                        (run_id, index, f"first_generated_token:{name}", json.dumps(list(tensor.shape)),
                         str(tensor.dtype), array_to_blob(tensor)),
                    )
            db.commit()
        return run_id

    def list_runs(self) -> list[sqlite3.Row]:
        with closing(self.connect()) as db:
            return db.execute("SELECT id, created_at, prompt, response FROM runs ORDER BY id DESC").fetchall()

    def load(self, run_id: int) -> RunCapture:
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"Run {run_id} was not found.")
            capture = RunCapture(
                prompt=row["prompt"], response=row["response"], token_ids=json.loads(row["token_ids_json"]),
                tokens=json.loads(row["tokens_json"]), metadata=json.loads(row["metadata_json"]),
                embedding=blob_to_array(row["embedding_blob"]), started_at=row["created_at"], completed_at=row["completed_at"],
            )
            capture.prompt_token_ids = capture.metadata.pop("prompt_token_ids", [])
            capture.prompt_tokens = capture.metadata.pop("prompt_tokens", [])
            for tensor_row in db.execute("SELECT * FROM tensors WHERE run_id = ? ORDER BY layer_index", (run_id,)):
                name = tensor_row["name"]
                tensor = blob_to_array(tensor_row["data_blob"])
                if name == "first_generated_token:embedding":
                    capture.generated_embedding = tensor
                elif name.startswith("first_generated_token:"):
                    index = tensor_row["layer_index"]
                    layer = capture.generated_layers.setdefault(index, LayerCapture(index))
                    layer.tensors[name.split(":", 1)[1]] = tensor
                else:
                    layer = capture.layers.setdefault(tensor_row["layer_index"], LayerCapture(tensor_row["layer_index"]))
                    layer.tensors[name] = tensor
        capture.validate()
        return capture


class LoadWorker(QThread):
    """Import torch and load model weights off the Qt event loop."""
    progress = pyqtSignal(str)
    loaded = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, capture_model: ModelCapture) -> None:
        super().__init__()
        self.capture_model = capture_model

    def run(self) -> None:
        try:
            self.capture_model.load(self.progress.emit)
            self.loaded.emit(self.capture_model.describe())
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc(limit=2)}")


class GenerationWorker(QThread):
    """Keep blocking local inference and capture outside Qt's event loop."""
    text_received = pyqtSignal(str)
    finished_capture = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, capture_model: ModelCapture, prompt: str, max_new_tokens: int = 128) -> None:
        super().__init__()
        self.capture_model, self.prompt, self.max_new_tokens = capture_model, prompt, max_new_tokens

    def run(self) -> None:
        try:
            capture = self.capture_model.generate(
                self.prompt, self.max_new_tokens, self.text_received.emit)
            self.finished_capture.emit(capture)
        except Exception as exc:  # worker must pass readable errors to Qt, not crash its thread
            self.failed.emit(f"{exc}\n\n{traceback.format_exc(limit=2)}")


class AttentionCanvas(FigureCanvas):
    """Matplotlib canvas displaying actual captured attention weights."""
    def __init__(self, weights: np.ndarray, original_shape: tuple[int, ...], parent: QWidget | None = None) -> None:
        figure = Figure(figsize=(5.8, 3.1), tight_layout=True)
        super().__init__(figure)
        self.setParent(parent)
        axes = figure.add_subplot(111)
        image = axes.imshow(display_matrix(weights), aspect="auto", cmap="viridis", interpolation="nearest")
        axes.set_title(f"Attention weights (head 0); original shape {original_shape}")
        axes.set_xlabel("key position")
        axes.set_ylabel("query position")
        figure.colorbar(image, ax=axes, fraction=0.035, pad=0.02)


# Display label for every captured tensor, in the order the model computes them.
TENSOR_LABELS = [
    ("normalized_input", "X — normalized layer input"),
    ("q", "Q = XW<sub>q</sub> — projection, before RoPE"),
    ("k", "K = XW<sub>k</sub> — projection, before RoPE"),
    ("v", "V = XW<sub>v</sub> — projection"),
    ("q_attended", "Q as attended — after RoPE, per head"),
    ("k_attended", "K as attended — after RoPE, KV heads expanded"),
    ("v_attended", "V as attended — KV heads expanded"),
    ("attention_scores", "QKᵀ / √d<sub>k</sub> + causal mask — the input to softmax"),
    ("attention_weights", "softmax(scores)"),
    ("attention_output", "W<sub>o</sub> · concat(heads) — attention block output"),
    ("layer_output", "Layer output"),
]


class LayerSection(QFrame):
    """One collapsible transformer layer.  Contents are built on first expand so a
    36-layer recap opens immediately instead of rendering 70+ heatmaps up front."""

    def __init__(self, index: int, layer: LayerCapture, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.layer = layer
        self.setFrameShape(QFrame.StyledPanel)
        self._layout = QVBoxLayout(self)
        self.toggle = QPushButton(f"▶  Layer {index}")
        self.toggle.setStyleSheet("font-size:15px;font-weight:bold;text-align:left;padding:6px;")
        self.toggle.clicked.connect(self._toggle)
        self._layout.addWidget(self.toggle)
        self.body: QWidget | None = None

    def _toggle(self) -> None:
        if self.body is None:
            self.body = QWidget()
            body_layout = QVBoxLayout(self.body)
            for key, label in TENSOR_LABELS:
                tensor = self.layer.tensors.get(key)
                if tensor is None:
                    continue
                text = QLabel(f"<b>{label}</b> — shape {tensor.shape}<br><pre>{sample_text(tensor)}</pre>")
                text.setTextFormat(Qt.RichText)
                text.setWordWrap(True)
                text.setTextInteractionFlags(Qt.TextSelectableByMouse)
                body_layout.addWidget(text)
            weights = self.layer.tensors["attention_weights"]
            body_layout.addWidget(AttentionCanvas(weights, weights.shape, self.body))
            self._layout.addWidget(self.body)
        visible = not self.body.isVisible()
        self.body.setVisible(visible)
        self.toggle.setText(f"{'▼' if visible else '▶'}  Layer {self.layer.index}")


class ComputationRecap(QDialog):
    """Scrollable, read-only view of one persisted or just-captured real forward pass."""
    def __init__(self, capture: RunCapture, run_id: int | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("TensorScope — Computation Recap")
        self.resize(1100, 800)
        root = QVBoxLayout(self)
        banner = QLabel("CAPTURED FROM THE RUNNING MODEL — no tensors are inferred or simulated")
        banner.setStyleSheet("background:#14532d;color:white;font-weight:bold;padding:9px;border-radius:4px;")
        root.addWidget(banner)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); root.addWidget(scroll)
        content = QWidget(); layout = QVBoxLayout(content); scroll.setWidget(content)

        title = QLabel(f"Run #{run_id}" if run_id else "Current run")
        title.setStyleSheet("font-size:20px;font-weight:bold;"); layout.addWidget(title)
        info = QFormLayout()
        info.addRow("Model", QLabel(str(capture.metadata.get("model", "unknown"))))
        info.addRow("Backend / GPU", QLabel(str(capture.metadata.get("backend", "unknown"))))
        info.addRow("Weight dtype", QLabel(str(capture.metadata.get("weight_dtype", "unknown"))))
        info.addRow("Capture scope", QLabel("prompt prefill + first generated token"))
        info.addRow("Captured layers", QLabel(str(len(capture.layers))))
        info.addRow("Attention verified", QLabel(
            "bitwise identical to upstream eager_attention_forward"
            if capture.metadata.get("faithful_to_upstream_eager") else "NOT VERIFIED"))
        layout.addLayout(info)

        layout.addWidget(self._section(
            "1. Prompt tokenization",
            "Token IDs: " + str(capture.prompt_token_ids) + "\nTokens: " + repr(capture.prompt_tokens)))
        layout.addWidget(self._section(
            "2. Prompt-prefill embedding output",
            f"Shape: {capture.embedding.shape}\nFirst values:\n{sample_text(capture.embedding)}"))
        layout.addWidget(self._section(
            "3. First generated-token embedding output",
            f"Shape: {capture.generated_embedding.shape}\nFirst values:\n{sample_text(capture.generated_embedding)}"))
        layout.addWidget(self._section(
            "4. Forward-pass equations",
            "Q = XWq    K = XWk    V = XWv        (projections, shown before RoPE)\n"
            "Attention = softmax(QKᵀ / √dₖ + mask)  (using the post-RoPE Q and K)\n"
            "Output = Wo · (Attention × V)\n\n"
            "Every value below was read out of the model as it computed these steps."))

        for phase_title, phase_layers in (("Prompt prefill forward pass", capture.layers),
                                          ("First generated-token forward pass", capture.generated_layers)):
            phase_heading = QLabel(phase_title)
            phase_heading.setStyleSheet("font-size:18px;font-weight:bold;margin-top:12px;")
            layout.addWidget(phase_heading)
            hint = QLabel("Click a layer to expand its captured tensors.")
            hint.setStyleSheet("color:#475569;"); layout.addWidget(hint)
            for index in sorted(phase_layers):
                layout.addWidget(LayerSection(index, phase_layers[index], content))
        layout.addStretch(1)

    @staticmethod
    def _section(title: str, body: str) -> QFrame:
        box = QFrame(); box.setFrameShape(QFrame.StyledPanel); layout = QVBoxLayout(box)
        label = QLabel(title); label.setStyleSheet("font-size:15px;font-weight:bold;"); layout.addWidget(label)
        text = QLabel(body); text.setTextInteractionFlags(Qt.TextSelectableByMouse); text.setWordWrap(True); layout.addWidget(text)
        return box


class HistoryDialog(QDialog):
    """Database browser that restores a prior recap without model inference."""
    def __init__(self, database: RunDatabase, parent: QWidget | None = None) -> None:
        super().__init__(parent); self.database = database
        self.setWindowTitle("TensorScope — Saved Runs"); self.resize(760, 480)
        layout = QVBoxLayout(self); self.runs = QListWidget(); layout.addWidget(self.runs)
        buttons = QHBoxLayout(); refresh = QPushButton("Refresh"); open_button = QPushButton("Open Recap")
        buttons.addWidget(refresh); buttons.addWidget(open_button); layout.addLayout(buttons)
        refresh.clicked.connect(self.reload); open_button.clicked.connect(self.open_selected)
        self.runs.itemDoubleClicked.connect(lambda _: self.open_selected()); self.reload()

    def reload(self) -> None:
        self.runs.clear()
        for row in self.database.list_runs():
            item_text = f"#{row['id']}  {row['created_at']}\nPrompt: {row['prompt'][:160]}\nResponse: {row['response'][:160]}"
            from PyQt5.QtWidgets import QListWidgetItem
            item = QListWidgetItem(item_text); item.setData(Qt.UserRole, row["id"]); self.runs.addItem(item)

    def open_selected(self) -> None:
        item = self.runs.currentItem()
        if not item:
            return
        try:
            recap = ComputationRecap(self.database.load(item.data(Qt.UserRole)), item.data(Qt.UserRole), self)
            recap.exec_()
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open run", str(exc))


class TensorScopeMainWindow(QMainWindow):
    """Primary application UI and coordination point for capture, storage, and recap."""
    def __init__(self) -> None:
        super().__init__()
        self.database = RunDatabase()
        self.capture_model = ModelCapture(os.getenv("TENSORSCOPE_MODEL", DEFAULT_MODEL_ID))
        self.worker: GenerationWorker | None = None
        self.loader: LoadWorker | None = None

        self.setWindowTitle("TensorScope"); self.resize(980, 700)
        central = QWidget(); self.setCentralWidget(central); layout = QVBoxLayout(central)
        title = QLabel("TensorScope"); title.setStyleSheet("font-size:27px;font-weight:bold;"); layout.addWidget(title)
        subtitle = QLabel("Local inference with exact, captured intermediate tensors")
        subtitle.setStyleSheet("color:#475569;"); layout.addWidget(subtitle)

        settings = QHBoxLayout(); settings.addWidget(QLabel("Model:"))
        self.model_id = QLineEdit(self.capture_model.model_id); settings.addWidget(self.model_id)
        self.load_button = QPushButton("Load model"); settings.addWidget(self.load_button)
        layout.addLayout(settings)

        layout.addWidget(QLabel("Prompt")); self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("Ask the model something...")
        self.prompt.setFixedHeight(125); layout.addWidget(self.prompt)
        buttons = QHBoxLayout(); self.run_button = QPushButton("Run Model")
        self.history_button = QPushButton("View Saved Runs")
        self.run_button.setEnabled(False)
        buttons.addWidget(self.run_button); buttons.addWidget(self.history_button)
        buttons.addStretch(1); layout.addLayout(buttons)
        layout.addWidget(QLabel("Model answer")); self.output = QTextEdit()
        self.output.setReadOnly(True); layout.addWidget(self.output)
        self.status = QLabel("Load a model to begin.")
        self.status.setStyleSheet("color:#334155;"); layout.addWidget(self.status)

        self.run_button.clicked.connect(self.run_model)
        self.history_button.clicked.connect(lambda: HistoryDialog(self.database, self).exec_())
        self.load_button.clicked.connect(self.load_model)

    def _say(self, message: str, colour: str = "#334155") -> None:
        self.status.setText(message); self.status.setStyleSheet(f"color:{colour};")

    def load_model(self) -> None:
        wanted = self.model_id.text().strip()
        if not wanted:
            QMessageBox.information(self, "Model required", "Enter a Hugging Face model id."); return
        self.capture_model = ModelCapture(wanted)
        self.load_button.setEnabled(False); self.run_button.setEnabled(False)
        self._say("Loading model...", "#1d4ed8")
        self.loader = LoadWorker(self.capture_model)
        self.loader.progress.connect(lambda message: self._say(message, "#1d4ed8"))
        self.loader.loaded.connect(self.model_ready)
        self.loader.failed.connect(self.load_failed)
        self.loader.start()

    def model_ready(self, description: str) -> None:
        self.load_button.setEnabled(True); self.run_button.setEnabled(True)
        self._say(f"Ready — {description}", "#166534")

    def load_failed(self, message: str) -> None:
        self.load_button.setEnabled(True)
        self._say("Model could not be loaded.", "#b91c1c")
        QMessageBox.critical(self, "Model load failed", message)

    def run_model(self) -> None:
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.information(self, "Prompt required", "Enter a prompt before running the model."); return
        if not self.capture_model.loaded:
            QMessageBox.information(self, "Model required", "Load a model first."); return
        self.output.clear(); self.run_button.setEnabled(False)
        self._say("Running the forward pass and capturing tensors...", "#1d4ed8")
        self.worker = GenerationWorker(self.capture_model, prompt)
        self.worker.text_received.connect(self.output.insertPlainText)
        self.worker.finished_capture.connect(self.persist_and_show)
        self.worker.failed.connect(self.run_failed)
        self.worker.start()

    def persist_and_show(self, capture: RunCapture) -> None:
        try:
            run_id = self.database.save(capture)
            self._say(f"Saved exact capture as run #{run_id}.", "#166534")
            ComputationRecap(capture, run_id, self).exec_()
        except Exception as exc:
            self._say("Generation completed but the capture could not be saved.", "#b91c1c")
            QMessageBox.critical(self, "Database error", str(exc))
        finally:
            self.run_button.setEnabled(True)

    def run_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self._say("Run failed: an exact capture was not available.", "#b91c1c")
        QMessageBox.critical(self, "Capture failed", message)


def self_test() -> None:
    """Small checks for persistence and real-array visualization helpers (no torch needed)."""
    import tempfile
    layer_data = {name: np.arange(36, dtype=np.float32).reshape(1, 6, 6) for name in REQUIRED_LAYER_TENSORS}
    capture = RunCapture(prompt="test", response="answer", token_ids=[1], tokens=["answer"],
                         prompt_token_ids=[0], prompt_tokens=["test"], embedding=np.arange(8, dtype=np.float16),
                         generated_embedding=np.arange(8, dtype=np.float16),
                         metadata={"capture_source": CAPTURE_SOURCE, "faithful_to_upstream_eager": True,
                                   "logits_match_stock_eager": True},
                         layers={0: LayerCapture(0, layer_data)}, generated_layers={0: LayerCapture(0, layer_data)})
    with tempfile.TemporaryDirectory() as temporary:
        database = RunDatabase(Path(temporary) / "test.sqlite3"); run_id = database.save(capture); restored = database.load(run_id)
        assert np.array_equal(restored.embedding, capture.embedding)
        assert np.array_equal(restored.layers[0].tensors["q"], layer_data["q"])
        assert restored.layers[0].tensors["q"].dtype == layer_data["q"].dtype
        assert numeric_sample(layer_data["q"]).shape == (5, 5)
        assert display_matrix(np.ones((300, 300))).shape[0] <= 96

    # An unverified attention implementation must be refused outright.
    unverified = RunCapture(prompt="p", response="r", token_ids=[1], tokens=["r"],
                            prompt_token_ids=[0], prompt_tokens=["p"],
                            embedding=np.zeros(4, dtype=np.float32),
                            generated_embedding=np.zeros(4, dtype=np.float32),
                            metadata={"capture_source": CAPTURE_SOURCE, "faithful_to_upstream_eager": False},
                            layers={0: LayerCapture(0, layer_data)},
                            generated_layers={0: LayerCapture(0, layer_data)})
    try:
        unverified.validate()
    except CaptureProtocolError:
        pass
    else:
        raise AssertionError("validate() accepted a capture with unverified attention")

    # Every label the recap renders must name a tensor the capture actually requires.
    labelled = {key for key, _ in TENSOR_LABELS}
    assert REQUIRED_LAYER_TENSORS <= labelled, REQUIRED_LAYER_TENSORS - labelled
    print("TensorScope self-test passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        application = QApplication(sys.argv)
        window = TensorScopeMainWindow(); window.show()
        sys.exit(application.exec_())
