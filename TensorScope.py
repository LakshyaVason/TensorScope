"""TensorScope — inspect captured, real local-model inference tensors.

Run the GUI with:  python TensorScope.py
Run non-GUI checks with: python TensorScope.py --self-test

TensorScope needs an instrumented local inference service.  Set
TENSORSCOPE_RUNNER_URL to its HTTP endpoint (default:
http://127.0.0.1:11435/v1/capture/generate).  The service must return JSONL
using the documented CaptureClient protocol below; plain Ollama endpoints are
deliberately rejected because they do not expose intermediate tensors.
"""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
import os
import sqlite3
import sys
import threading
import traceback
import zlib
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Keep matplotlib's cache beside the application when a user profile is locked down.
APP_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(APP_DIR / ".matplotlib"))

import numpy as np
import requests
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)


DB_PATH = APP_DIR / "tensorscope_runs.sqlite3"
MODEL_NAME = "gpt-oss:20b"
SCHEMA_VERSION = 1
DEFAULT_ENDPOINT = "http://127.0.0.1:11435/v1/capture/generate"
REQUIRED_LAYER_TENSORS = {
    "normalized_input", "q", "k", "v", "attention_scores",
    "attention_weights", "attention_output", "layer_output",
}


class CaptureProtocolError(RuntimeError):
    """Raised when a runner response cannot prove it contains real captures."""


def array_to_blob(array: np.ndarray) -> bytes:
    """Serialize an exact captured array without converting its numeric type."""
    stream = io.BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return zlib.compress(stream.getvalue(), level=6)


def blob_to_array(blob: bytes) -> np.ndarray:
    """Restore an array previously written by array_to_blob."""
    return np.load(io.BytesIO(zlib.decompress(blob)), allow_pickle=False)


def decode_tensor(value: Any) -> np.ndarray:
    """Decode runner tensor payloads. JSON lists and compressed .npy base64 are accepted."""
    if isinstance(value, list):
        return np.asarray(value)
    if isinstance(value, dict) and value.get("encoding") == "npy-zlib-base64":
        try:
            return blob_to_array(base64.b64decode(value["data"]))
        except (KeyError, ValueError, OSError) as exc:
            raise CaptureProtocolError("Invalid compressed tensor payload.") from exc
    if isinstance(value, dict) and value.get("encoding") == "raw-f32-base64":
        try:
            shape = tuple(int(dimension) for dimension in value["shape"])
            raw = base64.b64decode(value["data"], validate=True)
            array = np.frombuffer(raw, dtype="<f4")
            if array.size != int(np.prod(shape, dtype=np.int64)):
                raise ValueError("raw tensor byte count does not match its shape")
            return array.reshape(shape)
        except (KeyError, TypeError, ValueError) as exc:
            raise CaptureProtocolError("Invalid raw-f32-base64 tensor payload.") from exc
    raise CaptureProtocolError("Tensor data must be a JSON list or npy-zlib-base64 payload.")


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
    """An in-memory representation of a validated runner capture."""
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
        if self.metadata.get("capture_source") != "instrumented-runtime":
            raise CaptureProtocolError("Runner did not attest capture_source=instrumented-runtime.")


class CaptureClient:
    """HTTP/JSONL client for a patched local Ollama/llama.cpp-compatible runner.

    Request: {model, prompt, capture: {scope: prefill_plus_first_generated_token,
    tensors: [...]}}.  Each response line is JSON with one of: metadata, token,
    tensor, complete, error. Tensor records contain layer (or null), name, and data.
    """

    def __init__(self, endpoint: str | None = None) -> None:
        self.endpoint = endpoint or os.getenv("TENSORSCOPE_RUNNER_URL", DEFAULT_ENDPOINT)

    def generate(self, prompt: str, on_text: callable | None = None) -> RunCapture:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": True,
            "capture": {
                "schema_version": SCHEMA_VERSION,
                "scope": "prefill_plus_first_generated_token",
                "tensors": sorted(REQUIRED_LAYER_TENSORS | {"embedding"}),
            },
        }
        capture = RunCapture(prompt=prompt)
        try:
            with requests.post(self.endpoint, json=payload, stream=True, timeout=(4, 180)) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    self._consume_record(capture, json.loads(raw_line), on_text)
        except requests.RequestException as exc:
            raise CaptureProtocolError(
                f"Cannot reach instrumented runner at {self.endpoint}. "
                "Standard Ollama cannot supply exact intermediate tensors."
            ) from exc
        except json.JSONDecodeError as exc:
            raise CaptureProtocolError("Runner returned non-JSONL data; expected TensorScope capture protocol.") from exc
        capture.completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
        capture.validate()
        return capture

    @staticmethod
    def _consume_record(capture: RunCapture, record: dict[str, Any], on_text: callable | None) -> None:
        kind = record.get("type")
        if kind == "metadata":
            capture.metadata.update(record.get("data", {}))
        elif kind == "token":
            token = str(record.get("text", ""))
            capture.token_ids.append(int(record["id"]))
            capture.tokens.append(token)
            capture.response += token
            if on_text:
                on_text(token)
        elif kind == "tokenization":
            capture.prompt_token_ids = [int(token_id) for token_id in record.get("ids", [])]
            capture.prompt_tokens = [str(token) for token in record.get("tokens", [])]
        elif kind == "tensor":
            name = str(record.get("name", ""))
            tensor = decode_tensor(record.get("data"))
            phase = record.get("phase", "prefill")
            if phase not in {"prefill", "first_generated_token"}:
                raise CaptureProtocolError(f"Unknown capture phase: {phase!r}.")
            is_generated = phase == "first_generated_token"
            if name == "embedding":
                if is_generated:
                    capture.generated_embedding = tensor
                else:
                    capture.embedding = tensor
                return
            layer_index = record.get("layer")
            if layer_index is None:
                raise CaptureProtocolError(f"Tensor '{name}' has no layer index.")
            target = capture.generated_layers if is_generated else capture.layers
            layer = target.setdefault(int(layer_index), LayerCapture(int(layer_index)))
            layer.tensors[name] = tensor
        elif kind == "error":
            raise CaptureProtocolError(str(record.get("message", "Runner reported an error.")))
        elif kind == "complete":
            return
        else:
            raise CaptureProtocolError(f"Unknown capture record type: {kind!r}.")


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


class GenerationWorker(QThread):
    """Keep blocking local inference outside Qt's event loop."""
    text_received = pyqtSignal(str)
    finished_capture = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, prompt: str, endpoint: str) -> None:
        super().__init__()
        self.prompt, self.endpoint = prompt, endpoint

    def run(self) -> None:
        try:
            capture = CaptureClient(self.endpoint).generate(self.prompt, self.text_received.emit)
            self.finished_capture.emit(capture)
        except Exception as exc:  # worker must pass readable errors to Qt, not crash its thread
            self.failed.emit(f"{exc}\n\n{traceback.format_exc(limit=1)}")


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


class ComputationRecap(QDialog):
    """Scrollable, read-only view of one persisted or just-captured real forward pass."""
    def __init__(self, capture: RunCapture, run_id: int | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("TensorScope — Computation Recap")
        self.resize(1100, 800)
        root = QVBoxLayout(self)
        banner = QLabel("CAPTURED FROM INSTRUMENTED RUNTIME — no tensors are inferred or simulated")
        banner.setStyleSheet("background:#14532d;color:white;font-weight:bold;padding:9px;border-radius:4px;")
        root.addWidget(banner)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); root.addWidget(scroll)
        content = QWidget(); layout = QVBoxLayout(content); scroll.setWidget(content)

        title = QLabel(f"Run #{run_id}" if run_id else "Current run")
        title.setStyleSheet("font-size:20px;font-weight:bold;"); layout.addWidget(title)
        info = QFormLayout()
        info.addRow("Model", QLabel(str(capture.metadata.get("model", MODEL_NAME))))
        info.addRow("Backend / GPU", QLabel(str(capture.metadata.get("backend", "unknown"))))
        info.addRow("Capture scope", QLabel("prompt prefill + first generated token"))
        info.addRow("Captured layers", QLabel(str(len(capture.layers))))
        info.addRow("Run duration", QLabel(str(capture.metadata.get("duration_ms", "not supplied"))))
        layout.addLayout(info)

        layout.addWidget(self._section("1. Prompt tokenization", "Token IDs: " + str(capture.prompt_token_ids) + "\nTokens: " + repr(capture.prompt_tokens)))
        layout.addWidget(self._section("2. Prompt-prefill embedding output", f"Shape: {capture.embedding.shape}\nFirst values:\n{sample_text(capture.embedding)}"))
        layout.addWidget(self._section("3. First generated-token embedding output", f"Shape: {capture.generated_embedding.shape}\nFirst values:\n{sample_text(capture.generated_embedding)}"))
        layout.addWidget(self._section("4. Forward-pass equations", "Q = XWq    K = XWk    V = XWv\nAttention = softmax(QKᵀ / √dₖ)\nOutput = Attention × V\n\nAll values below are emitted by the runner after these GPU operations."))

        for phase_title, phase_layers in (("Prompt prefill forward pass", capture.layers), ("First generated-token forward pass", capture.generated_layers)):
            phase_heading = QLabel(phase_title); phase_heading.setStyleSheet("font-size:18px;font-weight:bold;margin-top:12px;"); layout.addWidget(phase_heading)
            for index in sorted(phase_layers):
                layer = phase_layers[index]
                box = QFrame(); box.setFrameShape(QFrame.StyledPanel); box_layout = QVBoxLayout(box)
                heading = QLabel(f"Layer {index}"); heading.setStyleSheet("font-size:16px;font-weight:bold;"); box_layout.addWidget(heading)
                mapping = [("normalized_input", "X (normalized input)"), ("q", "Q = XWq"), ("k", "K = XWk"), ("v", "V = XWv"), ("attention_scores", "QKᵀ / √dₖ"), ("attention_weights", "softmax(scores)"), ("attention_output", "Attention × V"), ("layer_output", "Layer output")]
                for key, label in mapping:
                    tensor = layer.tensors[key]
                    text = QLabel(f"<b>{label}</b> — shape {tensor.shape}<br><pre>{sample_text(tensor)}</pre>")
                    text.setTextFormat(Qt.RichText); text.setWordWrap(True); box_layout.addWidget(text)
                box_layout.addWidget(AttentionCanvas(layer.tensors["attention_weights"], layer.tensors["attention_weights"].shape, box))
                layout.addWidget(box)
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
        refresh.clicked.connect(self.reload); open_button.clicked.connect(self.open_selected); self.runs.itemDoubleClicked.connect(lambda _: self.open_selected()); self.reload()

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
    """Primary application UI and coordination point for runner, storage, and recap windows."""
    def __init__(self) -> None:
        super().__init__(); self.database = RunDatabase(); self.worker: GenerationWorker | None = None
        self.setWindowTitle("TensorScope"); self.resize(980, 700)
        central = QWidget(); self.setCentralWidget(central); layout = QVBoxLayout(central)
        title = QLabel("TensorScope"); title.setStyleSheet("font-size:27px;font-weight:bold;"); layout.addWidget(title)
        subtitle = QLabel("Local gpt-oss:20b inference with exact, instrumented runtime captures")
        subtitle.setStyleSheet("color:#475569;"); layout.addWidget(subtitle)
        settings = QHBoxLayout(); settings.addWidget(QLabel("Capture runner URL:"))
        self.endpoint = QLineEdit(os.getenv("TENSORSCOPE_RUNNER_URL", DEFAULT_ENDPOINT)); settings.addWidget(self.endpoint)
        self.check_button = QPushButton("Check runner"); settings.addWidget(self.check_button); layout.addLayout(settings)
        layout.addWidget(QLabel("Prompt")); self.prompt = QTextEdit(); self.prompt.setPlaceholderText("Enter a prompt for gpt-oss:20b..."); self.prompt.setFixedHeight(125); layout.addWidget(self.prompt)
        buttons = QHBoxLayout(); self.run_button = QPushButton("Run Model"); self.history_button = QPushButton("View Saved Runs")
        buttons.addWidget(self.run_button); buttons.addWidget(self.history_button); buttons.addStretch(1); layout.addLayout(buttons)
        layout.addWidget(QLabel("Model answer")); self.output = QTextEdit(); self.output.setReadOnly(True); layout.addWidget(self.output)
        self.status = QLabel("Configure the instrumented local runner, then check it."); self.status.setStyleSheet("color:#334155;"); layout.addWidget(self.status)
        self.run_button.clicked.connect(self.run_model); self.history_button.clicked.connect(lambda: HistoryDialog(self.database, self).exec_()); self.check_button.clicked.connect(self.check_runner)

    def check_runner(self) -> None:
        """Perform a harmless endpoint reachability check without attempting a model run."""
        try:
            response = requests.get(self.endpoint.text().rsplit("/", 1)[0] + "/health", timeout=2)
            if response.ok:
                self.status.setText("Instrumented runner is reachable. Ready to run."); self.status.setStyleSheet("color:#166534;"); return
        except requests.RequestException:
            pass
        self.status.setText("Runner unavailable. Standard Ollama is not enough for exact tensor capture."); self.status.setStyleSheet("color:#b91c1c;")

    def run_model(self) -> None:
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.information(self, "Prompt required", "Enter a prompt before running the model."); return
        self.output.clear(); self.run_button.setEnabled(False); self.status.setText("Generating and receiving GPU capture records..."); self.status.setStyleSheet("color:#1d4ed8;")
        self.worker = GenerationWorker(prompt, self.endpoint.text().strip()); self.worker.text_received.connect(self.output.insertPlainText)
        self.worker.finished_capture.connect(self.persist_and_show); self.worker.failed.connect(self.run_failed); self.worker.start()

    def persist_and_show(self, capture: RunCapture) -> None:
        try:
            run_id = self.database.save(capture)
            self.status.setText(f"Saved exact capture as run #{run_id}."); self.status.setStyleSheet("color:#166534;")
            ComputationRecap(capture, run_id, self).exec_()
        except Exception as exc:
            self.status.setText("Generation completed but capture could not be saved."); self.status.setStyleSheet("color:#b91c1c;")
            QMessageBox.critical(self, "Database error", str(exc))
        finally:
            self.run_button.setEnabled(True)

    def run_failed(self, message: str) -> None:
        self.run_button.setEnabled(True); self.status.setText("Run failed: exact capture was not available."); self.status.setStyleSheet("color:#b91c1c;")
        QMessageBox.critical(self, "Instrumented capture required", message)


def self_test() -> None:
    """Small, dependency-free checks for persistence and real-array visualization helpers."""
    import tempfile
    layer_data = {name: np.arange(36, dtype=np.float32).reshape(1, 6, 6) for name in REQUIRED_LAYER_TENSORS}
    capture = RunCapture(prompt="test", response="answer", token_ids=[1], tokens=["answer"],
                         prompt_token_ids=[0], prompt_tokens=["test"], embedding=np.arange(8, dtype=np.float16),
                         generated_embedding=np.arange(8, dtype=np.float16), metadata={"capture_source": "instrumented-runtime"},
                         layers={0: LayerCapture(0, layer_data)}, generated_layers={0: LayerCapture(0, layer_data)})
    with tempfile.TemporaryDirectory() as temporary:
        database = RunDatabase(Path(temporary) / "test.sqlite3"); run_id = database.save(capture); restored = database.load(run_id)
        assert np.array_equal(restored.embedding, capture.embedding)
        assert np.array_equal(restored.layers[0].tensors["q"], layer_data["q"])
        assert numeric_sample(layer_data["q"]).shape == (5, 5)
        assert display_matrix(np.ones((300, 300))).shape[0] <= 96
    print("TensorScope self-test passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        application = QApplication(sys.argv)
        window = TensorScopeMainWindow(); window.show()
        sys.exit(application.exec_())
