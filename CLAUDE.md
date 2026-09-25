# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TensorScope inspects the *exact* intermediate tensors produced during a real local LLM
forward pass, so a reader can watch an actual model answer an actual prompt. It is a
single PyQt5 desktop app:

- **`TensorScope.py`** — loads a Hugging Face causal LM **in-process**, runs it with a
  custom eager attention implementation plus forward hooks, persists every captured
  tensor to SQLite, and renders a read-only, two-mode "Computation Recap" (tokenization,
  embeddings, per-layer Q/K/V before and after RoPE, attention scores/weights, attention
  block output, layer output, plus an attention-weight heatmap).
- **`verify_capture.py`** — loads a real model and proves the capture is faithful and
  complete. This is the test suite that matters; `--self-test` only covers the
  torch-free helpers.

The design invariant that shapes everything: **no tensor is ever inferred, simulated, or
reconstructed.** Every displayed value must be a tensor the running model actually
produced. The client refuses to save or display a capture that cannot prove this.

### History: there is no llama.cpp any more

Earlier versions shipped `instrumented-llama.cpp/`, a fork whose
`tensorscope-capture-runner` streamed tensors over HTTP/JSONL. **That fork is deleted and
must not come back.** Requiring users to build a CUDA C++ project made the app
undistributable as open source. PyTorch forward hooks give the same guarantee — real GPU
tensors, no reconstruction — with `pip install -r requirements.txt`. If you find leftover
references to `TENSORSCOPE_RUNNER_URL`, `CaptureClient`, `/v1/capture/generate`,
`k_tensor_names`, or `gpt-oss:20b`, they are stale.

## Commands

```bash
python -m venv .venv                                        # see requirements.txt: MUST be a venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

./.venv/Scripts/python.exe TensorScope.py                   # launch the GUI
./.venv/Scripts/python.exe TensorScope.py --self-test       # torch-free: persistence + display helpers
./.venv/Scripts/python.exe verify_capture.py                # real model, real capture, full assertions
./.venv/Scripts/python.exe verify_capture.py Qwen/Qwen3-4B  # verify the default model too
```

`verify_capture.py` defaults to `Qwen/Qwen2.5-0.5B-Instruct` because it exercises the
mechanism cheaply. Run it against `Qwen/Qwen3-4B` as well before trusting a change: Qwen3
adds QK-norm and a head_dim decoupled from `hidden_size / num_heads`, which has caught
shape assumptions that the 0.5B model does not.

### Environment landmines (both cost real debugging time)

- **Never install torch into an Anaconda base env.** Conda's MKL and pip's torch both
  ship `libiomp5md.dll`, giving `OMP: Error #15`. The documented workaround
  (`KMP_DUPLICATE_LIB_OK=TRUE`) warns it "may silently produce incorrect results", which
  disqualifies it for a tool whose entire purpose is exact numerics. Use a venv.
- **Blackwell / sm_120 (RTX 50-series) needs cu128 wheels.** Default PyPI torch has no
  kernels for it. `pip install --index-url https://download.pytorch.org/whl/cu128 torch`.

## How capture works

`ModelCapture` (in `TensorScope.py`) installs two mechanisms:

1. **A registered attention backend.** `CaptureAttention` mirrors the model file's own
   `eager_attention_forward` line for line and keeps the pre-softmax `scores` tensor it
   hands to softmax. Scores are not recoverable from hooks — nothing exposes them — so
   mirroring the function is the only way, and the mirror is checked (below).
2. **Forward hooks** on `embed_tokens`, `input_layernorm`, `q_proj`/`k_proj`/`v_proj`,
   `o_proj`, and the decoder layer itself, for the tensors that *are* module outputs.
   A `register_forward_pre_hook` on each `self_attn` points `CaptureAttention.sink` at
   that layer's slot before the module runs.

`attn_implementation="eager"` is mandatory. Fused/flash/SDPA kernels never materialize the
full attention matrix, so it cannot be captured — the same reason the old llama.cpp runner
disabled Flash Attention. Do not "optimize" this.

### Registering the backend: the trap that silently corrupts every capture

Register with `.register(name, fn)`, **never** `registry[name] = fn`, and register in
*both* registries:

```python
ALL_ATTENTION_FUNCTIONS.register(name, self.attention)          # transformers.modeling_utils
ALL_MASK_ATTENTION_FUNCTIONS.register(name, masks["eager"])     # transformers.masking_utils
```

Why: `GeneralInterface.__setitem__` writes to a per-instance `_local_mapping`, while
`masking_utils._preprocess_mask_arguments` tests membership against the class-wide
`_global_mapping` and treats a miss as "custom backend that needs no mask" — returning
`None`. A `None` mask makes the eager path skip masking entirely, so **attention becomes
bidirectional**: the model reads the future, every tensor still looks plausible, and every
value is genuinely computed, so nothing looks wrong. This bug produced `<|im_end|>` as the
answer to "What is the capital of France?". `_register_implementation` now asserts the name
reached `_global_mapping` and raises if it did not.

### The three verification gates

Any one of these failing makes `RunCapture.validate()` reject the run, so a corrupt
capture can never be saved or displayed:

| Gate | Metadata key | What it catches |
| --- | --- | --- |
| `capture_source` | `capture_source == "pytorch-forward-hooks"` | tensors from anywhere but the real capture path |
| Attention mirrors upstream, **on every captured call** | `faithful_to_upstream_eager`, `attention_calls_verified` | a transformers upgrade changing the eager math |
| Prefill logits identical to stock eager, bit for bit | `logits_match_stock_eager`, `logits_max_abs_diff_vs_stock_eager == 0.0` | capture perturbing the computation at all |

The last gate is the one that actually matters, and it is why the bidirectional-mask bug
was caught rather than shipped. Faithfulness alone is not sufficient: a perfect mirror of
the eager math fed a wrongly-built mask is still wrong, and it reported
`faithful: True` while the answer was garbage. `_verify_undisturbed` re-runs the same
prefill under stock `"eager"` and requires `delta == 0.0` exactly — not a tolerance.

Verification must also *cover* the run. `faithful` originally checked only layer 0's first
call and gave a false pass; `CaptureAttention` now re-runs upstream's implementation on
every call while a sink is attached, i.e. every layer of both captured phases.

## Capture scope and tensor set

Fixed scope: **prompt prefill + the first generated token.** Remaining tokens are
generated with capture off, purely so the reader sees a complete answer.

`REQUIRED_LAYER_TENSORS`: `normalized_input, q, k, v, q_attended, k_attended, v_attended,
attention_scores, attention_weights, attention_output, layer_output` — plus an
`embedding`. Both phases must cover the same layers and every layer must carry all of
them; `RunCapture.validate()` enforces it.

Two labeling distinctions the UI must keep honest, both of which were wrong once:

- `q`/`k`/`v` are raw projection outputs, **before RoPE** (and before QK-norm). The
  tensors that actually enter QKᵀ are `q_attended`/`k_attended`/`v_attended`, captured
  inside the attention call, post-RoPE and post-`repeat_kv` (so GQA is already expanded).
- `attention_output` is `o_proj`'s output — `W_o · concat(heads)`, the attention block's
  output. It is **not** "attention × V". Calling it that was a real mislabel.

If you change the captured set, update `REQUIRED_LAYER_TENSORS`, `TENSOR_LABELS` **and
`TENSOR_EXPLANATIONS`** together. `--self-test` asserts every required tensor has a label,
and asserts `TENSOR_EXPLANATIONS` and `TENSOR_LABELS` have *exactly* the same keys in both
directions — so a new tensor cannot reach the screen unexplained, and copy for a deleted
tensor cannot linger.

Beyond the per-layer set, `RunCapture.logits` holds the prefill logits row for the final
prompt position — the vector that chose the first generated token. It is stored, not
derived, and `validate()` requires `logits.argmax() == token_ids[0]` when present so a real
score list can never be shown next to the wrong word. It is optional purely for
back-compatibility: runs saved under schema 2 have none, and the recap degrades to a notice
rather than refusing them.

`MAX_CAPTURE_TOKENS = 256` bounds the prompt, because attention tensors grow with the
square of prompt length.

## Model support

Default is `DEFAULT_MODEL_ID = "Qwen/Qwen3-4B"`, configurable in the GUI. Requirements for
any model you point it at:

- It must answer real prompts coherently — the point is showing how an LLM *works*, and a
  tiny model that emits nonsense teaches nothing.
- It must be **dense, not MoE.** `REQUIRED_LAYER_TENSORS` describes a dense
  attention-then-MLP block; MoE routes through a router and experts between
  `attention_output` and `layer_output`, which the tensor set does not model. This is why
  the old `gpt-oss:20b` target is gone.
- The model file must expose `repeat_kv` and `eager_attention_forward` at module level;
  `CaptureAttention.__init__` raises `CaptureProtocolError` if not.

Measured VRAM: Qwen2.5-0.5B ≈ 1.0 GiB, Qwen3-4B ≈ 7.6 GiB peak (bf16, 36 layers, all
attention matrices retained). An 8B model in bf16 (~16.4 GB) does not fit 15.9 GB.

## Client architecture notes

- `ModelCapture` does the loading and capture. `LoadWorker` and `GenerationWorker`
  (`QThread`) keep both off the Qt event loop and marshal progress/text/results/errors
  back via signals.
- `RunDatabase` (SQLite, `tensorscope_runs.sqlite3`) stores runs + tensors.
  Generated-token tensors are stored under a `first_generated_token:` name prefix and the
  embedding at `layer_index = -1`; `load()` reverses this convention. `final_logits` also
  rides at `layer_index = -1`, so **its branch in `load()` must precede the by-layer
  `else`** — falling through invents a `layers[-1]` that `validate()` then rejects for
  missing every required tensor, breaking every saved run. Arrays are `np.save`
  + zlib blobs (`array_to_blob`/`blob_to_array`) — dtype is preserved exactly, never
  downcast. `verify_capture.py` asserts a bit-exact round-trip.
- `tensor_to_numpy` widens bf16 to float32 because numpy has no bfloat16 dtype. This is a
  *widening* cast: lossless, and not the forbidden downcast. Everything else passes
  through untouched.
- `numeric_sample` / `display_matrix` sample tensors **for display only** (5×5 text
  preview, ≤96×96 heatmap). They must never mutate or replace persisted data.

### The recap is four modules, and the layering is the point

```
tensorscope_content.py   pure copy + data; imports nothing of ours
tensorscope_common.py    presentation primitives (no journey logic)
tensorscope_views.py     the three journeys
tensorscope_ui.py        ComputationRecap, the shell
TensorScope.py           capture, persistence, app; imports the shell
```

`tensorscope_common` exists to break an import cycle: `MiniHeatmap` needs
`display_matrix` from `TensorScope`, so it imports it *inside* `__init__` — a module-level
import would run while `TensorScope` is still half-initialised. Don't move that import out.

- `ComputationRecap` (`tensorscope_ui`) owns the window, the palette, the banner, the
  provenance evidence card and the navigation. It knows nothing about tensors. Each view
  exposes the same small contract — a `NAV` list of `(stage key, label)`, `go_to(key)`,
  `current_stage()`, and the `stage_changed` / `context_changed` signals — and the shell
  drives its navigator, sidebar, breadcrumb and footer from those alone, so adding a mode
  is a row in `MODES`, not a layout change. Views other than the default are built on
  first visit.
- Three modes, in the order a reader should meet them. **Understand** (`LearnView`, the
  default) asks five questions a person actually asks — prompt & response, tokens,
  building context, choosing each token, limits — and puts the explanation *above* the
  numbers. **Internals** (`InternalsView`) walks the same capture by architecture:
  embedding, one decoder layer's six steps, vocabulary scores, the next pass. **Raw
  tensors** (`RawView`) is phase/layer/tensor pickers over a full `TensorInspector`, so
  retiring the old exhaustive view lost nothing — every array stays reachable in two
  clicks, unframed.
- `InternalsView` follows one fixed disclosure ladder per layer step: purpose → diagram →
  equation → dimensions → tensor. The array is the *last* rung, never the first thing on
  screen. Level 2 (`STEP_FLOW`) is deliberately numberless; level 4 reads its dimensions
  off the captured arrays rather than the config, so a shape on screen is always this
  run's.
- Copy never lives inline in a widget. `TENSOR_LABELS`, `TENSOR_EXPLANATIONS`,
  `LEARN_STAGES`, `INTERNALS_STAGES`, `INTERNALS_STEPS`, `WHAT_WE_KNOW`,
  `WHAT_WE_CANNOT_CONCLUDE` and `STOP_REASONS` are all in `tensorscope_content`.
  `--self-test` formats every stage's copy against `learn_facts()`, so a stage naming a
  field that function does not supply fails the test instead of raising `KeyError` in front
  of a reader.
- Every number on screen carries an evidence tier (`EVIDENCE_OBSERVED`,
  `EVIDENCE_UNATTESTED`, `EVIDENCE_DERIVED`, `EVIDENCE_CONCEPTUAL`). Four tiers, not
  three: candidate scores for generated tokens after the first are real forward-pass
  output that no verification gate re-derived, and they must not look identical to a tensor
  that survives all three gates. Anything TensorScope itself computed for the screen —
  a ranking, a margin, an argmax — is `EVIDENCE_DERIVED`, and anything the architecture has
  but this capture did not keep (A@V, the residual sum, the MLP interior, the learned weight
  matrices) is described and badged `EVIDENCE_CONCEPTUAL` rather than reconstructed.
- `Disclosure` builds its body only when toggled, which is what keeps a 36-layer model from
  constructing 70+ matplotlib canvases up front — and is why `test_recap.py` clicks every
  one of them open. `test_every_view_stage_builds_from_real_captures_without_mutating_them`
  walks all three views over every saved run (both phases, every layer, all six layer
  steps, every generated token) and re-hashes the capture afterwards; a lazily-built stage
  nobody visits would otherwise carry a stale attribute name indefinitely, which is exactly
  how the previous rework broke.

### On the softmax cross-check

`softmax(captured_scores)` matches `captured_weights` to ~2e-3, not exactly, and the
tolerance is `4e-3`. That is not slop: upstream computes softmax in float32 then casts
back to the model dtype, and bf16 has an 8-bit mantissa, so the relative step is
2⁻⁸ ≈ 0.0039. The measured disagreement (1.9e-3) sits right where that predicts. Don't
"fix" it by loosening further or by recomputing in float32 — the captured weights are the
ones the model used.
