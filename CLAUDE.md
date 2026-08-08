# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TensorScope inspects the *exact* intermediate tensors produced during a real local LLM
forward pass (model `gpt-oss:20b`). It has two halves that talk over HTTP/JSONL:

- **`TensorScope.py`** — a PyQt5 desktop app (client). Sends a prompt, receives streamed
  capture records, validates them, persists them to SQLite, and renders a read-only
  "Computation Recap" (tokenization, embeddings, per-layer Q/K/V, attention scores/weights,
  attention output, layer output, plus an attention-weight heatmap).
- **`instrumented-llama.cpp/`** — a fork of llama.cpp. The added
  `examples/tensorscope-runner/` builds `tensorscope-capture-runner`, a single-purpose
  local HTTP server that runs inference and streams the real GPU tensors back.

The design invariant that shapes almost everything: **no tensor is ever inferred,
simulated, or reconstructed in Python.** Every displayed value must come from the
instrumented runtime. The client aggressively validates this and rejects captures
(and plain Ollama endpoints) that can't prove it.

## Commands

### Python client
```bash
python TensorScope.py            # launch the GUI
python TensorScope.py --self-test  # dependency-light checks for persistence + display helpers
```
Dependencies (no requirements file at repo root — install manually): `numpy`, `requests`,
`matplotlib`, `PyQt5`. There is no lint/test framework beyond `--self-test`.

Configure the runner endpoint via env var `TENSORSCOPE_RUNNER_URL`
(default `http://127.0.0.1:11435/v1/capture/generate`) or the URL field in the GUI.

### C++ capture runner
Built as part of the llama.cpp CMake tree (it's registered in
`instrumented-llama.cpp/examples/CMakeLists.txt`):
```bash
cd instrumented-llama.cpp
cmake -B build -DGGML_CUDA=ON        # CUDA build; the runner is designed for GPU capture
cmake --build build --target tensorscope-capture-runner -j

./build/bin/tensorscope-capture-runner --model /path/to/gpt-oss-20b.gguf \
    --port 11435 --context-size 2048 --max-tokens 128
```
For general llama.cpp build/lint/test guidance, see `instrumented-llama.cpp/CLAUDE.md`
and `instrumented-llama.cpp/AGENTS.md`.

## Capture protocol (the contract between the two halves)

The runner replies to `POST /v1/capture/generate` with newline-delimited JSON. Health check
is `GET /v1/capture/health`. Record `type`s the client consumes (`_consume_record` in
`TensorScope.py`, produced in `runner::generate` in `tensorscope-runner.cpp`):

- `metadata` — must include `capture_source: "instrumented-runtime"` or the client rejects it.
- `tokenization` — prompt `ids` + `tokens` (lengths must match).
- `token` — one generated token (`id`, `text`); appended to the response.
- `tensor` — `name`, optional `layer`, `phase` (`prefill` | `first_generated_token`), and
  `data`. Tensor encodings: JSON list, `npy-zlib-base64`, or `raw-f32-base64`.
- `complete` / `error`.

**Capture scope is fixed**: prompt prefill + the first generated token only. Both phases must
cover the same layers, and every layer must carry all of `REQUIRED_LAYER_TENSORS`
(`normalized_input, q, k, v, attention_scores, attention_weights, attention_output,
layer_output`) plus an `embedding`. `RunCapture.validate()` enforces all of this; changing the
captured tensor set means editing **both** `REQUIRED_LAYER_TENSORS` (Python) and
`k_tensor_names` (C++) together.

## How the runner captures tensors

- It registers `capture_callback` as llama.cpp's `cb_eval` (`ggml` scheduler post-node hook).
  The callback is invoked *after* the assigned backend (incl. CUDA) computes each node, so
  values are real GPU outputs copied to host, converted to f32, and streamed as base64.
- Tensors are matched by graph-node name: llama.cpp's `cb(tensor, "<stem>", layer)` produces
  names like `q_projection-0`. `k_tensor_names` maps those stems to protocol names. The
  gpt-oss graph instrumentation that emits these names lives in
  `instrumented-llama.cpp/src/models/openai-moe.cpp` (e.g. `q_projection`, `k_projection`,
  `attn_norm`); `split_layer_name` parses the `-<layer>` suffix.
- **Flash Attention is deliberately disabled** (`LLAMA_FLASH_ATTN_TYPE_DISABLED`). A fused
  flash-attention kernel never materializes the full attention matrix, so it couldn't be
  captured. Do not "optimize" this back on.
- The runner is **single-request-at-a-time** (a `request_mutex`), clears KV memory before and
  after each request, and locks to `gpt-oss:20b`. Each capture is a complete, isolated forward
  pass — concurrency would corrupt captures.

## Client architecture notes

- `CaptureClient` speaks the protocol; `GenerationWorker` (a `QThread`) runs it off the Qt
  event loop and marshals text/results/errors back via signals.
- `RunDatabase` (SQLite, `tensorscope_runs.sqlite3`) stores runs + tensors. Generated-token
  tensors are stored under a `first_generated_token:` name prefix and the embedding at
  `layer_index = -1`; `load()` reverses this convention. Arrays are `np.save` + zlib blobs
  (`array_to_blob`/`blob_to_array`) — exact dtype is preserved, never downcast.
- `numeric_sample` / `display_matrix` sample tensors **for display only** (5×5 text preview,
  ≤96×96 heatmap). They must never mutate or replace persisted data.
