# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TensorScope inspects the *exact* intermediate tensors produced during a real local LLM
forward pass, so a reader can watch an actual model answer an actual prompt. It is a
single PyQt5 desktop app:

- **`TensorScope.py`** — loads a Hugging Face causal LM **in-process**, runs it with a
  custom eager attention implementation plus forward hooks, persists every captured
  tensor to SQLite, and renders a read-only two-page "Computation Recap": one scrollable
  worked calculation over the capture (tokenization, embeddings, per-layer Q/K/V before and
  after RoPE, attention scores/weights, attention block output, layer output, the token
  decision and the continuation), plus an illustrated reference in the same order and
  notation.
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
./.venv/Scripts/python.exe TensorScope.py --self-test        # torch-free: copy, parity, display helpers
./.venv/Scripts/python.exe test_recap.py                    # the recap, over the real saved runs
./.venv/Scripts/python.exe test_threading.py                # where torch is imported, and on which thread
./.venv/Scripts/python.exe tensorscope_diagrams.py          # the diagram gallery, both themes
./.venv/Scripts/python.exe verify_capture.py                # real model, real capture, full assertions
./.venv/Scripts/python.exe verify_capture.py Qwen/Qwen3-4B  # verify the default model too
./.venv/Scripts/python.exe verify_capture.py --reload       # load twice: unload() frees VRAM, gates still pass
```

`verify_capture.py` defaults to `Qwen/Qwen2.5-0.5B-Instruct` because it exercises the
mechanism cheaply. Run it against `Qwen/Qwen3-4B` as well before trusting a change: Qwen3
adds QK-norm and a head_dim decoupled from `hidden_size / num_heads`, which has caught
shape assumptions that the 0.5B model does not.

`test_recap.py` is the one to run after any UI change: 53 tests over every saved run, no
torch, no GPU, about three minutes. It takes an optional database path first, then
test names — `test_recap.py SavedCaptureTests.test_token_span_selection_matches_the_recorded_candidates`
runs one. (It only treats the first argument as a database when it looks like one, so
passing a bare test name works.)

`test_threading.py` is separate from it on purpose, and `test_recap.py` must never import
it. The recap suite's hard property is that the recap reads with no model library present
at all — it installs a meta path finder that fails the run the moment `torch` or
`transformers` is imported. The threading tests need the opposite: *stub* torch and
transformers modules in `sys.modules`, so they can watch which thread calls into them.
Twelve tests, a second, no GPU.

The diagram gallery is a visual check, not a test: it lays every QPainter widget out in both
token sets so they can be eyeballed without loading a model or opening a capture.

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

If you change the captured set, four tables in `tensorscope_content` move together:
`TENSOR_LABELS`, `TENSOR_EXPLANATIONS`, `TENSOR_SYMBOLS` (the symbol the tensor is written
as) and `VARIABLE_ROLES` (its one-line purpose, which is what the variable panel shows at a
glance). `--self-test` asserts `REQUIRED_LAYER_TENSORS ⊆ TENSOR_SYMBOLS`, and asserts
exact key parity in *both* directions for `TENSOR_LABELS`↔`TENSOR_EXPLANATIONS` and
`TENSOR_SYMBOLS`↔`VARIABLE_ROLES` — so a new tensor cannot reach a variable chip unnamed or
a panel unexplained, and copy for a deleted tensor cannot linger.

`TENSOR_SYMBOLS` carries one key that is not a stored tensor name: `layer_input`, written
`X`. A layer's input really is captured, just addressed differently — it is the embedding
output at layer 0 and the previous layer's own output at every later layer — and
`VariableRegistry._captured_array` resolves it and says which of the two it handed you.
That is reading a capture, not reconstructing one, and it is the only indirection of its
kind.

Beyond the per-layer set, `RunCapture.logits` holds the prefill logits row for the final
prompt position — the vector that chose the first generated token. It is stored, not
derived, and `validate()` requires `logits.argmax() == token_ids[0]` when present so a real
score list can never be shown next to the wrong word. It is optional purely for
back-compatibility: runs saved under schema 2 have none, and the recap degrades to a notice
rather than refusing them.

`MAX_CAPTURE_TOKENS = 256` bounds the prompt, because attention tensors grow with the
square of prompt length.

### The shipped database has three capability tiers, and the tests need all three

`tensorscope_runs.sqlite3` is a test fixture as much as a history. Every piece of copy and
every section has to be true for the thinnest run in it, and `test_recap.py` selects those
runs **by capability rather than by id**, asserting each group is non-empty — so a database
that lost its old runs fails loudly instead of quietly dropping the degradation paths.

| Tier | Has | Lacks | What it exercises |
| --- | --- | --- | --- |
| schema 2 | all 36 layers × 11 tensors, both phases | `logits`, `generation_trace`, `stop_reason` | `LOGITS_UNAVAILABLE`, `TELEMETRY_UNAVAILABLE`, the unknown stop reason, an unrecorded vocabulary size |
| schema 3 | `logits` | `generation_trace`, `stop_reason` | per-token telemetry missing while the first token's vector is present |
| schema 4 | everything | — | the full path, including the EOS stopping decision |

One more thing that database is good for: its runs are not all the same size (14, 15 and 16
prompt positions; 48, 75 and 81 emitted tokens), which is what catches a count hardcoded to
the run someone happened to develop against. One response also ends in replacement
characters from a truncated multi-byte sequence, so it exercises rendering those as text.

Two degradation paths the saved runs do **not** reach, so don't assume they are covered:

- `ResponseTextView`'s unfaithful-spellings fallback to `TokenChipStrip`. All six runs
  reassemble correctly, so the branch is reachable in principle and untested in practice.
  `test_token_span_selection_matches_the_recorded_candidates` asserts `is_faithful()`
  against an independently computed comparison rather than assuming the faithful branch, so
  it will follow whichever is real — but today that is always the faithful one.
- `PROMPT_SPAN_UNKNOWN`. `prompt_token_bands` resolves for all six runs, so the "could not
  line your typed text up" notice never appears. Cover either of these with a synthetic
  fixture in `--self-test` if you touch them.

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

- `ModelCapture` does the loading and capture. One `ModelWorker` (a `QObject` moved onto
  a single long-lived `QThread`) keeps all of it off the Qt event loop and marshals
  progress/text/results/errors back via signals. See "Torch and threads" below before
  changing anything about it — both halves of that arrangement exist to stop a native
  crash, and the application appears to work without either.
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
  preview, ≤96×96 heatmap) and return *views*, so they must never be written through and
  must never replace persisted data. Both are now vestigial — see
  "Vestigial after the two-page redesign" at the end of this file.

### Crashes: the three that looked identical, and the rules that stop them

Pressing **Run Capture** used to kill the process with a Windows access violation — Git
Bash prints it as `Segmentation fault`. There is no Python traceback, so the
`except` handlers in the worker never see it, and the GUI prints nothing to the console,
so the last thing on screen is tqdm's `Loading weights: 100%` regardless of where it
actually died.

What the diagnosis found: driving `ModelCapture.load()` → `.generate()` → `validate()`
headlessly on the main thread works every time, but the same stack driven through the
real window crashed with the faulting thread **inside the process's first `import torch`,
running on a worker `QThread`** (`torch/_strobelight/compile_time_profiler.py`, under
`importlib._bootstrap`) while the Qt event loop pumped Windows messages. It is a race,
not a certainty — observed once in twelve runs, on the coldest start, where the import
spends longest reading and compiling bytecode. **A green run proves nothing here**, which
is why the two rules below are rules rather than something to re-measure.

One red herring, recorded so nobody chases it twice: faulthandler also reports
`code 0x8001010d` (`RPC_E_CANTCALLOUT_ININPUTSYNCCALL`) on every launch. It looks like a
smoking gun and is not — it appears in `--demo` browse-only runs that never import torch
at all, on the main thread, at `application.exec_()`. It is a benign first-chance Qt/COM
exception. The crash-log banner says so, because the entry worth reading is
`access violation`.

Two rules follow, and both are tested in `test_threading.py`:

1. **`preimport_model_stack()` runs on the GUI thread**, from `_start_load`, before any
   worker is started. It is *not* a module-scope import: browse-only mode must still open
   saved runs on a machine with no torch installed (the `find_spec("torch")` probe in
   `__main__`), and `test_recap.ForbidModelImports` fails the moment `TensorScope`
   imports torch at module scope. It warms `torch`, `transformers`,
   `transformers.modeling_utils`, `transformers.masking_utils` **and** the auto classes —
   transformers 5.x is a `_LazyModule`, so importing the package alone would leave the
   `tokenizers` Rust extension and friends for whoever first touched the names, i.e. the
   worker. The cost is a bounded one-off GUI freeze while the import blocks, labelled in
   the status line and the title bar. That is the trade; do not move the import back onto
   a worker to make the window feel faster.
2. **One thread owns every torch call for the session.** The old `LoadWorker` created the
   CUDA context and then *exited*, leaving every later kernel to run on `GenerationWorker`
   — formally legal, since a CUDA primary context is per-process, but a known Windows
   crash source, and it best explains a fault arriving *after* the weights finished
   loading. `ModelWorker` does load, generate and unload on one thread. Nothing on it may
   be called directly from the GUI thread: a direct Python call runs on the caller's
   thread and silently undoes the whole point, so every entry point is a slot reached
   only through a queued signal. Its handlers catch `BaseException`, not `Exception` —
   anything escaping a slot invoked from a QThread's event loop makes PyQt abort the
   process with no traceback, which is indistinguishable from the crash above.

Supporting pieces:

- **`faulthandler` writes to `tensorscope-crash.log`** beside the run database, enabled
  first thing in `__main__`'s GUI branch with `all_threads=True`. It must be a file and
  not stderr: `build/tensorscope.spec` sets `console=False`, so `sys.stderr` is `None` in
  a packaged build. `TENSORSCOPE_CRASH_LOG` overrides the path, or disables it with `0`.
  It is deliberately not at module scope — `test_recap.py`, `verify_capture.py` and
  `--self-test` all import `TensorScope`, and none of them should drop a log file next to
  the hashed database fixture. It earned its keep immediately: it is what identified the
  `_on_scrolled` crash above, from two lines in a user's log.
- **`sys.excepthook` writes to the same file, and faulthandler does not cover what it
  catches.** PyQt responds to an unhandled Python exception raised *inside a slot* by
  calling Qt's `abort()`. That is not a signal, so faulthandler never sees it, and the
  process dies with no traceback and an exit code a user cannot tell apart from a native
  fault — **a plain `KeyError` in a navigation slot reads to them as
  "Segmentation fault"**. Measured: with the default hook, exit 127 and not one line of
  output; with a hook installed, the traceback is recorded *and the application survives*.
  So the hook is both the diagnostic and the containment. It does not hide integrity
  failures: `validate()` is called inside `ComputationRecap.__init__`, which both
  `persist_and_show` and `_open_selected_run` already wrap in `try/except` with a dialog,
  so a capture that cannot prove itself still refuses loudly.
- **`clear_layout` (`tensorscope_common`) is the only sanctioned way to empty a layout.**
  `takeAt` removes the layout *item*; the widget keeps its parent, its geometry and its
  visibility, and `deleteLater` changes none of that until the event loop reaches the
  deferred delete. Until then the old widgets are painted over the new ones, returned by
  `findChildren`, and **clickable** — and clicking a stale navigator chip calls `go_to`
  with a key the current page does not have, which raises inside a slot, which is the
  abort above. `PipelineNavigator.set_stages` and the shell's `_rebuild_stage_buttons` had
  exactly that bug: the pipeline chips accumulated 13 → 25 → 38 across two mode switches,
  all still reporting `isVisible()`. `replace_in` already had the right idiom
  (`setParent(None)` first, binding the widget before the reparent, because
  `setParent(None)` can leave `item.widget()` returning `None`); all four call sites now
  share it. `_nav_to` is additionally guarded with `key in self.view.stage_keys()`,
  because a slot should not raise whatever reaches it.
- **`closeEvent` stops the thread** before Qt tears the window down under it, and
  `self._busy` (`"load"` / `"run"` / `None`) is the in-flight guard. `isRunning()` cannot
  serve: the thread is always running. Disabled buttons cannot serve either, because
  every `QMessageBox` in the file spins a nested event loop during which a queued click
  can still arrive. Closing mid-generation blocks for about five seconds on a 4B model
  while the current `generate()` finishes, and that is the intended behaviour: a
  `terminate()` on a thread inside CUDA leaves driver state undefined, which is a worse
  crash than none. Only if the 20-second `wait` expires does it `os._exit(0)`, so no
  destructor races a live worker.
- **`ModelCapture.unload()` is what makes a second load survivable.** `_install()` now
  retains every `register_forward_hook` handle, because the hook closures capture the
  `ModelCapture`, so `model → _forward_hooks → closure → self → model` is a cycle that
  nothing in this application collected; two Qwen3-4B copies do not fit on a 16 GB card.
  Removing the handles frees the weights on refcount zero. Measured with
  `verify_capture.py --reload`: allocated goes 7.55 GiB → 0.01 GiB for Qwen3-4B and
  0.94 GiB → 0.01 GiB for the 0.5B model, and the three gates still pass on the second
  load. Measure `memory_allocated()`, never `memory_reserved()` — the caching allocator
  keeps reserved high by design, so reserved would never show the drop.
  It deliberately **does not unregister the attention backend.** The invariant
  `_register_implementation` protects is that the name is in the class-wide
  `_global_mapping` whenever a capture runs, and `_install()` re-registers and re-asserts
  on every load. Clearing `attention.sink` leaves the retained `CaptureAttention` holding
  no tensors, and `GeneralInterface.register` replaces the value under the same key, so
  the next load drops the old object anyway. Do not "finish the job" by deleting the key:
  nothing unverified may ever be left sitting in that registry.

### The recap is ten modules, and the layering is the point

```
tensor_widgets.py          numpy + PyQt5 only; imports nothing of ours
tensorscope_content.py     pure copy + data; imports nothing of ours
tensorscope_diagrams.py    QPainter sketches; content + common, never TensorScope
tensorscope_common.py      presentation primitives (no page logic)
tensorscope_variables.py   the variable registry, chips, the one shared panel
tensorscope_operations.py  the body of each of the notebook's operations
tensorscope_notebook.py    NotebookView  — "Follow this run"
tensorscope_reference.py   ReferenceView — "Understand the calculation"
tensorscope_views.py       RawView + RawTensorDialog, and re-exports both pages
tensorscope_ui.py          ComputationRecap, the shell
TensorScope.py             capture, persistence, app; imports the shell
```

The import graph is a one-directional DAG and
`test_recap.ModuleLayeringTests` ast-parses the modules to enforce it, rather than trusting
this comment. Two rules it checks that are easy to break by accident:

- **No recap module may import `TensorScope` at module scope.** `MiniHeatmap` needs
  `display_matrix`, so it imports it *inside* `__init__`; a module-level import would run
  while `TensorScope` is still half-initialised. Don't move it out.
- **A diagram may not import `TensorScope` at all.** Diagram widgets take ints, strings and
  small numpy slices the caller already produced with basic indexing. A diagram that could
  reach a capture would be a diagram that could decide what a tensor means.

`tensorscope_common` exists to break a cycle rather than for tidiness: the colour and stage
constants are read by both the shell and the pages.

### Two pages, two bands of chrome

- `ComputationRecap` (`tensorscope_ui`) owns the window, the palette and two rows of chrome,
  and knows nothing about tensors. The **banner**: which run, the mode toggle, "✓ Validated
  capture" with a **Details…** button (the provenance card, in its own dialog), and
  **All captured arrays…**. The **reading bar**: the prompt, `ChapterProgress` (which
  chapter, "Operation 7 of 12"), the page's own toolbar control, a `ContentsButton` menu,
  and Previous / Next (also Alt+←/→). That replaced a pipeline of 13 chips, a sidebar list
  of the same 13, a breadcrumb, progress pills, a footer and a pinned ribbon of pickers —
  six bands and five ways of navigating one list, which left the page a quarter of a laptop
  screen. `test_the_shell_fits_a_laptop_screen` holds the line.
- The page contract the shell drives all of that from: `NAV`, `go_to(key)`,
  `current_stage()`, `stage_keys()`, `stage_changed` / `context_changed`, and optionally
  `CHAPTERS` (a list of `(title, section keys)`), `progress_text(key)` and
  `toolbar_widget()`. The shell may import only `tensorscope_common` and
  `tensorscope_views`, so chapters reach it through this contract and never by importing
  content.
- `MODES` has exactly two rows, **Follow this run** first: it is the page a reader opening
  a finished capture wants and the one that opens (`DEFAULT_MODE = 'run'`). **Understand
  the calculation** (`ReferenceView`) is twelve illustrated topics, each kicked with the
  operation it explains ("Explains operation 4"), because topic and operation numbers do
  not line up one to one.
- Every captured array is still reachable, through **All captured arrays…** (banner, and the
  variable panel's header), which opens `RawView` in a modeless `RawTensorDialog`. It is
  deliberately *not* a third `MODES` row.

### The notebook's teaching structure

The notebook is built to be followed by an engineering student who knows vectors and matrix
multiplication, and its structure is mechanical rather than editorial:

- **It opens on its destination.** The `input` section's `preface` is a card with the
  prompt, the start of the real answer (`response_excerpt`) with its first token marked, and
  one sentence saying the page follows the calculation that chose that token.
- **Four chapters** (`CHAPTERS` in content): Represent the input, Exchange information,
  Refine the representation, Select a token. They are read, not navigated: a kicker on the
  first section of each, the progress indicator, and the contents menu's groups. Nothing
  moves by chapter. `--self-test` asserts the four titles and that they cover the 13
  anchors in order.
- **Every operation reads in the same order**, and the `Operation` fields are that order:
  `lede` (purpose, one sentence, ≤260 characters by assertion) → `terms` (clickable
  formula) → `inputs` / `action` / `outputs` (`ctx.io_row`: IN · DOES · OUT) → a diagram →
  a few values → `note` (the one misconception worth heading off here, as a callout) →
  `takeaway` ("NEXT", how the result is used). The variable list, the optional check and
  heavy extras sit behind `Disclosure`s. Captions live in `NOTEBOOK_TEXT`; the ones that
  need a builder value (a weight, a count) declare it in `NOTEBOOK_TEXT_FIELDS`, and the
  self-test formats every caption against both fixtures with zeros for those.
- **One position is followed.** `default_focus_position` picks the last typed token with a
  letter or digit (the last word of the question, not its `?`), only when
  `prompt_token_bands` resolves; otherwise the last typed token, then the last position.
  The registry starts there, the `FocusChip` in the reading bar shows and changes it, and
  every diagram highlights that real row. `Diagram.row_window` is what makes "real row"
  true: the old rule clamped a selection onto the last drawn row, so position 13 of 15
  was painted on the row that stood for 14 and labelled "row 13".
- **The hand-off is explicit.** Operation 12 says the decision is made at the final prompt
  position and names both positions when they differ (`focus_shift`), the
  `VocabularyProjection` draws the followed row dashed beside the final one, and a button
  offers to follow the final position. It never moves the selection by itself.
- **Controls appear where they explain something.** `HeadPicker` in operations 6–8,
  `LayerPicker` in 11, `PhaseToggle` in the continuation. Several copies feed one registry,
  so each syncs from it with its own signals blocked, through a *bound method* so Qt
  disconnects it when the widget dies (a closure would outlive a slot rebuild and raise in
  a slot — the abort described above). `test_controls_appear_where_they_explain_something`.
- **The variable panel is closed until a variable is chosen**, has a close button and Esc,
  scrolls its own content, and carries the colour key (`RoleLegend`), which no longer pins
  a band above the page. Opening or closing it changes the page width, so it goes through
  `AnchoredScrollPage.preserve_place()`, which records (section, offset), sets
  `_scrolling`, settles the layout synchronously and restores the place.
- **The continuation has a timeline** (`PassTimeline`, buttons not a table — the
  candidate-table helpers in the tests read any `QTableWidget` with a "Token ID" column).
  One row per pass, from the run: pass 1 is the prefill (layer tensors, full vector when
  `logits` exists); pass 2 processed token 1 and chose token 2, and is the one later pass
  with layer tensors; later passes have leading candidates where `generation_trace` has
  them, "not recorded" otherwise; a stop row when the stopping token is known. Collapsed to
  the first two, the selected and the last.

### `go_to` scrolls; it does not swap

`AnchoredScrollPage` (`tensorscope_common`) is the base of both pages. Sections are built
once, in order, and navigating moves the reader to one. Nothing is destroyed, so selections,
expansions and reveals survive navigation **by construction** rather than by being saved and
replayed — `test_the_notebook_keeps_the_readers_place_across_navigation` asserts
`id(section)` is unchanged to prove it.

Three details are load-bearing:

- The shell no longer owns a `QScrollArea`; each page owns its own. The shell's two
  unconditional `scroll.verticalScrollBar().setValue(0)` calls are gone, replaced by
  `_reset_scroll()`, which returns early for any view offering `reset_scroll` — forcing a
  page to the top right after `go_to` would undo exactly what `go_to` just did.
  `_host` (which wraps a view in a scroll area) and the body of `_reset_scroll` are a
  safety net with no current caller, because both modes are `AnchoredScrollPage`s: they are
  there so that adding a plain tall view later does not silently become unscrollable.
- Inside `go_to`, the state change and `stage_changed` happen **first and completely**; the
  scroll and the flash are presentation afterwards. The scroll puts the section's first
  line at the top of the viewport (`_scroll_to`: the anchor's offset in the section plus
  the section's in the page, no `mapTo` at all) — `ensureWidgetVisible` only made it
  *visible*, so going forward the new title sat at the bottom edge. Freshly built sections
  have only posted their layout requests, so `go_to` delivers them first and re-applies the
  scroll once more on a zero timer (`_settle_scroll`, only if the reader has not moved). So tests assert
  `current_stage()` and never a pixel offset. `_on_scrolled` keeps the navigator in step
  with free scrolling and is guarded by `_scrolling` against the scroll `go_to` performs.
  That guard also made a crash invisible for a whole branch, so it is worth knowing how:
  **because `go_to` sets `_scrolling` around its own scroll, `_on_scrolled`'s body never
  ran during navigation — and navigation is all the tests did.** A reader's first wheel
  turn ran it, and it mapped the viewport into the page with
  `viewport.mapTo(scroll.widget(), …)`. The page is a **child** of the viewport, and
  `QWidget.mapTo` walks the parent chain *upwards*, so the walk never found the page, ran
  off the top of the hierarchy and dereferenced null. Opening a saved run and scrolling
  killed the process every single time, with no Python traceback — just
  `Segmentation fault`. The body now lives in `_track`, uses `page.mapFrom(viewport, …)`
  (the direction that is actually valid), and sets `_scrolling` around itself so that
  `ensure_built` growing the page mid-scroll cannot re-enter it.
  `test_free_scrolling_tracks_the_section_at_the_top` scrolls both pages by hand, which is
  the only thing that exercises any of this; a recurrence takes the interpreter down
  rather than failing an assertion, because an access violation cannot be caught.
- A section's header and lede are built up front (cheap text); its **body** builds on first
  arrival. Operation 11 is one summary table (a row per layer, for the followed position)
  plus a single shape table behind a `Disclosure` — it used to be 36 identical shape
  tables. `open_every_disclosure` still *loops* until no new ones appear, because
  disclosures can contain disclosures.

### Variables are the interactive objects

`VariableRegistry` (`tensorscope_variables`) holds the reader's phase, layer, head and
selected position and resolves any variable against them. A `VariableChip` holds a *key*,
not a tensor, so changing the selection repaints chips rather than rebuilding sections.
`VariablePanel` is a single docked instance, so clicking `X_norm` in operation 4 and in
operation 5 lands in exactly the same place.

Four provenances, mapping onto the evidence tiers: `captured` (the stored array exists),
`learned` (a weight matrix — the model has it, the capture kept only its output), `absent`
(an intermediate the model computed and discarded), `display` (something TensorScope
calculated for the screen). The last three have **no code path to a tensor widget**: the
panel builds no sample row and no inspector rung for them, so an uncaptured value cannot be
shown as data even in a section nobody reviewed. `test_the_panel_builds_no_tensor_widget_for_a_variable_with_no_array`
asserts it.

`display` is reserved and currently has no variables: every `CONCEPTUAL_VARIABLES` entry is
`learned` or `absent`, and `--self-test` asserts that. Numbers TensorScope worked out reach
the screen badged `EVIDENCE_DERIVED` — operation 8's worked scalar, the weight-row total,
the argmax, the top-two margin, the head-mean in operation 11 — rather than as a clickable
variable, because there is no array behind them for a panel to be about.

Values in the teaching view are rounded for reading (`approx`: "≈0.252"), and every value
box carries the exact stored digits (`exact_scalar`) as its tooltip; the panel's complete
array and `RawView` show exact digits only. `approx_share` turns an **attention weight**
into "≈25.2% of this head's mixing weight" — the only percentage anywhere, always beside
operation 7's note that attention weights are not next-token probabilities. Logits are
never shown as percentages. `test_rounded_values_carry_their_exact_value`.

`layer_input` (`X`) is the one indirection worth knowing: a layer's input *is* captured, just
under another name — the embedding output at layer 0, the previous layer's output after that.
The registry returns that array and says which one it is. That is reading a capture, not
reconstructing one.

Shapes written short always carry `shape_line`'s caption naming the stored shape, so the
batch axis can never be dropped quietly. One helper, used everywhere.

### Copy never lives inline in a widget

`TENSOR_LABELS`, `TENSOR_EXPLANATIONS`, `EXTRA_TENSOR_LABELS`/`EXTRA_TENSOR_EXPLANATIONS`,
`TENSOR_SYMBOLS`, `VARIABLE_ROLES`, `ROLE_LEGEND`, `CONCEPTUAL_VARIABLES`,
`REFERENCE_TOPICS`, `NOTEBOOK_OPERATIONS`, `CONTINUATION`, `CHAPTERS`, `NOTEBOOK_TEXT`,
`NOTEBOOK_CHECKS`, `ILLUSTRATIONS`, `STOP_REASONS`, `WHAT_WE_KNOW` and
`WHAT_WE_CANNOT_CONCLUDE` are all in `tensorscope_content`. `WHAT_WE_KNOW` lines carry the
capability they need (`logits`, `trace`, `decoding`) and are read through
`what_we_know(facts)`: a schema-2 run must not be told it has a score vector or a recorded
greedy rule. Likewise `selection_rule_sentence` has three forms — greedy recorded; schema 3,
where `validate()` still proves the stored vector peaks at the emitted token; and neither.
`--self-test` formats every one of them against `learn_facts()` for a full *and* a legacy
fixture, so a line naming a field that function does not supply fails the test instead of
raising `KeyError` in front of a reader.

Several of those assertions encode promises that are otherwise only editorial:

- `len(NOTEBOOK_OPERATIONS) == 12` while the page has thirteen anchors — the continuation is
  a section but not an operation, and stating both stops either drifting.
- Every topic's `operations` must cover `OPERATION_ORDER`, and the topics' first operations
  must not run backwards. That is how "the reference follows the same order as the worked
  computation" becomes mechanical.
- Invented content lives in an `Illustration`, which renders in a dashed `#illustrationBox`
  with a disclaimer. The words *noun* and *adjective* may appear **only** inside an
  `Illustration.body` — a cheap check that the Q/K analogy cannot leak into a sentence that
  reads as a fact about a head.
- `ffn_interior_note` must contain no digits: the feed-forward interior width is genuinely
  absent from a capture, and a plausible-looking number there would be invented.

### Four evidence tiers, and where each may appear

Every number on screen carries one of `EVIDENCE_OBSERVED`, `EVIDENCE_UNATTESTED`,
`EVIDENCE_DERIVED`, `EVIDENCE_CONCEPTUAL`. Four, not three: candidate scores for generated
tokens after the first are real forward-pass output that no verification gate re-derived, and
they must not look identical to a tensor that survives all three gates. Each `Operation`
declares the `tiers` it may show and a test enforces it, and `EVIDENCE_UNATTESTED` is
confined to two places — a later generation step's scores, and the stopping score, which
also comes from a non-prefill pass.

Operation 8 is the only place an intermediate is *computed*: one scalar of `A @ V`, in
float64, from two captured operands, badged derived and shown beside `DISPLAY_CALC_NOTE`.
It exists because that is the step whose result the model discarded, so a reader who has
never watched a weighted average happen has not seen attention. It shows the heaviest six
keys and one "other N positions" remainder, so the visible products plus the remainder are
exactly the total (`test_mix_products_add_up_to_the_shown_total`); it used to show the first
eight keys in position order beside a total over all of them.

Everything else derived is *selection or summary*, never reconstruction: which features
are largest (`tracked_features`, followed through operations 2, 4, 9 and 10), which keys
carry the weight, a head-mean of one weight row, a row total, the margin between two
scores. **Forbidden**, because each would recreate something the capture discarded: the RMS
or vector length of X or X_norm (`rms` is a named absent variable, and ‖X‖ = rms·√d), any
difference of two captured tensors (`X_next − X` is O + F), and H, F or Z beyond that one
scalar. It is also the only surviving user of `StepReveal`:
four steps at the reader's pace, captioned with `STRUCTURE_NOT_TIMING_NOTE`, which says the
order represents how the operation is built and not how long any of it took. No copy
anywhere states a duration.

### Vestigial after the two-page redesign

Nothing on either page uses these any more. They still import and still work, so nothing
is broken — but do not reach for one assuming it is part of the current design. Only
`numeric_sample` is still covered by a test; the others have no remaining caller and
therefore no remaining coverage:

| Dead | Why, and what replaced it |
| --- | --- |
| `MiniHeatmap` | The notebook paints attention with `AttentionGrid` (`tensorscope_diagrams`). A 15×15 or 1×16 matrix is not worth a `FigureCanvasQTAgg`, and one canvas per layer is exactly the cost `Disclosure` laziness exists to avoid. `MiniHeatmap` is the **only** caller of `display_matrix`, so with it unused the `tensorscope_common → TensorScope` back-edge now has no live traffic. The deferred-import rule in "The recap is ten modules" above still stands for as long as the class exists. |
| `FlowDiagram` | The reference page draws with QPainter, so a numberless box-and-arrow sketch has no remaining caller. |
| `ShapeDiagram`, `Card.shape_diagram` | Replaced by `MatMulShapes`, which also ties the shared inner dimension across both operands. |
| `sample_text` | Had no caller before this change either. `numeric_sample` survives only as its helper and as a `--self-test` assertion. |
| `PipelineNavigator`, `StageProgress` | The shell's chip pipeline and pills, replaced by `ContentsButton` and `ChapterProgress`. |
| `PhaseBadge` | Lived in the removed context ribbon; dark-palette only. The `FocusChip` says when the generated-token pass is on screen. |

`TokenChipStrip` looks dead and is not quite: `ResponseTextView` falls back to it when the
stored token spellings do not reassemble into the stored response. No run in the shipped
database trips that, so the branch is reachable and currently unexercised — see the
capability-tiers section above. Do not delete it on the strength of a coverage report.

Deleting the dead ones is a reasonable cleanup; it would also leave `display_matrix` with
no caller, which is capture-adjacent code, so it is a decision rather than a tidy-up.

### On the softmax cross-check

`softmax(captured_scores)` matches `captured_weights` to ~2e-3, not exactly, and the
tolerance is `4e-3`. That is not slop: upstream computes softmax in float32 then casts
back to the model dtype, and bf16 has an 8-bit mantissa, so the relative step is
2⁻⁸ ≈ 0.0039. The measured disagreement (1.9e-3) sits right where that predicts. Don't
"fix" it by loosening further or by recomputing in float32 — the captured weights are the
ones the model used.
