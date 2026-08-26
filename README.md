# TensorScope

Watch a real language model answer a real prompt — and inspect the *exact* tensors it
produced while doing it.

TensorScope loads a Hugging Face causal LM into its own process, runs it with a custom
eager attention implementation plus PyTorch forward hooks, and shows you every
intermediate tensor from the run: tokenization, embeddings, per-layer Q/K/V before and
after RoPE, the pre-softmax attention scores, the attention weights (as a heatmap), the
attention block output, and each layer's output.

**No value is ever inferred, simulated, or reconstructed.** Everything displayed is a
tensor the running model actually computed. Before a run can be saved or displayed,
TensorScope re-runs the same prefill with stock attention and requires the logits to match
bit for bit — if the capture machinery perturbed the computation at all, the run is thrown
away rather than shown to you.

---

## Requirements

- **Python 3.10+**
- **An NVIDIA GPU is strongly recommended.** CPU works but is slow. The default model needs
  about **8 GB of VRAM**; see [Choosing a model](#choosing-a-model) for smaller options.
- ~8 GB of disk for the default model weights (downloaded once, cached by Hugging Face).

## Install

> **Use a virtual environment.** Do not install into an Anaconda base environment — conda's
> MKL and pip's torch both ship `libiomp5md.dll`, which produces `OMP: Error #15`. The
> usual workaround for that error warns it "may silently produce incorrect results", which
> is unacceptable for a tool whose entire purpose is exact numbers.

```bash
git clone <this-repo>
cd TensorScope
python -m venv .venv
```

**Windows**

```bash
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

**Linux / macOS**

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

### If you have an RTX 50-series card (Blackwell, sm_120)

Default PyPI torch ships no kernels for these GPUs. Install torch from the CUDA 12.8 index
*first*, then the rest:

```bash
./.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Check that your card is covered:

```bash
./.venv/Scripts/python.exe -c "import torch; print(torch.cuda.get_arch_list())"
```

## Run it

```bash
./.venv/Scripts/python.exe TensorScope.py        # Windows
./.venv/bin/python TensorScope.py                # Linux/macOS
```

Then, in the window:

1. **Model** — leave the default (`Qwen/Qwen3-4B`) or type another Hugging Face model id.
2. Click **Load model.** The first run downloads the weights, so expect a wait; the status
   line reports progress and finishes with the layer count, hidden size and device.
3. Type a question into **Prompt** (e.g. *What is the capital of France?*).
4. Click **Run Model.** The answer streams into **Model answer**, the run is saved, and a
   **Computation Recap** window opens with the captured tensors. The recap is modal — close
   it to get back to the main window.
5. In the recap, click any **Layer *n*** header to expand it. Layers build their contents
   on first expand, which is why a 36-layer model opens instantly.
6. **Saved Runs** lists every past capture; select one and click **Open Recap** to
   reopen it. The stored tensors are the original values, at their original dtype.
   To browse a database from somewhere else, click **Choose Database…** (also on the
   **Settings** screen) — no GPU or model load needed. `--db /path/to/runs.sqlite3`
   does the same thing from a terminal.

Prompts are capped at **256 tokens**, because attention tensors grow with the square of the
prompt length.

## Choosing a model

Any model you point TensorScope at must:

- **Answer real prompts coherently.** The point is to show how an LLM works; a tiny model
  emitting nonsense teaches nothing.
- **Be dense, not MoE.** The captured tensor set describes a dense attention-then-MLP
  block. Mixture-of-experts models route through a router and experts, which the recap does
  not model.
- Fit in your VRAM.

| Model | Layers | Peak VRAM (bf16, measured) |
| --- | --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct` | 24 | ~1.0 GiB |
| `Qwen/Qwen3-4B` *(default)* | 36 | ~7.6 GiB |

If you have less than 8 GB of VRAM, start with `Qwen/Qwen2.5-0.5B-Instruct`. It answers
simple factual prompts well enough to follow the computation, though a 4B model is a much
better demonstration.

## Verifying the capture yourself

Don't take the "no simulated tensors" claim on trust — check it:

```bash
./.venv/Scripts/python.exe verify_capture.py                # Qwen2.5-0.5B-Instruct: quick
./.venv/Scripts/python.exe verify_capture.py Qwen/Qwen3-4B  # the default model
```

This loads a real model, runs a real capture, and asserts that:

- the capture attention implementation is **bitwise identical** to the transformers eager
  path, on **every** captured call (every layer, both captured passes);
- the prefill logits are identical to stock eager attention — max abs difference exactly
  `0.0`, not a tolerance;
- `softmax(captured scores)` reproduces the captured weights within bf16 rounding;
- every required tensor is present for every layer of both passes;
- the SQLite round-trip is bit-exact and never changes a dtype.

There is also a fast check with no torch or GPU required, covering the persistence and
display helpers:

```bash
./.venv/Scripts/python.exe TensorScope.py --self-test
```

## What gets captured

Capture scope is fixed: the **prompt prefill** plus the **first generated token.** Any
remaining tokens are generated with capture switched off, purely so you see a complete
answer.

Per layer, per pass:

| Tensor | What it is |
| --- | --- |
| `normalized_input` | layer input after RMSNorm |
| `q`, `k`, `v` | raw projection outputs — **before** RoPE and QK-norm |
| `q_attended`, `k_attended`, `v_attended` | the tensors that actually enter QKᵀ — after RoPE, with GQA heads already expanded |
| `attention_scores` | the pre-softmax matrix, exactly as handed to softmax |
| `attention_weights` | post-softmax attention (shown as the heatmap) |
| `attention_output` | `W_o · concat(heads)` — the attention block's output |
| `layer_output` | the decoder layer's output |

Plus the token embeddings for the pass.

## Where things are stored

- `tensorscope_runs.sqlite3` — saved runs and their tensors, in the project directory.
  Arrays are stored as compressed `np.save` blobs with the dtype preserved exactly. This
  file grows quickly; deleting it just clears your run history.
- Model weights live in the standard Hugging Face cache, not here.

## Troubleshooting

**`OMP: Error #15` / duplicate `libiomp5md.dll`** — torch was installed alongside conda's
MKL. Create a fresh venv and install there. Don't set `KMP_DUPLICATE_LIB_OK`.

**`no kernel image is available for execution on the device`** — your GPU needs newer CUDA
wheels than PyPI's default build. See the cu128 instructions above.

**CUDA out of memory** — use a smaller model (`Qwen/Qwen2.5-0.5B-Instruct`) or a shorter
prompt. Attention tensors are retained for every layer, so memory scales with layer count
and with the square of the prompt length.

**The run was discarded with a message about logits differing** — that is the safety gate
working. It means the capture would not have shown you the tensors this model normally
computes, so nothing was saved. Most likely cause is a transformers version whose eager
attention path changed.

>claude --resume c3c9511e-a38a-4954-a3fc-0aabceaa5e19

