# TensorScope

Watch a real language model answer a real prompt — and inspect the *exact* tensors it
produced while doing it.

TensorScope loads a Hugging Face causal LM into its own process, runs it with a custom
eager attention implementation plus PyTorch forward hooks, and keeps every intermediate
tensor the run produced: tokenization, embeddings, per-layer Q/K/V before and after RoPE,
the pre-softmax attention scores, the attention weights, the attention block output, and
each layer's output.

Then it shows you the calculation those tensors came from, as one scrollable worked
notebook: twelve operations from the tokens the model received to the token it selected,
each with its formula, a picture of what the operation does, and the real captured numbers
beside it. Every variable in every formula is something you can click. A second page
explains the idea behind each operation, in the same order and the same notation, for when
a step stops making sense.

**No value is ever inferred, simulated, or reconstructed.** Everything displayed is a
tensor the running model actually computed. Before a run can be saved or displayed,
TensorScope re-runs the same prefill with stock attention and requires the logits to match
bit for bit — if the capture machinery perturbed the computation at all, the run is thrown
away rather than shown to you.

The corollary matters just as much. A forward pass computes plenty that it then throws
away — the learned weight matrices are never activations at all, and the residual sums, the
per-head value mix and everything inside the feed-forward network are computed and
discarded. TensorScope **names those, explains what they do, and shows you what it does
have instead.** It never fills the gap with a plausible-looking number.

So every set of values on screen is labelled with where it came from, in one of four ways:
a tensor the model produced and the checks above cover; real output of the real forward pass
that no check independently re-derived (the candidate scores for the second generated token
onwards); something TensorScope worked out from captured numbers for the display; or an
operation the model performed whose result this capture did not keep, described rather than
shown.

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
2. Click **Load**. The first run downloads the weights, so expect a wait; the status
   line reports progress and finishes with the layer count, hidden size and device.
3. Type a question into **Prompt** (e.g. *What is the capital of France?*).
4. Click **Run Capture.** The answer streams into **Response**, the run is saved, and a
   **Computation Recap** window opens with the captured tensors. It is a separate window,
   so you can leave it open and run another prompt beside it.
5. The recap has two pages.

   **Follow this run** is where it opens. It starts with where it is going — your prompt,
   the start of the model's actual answer, and the first token of it — and then works the
   calculation that chose that token, in four chapters: *Represent the input*, *Exchange
   information*, *Refine the representation*, *Select a token*. Each of the twelve
   operations reads the same way: its purpose in a sentence, the formula, what went in and
   what came out, a picture, a few real values, and how the result is used next. It
   follows one prompt position (the last word you typed) through every step, then hands
   over explicitly to the final position, whose row actually chose the token. Nothing above
   you disappears as you go, so you can always scroll back to a value you met earlier.

   Every named variable is clickable. Clicking one opens a panel with what it is, its
   shape in this run, where its numbers came from, a few values (rounded for reading; hover
   for the exact stored number), and the complete stored array one click further on. A
   variable the model computed but this capture did not keep says so instead, and shows you
   what *was* captured in its place. Nothing is ever reconstructed to fill a gap.

   **Understand the calculation** is the reference: twelve topics, each with the purpose,
   the formula, a picture and an optional longer explanation, each labelled with the
   operation it explains. Every operation has an **Explain this operation →** button, and
   every topic a **Follow this in the run →** button back.

   The bar at the top keeps the prompt, your chapter, the position being followed,
   **Contents** and **Previous / Next** in view (Alt+← / Alt+→ work too). **All captured
   arrays…** opens every stored tensor with no framing at all, and **Details…** shows what
   TensorScope verified for the run.
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

Two more checks need no GPU, no model download and no torch at all. The first covers the
persistence and display helpers and every line of copy on both pages; the second drives the
whole recap over your own saved runs — every operation, every variable, every disclosure,
both themes — and re-hashes the database afterwards to prove that reading a capture never
rewrites it:

```bash
./.venv/Scripts/python.exe TensorScope.py --self-test
./.venv/Scripts/python.exe test_recap.py
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
| `attention_weights` | post-softmax attention — each query's row sums to 1 |
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

