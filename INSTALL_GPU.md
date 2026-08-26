# Installing GPU-accelerated torch for TensorScope

The pre-built TensorScope installer bundles CPU-only PyTorch, which works but is slow
for large models like Qwen3-4B.  To enable GPU inference, replace the bundled torch with
a CUDA or MPS build after installation.

---

## NVIDIA — standard (RTX 30/40-series, GTX 16xx, most current GPUs)

```bash
pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
```

## NVIDIA — Blackwell (RTX 50-series, sm_120)

The default PyPI torch build has no kernels for sm_120.  Use the cu128 channel:

```bash
pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

Verify your card is covered:
```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

## macOS — Apple Silicon (MPS)

```bash
pip install --upgrade --force-reinstall torch
```

MPS acceleration is automatic on Apple Silicon.  TensorScope uses
`attn_implementation="eager"` which runs on MPS.

## Anaconda users — important

**Do not install into an Anaconda base environment.**  Conda's MKL and PyPI torch both
ship `libiomp5md.dll`, which produces `OMP: Error #15` at startup.  The only workaround
(`KMP_DUPLICATE_LIB_OK=TRUE`) "may silently produce incorrect results" — disqualifying
for a tool whose purpose is exact numerics.

Use a plain venv:
```bash
python -m venv tensorscope-env
tensorscope-env\Scripts\python.exe -m pip install -r requirements.txt  # Windows
```

## Checking VRAM requirements

| Model | Peak VRAM (bf16) |
|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | ~1.0 GB |
| Qwen/Qwen3-4B | ~7.6 GB |

TensorScope warns at model-load time if your GPU has less than 8 GB.
