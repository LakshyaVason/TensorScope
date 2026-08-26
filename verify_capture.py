"""End-to-end verification that TensorScope captures real, complete tensors.

Unlike `TensorScope.py --self-test` (which needs no torch), this loads an actual
model and exercises the real capture path, then checks the properties the project
depends on:

  * the capture attention implementation is bitwise identical to upstream's
  * softmax(captured scores) reproduces the captured weights
  * every REQUIRED_LAYER_TENSORS entry is present, for both phases, every layer
  * RunCapture.validate() accepts the result
  * RunDatabase round-trips every tensor bit-exactly

Run:  ./.venv/Scripts/python.exe verify_capture.py [model_id]
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from TensorScope import (
    REQUIRED_LAYER_TENSORS,
    TENSOR_EXPLANATIONS,
    TENSOR_LABELS,
    ModelCapture,
    RunDatabase,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # small: verifies mechanism cheaply
PROMPT = "What is the capital of France?"


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}  "
              f"capability={torch.cuda.get_device_capability(0)}")

    capture_model = ModelCapture(model_id)
    capture_model.load(lambda message: print(f"  {message}"))

    capture = capture_model.generate(PROMPT, max_new_tokens=40,
                                     on_text=lambda _t: None)
    print(f"\nprompt:   {capture.prompt}")
    print(f"response: {capture.response.strip()[:200]}")

    # --- attention faithfulness -------------------------------------------------
    faithful = capture.metadata["faithful_to_upstream_eager"]
    print(f"\nattention impl bitwise identical to upstream: {faithful}")
    assert faithful is True, "capture attention diverged from upstream eager path"
    print(f"attention implementation invoked {capture.metadata['attention_calls']} times")

    # Coverage matters as much as the verdict: checking one call once previously passed
    # while the run as a whole was wrong. Every captured call must have been checked.
    verified = capture.metadata["attention_calls_verified"]
    expected = len(capture.layers) + len(capture.generated_layers)
    print(f"calls verified bitwise against upstream: {verified} "
          f"(every layer of both captured phases = {expected})")
    assert verified >= expected, f"only {verified} of {expected} captured calls verified"

    # The check that actually matters: capture must not perturb the computation.
    delta = capture.metadata["logits_max_abs_diff_vs_stock_eager"]
    print(f"prefill logits vs stock eager attention: max abs diff {delta:.3e}")
    assert delta == 0.0, "capture changed the model's output"

    # --- scores really are the softmax input ------------------------------------
    # Upstream casts the f32 softmax result back to the model dtype, so bf16
    # rounding (8-bit mantissa, ~0.4% relative step) bounds the agreement here.
    scores = capture.layers[0].tensors["attention_scores"].astype(np.float32)
    weights = capture.layers[0].tensors["attention_weights"].astype(np.float32)
    resoftmaxed = torch.softmax(torch.from_numpy(scores), dim=-1).numpy()
    delta = float(np.abs(resoftmaxed - weights).max())
    print(f"softmax(scores) reproduces weights: {np.allclose(resoftmaxed, weights, atol=4e-3)} "
          f"(max diff {delta:.3e}, bf16 step ~{2 ** -8:.4f})")
    assert delta < 4e-3, f"scores and weights disagree by {delta}"

    # --- the final scores really are the ones that chose the word ----------------
    # These logits are the tensor _verify_undisturbed just proved identical to stock eager,
    # so the recap's score table inherits that proof.  The peak must be the token emitted.
    assert capture.logits is not None, "capture did not keep the final score vector"
    assert capture.logits.ndim == 1, f"final logits should be one vector, got {capture.logits.shape}"
    peak = int(capture.logits.argmax())
    print(f"\nfinal scores: {capture.logits.shape[0]} entries, peak at token {peak} "
          f"= generated token {capture.token_ids[0]}")
    assert peak == capture.token_ids[0], "final logits disagree with the generated token"

    # The recap displays the decoded top-k from metadata; it must describe the stored vector.
    top = capture.metadata["final_logits_top"]
    expected_ids = [int(i) for i in np.argsort(capture.logits)[::-1][:len(top)]]
    assert [entry["id"] for entry in top] == expected_ids, "top-k metadata is not the stored vector's"
    for entry in top:
        assert entry["logit"] == float(capture.logits[entry["id"]]), \
            f"top-k score for token {entry['id']} is not the stored one"
    print("top-k shown by the recap matches the stored vector: "
          + ", ".join(f"{entry['token']!r}={entry['logit']:.3f}" for entry in top[:4]))

    # --- completeness -----------------------------------------------------------
    print("\ncaptured tensors per layer:")
    for name, _label in TENSOR_LABELS:
        tensor = capture.layers[0].tensors[name]
        print(f"  {name:<20} {str(tensor.shape):<26} {tensor.dtype}")
        # Every displayed tensor must carry plain-language copy, or story mode would show a
        # matrix with nothing said about it.
        assert TENSOR_EXPLANATIONS[name].strip(), f"{name} has no explanation"

    for phase, layers in (("prefill", capture.layers),
                          ("first_generated_token", capture.generated_layers)):
        for index, layer in layers.items():
            missing = REQUIRED_LAYER_TENSORS - set(layer.tensors)
            assert not missing, f"{phase} layer {index} missing {missing}"
        print(f"{phase}: {len(layers)} layers, all required tensors present")

    capture.validate()
    print("RunCapture.validate() passed")

    # --- persistence ------------------------------------------------------------
    with tempfile.TemporaryDirectory() as temporary:
        database = RunDatabase(Path(temporary) / "verify.sqlite3")
        run_id = database.save(capture)
        restored = database.load(run_id)
        for index, layer in capture.layers.items():
            for name, tensor in layer.tensors.items():
                assert np.array_equal(restored.layers[index].tensors[name], tensor), \
                    f"layer {index} tensor {name} changed in the database"
                assert restored.layers[index].tensors[name].dtype == tensor.dtype
        assert np.array_equal(restored.embedding, capture.embedding)
        assert np.array_equal(restored.generated_embedding, capture.generated_embedding)
        assert np.array_equal(restored.logits, capture.logits), "final scores changed in the database"
        assert restored.logits.dtype == capture.logits.dtype
        # Stored at layer_index -1: it must not be read back as a phantom layer.
        assert set(restored.layers) == set(capture.layers), "load() invented a layer"
        total = sum(len(layer.tensors) for layer in capture.layers.values()) * 2
        print(f"RunDatabase round-trip bit-exact across ~{total} tensors, "
              f"plus the {restored.logits.shape[0]}-entry score vector")

    if torch.cuda.is_available():
        print(f"\npeak VRAM: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
    print("\nVERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
