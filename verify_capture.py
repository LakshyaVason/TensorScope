"""End-to-end verification that TensorScope captures real, complete tensors.

Unlike `TensorScope.py --self-test` (which needs no torch), this loads an actual
model and exercises the real capture path, then checks the properties the project
depends on:

  * the capture attention implementation is bitwise identical to upstream's
  * softmax(captured scores) reproduces the captured weights
  * every REQUIRED_LAYER_TENSORS entry is present, for both phases, every layer
  * RunCapture.validate() accepts the result
  * RunDatabase round-trips every tensor bit-exactly

Run:  ./.venv/Scripts/python.exe verify_capture.py [model_id] [--reload]

`--reload` additionally loads the model twice in one process, checking that
`ModelCapture.unload()` really gives the VRAM back and that the gates still pass after
the attention backend has been re-registered over itself.  It is opt-in because it
doubles the run.
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


def check_the_three_gates(capture, label: str) -> None:
    """The gates RunCapture.validate() enforces, re-asserted by name.

    Factored out so the --reload pass can prove they still hold on a second model loaded
    into the same process -- the only place `.register()` replacing an existing key, and
    the _global_mapping assertion firing twice, are ever exercised.
    """
    assert capture.metadata["capture_source"] == "pytorch-forward-hooks", label
    assert capture.metadata["faithful_to_upstream_eager"] is True, label
    expected = len(capture.layers) + len(capture.generated_layers)
    assert capture.metadata["attention_calls_verified"] >= expected, label
    assert capture.metadata["logits_max_abs_diff_vs_stock_eager"] == 0.0, label
    capture.validate()
    print(f"{label}: all three gates hold "
          f"({expected} captured attention calls re-verified against upstream)")


def check_reload(model_id: str) -> None:
    """Load, unload, and load again -- then prove the gates still pass.

    `ModelCapture.unload()` exists because re-loading used to leak the previous model for
    the process lifetime: the forward-hook closures capture the ModelCapture, so the
    model was held by a reference cycle nothing collected, and two 4B models do not fit
    on a 16 GB card.  Allocated bytes, not reserved: the caching allocator keeps reserved
    high on purpose, so reserved would never show the drop.
    """
    print(f"\n--- reload: {model_id} twice in one process ---")
    first = ModelCapture(model_id)
    first.load(lambda message: print(f"  {message}"))
    check_the_three_gates(first.generate(PROMPT, max_new_tokens=8, on_text=lambda _t: None),
                          "first load")
    loaded_bytes = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    first.unload()
    freed_bytes = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        print(f"allocated after unload: {freed_bytes / 2 ** 30:.2f} GiB "
              f"(was {loaded_bytes / 2 ** 30:.2f} GiB)")
        # The weights are the bulk of it; anything above a tenth of what was held means
        # something is still pinning the model.
        assert freed_bytes < max(loaded_bytes * 0.1, 64 * 2 ** 20), \
            f"unload() freed only {(loaded_bytes - freed_bytes) / 2 ** 30:.2f} GiB"

    second = ModelCapture(model_id)
    second.load(lambda message: print(f"  {message}"))
    check_the_three_gates(second.generate(PROMPT, max_new_tokens=8, on_text=lambda _t: None),
                          "second load")
    second.unload()
    print("reload verified: re-registration is clean and the gates still pass")


def main() -> int:
    arguments = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
    model_id = arguments[0] if arguments else DEFAULT_MODEL

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
        for phase, original_layers, restored_layers in (
            ("prefill", capture.layers, restored.layers),
            ("first_generated_token", capture.generated_layers, restored.generated_layers),
        ):
            assert set(restored_layers) == set(original_layers), f"{phase} layer set changed"
            for index, layer in original_layers.items():
                assert set(restored_layers[index].tensors) == set(layer.tensors)
                for name, tensor in layer.tensors.items():
                    actual = restored_layers[index].tensors[name]
                    assert np.array_equal(actual, tensor), \
                        f"{phase} layer {index} tensor {name} changed in the database"
                    assert actual.dtype == tensor.dtype
        for name in ("embedding", "generated_embedding"):
            original, actual = getattr(capture, name), getattr(restored, name)
            assert np.array_equal(actual, original), f"{name} changed in the database"
            assert actual.dtype == original.dtype, f"{name} dtype changed in the database"
        assert np.array_equal(restored.logits, capture.logits), "final scores changed in the database"
        assert restored.logits.dtype == capture.logits.dtype
        # Stored at layer_index -1: it must not be read back as a phantom layer.
        assert set(restored.layers) == set(capture.layers), "load() invented a layer"
        total = sum(len(layer.tensors) for layers in (capture.layers, capture.generated_layers)
                    for layer in layers.values())
        print(f"RunDatabase round-trip bit-exact across {total} layer tensors, "
              f"plus the {restored.logits.shape[0]}-entry score vector")

    if torch.cuda.is_available():
        print(f"\npeak VRAM: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")

    # Opt-in, because it loads the model twice and would double the Qwen3-4B run.
    if "--reload" in sys.argv:
        capture_model.unload()
        check_reload(model_id)

    print("\nVERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
