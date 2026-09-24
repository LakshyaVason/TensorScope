"""Educational copy and evidence taxonomy for the saved-capture UI.

No model, numpy or GUI imports: this module is pure data, so it can be read by the
capture engine, the views and the tests alike.

Two rules govern everything here:

* Equations describe the architecture.  Numerical displays must use the saved arrays,
  including when a mathematically intermediate result was not retained by capture.
* Every number shown to a reader carries an evidence tier (see EVIDENCE_KINDS).  A
  value the model produced and a value TensorScope calculated for presentation must
  never look alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Per-token decision telemetry ─────────────────────────────────────────────────
# Defined here rather than in TensorScope.py because the view modules need it and
# TensorScope.py imports them: a view importing it from TensorScope would run while
# TensorScope is still mid-import.  This module imports nothing, so it is always safe.

@dataclass
class GenerationDecision:
    """The real scores that chose one generated token.

    Recorded inside the generation loop from the logits tensor the forward pass had
    already produced.  Nothing here is recomputed, re-run or reconstructed afterwards.

    `attested` marks whether the vector behind this decision is covered by the
    stock-eager comparison in ModelCapture._verify_undisturbed.  Only the prefill
    logits are, so only step 0 is True; later steps are real model output that no
    verification gate independently re-derived.  The UI must not present the two at
    the same confidence.
    """

    step: int                                          # 0 == first generated token
    selected_token_id: int
    selected_logit: float
    # Descending by score, ties broken by ascending token id so entry 0 always agrees
    # with argmax.  Shape: [{"id": int, "token": str, "logit": float}, ...]
    top_candidates: list[dict[str, Any]] = field(default_factory=list)
    attested: bool = False

    @property
    def runner_up(self) -> dict[str, Any] | None:
        """The highest-scoring candidate that was not selected, if one was recorded."""
        for entry in self.top_candidates:
            if int(entry["id"]) != self.selected_token_id:
                return entry
        return None

    @property
    def margin(self) -> float | None:
        """How far ahead the winner scored.  Derived, not captured."""
        other = self.runner_up
        return None if other is None else self.selected_logit - float(other["logit"])


# ── Evidence taxonomy ────────────────────────────────────────────────────────────
# Four tiers, not three.  The spec asks for observed / derived / conceptual, but
# per-token telemetry introduced a fourth case that would otherwise be mislabelled
# as plain "observed": real forward-pass output that no verification gate covers.

EVIDENCE_OBSERVED = "observed"
EVIDENCE_UNATTESTED = "unattested"
EVIDENCE_DERIVED = "derived"
EVIDENCE_CONCEPTUAL = "conceptual"

EVIDENCE_KINDS = {
    EVIDENCE_OBSERVED: (
        "Observed",
        "A tensor this run captured from the model and stored exactly, covered by the "
        "stock-eager comparison: embeddings, Q/K/V, attention scores and weights, layer "
        "outputs, the prefill logits vector, and the selected token.",
    ),
    EVIDENCE_UNATTESTED: (
        "Observed · unattested",
        "Real output of the real forward pass, recorded as it happened, but not "
        "independently re-derived. Only the prompt prefill is re-run under stock eager "
        "attention, so candidate scores for generated tokens after the first carry no "
        "such comparison.",
    ),
    EVIDENCE_DERIVED: (
        "Derived",
        "Calculated by TensorScope from stored values for presentation: candidate "
        "rankings, score differences, optional softmax percentages, summary statistics. "
        "The model did not compute these.",
    ),
    EVIDENCE_CONCEPTUAL: (
        "Conceptual",
        "An operation the architecture performs whose intermediate value this capture "
        "did not retain: the residual sum, the MLP interior, A @ V before the output "
        "projection, and every learned weight matrix. Described, never shown as data.",
    ),
}


# These keys deliberately match REQUIRED_LAYER_TENSORS in TensorScope.py exactly.
TENSOR_LABELS = [
    ("normalized_input", "Normalized layer input · X_norm"),
    ("q", "Query projection · Q_raw"),
    ("k", "Key projection · K_raw"),
    ("v", "Value projection · V_raw"),
    ("q_attended", "Prepared queries · Q"),
    ("k_attended", "Prepared keys · K"),
    ("v_attended", "Prepared values · V"),
    ("attention_scores", "Scaled and masked attention scores · S"),
    ("attention_weights", "Attention weights · A"),
    ("attention_output", "Attention block output · after o_proj"),
    ("layer_output", "Decoder layer output · X_next"),
]

TENSOR_LABEL_BY_KEY = dict(TENSOR_LABELS)

TENSOR_EXPLANATIONS = {
    "normalized_input": (
        "The input normalization module rescales each token's feature vector before the "
        "attention projections. In Qwen2.5 and Qwen3 this is RMSNorm: divide by a measure "
        "of the vector's magnitude, then apply learned feature scales. This helps control "
        "activation scale as information passes through many layers. Axes are "
        "[batch, query token, model feature]; normalization preserves their sizes. "
        "The captured output supplies all three Q/K/V projections. The normalization's "
        "learned scales and intermediate statistics are not saved."
    ),
    "q": (
        "A query is a learned description of what information a token will look for at "
        "this layer. A linear projection maps each normalized feature row into query "
        "features: Q_raw = Linear_Q(X_norm). Using PyTorch's stored weight convention, "
        "Linear(X) = X Wᵀ + b, with a bias only where the architecture uses one. "
        "Axes are [batch, query token, query projection feature]; the final axis will "
        "be split into heads. These are the actual q_proj outputs, before this layer's "
        "QK normalization (where present) and rotary position transformation. Earlier "
        "layers may already have encoded positional context. Learned weights and biases "
        "are not part of this saved capture."
    ),
    "k": (
        "A key is the learned feature vector against which a query is compared by a dot "
        "product. K_raw = Linear_K(X_norm) uses a different learned projection from Q. "
        "Axes are [batch, current token, key projection feature]. Grouped-query attention "
        "can use fewer key heads than query heads, so this width need not equal Q's width. "
        "This is the actual k_proj output before head preparation, optional QK normalization, "
        "and RoPE. In the generated-token phase it contains only the newly processed token; "
        "the prepared K tensor also includes cached prompt keys. Projection weights are not saved."
    ),
    "v": (
        "Values carry the content that attention will combine. V_raw = Linear_V(X_norm) "
        "produces a third learned view of each token. Queries and keys determine the "
        "mixing coefficients; values supply the vectors being mixed. Axes are "
        "[batch, current token, value projection feature]. This is the actual v_proj "
        "output before reshaping into heads, cache assembly, and any repetition of "
        "shared value heads. The learned projection matrix is not saved."
    ),
    "q_attended": (
        "These are the queries actually supplied to the attention dot products, with axes "
        "[batch, query head, query token, head feature]. A head is one parallel attention "
        "channel with its own query features and mixing pattern. The projection features "
        "have been separated into heads and position transformations have been applied. "
        "Qwen3 applies per-head Q normalization before RoPE; Qwen2.5 does not have that "
        "Q normalization step. RoPE rotates pairs of query/key features as a function of "
        "token position, allowing their dot product to depend on relative position. "
        "Only the resulting prepared tensor is saved, not each preparation intermediate."
    ),
    "k_attended": (
        "These are the keys actually multiplied with Q, with axes "
        "[batch, query head, key token, head feature]. Qwen keys have been reshaped and "
        "rotated by RoPE, with per-head K normalization first in Qwen3. The attention "
        "implementation then repeats shared KV heads to match the query-head count where "
        "grouped-query attention (GQA) is used. In the generated-token pass, the key-token "
        "axis includes cached prompt keys followed by the new token's key. That cache "
        "content is present in this captured tensor; a separate cache object is not saved."
    ),
    "v_attended": (
        "These are the values the attention weights actually mix, with axes "
        "[batch, query head, key token, value feature]. In Qwen, V is reshaped into heads, "
        "combined with cached values when decoding, and repeated to match query heads "
        "where GQA is used. RoPE is applied directly to Q and K, not to V; this does "
        "not mean V is free of positional context inherited from previous layers. "
        "Its key-token axis matches the attention weights' column axis so A @ V can "
        "sum over the same key positions."
    ),
    "attention_scores": (
        "Each cell compares one query token with one key token in one head. "
        "S = s · (Q @ Kᵀ) + mask, where s is the attention scale "
        "(1/√d_head in Qwen). Axes are [batch, head, query token, key token]. "
        "This captured tensor is already scaled and masked: it is exactly the input "
        "handed to softmax. The unscaled product and the mask are not separately saved. "
        "A causal mask prevents a query from reading later positions, using extremely "
        "negative values or negative infinity according to the implementation and dtype. "
        "A generated-token query can read the cached prompt and itself; its row is not "
        "a square prompt-prefill grid."
    ),
    "attention_weights": (
        "Softmax operates across the key-token axis of each score row to produce "
        "nonnegative mixing weights: A = softmax(S). The axes remain "
        "[batch, head, query token, key token]. Each row is mathematically a distribution "
        "over available keys; stored values can sum only approximately to one after "
        "rounding to the model dtype. Capture runs in evaluation mode, so dropout is "
        "inactive. Each cell weights that key's V vector when computing a query's output. "
        "These are attention mixing weights over input positions, not probabilities over "
        "next output tokens — the two are different sizes and different things. They "
        "describe one head's computation and are not an explanation of why the model "
        "answered as it did."
    ),
    "attention_output": (
        "The captured o_proj output is the attention block's contribution in model-feature "
        "space. Conceptually, each head first computes A @ V, the head results are joined, "
        "then Linear_o maps the joined features back to d_model. With row vectors, "
        "this projection is concat(heads) W_oᵀ, plus a bias where the architecture uses one. "
        "Axes are [batch, query token, model feature]. The A @ V result, joined-head tensor, "
        "and projection weights are not individually persisted. The displayed tensor "
        "is after the output projection; it is not the A @ V intermediate. Next the "
        "decoder combines this contribution with its residual stream and feed-forward path."
    ),
    "layer_output": (
        "This is the decoder layer's completed output, with axes "
        "[batch, query token, model feature]. In the supported Qwen blocks, an attention "
        "residual addition is followed by another normalization, a gated feed-forward "
        "network (MLP), and another residual addition. The MLP transforms each position's "
        "features; attention is the part that mixes information across positions. Those "
        "residual, normalization, and MLP intermediates are not individually saved. "
        "The layer output becomes the next decoder layer's input, keeping the same width. "
        "After the last layer, a final model normalization and vocabulary projection "
        "produce logits; the final normalization output is not separately captured."
    ),
}


# ── The default learning journey ─────────────────────────────────────────────────
# Each stage answers a question a human actually asks, and says so.  The old journey
# was organised around operations (normalize, project, rotate, softmax); a reader who
# has never studied transformers met the math before the meaning.

@dataclass(frozen=True)
class LearnStage:
    """One stage of the Understand journey; format fields come from learn_facts()."""

    key: str
    nav: str        # short label for the sidebar and pipeline chips
    question: str   # the question this screen answers, shown as its heading
    plain: str      # two or three sentences, no more -- the screens carry the weight


LEARN_STAGES = [
    LearnStage(
        "overview", "Prompt & response",
        "What did I ask, and what did the model answer?",
        "This is one real run of {model}. The response was not written all at once: the "
        "model produced {generated_count} tokens one at a time, each one chosen by a "
        "fresh pass over everything written so far. TensorScope recorded that run as it "
        "happened rather than reproducing it afterwards."
    ),
    LearnStage(
        "tokens", "What it received",
        "What did the model actually receive?",
        "Not your sentence. A tokenizer splits text into vocabulary entries — whole words, "
        "word fragments, punctuation, spaces — and the model only ever sees their ID "
        "numbers. This prompt became {prompt_token_count} tokens. Select any one to see "
        "what the model was handed in its place."
    ),
    LearnStage(
        "context", "Building context",
        "How does the model build context between tokens?",
        "Each token starts as a vector that depends only on which token it is. Across "
        "{layer_count} layers that vector is repeatedly updated by pulling in information "
        "from other positions in the text, so by the end it reflects its context and not "
        "just its own identity. The mechanism that does the pulling is attention."
    ),
    LearnStage(
        "generation", "Choosing each token",
        "How does it generate one token at a time, and why this token?",
        "At every step the model scores the entire {vocab_size}-entry vocabulary and the "
        "highest score wins. Click any token of the response to see the real scores that "
        "chose it and what came second."
    ),
    LearnStage(
        "limits", "What we can and cannot say",
        "What can we observe inside the model, and what can we not conclude?",
        "TensorScope shows exactly what this run computed. That is a strong claim about "
        "values and a weak one about meaning: knowing every number a model produced is "
        "not the same as knowing why it learned to produce them."
    ),
]

LEARN_STAGE_INDEX = {stage.key: stage for stage in LEARN_STAGES}


# ── The optional technical journey ───────────────────────────────────────────────
# Everything the old default view showed, kept in full and reachable in two clicks --
# but ordered purpose -> diagram -> equation -> dimensions -> tensor, so the tensor is
# the last level of detail rather than the first.

@dataclass(frozen=True)
class InternalsStage:
    key: str
    nav: str
    heading: str
    plain: str


INTERNALS_STAGES = [
    InternalsStage(
        "embedding", "Embedding",
        "Token IDs become feature vectors",
        "The embedding module looks up a learned vector for each token ID. Each vector has "
        "{hidden_size} model features, with axes [batch, token position, model feature]. "
        "Individual features are learned coordinates, not named concepts. The complete "
        "lookup table is not saved, only the rows this pass produced."
    ),
    InternalsStage(
        "layers", "Decoder layer",
        "Inside one of {layer_count} decoder layers",
        "Every layer repeats the same structure with its own learned parameters: normalize, "
        "project into queries/keys/values, mix information across positions with attention, "
        "then a residual addition and a feed-forward network. Pick a layer and walk its six "
        "steps; its output is the next layer's input."
    ),
    InternalsStage(
        "head", "Vocabulary scores",
        "The last position becomes one score per vocabulary entry",
        "Only the final position has attended to the whole prompt, so its representation is "
        "the one that chooses the next token. A final RMSNorm and a learned vocabulary "
        "projection turn it into {vocab_size} scores. The normalization output and the "
        "projection matrix are not saved; the resulting score vector is."
    ),
    InternalsStage(
        "decode", "The next pass",
        "The selected token starts another forward pass",
        "Generation is autoregressive: the chosen token must itself go through the model to "
        "score the token after it. TensorScope captures this second pass in full, which is "
        "why two phases exist. Passes after it are not captured at tensor level."
    ),
]

INTERNALS_STAGE_INDEX = {stage.key: stage for stage in INTERNALS_STAGES}

# Per-step copy for the layer walk.  `purpose` is level 1 (plain language), `equation`
# is level 3; the diagram, dimensions and tensor inspector are built from the captured
# shapes by the view.  `concept` is the one-line caption under the diagram.
INTERNALS_STEPS = {
    "input": {
        "purpose": (
            "Before a layer does anything else it puts its input on a predictable scale. "
            "Without this, values can grow or shrink as they pass through dozens of layers "
            "until the arithmetic stops being useful."
        ),
        "concept": "every token's vector is rescaled independently; no mixing happens here",
        "equation": (
            "X_norm  =  X / RMS(X)  ·  g<br>"
            "RMS(X)  =  √(mean(X²) + ε)<br><br>"
            "<i>The learned scales g and the intermediate statistics are not saved.</i>"
        ),
    },
    "qkv": {
        "purpose": (
            "Each token is turned into three different learned views of itself: what it is "
            "looking for (query), what it can be found by (key), and what it will contribute "
            "if found (value). Attention is built entirely out of these three."
        ),
        "concept": "three independent linear maps read the same normalized vector",
        "equation": (
            "Q_raw = X_norm W_Qᵀ<br>K_raw = X_norm W_Kᵀ<br>V_raw = X_norm W_Vᵀ<br><br>"
            "<i>PyTorch stores these transposed, hence Wᵀ. The weight matrices "
            "themselves are not saved — only their outputs.</i>"
        ),
    },
    "prepare": {
        "purpose": (
            "The projections are split into heads so several comparisons can run in parallel, "
            "and position information is rotated into the queries and keys. Until this "
            "happens the model has no notion of word order."
        ),
        "concept": "split into heads, then rotate by position (RoPE); Q and K only, never V",
        "equation": (
            "reshape to [batch, head, token, head feature]<br>"
            "Q = RoPE(norm(Q_raw))   K = RoPE(norm(K_raw))<br>"
            "V = repeat_kv(V_raw)<br><br>"
            "<i>Per-head QK normalization is a Qwen3 addition; Qwen2.5 has no such step. "
            "repeat_kv expands shared KV heads for grouped-query attention.</i>"
        ),
    },
    "scores": {
        "purpose": (
            "Every query is compared against every key it is allowed to see. A large score "
            "means that position is relevant to this one. Softmax then turns each row of "
            "scores into mixing weights that sum to one."
        ),
        "concept": "one number per (query, key) pair, per head; the future is masked out",
        "equation": (
            "S  =  (Q @ Kᵀ) / √d_head  +  mask<br>"
            "A  =  softmax(S)   along the key axis<br><br>"
            "<i>Both S and A are captured. The mask sets future positions to a very "
            "negative value so softmax gives them almost zero weight.</i>"
        ),
    },
    "mix": {
        "purpose": (
            "Each head builds its output as a weighted blend of the values it attended to. "
            "The heads are then joined back together and projected once, which is how the "
            "attention block returns to the model's own feature width."
        ),
        "concept": "weights select which values to blend; one projection rejoins the heads",
        "equation": (
            "head_i  =  A_i @ V_i          <b>not captured</b><br>"
            "output  =  concat(head_i) W_oᵀ  <b>captured</b><br><br>"
            "<i>The captured attention_output is after the projection. It is not A @ V.</i>"
        ),
    },
    "output": {
        "purpose": (
            "The attention result is added back onto the running representation rather than "
            "replacing it, then a feed-forward network refines each position on its own. The "
            "layer hands on a vector of the same width it received."
        ),
        "concept": "two residual additions around attention and the MLP",
        "equation": (
            "H  =  X + attention_output<br>"
            "X_next  =  H + MLP(RMSNorm(H))<br><br>"
            "<i>Only X_next is captured. H, the post-attention normalization and the "
            "MLP interior are conceptual here.</i>"
        ),
    },
}


# Execution order of the six steps above, asserted against the UI's own list so the
# copy and the navigator can never drift apart silently.
LAYER_STEP_ORDER = ["input", "qkv", "prepare", "scores", "mix", "output"]

assert set(LAYER_STEP_ORDER) == set(INTERNALS_STEPS), "INTERNALS_STEPS is missing a step"


# ── What can and cannot be concluded ─────────────────────────────────────────────

WHAT_WE_KNOW = [
    "The exact token IDs the model was given, after any chat template was applied.",
    "The exact embedding vectors those IDs produced in this run.",
    "The exact queries, keys, values, attention scores and attention weights for every "
    "head of every layer, in both captured passes.",
    "The exact output of every decoder layer.",
    "The full vocabulary score vector that chose the first generated token.",
    "The leading candidate scores at every later generation step, and which token won.",
    "The decoding rule: greedy, so the highest score is selected with no sampling.",
    "That capture did not change the computation — the prompt pass was re-run under "
    "stock eager attention and the logits matched bit for bit.",
]

WHAT_WE_CANNOT_CONCLUDE = [
    "Why the model learned this association. Training data and training dynamics are "
    "outside anything a forward pass can show.",
    "That any single attention head caused the answer. Removing it is the experiment "
    "that would test that, and TensorScope does not run it.",
    "That an individual neuron or feature corresponds to a human concept. Features are "
    "learned coordinates, and meaning is generally spread across many of them.",
    "That an attention heatmap is the model's reasoning. It shows where one head's "
    "weights went, which is a mechanism, not an intention.",
    "What the model would have answered to a different prompt. That needs another run.",
]

TELEMETRY_UNAVAILABLE = (
    "Detailed candidate scores were not recorded for this generation step. This run was "
    "saved before TensorScope recorded per-token decisions; the token it selected is "
    "stored, but the scores that competed with it are not recoverable from this capture. "
    "A new run records them for every token."
)

LOGITS_UNAVAILABLE = (
    "This saved run has no retained logits vector. Earlier capture versions did not "
    "persist it. The selected first token is available, but its vocabulary scores "
    "cannot be recovered from this capture. A new run can capture those scores."
)

STOP_REASONS = {
    "eos": "The model emitted an end-of-sequence token, so generation stopped there.",
    "max_new_tokens": "Generation hit TensorScope's token limit for this run, so the "
                      "response may be cut off mid-sentence.",
    "unknown": "This run does not record why generation stopped.",
}


def learn_facts(capture: Any) -> dict[str, Any]:
    """Read structural facts from a saved capture without loading a model.

    Every field any stage's copy interpolates must appear here, or a reader would meet
    a KeyError instead of a sentence.  self_test() formats every stage against this.
    """
    embedding = getattr(capture, "embedding", None)
    logits = getattr(capture, "logits", None)
    vocabulary = int(logits.shape[-1]) if logits is not None else 0
    return {
        "model": capture.metadata.get("model", "an open-weight model"),
        "layer_count": len(capture.layers),
        "prompt_token_count": len(capture.prompt_token_ids),
        "generated_count": len(capture.token_ids),
        "hidden_size": int(embedding.shape[-1]) if embedding is not None else 0,
        "vocab_size": f"{vocabulary:,}" if vocabulary else "full",
    }
