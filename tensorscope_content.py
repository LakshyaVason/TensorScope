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

# ── Byte-level BPE spellings ─────────────────────────────────────────────────────
# A tokenizer vocabulary stores a leading space as U+0120 and a newline as U+010A, so
# the saved spellings are not readable as text.  These are presentation substitutions
# only: `prompt_tokens` itself is never rewritten, and the stored form is shown beside
# the readable one wherever a reader might otherwise think the model saw a 'Ġ'.

BYTE_BPE_REPLACEMENTS = (("Ġ", " "), ("Ċ", "\n"), ("ĉ", "\t"), ("▁", " "))


def readable_spelling(token: str) -> str:
    """What a stored token spelling stands for, as text.

    Special markers such as <|im_start|> contain none of these characters and pass
    through unchanged, which is what lets the bands below tell them apart from words.
    """
    text = token
    for stored, plain in BYTE_BPE_REPLACEMENTS:
        text = text.replace(stored, plain)
    return text


TEMPLATE_BAND = "Added by the chat template"
TYPED_BAND = "Your text, as the tokenizer split it"


def prompt_token_bands(prompt: str, prompt_tokens: list[str]) -> list[tuple[int, int, str]] | None:
    """Group prompt tokens into template / typed / template spans.

    Returns None rather than guessing.  There is no tokenizer and no chat template in a
    saved run, so the only honest way to find the reader's own words is to reassemble the
    stored spellings and look for the typed prompt inside them.  The match must be unique
    and must start and end exactly on token boundaries; anything else would put a band
    edge in the middle of a token and claim more than the capture supports.
    """
    tokens = list(prompt_tokens or [])
    target = (prompt or "").strip()
    if not tokens or not target:
        return None
    spellings = [readable_spelling(token) for token in tokens]
    starts, joined = [], ""
    for text in spellings:
        starts.append(len(joined))
        joined += text
    ends = [start + len(text) for start, text in zip(starts, spellings)]
    first = joined.find(target)
    if first < 0 or joined.find(target, first + 1) >= 0:
        return None                                   # absent, or ambiguous
    last = first + len(target)
    if first not in starts or last not in ends:
        return None                                   # would split a token
    begin = starts.index(first)
    stop = max(index for index, end in enumerate(ends) if end == last) + 1
    bands = [(0, begin, TEMPLATE_BAND), (begin, stop, TYPED_BAND), (stop, len(tokens), TEMPLATE_BAND)]
    return [band for band in bands if band[1] > band[0]]


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



# ── Variables: the objects the reader clicks ─────────────────────────────────────
# A variable is a symbol in an equation plus a claim about where its numbers came from.
# The claim is the part that has to be exactly right, so the copy for it lives here and
# the resolution lives in `tensorscope_variables`, which reads the capture.
#
# Four provenances, mapping onto the four evidence tiers:
#
#   captured  — the stored array exists in this run.  EVIDENCE_OBSERVED.
#   learned   — a learned parameter.  The model has it; this capture kept only its output.
#   absent    — an intermediate the architecture computes and this capture did not retain.
#   display   — a number TensorScope calculated from captured operands, for the screen.
#
# `learned` and `absent` are both EVIDENCE_CONCEPTUAL: they are described, never shown as
# data.  They are separate words because "nobody saved the weights" and "nobody saved this
# step's output" are different facts, and a reader who is told the second when the first is
# true will look for a tensor that was never going to be there.

PROVENANCE_CAPTURED = "captured"
PROVENANCE_LEARNED = "learned"
PROVENANCE_ABSENT = "absent"
PROVENANCE_DISPLAY = "display"

PROVENANCE_TIERS = {
    PROVENANCE_CAPTURED: EVIDENCE_OBSERVED,
    PROVENANCE_LEARNED: EVIDENCE_CONCEPTUAL,
    PROVENANCE_ABSENT: EVIDENCE_CONCEPTUAL,
    PROVENANCE_DISPLAY: EVIDENCE_DERIVED,
}

LEARNED_VALUES_NOT_SAVED = (
    "This is a learned parameter. Its values were fixed during training and the model used "
    "them in this very run, but TensorScope captures activations, not weights, so the "
    "numbers themselves are not in this capture. What is here is what they produced."
)

INTERMEDIATE_NOT_KEPT = (
    "The architecture computes this and the model computed it in this run. This capture "
    "did not retain it, so it is described here rather than shown. Nothing on this screen "
    "reconstructs it."
)

SHAPE_IS_NOT_CONTENTS = (
    "This line is the arrangement of the numbers, not the numbers. It says how many there "
    "are and how they are grouped; open the variable to see the values themselves."
)

BATCH_AXIS_NOTE = (
    "The leading axis is the batch: how many sequences were run at once. TensorScope runs "
    "one, so it is always 1 here, and it is dropped from shapes written for reading."
)

ROW_VECTOR_NOTE = (
    "Throughout, a token's features are written as a row, so a sequence is rows stacked "
    "downwards and a projection is applied on the right as X Wᵀ. That is why the weight "
    "matrices appear transposed: PyTorch stores a linear layer as "
    "[output features × input features]."
)

STRUCTURE_NOT_TIMING_NOTE = (
    "Shown one part at a time, for reading. The order represents how the operation is "
    "built, not how long any of it took: a layer's arithmetic happens at once, in "
    "parallel, on the GPU, and nothing on these pages measures execution time."
)

PROMPT_SPAN_UNKNOWN = (
    "TensorScope could not line your typed text up with these tokens unambiguously, so no "
    "grouping is shown. Every token listed is still exactly what the model received."
)

SYMBOLIC_WEIGHTS_NOTE = (
    "The learned weight matrices appear as symbols because they were never part of this "
    "capture — only their outputs were. Nothing here stands in for their values."
)

CACHED_KV_NOTE = (
    "A generation pass processes one new token but compares it against every earlier "
    "position: the prompt's keys and values were kept from the previous pass rather than "
    "recomputed. Those cached rows are inside the captured prepared tensors, which is why "
    "that pass's attention is one row wide and many columns long; the cache object itself "
    "is not saved."
)

DISPLAY_CALC_NOTE = (
    "Calculated here from captured operands, for this display. The model computed in "
    "bfloat16 and this arithmetic was done in double precision, so expect close agreement "
    "rather than identical digits. Where the model's own result was captured, it is shown "
    "beside this one rather than replaced by it."
)

EOS_IS_A_SCORED_ENTRY = (
    "End-of-sequence is an ordinary vocabulary entry with an ordinary learned score. "
    "Nothing separate decides to stop: the same scoring step simply ranked that entry "
    "highest, and the generation code treats winning as the signal to stop."
)

ILLUSTRATION_DISCLAIMER = (
    "Invented numbers, chosen to make the shape of the operation visible. Nothing in this "
    "box came from the model or from this run."
)

# Labels for the two captured arrays that are not part of the per-layer set, so have no
# TENSOR_LABELS row.  Kept here rather than in a view so every surface names them alike.
EXTRA_TENSOR_LABELS = {
    "layer_input": "This layer's input · X",
    "embedding": "Embedding rows for this pass",
    "final_logits": "Vocabulary scores for the final prompt position",
}

EXTRA_TENSOR_EXPLANATIONS = {
    "layer_input": (
        "The tensor this layer received. It is captured, but under another name: at layer "
        "0 it is the embedding output, and at every later layer it is the previous layer's "
        "own output. Nothing here is reconstructed — the array shown is whichever of "
        "those two the selected layer actually read. Axes are "
        "[batch, token, model feature]."
    ),
    "embedding": (
        "Captured embedding output for this pass: one row per token position, each the "
        "vector the lookup table holds for that token's ID. The learned lookup table "
        "itself is not persisted, only the rows this pass produced. Axes are "
        "[batch, token, model feature]."
    ),
    "final_logits": (
        "Captured vocabulary scores at the final prompt position — the one vector that "
        "chose the first generated token. One score per vocabulary entry. These are "
        "scores, not probabilities: nothing here was passed through softmax, and greedy "
        "decoding never needed to. The axis is [vocabulary token ID]."
    ),
}

# The symbol each captured array is written as, declared once so the reference page, the
# equations, the chips and the panel cannot drift into three notations for one tensor.
# `layer_input` is not a stored name: it is the layer's input X, which *is* captured, just
# addressed differently -- the embedding for layer 0 and the previous layer's output after
# that.  The registry resolves it; nothing invents it.
TENSOR_SYMBOLS = {
    "layer_input": "X",
    "normalized_input": "X_norm",
    "q": "Q_raw",
    "k": "K_raw",
    "v": "V_raw",
    "q_attended": "Q",
    "k_attended": "K",
    "v_attended": "V",
    "attention_scores": "S",
    "attention_weights": "A",
    "attention_output": "O",
    "layer_output": "X_next",
    "embedding": "E[id]",
    "final_logits": "logits",
}


# What each captured array is *for*, in one sentence, in the reader's terms.  Deliberately
# shorter than TENSOR_EXPLANATIONS: this is what the variable panel shows at a glance,
# above the shape and the sample, with the full paragraph a click further on.
VARIABLE_ROLES = {
    "layer_input": "What this layer received: the embedding rows at layer 0, and the "
                   "previous layer's output after that.",
    "normalized_input": "The layer's input after RMSNorm, ready for the projections.",
    "q": "What each position is looking for, before it is split into heads.",
    "k": "What each position advertises it can be matched on, before the split.",
    "v": "What each position will contribute if it is matched, before the split.",
    "q_attended": "The queries that actually entered the comparison: split into heads, "
                  "rotated by position.",
    "k_attended": "The keys that actually entered the comparison: split into heads, "
                  "rotated by position, with shared KV heads already copied out.",
    "v_attended": "The values the weights actually mixed, with shared KV heads already "
                  "copied out.",
    "attention_scores": "How well each query matched each key — already scaled and "
                        "already masked, exactly as softmax received it.",
    "attention_weights": "How much of each value each position took, after softmax. Each "
                         "query's row adds up to 1.",
    "attention_output": "The attention block's contribution, after the output projection. "
                        "Not A @ V.",
    "layer_output": "This layer's output, which is the next layer's input.",
    "embedding": "Each token's starting vector, looked up by its ID.",
    "final_logits": "One score per vocabulary entry, for the final prompt position.",
}


@dataclass(frozen=True)
class ConceptualVariable:
    """A symbol whose numbers this capture does not contain.

    `shape_text` is formatted against `learn_facts()`, and every number in it must be one
    read off a captured array — a weight's width is known because the activation it
    produced has that width, not because a config said so.  Where even that is unavailable
    the shape stays symbolic, which is the honest answer.
    """

    symbol: str
    title: str
    role: str
    provenance: str                  # PROVENANCE_LEARNED or PROVENANCE_ABSENT
    shape_text: str = ""
    axes_text: str = ""
    colour: str = "layer"
    captured_instead: tuple = ()     # keys of the captured arrays that *are* here


CONCEPTUAL_VARIABLES = {
    "W_Q": ConceptualVariable(
        "W_Q", "Query projection weights",
        "Turns a normalized token vector into query features. Learned during training.",
        PROVENANCE_LEARNED, "[{q_projection_width_text} × {hidden_size_text}]",
        "output feature × input feature", "q", ("q",)),
    "W_K": ConceptualVariable(
        "W_K", "Key projection weights",
        "Turns a normalized token vector into key features. Learned during training.",
        PROVENANCE_LEARNED, "[{kv_projection_width_text} × {hidden_size_text}]",
        "output feature × input feature", "k", ("k",)),
    "W_V": ConceptualVariable(
        "W_V", "Value projection weights",
        "Turns a normalized token vector into the features it will contribute. Learned "
        "during training.",
        PROVENANCE_LEARNED, "[{kv_projection_width_text} × {hidden_size_text}]",
        "output feature × input feature", "v", ("v",)),
    "W_O": ConceptualVariable(
        "W_O", "Output projection weights",
        "Maps the joined head results back to the model's own feature width. Learned "
        "during training.",
        PROVENANCE_LEARNED, "[{hidden_size_text} × {q_projection_width_text}]",
        "output feature × input feature", "output", ("attention_output",)),
    "W_U": ConceptualVariable(
        "W_U", "Vocabulary projection weights",
        "Scores every vocabulary entry against a final token vector. Learned during "
        "training.",
        PROVENANCE_LEARNED, "[{vocab_size} × {hidden_size_text}]",
        "vocabulary entry × model feature", "scores", ("final_logits",)),
    "E": ConceptualVariable(
        "E", "The embedding table",
        "One learned row per vocabulary entry. A token ID is a row number into it.",
        PROVENANCE_LEARNED, "[{vocab_size} × {hidden_size_text}]",
        "vocabulary entry × model feature", "norm", ("embedding",)),
    "g": ConceptualVariable(
        "g", "RMSNorm feature scales",
        "One learned multiplier per feature, applied after the division by magnitude.",
        PROVENANCE_LEARNED, "[{hidden_size_text}]", "model feature", "norm",
        ("normalized_input",)),
    "epsilon": ConceptualVariable(
        "ε", "RMSNorm stabiliser",
        "A tiny constant added inside the square root so a near-zero vector cannot divide "
        "by zero. Part of the architecture, not captured.",
        PROVENANCE_LEARNED, "a single number", "", "norm", ("normalized_input",)),
    "rms": ConceptualVariable(
        "rms(X)", "The magnitude statistic",
        "The root-mean-square of one position's features, which RMSNorm divides by. "
        "Computed and discarded.",
        PROVENANCE_ABSENT, "[{position_count_text} × 1]", "token position", "norm",
        ("normalized_input",)),
    "W_gate": ConceptualVariable(
        "W_gate", "Feed-forward gate weights",
        "Produces the values that are passed through SiLU to decide how much of the other "
        "branch to let through. Learned during training.",
        PROVENANCE_LEARNED, "[d_ff × {hidden_size_text}]",
        "interior feature × model feature", "norm", ("layer_output",)),
    "W_up": ConceptualVariable(
        "W_up", "Feed-forward up-projection weights",
        "Widens each position's vector into the feed-forward interior. Learned during "
        "training.",
        PROVENANCE_LEARNED, "[d_ff × {hidden_size_text}]",
        "interior feature × model feature", "norm", ("layer_output",)),
    "W_down": ConceptualVariable(
        "W_down", "Feed-forward down-projection weights",
        "Brings the interior back to the model's feature width. Learned during training.",
        PROVENANCE_LEARNED, "[{hidden_size_text} × d_ff]",
        "model feature × interior feature", "output", ("layer_output",)),
    "QK_raw": ConceptualVariable(
        "QKᵀ", "The unscaled, unmasked comparison",
        "Every query against every key, before the scale factor and before the mask. The "
        "captured scores already have both applied.",
        PROVENANCE_ABSENT,
        "[{query_head_count_text} × {position_count_text} × {position_count_text}]",
        "head × query token × key token", "scores", ("attention_scores",)),
    "M": ConceptualVariable(
        "M", "The causal mask",
        "Added to the comparison to stop a position reading later ones. A large negative "
        "number where reading is not allowed, zero where it is.",
        PROVENANCE_ABSENT,
        "[{position_count_text} × {position_count_text}]", "query token × key token",
        "layer", ("attention_scores",)),
    "Z_head": ConceptualVariable(
        "Z_h", "One head's value mix",
        "A_h V_h: this head's weighted blend of the values it attended to, before the "
        "heads are joined.",
        PROVENANCE_ABSENT,
        "[{position_count_text} × {head_dim_text}]", "query token × feature within head",
        "v", ("attention_weights", "v_attended", "attention_output")),
    "Z_joined": ConceptualVariable(
        "Z_joined", "The joined heads",
        "Every head's mix laid side by side, immediately before the output projection.",
        PROVENANCE_ABSENT,
        "[{position_count_text} × {q_projection_width_text}]",
        "query token × joined feature", "v", ("attention_output",)),
    "H": ConceptualVariable(
        "H", "The residual sum after attention",
        "X + O: the layer's input with the attention block's contribution added back in.",
        PROVENANCE_ABSENT, "[{position_count_text} × {hidden_size_text}]",
        "token position × model feature", "layer", ("attention_output", "layer_output")),
    "H_norm": ConceptualVariable(
        "RMSNorm(H)", "The feed-forward input",
        "H after the layer's second normalization, which is what the feed-forward network "
        "reads.",
        PROVENANCE_ABSENT, "[{position_count_text} × {hidden_size_text}]",
        "token position × model feature", "norm", ("layer_output",)),
    "F": ConceptualVariable(
        "F", "The feed-forward output",
        "What the feed-forward network contributed, before it was added back to H.",
        PROVENANCE_ABSENT, "[{position_count_text} × {hidden_size_text}]",
        "token position × model feature", "norm", ("layer_output",)),
    "X_final_norm": ConceptualVariable(
        "RMSNorm(X_final)", "The final normalized representation",
        "The last layer's output after the model's final normalization — the vector the "
        "vocabulary projection actually reads.",
        PROVENANCE_ABSENT, "[{position_count_text} × {hidden_size_text}]",
        "token position × model feature", "norm", ("layer_output", "final_logits")),
    "prefill_logits_full": ConceptualVariable(
        "logits(all positions)", "Scores for every prompt position",
        "The model scored every position in the prompt, not just the last one. Only the "
        "final position's row was stored, because that is the only row greedy decoding "
        "read — and keeping all of them would be one number per position per vocabulary "
        "entry.",
        PROVENANCE_ABSENT, "[{position_count_text} × {vocab_size}]",
        "token position × vocabulary token ID", "scores", ("final_logits",)),
}

# Order fixes the reading sequence and, with the two-way assertion in self_test(), makes it
# impossible to add a conceptual variable without describing it or to describe one that no
# longer exists.
CONCEPTUAL_VARIABLE_ORDER = [
    "E", "g", "epsilon", "rms", "W_Q", "W_K", "W_V", "QK_raw", "M", "Z_head", "Z_joined",
    "W_O", "H", "H_norm", "W_gate", "W_up", "W_down", "F", "X_final_norm", "W_U",
    "prefill_logits_full",
]

assert set(CONCEPTUAL_VARIABLE_ORDER) == set(CONCEPTUAL_VARIABLES), \
    "CONCEPTUAL_VARIABLES and its order disagree"



# ── Page 1: Understand the calculation ──────────────────────────────────────────
# An illustrated reference in the same order and the same notation as the worked
# notebook, so a reader can leave the run at any operation, read the idea behind it, and
# come back.  Each topic is: what this is for, the formula, the symbols in it, a picture,
# and -- only if the reader asks -- a longer explanation.
#
# Two rules this copy lives by.  First, nothing here describes *this run*: the reference
# is about the architecture, the notebook is about the capture, and mixing them is how a
# reader ends up believing an invented number came from the model.  Second, every claim
# about what a part of the model *means* is either something the mathematics forces or is
# labelled as an example of what could be learned -- never as a fact about a head.


@dataclass(frozen=True)
class Illustration:
    """Invented content, fenced off so it can never read as a captured result.

    The analogy for what a head might match on, and the small arithmetic example of a
    nonlinearity, are both genuinely useful and both genuinely made up.  Giving them their
    own type means the page renders every one of them inside the same visibly separate box
    with the same disclaimer, instead of relying on an author to remember the hedge.
    """

    heading: str
    body: str
    disclaimer: str = ILLUSTRATION_DISCLAIMER


ILLUSTRATIONS = {
    "qkv": Illustration(
        "An example of a relationship a head could learn",
        "Suppose some head had learned to connect an adjective to the noun it describes. "
        "In “the blue sky”, its query for the position holding “blue” would be "
        "shaped to ask “am I describing a noun nearby?”, and its key for the position "
        "holding “sky” would advertise “I am a noun that can be described”. Those "
        "two vectors would then have a large dot product, so that position's value would be "
        "mixed in heavily, and “blue” would end up carrying information about "
        "“sky”.\n\n"
        "That is a picture of what the query-key mechanism makes possible, and it is why "
        "having separate queries and keys buys anything at all. It is not a claim about any "
        "head in any model. Heads are not assigned jobs: whatever a particular head matches "
        "on is whatever happened to reduce the training loss, it frequently has no tidy "
        "description in grammatical terms, and this capture contains no evidence either "
        "way. Where the run page shows a head's weights, it says where that head's weights "
        "went and stops there."),
    "block": Illustration(
        "Why a nonlinearity is needed at all",
        "Take two made-up weights, 2 and −3, and feed them a made-up input of 4. A "
        "purely linear path gives 2 × 4 = 8 and −3 × 4 = −12, and stacking more "
        "linear steps on top only ever multiplies those numbers again — any number of "
        "stacked linear layers collapses into one linear layer, which can only ever draw a "
        "straight boundary.\n\n"
        "Now send each through SiLU, which leaves comfortably positive values nearly alone "
        "and squeezes negative ones towards zero: 8 stays about 8, while −12 becomes "
        "about −0.0001. The two paths no longer scale together, so the next layer can "
        "treat “both features present” differently from “only one present”. That "
        "is the whole purpose of the gate: it lets later layers express conditions, not "
        "just sums.\n\n"
        "Note what this is not. SiLU does not confine values to 0…1 — 8 came out as 8 "
        "— and the feed-forward network is not a normalizer. Every number in this box was "
        "chosen by hand to make the arithmetic small."),
}


@dataclass(frozen=True)
class ReferenceTopic:
    """One topic of the illustrated reference.

    `visual` names diagram factories that `tensorscope_reference` knows how to build from
    this run's geometry; `symbols` names registry keys, so the notation in a topic cannot
    drift from the notation in the notebook.  `operations` says which of the worked
    notebook's steps this topic explains, and `--self-test` uses it to check that the two
    pages really are in the same order.
    """

    key: str
    nav: str
    heading: str
    purpose: str
    formula: str = ""
    caption: str = ""
    deeper: str = ""
    visual: tuple = ()
    symbols: tuple = ()
    operations: tuple = ()
    illustration: str = ""


REFERENCE_TOPICS = [
    ReferenceTopic(
        "notation", "How to read the formulas",
        "The notation used on both pages",
        "A token's features are written as a row, so a sequence of tokens is rows stacked "
        "downwards and every transformation is applied on the right. Fix that one "
        "convention and every formula on these pages has the same shape.",
        formula="X Wᵀ   —   rows in, rows out",
        caption="Row-vector convention, used everywhere on both pages.",
        deeper="The transpose is not decoration. PyTorch stores a linear layer's weights "
               "as [output features × input features], so the multiplication that maps "
               "{hidden_size_text} input features to a wider output is written X Wᵀ. "
               "Shapes are written with the batch axis dropped — "
               "[{position_count_text} × {hidden_size_text}] rather than "
               "[1 × {position_count_text} × {hidden_size_text}] — and wherever that "
               "happens the page says so and gives the stored shape too. One other habit "
               "worth knowing: a subscript h, as in Q_h, means “the slice belonging to "
               "head h”.",
        symbols=("layer_input", "normalized_input", "q_attended", "k_attended",
                 "v_attended", "attention_scores", "attention_weights",
                 "attention_output", "layer_output"),
    ),
    ReferenceTopic(
        "ids", "Tokens and IDs",
        "What the model actually receives",
        "A model does not read text. A tokenizer cuts the text into vocabulary entries — "
        "whole words, word fragments, punctuation, single spaces — and hands the model "
        "the entries' ID numbers. An ID is a row number in a table, nothing more: there is "
        "no sense in which ID 4096 is twice ID 2048.",
        formula="text  →  [id₀, id₁, …]",
        caption="Every position is one vocabulary entry.",
        deeper="Two consequences catch people out. First, a chat model's input is not what "
               "you typed: a chat template wraps it in control markers that say where your "
               "turn starts and where the model's begins, and those become tokens too, so "
               "a three-word question can arrive as fifteen positions. Second, token "
               "boundaries are a property of the vocabulary, not of English, so “ "
               "color” with its leading space can be one entry while an unusual word "
               "splits into four. The run page lists every position this run received, "
               "with the ones you typed marked apart from the ones the template added.",
        visual=("token_ids",),
        operations=("input",),
    ),
    ReferenceTopic(
        "shapes", "Vectors, tensors and shapes",
        "From an ID to a vector, and what a shape tells you",
        "Each ID is used once, as a row number: the model looks that row up in a learned "
        "table and gets a list of {hidden_size_text} numbers. That list is the token's "
        "starting vector. Stack one per position and you have the tensor the first layer "
        "reads.",
        formula="shape(X) = [batch, token positions, features]",
        caption="One row per position; one number per feature.",
        deeper="A vector of {hidden_size_text} numbers is one point in a "
               "{hidden_size_text}-dimensional space — not a "
               "{hidden_size_text} × {hidden_size_text} grid of anything. Its "
               "coordinates were learned during training, and the only reason they mean "
               "something is that training arranged for tokens used in similar ways to "
               "land near each other.\n\nA shape is the arrangement of the numbers, not "
               "the numbers. [batch, token positions, features] says: how many sequences "
               "at once, how many positions in each, how many numbers per position. "
               "TensorScope runs one sequence, so the batch axis is always 1 here. Reading "
               "a shape as though it were content — “the tensor is 2,560” — is the "
               "single most common slip; the shape line and the values are kept apart on "
               "the run page for exactly that reason.",
        visual=("embedding_lookup", "input_tensor"),
        symbols=("embedding", "layer_input"),
        operations=("embed", "shape"),
    ),
    ReferenceTopic(
        "weights", "Where the numbers come from",
        "Learned weights, and why nothing is learning now",
        "Every matrix in this architecture is a block of numbers that were adjusted during "
        "training by gradient descent and then frozen. Running the model — what this "
        "capture recorded — changes none of them. Each weight matrix is read, used, and "
        "left exactly as it was.",
        formula="training: W ← W − η ∇W        inference: W is read, never written",
        caption="Training sets the weights; inference only uses them.",
        deeper="During training the model is shown text, asked to predict the next token, "
               "and scored on how wrong it was; the gradient of that error with respect to "
               "every weight says which way to nudge each one. Repeat on enough text and "
               "the weights end up encoding whatever regularities reduce the error. That "
               "is the only place the model's knowledge lives.\n\nAt inference time there "
               "is no error signal and no update — the same prompt run twice with greedy "
               "decoding gives the same answer twice. This matters for reading the run "
               "page: TensorScope captures activations, the numbers that flow *through* "
               "the weights, not the weights themselves. So a weight matrix on the run "
               "page is always described by its shape and its job, with its own values "
               "marked as not saved, while the tensor it produced is right there to open.",
        visual=("training_vs_inference",),
        symbols=("W_Q", "W_K", "W_V", "W_O", "W_U", "E", "g"),
    ),
    ReferenceTopic(
        "norm", "X and X_norm",
        "Why the input is normalized first",
        "Across dozens of layers, the numbers in the residual stream can drift to very "
        "different magnitudes, and a projection fed wildly scaled inputs produces wildly "
        "scaled outputs. RMSNorm puts each position's vector on a predictable scale before "
        "the projections read it.",
        formula="X_norm[i] = X[i] / √(mean(X[i]²) + ε)  ⊙  g",
        caption="Same shape, different values. Nothing moves between positions.",
        deeper="Read it in three parts. mean(X[i]²) is the average of the squared "
               "features of one position, so its square root is a measure of that "
               "position's overall magnitude; dividing by it leaves the direction of the "
               "vector intact and standardises its length. ε is a tiny constant inside "
               "the square root, there only so a near-zero vector cannot divide by zero. "
               "g is a learned multiplier per feature, which lets training decide that "
               "some features should carry more weight than the plain division would give "
               "them.\n\nThe important structural point: X and X_norm have exactly the "
               "same shape and both hold one vector per position. Normalization changes "
               "values, not arrangement, and it works on each position independently — "
               "no information crosses between positions here. That is attention's job, "
               "and it happens next. Neither g nor ε nor the magnitude statistic is "
               "saved in a capture; X_norm, the result, is.",
        visual=("norm_pair",),
        symbols=("layer_input", "normalized_input", "g", "epsilon", "rms"),
        operations=("norm",),
    ),
    ReferenceTopic(
        "qkv", "Q, K and V",
        "Three questions asked of the same vector",
        "The same normalized vector is read by three different learned projections. The "
        "query is what this position is looking for; the key is what it advertises it can "
        "be matched on; the value is what it will contribute to whoever matches it. "
        "Nothing about the vector says which is which — the three weight matrices do.",
        formula="Q_raw = X_norm W_Qᵀ    K_raw = X_norm W_Kᵀ    V_raw = X_norm W_Vᵀ",
        caption="One input, three learned views of it.",
        deeper="The separation is what makes attention asymmetric, and the asymmetry is "
               "the point. If a position used one vector for both asking and advertising, "
               "“A is relevant to B” would force “B is relevant to A”. Splitting "
               "the two lets a position seek something it is not itself. The value is "
               "separate again for a different reason: what a position is worth reading "
               "*for* need not be the same as what makes it findable.\n\nThe widths differ "
               "and the difference is architectural, not arbitrary: in this model the query "
               "projection produces {q_projection_width_text} features while the key and "
               "value projections produce {kv_projection_width_text}, because several query "
               "heads share each key/value head. The dot product that follows is still "
               "defined, because the sharing is undone by copying before any comparison "
               "happens.",
        visual=("qkv_projection",),
        symbols=("normalized_input", "W_Q", "W_K", "W_V", "q", "k", "v"),
        operations=("project",),
        illustration="qkv",
    ),
    ReferenceTopic(
        "heads", "Heads and layers",
        "Two different kinds of repetition",
        "A head is a parallel slice *within* one attention step: the projected features are "
        "cut into groups, and each group runs the whole compare-and-mix independently, at "
        "the same time, on the same input. A layer is a sequential repetition of the entire "
        "block, and it reads what the layer before it produced.",
        formula="{q_projection_width_text} = {query_head_count_text} heads "
                "× {head_dim_text} features",
        caption="Heads run side by side; layers run one after another.",
        deeper="Splitting into heads costs nothing and buys independence. One head can "
               "attend to the position two words back while another attends to the start "
               "of the sentence, because each compares only its own slice of the features. "
               "Note what the split is *not*: it is a regrouping of numbers that already "
               "exist, with no arithmetic at all. The arithmetic happened in the "
               "projection, which is why the run page draws those two as separate stages "
               "with a labelled boundary.\n\nTwo details that trip people up. The head "
               "width is not necessarily the hidden size divided by the head count — in "
               "this model it is {head_dim_text}, and TensorScope reads it off the captured "
               "tensors rather than computing it. And several query heads can share one "
               "key/value head: {kv_head_count_text} key/value heads serve "
               "{query_head_count_text} query heads here, so each is copied "
               "{kv_group_size} times before the comparison. Through all of it the "
               "{position_count_text} token positions stay separate rows. Layers, by "
               "contrast, are genuinely sequential and each has its own full set of learned "
               "parameters — nothing is shared between them.",
        visual=("head_split", "kv_share", "layer_stack"),
        symbols=("q", "q_attended", "k_attended", "v_attended"),
        operations=("prepare", "layers"),
    ),
    ReferenceTopic(
        "matching", "Comparing, then mixing",
        "Why compare Q with K, and then use the result on V",
        "Attention is two steps that are easy to conflate. First every query is compared "
        "with every key, which produces a number per pair saying how relevant that key's "
        "position is. Those numbers are turned into weights. Only then are the weights used "
        "to take a weighted average of the values — so what gets read is decided "
        "separately from what is read.",
        formula="S = s·(Q Kᵀ) + M      A = softmax(S)      Z_h = A_h V_h",
        caption="Compare to decide; mix to collect.",
        deeper="The comparison is a dot product: multiply the two vectors entry by entry "
               "and add up the {head_dim_text} products. A large positive total means the "
               "two vectors point the same way along the features this head cares about, "
               "which is what “relevant” means here. There is nothing deeper to it — "
               "the whole of attention's decision is one dot product per pair.\n\nTwo "
               "adjustments before softmax. The totals are multiplied by "
               "{attention_scale_text}, because adding {head_dim_text} products gives "
               "numbers that grow with the head width, and softmax on very large numbers "
               "collapses to picking exactly one key. And a causal mask is added, so no "
               "position can read a later one.\n\nThe separation of mixing from comparing "
               "is what makes attention a *soft* lookup: a hard lookup would take one "
               "position's value, while this takes a blend of all permitted ones in "
               "proportion to the weights. A capture keeps the comparison's result and the "
               "weights, but not the per-head blend itself — what it keeps next is the "
               "output after the heads have been joined and projected.",
        visual=("qk_compare", "attention_cells"),
        symbols=("q_attended", "k_attended", "v_attended", "attention_scores",
                 "attention_weights", "QK_raw", "Z_head"),
        operations=("scores", "mix"),
    ),
    ReferenceTopic(
        "softmax", "Softmax and the causal mask",
        "Turning scores into a mixture, and keeping the future out",
        "Softmax takes a row of scores and returns a row of positive weights that add up to "
        "1, keeping the order and making the gaps proportional. The causal mask is added "
        "before it: a large negative number on every pair a position is not allowed to "
        "read, so those weights come out at effectively zero.",
        formula="A[i,j] = exp(S[i,j]) / Σₖ exp(S[i,k])",
        caption="Each query's row of weights sums to 1.",
        deeper="Why exponentiate at all? Scores can be negative and a weighted average "
               "needs non-negative weights; exp sends every real number to a positive one "
               "and preserves order. Dividing by the row's total then makes the row sum to "
               "1, which is what makes the result a mixture. The consequence worth "
               "remembering is that softmax cares only about the *differences* between "
               "scores: add 10 to every score in a row and the weights do not change.\n\n"
               "The mask exists because the model is trained to predict the next token. If "
               "position 3 could see position 7, it could read the answer instead of "
               "predicting it, and the training signal would be worthless. Note what the "
               "mask is, mechanically: a value *added* to the scores, large and negative "
               "where reading is not allowed — large enough that exp of it is "
               "indistinguishable from zero, but a real finite number in a real dtype, not "
               "necessarily negative infinity. The run page shows the captured scores with "
               "the scale and the mask already in them, because that is exactly the tensor "
               "softmax received.",
        visual=("softmax_flow", "causal_mask"),
        symbols=("attention_scores", "attention_weights", "M"),
        operations=("scores",),
    ),
    ReferenceTopic(
        "block", "Finishing the layer",
        "Output projection, residual connections, and the feed-forward network",
        "Three jobs remain. The heads' results are joined and projected once, to get back "
        "to the model's own feature width. The result is added to what came in, rather than "
        "replacing it. And a feed-forward network then processes each position on its own, "
        "with its own residual addition.",
        formula="O = Z_joined W_Oᵀ     H = X + O     "
                "F = {ffn_description}     X_next = H + F",
        caption="Attention gathers; the feed-forward network processes; both are added back.",
        deeper="**The output projection** does two things at once. It maps "
               "{q_projection_width_text} joined features back to {hidden_size_text}, so "
               "the result can rejoin the stream, and it lets the model mix the heads — "
               "until this point each head's output sat in its own slice, unable to "
               "influence the others.\n\n**The residual connections** are the additions. "
               "Writing H = X + O rather than H = O means each block proposes a change to "
               "the stream instead of replacing it, so information can survive dozens of "
               "layers untouched, and the gradient during training has a short path back to "
               "early layers. It is also why it is reasonable to call the running vector a "
               "“stream” at all.\n\n**The feed-forward network** is learned nonlinear "
               "processing of one position at a time. Attention is the only step that moves "
               "information between positions; this step takes what attention gathered for "
               "a position and computes something new from it, with no reference to any "
               "other position. In this model it is gated: one projection produces values "
               "that are passed through SiLU, those multiply a second projection's output "
               "element by element, and a third projection brings the result back to "
               "{hidden_size_text} features. The gate is what lets the network express "
               "“only if” rather than merely “how much”.\n\nWhat it is not: it does "
               "not normalize, it does not turn anything into a probability, and it does "
               "not squash values into 0…1. Several of its intermediates — the joined "
               "heads, the residual sums, everything inside the feed-forward network — are "
               "computed and discarded, so the run page describes them and shows the one "
               "tensor that does survive: the layer's output.",
        visual=("residual", "gated_ffn"),
        symbols=("Z_head", "Z_joined", "W_O", "attention_output", "H", "H_norm",
                 "W_gate", "W_up", "W_down", "F", "layer_output"),
        operations=("join", "block"),
        illustration="block",
    ),
    ReferenceTopic(
        "vocabulary", "From vectors to a token",
        "How vocabulary scores become a selected token",
        "After the last layer, one more learned matrix compares a position's final vector "
        "against every entry in the vocabulary, giving one score per entry. The generation "
        "code then applies a selection rule to those scores. The model produces the scores; "
        "it does not choose.",
        formula="logits = RMSNorm(X_final) W_Uᵀ      "
                "next = argmax(logits[last])",
        caption="One score per vocabulary entry; the rule picks the winner.",
        deeper="The scores are called logits, and they are scores, not probabilities. "
               "Nothing here divides by a total, so they do not sum to anything in "
               "particular and a score of 34.5 is only meaningful next to the other scores "
               "in the same vector. Softmax would turn them into probabilities, and for "
               "greedy decoding there is no reason to: the largest score is the largest "
               "score either way.\n\nTwo things are worth separating carefully. The model "
               "computes a vector for *every* position in its input, not just the last one "
               "— that is how it can be trained on every position at once. But generation "
               "reads only the final position's row, because that is the only one whose "
               "“next token” has not been written yet. And the selection rule lives in "
               "the generation code, not the model: argmax always takes the highest score, "
               "while sampling rules would sometimes take another, from the very same "
               "scores. TensorScope records which rule a run used.",
        visual=("vocabulary_projection",),
        symbols=("layer_output", "X_final_norm", "W_U", "final_logits",
                 "prefill_logits_full"),
        operations=("decide",),
    ),
    ReferenceTopic(
        "stopping", "Why generation stops",
        "What ends an answer",
        "Generating one token does not end anything: the token is appended to the input and "
        "the whole thing runs again. What stops it is that one of the vocabulary entries "
        "means “I am finished”, and sooner or later that entry wins.",
        formula="loop:  score  →  select  →  append        until  selected = EOS",
        caption="Stopping is a scored entry winning, not a separate decision.",
        deeper="End-of-sequence is an ordinary vocabulary entry with an ordinary learned "
               "score. Training taught the model to score it highly where a complete "
               "answer has been given, which is why a well-trained model stops at the end "
               "of a sentence rather than mid-word. Nothing separate watches for "
               "completion.\n\nBecause the marker is a selection like any other, it is not "
               "part of the answer: the generation code sees that it won and stops, without "
               "adding it to the text. So an answer of {generated_count} tokens had "
               "{generated_count} tokens scored and selected, plus one further scoring step "
               "whose winner was the stopping marker. Other limits can end a run first — a "
               "cap on how many tokens to generate will cut an answer off mid-sentence, "
               "which looks quite different from stopping properly. The run page says which "
               "of these actually happened.\n\nOne efficiency note that explains an "
               "otherwise confusing shape: later steps do not recompute the whole prompt. "
               "The keys and values for positions already processed are cached and reused, "
               "so a generation step computes a query for exactly one new position while "
               "attending to every cached key — which is why that pass's attention is one "
               "row wide and many columns long.",
        visual=("stopping_loop",),
        symbols=("final_logits",),
    ),
]

REFERENCE_TOPIC_INDEX = {topic.key: topic for topic in REFERENCE_TOPICS}

assert len(REFERENCE_TOPIC_INDEX) == len(REFERENCE_TOPICS), "duplicate reference topic key"



# ── Page 2: Follow this run ─────────────────────────────────────────────────────
# One scrollable worked calculation over one real capture, in the order the model did it.
# Twelve operations and then the continuation, each an anchor on the same page, so a
# reader can scroll back to the variable they met four steps ago instead of navigating
# away from it.
#
# `terms` is the clickable equation: a registry key becomes a chip, anything else is an
# operator.  `variables` is the list beneath it, with shapes and provenance.  `formula` is
# the plain-text fallback for an equation whose terms are not all variables.  `tiers`
# declares which evidence badges an operation is allowed to show, which is what lets a
# test catch an operation that quietly started presenting a derived number as captured.


@dataclass(frozen=True)
class Operation:
    """One anchored step of the worked notebook."""

    key: str
    number: int
    nav: str
    heading: str
    lede: str
    formula: str = ""
    terms: tuple = ()
    variables: tuple = ()
    tiers: tuple = (EVIDENCE_OBSERVED,)
    check: str = ""
    topic: str = ""
    fields: tuple = ()


NOTEBOOK_OPERATIONS = [
    Operation(
        "input", 1, "The model's input",
        "What this run actually received",
        "Your prompt reached the model as {position_count} positions, each one vocabulary "
        "entry. {typed_token_summary} The rest were added by the chat template, which is "
        "part of how an instruction-tuned model is told whose turn it is. Every position "
        "below is a token this run really was given, in the order it was given them.",
        terms=("text", "→", "[id₀, id₁, …]"),
        tiers=(EVIDENCE_OBSERVED,),
        topic="ids",
    ),
    Operation(
        "embed", 2, "Embedding lookup",
        "From a token ID to a vector",
        "A token ID is used exactly once: as a row number. The model reads that row out of "
        "a learned table and gets {hidden_size_text} numbers — one point in a "
        "{hidden_size_text}-dimensional space, not a {hidden_size_text} × "
        "{hidden_size_text} grid. Those coordinates were fixed during training; this run "
        "only looked them up.",
        terms=("E", "[", "selected token ID", "]", "→", "embedding"),
        variables=("embedding", "E"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL),
        check="embed",
        topic="shapes",
    ),
    Operation(
        "shape", 3, "The input tensor",
        "One tensor for the whole sequence",
        "Stack one embedding row per position and you have the tensor the first layer "
        "reads. The line below is the *arrangement* of those numbers: one batch, "
        "{position_count_text} token positions, {hidden_size_text} features each. It is "
        "not the numbers themselves — open X to see those.",
        formula="shape(X) = [{batch_size}, {position_count}, {hidden_size}]",
        variables=("layer_input",),
        tiers=(EVIDENCE_OBSERVED,),
        check="shape",
        topic="shapes",
    ),
    Operation(
        "norm", 4, "Input normalization",
        "Putting each position's vector on a predictable scale",
        "RMSNorm divides each position's vector by a measure of its own magnitude and then "
        "scales it feature by feature with learned values. X and X_norm hold the same "
        "number of token vectors in the same arrangement; only the values differ, and no "
        "number crossed between positions. The result was captured; the magnitude "
        "statistic, the learned scales and ε were not.",
        formula="X_norm[i] = X[i] / √(mean(X[i]²) + ε)  ⊙  g",
        terms=("normalized_input", "=", "RMSNorm", "(", "layer_input", ")"),
        variables=("layer_input", "g", "epsilon", "rms", "normalized_input"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL),
        topic="norm",
    ),
    Operation(
        "project", 5, "Raw projections",
        "Three learned views of the same vector",
        "The same normalized vector is read by three different learned matrices. The query "
        "is what a position is looking for, the key is what it advertises, the value is "
        "what it contributes when matched. All three weight matrices were learned during "
        "training; none of their values are in this capture, and all three of their "
        "outputs are.",
        formula="Q_raw = X_norm W_Qᵀ        K_raw = X_norm W_Kᵀ        "
                "V_raw = X_norm W_Vᵀ",
        terms=("q", "=", "normalized_input", "@", "W_Q"),
        variables=("normalized_input", "W_Q", "q", "W_K", "k", "W_V", "v"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL),
        topic="qkv",
    ),
    Operation(
        "prepare", 6, "Preparing for attention",
        "Four distinct things happen here",
        "Between the raw projections and the comparison there are four separate steps, and "
        "they are different kinds of operation. The projection computed new numbers. "
        "Reshaping into {query_head_count} heads of {head_dim} only regroups the numbers "
        "that already exist — no arithmetic at all. A rotation by position is then applied, "
        "and any per-head normalization the architecture uses. Finally each of the "
        "{kv_head_count} key/value heads is copied out to the {kv_group_size} query heads "
        "that share it. Only the finished tensors were captured, so the steps between are "
        "described here rather than shown.",
        formula="Q_raw → reshape to [{query_head_count} × {head_dim}] → "
                "rotate by position → Q",
        terms=("q", "→", "q_attended"),
        variables=("q", "q_attended", "k", "k_attended", "v", "v_attended"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL),
        topic="heads",
    ),
    Operation(
        "scores", 7, "Scores and weights",
        "Deciding what to read",
        "Every query is compared with every key by a dot product, the totals are scaled by "
        "{attention_scale_text}, and a causal mask is added so no position can read a later "
        "one. Softmax then turns each query's row of scores into weights that add up to 1. "
        "The captured scores already have the scale and the mask in them — they are exactly "
        "the tensor softmax received — and the mask alone was not saved separately.",
        formula="S = s·(Q Kᵀ) + M            A = softmax(S)",
        terms=("attention_scores", "→ softmax →", "attention_weights"),
        variables=("q_attended", "k_attended", "QK_raw", "M", "attention_scores",
                   "attention_weights"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED),
        check="scores",
        topic="softmax",
    ),
    Operation(
        "mix", 8, "Mixing the values",
        "Collecting what was decided",
        "Each head now takes a weighted average of the values it attended to, using the "
        "weights from the previous step. That per-head result was computed by the model and "
        "then discarded: what this capture keeps is the attention block's output after the "
        "heads have been joined and projected. One entry of it is worked through below from "
        "captured operands, marked as calculated for this display.",
        formula="Z_h = A_h V_h",
        terms=("Z_head", "=", "attention_weights", "@", "v_attended"),
        variables=("attention_weights", "v_attended", "Z_head"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED),
        topic="matching",
    ),
    Operation(
        "join", 9, "Joining the heads and projecting",
        "Back to the model's own feature width",
        "Every head's result is laid side by side, giving {q_projection_width} features, and "
        "one more learned matrix maps those back to {hidden_size}. This is also the only "
        "point at which heads can influence each other. The captured tensor here is the "
        "output *after* that projection — it is not the per-head mix, and it is not "
        "A @ V.",
        formula="Z_joined = concat(Z_h)            O = Z_joined W_Oᵀ",
        terms=("attention_output", "=", "Z_joined", "@", "W_O"),
        variables=("Z_joined", "W_O", "attention_output"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL),
        topic="block",
    ),
    Operation(
        "block", 10, "Residual and feed-forward",
        "Finishing one layer",
        "The attention block's output is added to what came into the layer, rather than "
        "replacing it. The sum is normalized again and passed through the feed-forward "
        "network — learned nonlinear processing of each position on its own, gated in this "
        "model — and that result is added back in turn. The layer's output was captured; "
        "the residual sums and everything inside the feed-forward network were not.",
        formula="H = X + O            F = {ffn_description}            X_next = H + F",
        terms=("layer_output", "=", "H", "+", "F"),
        variables=("layer_input", "attention_output", "H", "H_norm", "W_gate", "W_up",
                   "W_down", "F", "layer_output"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL),
        check="block",
        topic="block",
    ),
    Operation(
        "layers", 11, "The remaining layers",
        "That completed one layer of {layer_count}",
        "X_next is the next layer's X, and the whole of operations 4 to 10 happens again "
        "with a different set of learned parameters — nothing is shared between layers. "
        "The {position_count} token positions stay separate rows throughout: attention is "
        "the only step that moves anything between them. Every one of the "
        "{remaining_layer_count} other layers was captured in full and is listed below.",
        formula="X → layer 0 → layer 1 → … → "
                "layer {remaining_layer_count} → X_final",
        variables=("layer_output",),
        tiers=(EVIDENCE_OBSERVED,),
        topic="heads",
    ),
    Operation(
        "decide", 12, "The token decision",
        "Choosing the first generated token",
        "After the last layer, attention is finished and only one position matters: the "
        "final prompt position, because it is the only one whose next token has not been "
        "written yet. Its vector is normalized once more and compared against every "
        "vocabulary entry, giving one score per entry. The generation code then applies the "
        "recorded selection rule — {decoding_rule} — to those scores.",
        formula="logits = RMSNorm(X_final[{position_count}−1]) W_Uᵀ        "
                "selected_token_id = argmax(logits)",
        terms=("final_logits", "→ argmax →", "selected token ID"),
        variables=("layer_output", "X_final_norm", "W_U", "final_logits",
                   "prefill_logits_full"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED),
        check="decide",
        topic="vocabulary",
    ),
]

OPERATION_INDEX = {operation.key: operation for operation in NOTEBOOK_OPERATIONS}
OPERATION_ORDER = [operation.key for operation in NOTEBOOK_OPERATIONS]

# The continuation is a section of the same page but not one of the twelve operations: it
# is what happened *after* the calculation the notebook walks.  Keeping it out of
# NOTEBOOK_OPERATIONS is what makes `len(NOTEBOOK_OPERATIONS) == 12` while the page has
# thirteen anchors, and both numbers are asserted so neither can drift.
CONTINUATION = Operation(
    "continuation", 0, "The rest of the answer",
    "What happened after that token",
    "The calculation above produced one token. {continuation_summary} Each later step ran "
    "the same layers with the same learned parameters over the text so far, and reused the "
    "keys and values already computed for earlier positions rather than recalculating the "
    "prompt. Detailed layer tensors were captured for the prompt pass and the first "
    "generated-token pass only; for the later steps this run kept the candidate scores and "
    "nothing else.",
    variables=("final_logits",),
    tiers=(EVIDENCE_OBSERVED, EVIDENCE_UNATTESTED, EVIDENCE_DERIVED, EVIDENCE_CONCEPTUAL),
    topic="stopping",
)

assert len(NOTEBOOK_OPERATIONS) == 12, len(NOTEBOOK_OPERATIONS)
assert len(OPERATION_INDEX) == 12, "duplicate operation key"
assert [operation.number for operation in NOTEBOOK_OPERATIONS] == list(range(1, 13))


# ── Optional checks ─────────────────────────────────────────────────────────────
# Nothing in the notebook is gated on answering one of these: picking an option and
# pressing "show me the answer" reveal the same explanation.  They exist because a reader
# who has predicted an answer notices the real one, and because each of the five targets a
# specific misconception that the surrounding prose can state but not dislodge.

NOTEBOOK_CHECKS = {
    "embed": {
        "operation": "embed",
        "question": "The selected token's embedding is a list of {hidden_size} numbers. "
                    "What does that describe?",
        "options": [
            "A point in a {hidden_size}-dimensional space",
            "A {hidden_size} × {hidden_size} grid of relationships",
            "One number for each of the model's {layer_count} layers",
        ],
        "answer": 0,
        "explanation": "One vector, {hidden_size} coordinates, one point. The table it was "
                       "read from does have two dimensions — one row per vocabulary entry, "
                       "{hidden_size} columns — but a single token's embedding is one of "
                       "those rows, and the number of layers has nothing to do with its "
                       "width.",
        "fields": (),
    },
    "shape": {
        "operation": "shape",
        "question": "shape(X) = [{batch_size}, {position_count}, {hidden_size}]. What have "
                    "you just been told?",
        "options": [
            "How the numbers are arranged, and how many there are",
            "The values in the tensor",
            "The strength of the connection between each pair of tokens",
        ],
        "answer": 0,
        "explanation": "A shape is an arrangement, not contents. This one says: one "
                       "sequence, {position_count} positions, {hidden_size} features per "
                       "position — so there are {batch_size} × {position_count} × "
                       "{hidden_size} numbers in there somewhere, and the shape says "
                       "nothing whatever about what they are. Connections between tokens "
                       "do not exist yet; that is attention's doing, three operations "
                       "further on.",
        "fields": (),
    },
    "scores": {
        "operation": "scores",
        "question": "In the captured weight row for a query position, what weight do the "
                    "positions *after* it get?",
        "options": [
            "Effectively zero, because the mask was added before softmax",
            "A small but real share, since softmax never outputs exactly zero",
            "Whatever the dot products happened to give them",
        ],
        "answer": 0,
        "explanation": "The mask adds a large negative number to every disallowed pair "
                       "before softmax runs, and the exponential of a large negative number "
                       "is indistinguishable from zero in the model's dtype. The second "
                       "answer is true of softmax in general and false here, which is "
                       "exactly why the mask is additive rather than a step applied "
                       "afterwards. Note that the captured scores already include the mask.",
        "fields": (),
    },
    "block": {
        "operation": "block",
        "question": "What does the feed-forward network do with what attention gathered?",
        "options": [
            "Learned nonlinear processing of each position on its own",
            "Squashes the values into the range 0 to 1",
            "Mixes information between token positions once more",
        ],
        "answer": 0,
        "explanation": "It processes one position at a time, with no reference to any other "
                       "position — attention is the only step that moves information "
                       "between positions, and the feed-forward network is where the model "
                       "computes something new from what was gathered. It is not a "
                       "normalizer and it does not bound its output: this model's gate uses "
                       "SiLU, which leaves large positive values roughly as they were.",
        "fields": (),
    },
    "decide": {
        "operation": "decide",
        "question": "The winning score was {winner} and the runner-up was {runner_up}. Was "
                    "the winner {ratio} times as likely?",
        "options": [
            "No — these are scores, and nothing here turned them into probabilities",
            "Yes, the ratio of the scores is the ratio of the probabilities",
            "Yes, but only because this run used greedy decoding",
        ],
        "answer": 0,
        "explanation": "Logits are scores. Nothing in this run divided them by a total, so "
                       "their ratio means nothing: softmax would turn them into "
                       "probabilities, and it works on the *differences* between scores, "
                       "not their ratio. Greedy decoding never needed probabilities at all "
                       "— the highest score wins either way, which is why this run has "
                       "scores and no probabilities to show you.",
        "fields": ("winner", "runner_up", "ratio"),
    },
}

NOTEBOOK_CHECK_ORDER = ["embed", "shape", "scores", "block", "decide"]

assert set(NOTEBOOK_CHECK_ORDER) == set(NOTEBOOK_CHECKS), \
    "NOTEBOOK_CHECKS and its order disagree"
assert len(NOTEBOOK_CHECK_ORDER) == 5, len(NOTEBOOK_CHECK_ORDER)


def _head_geometry(capture: Any) -> dict[str, int]:
    """Head geometry read off the captured arrays, never off a model config.

    Degrades to zeros instead of raising.  learn_facts() is formatted against synthetic
    captures in the self-test and against runs saved by older schema versions, and a
    tensor this function cannot find must become a number rather than a KeyError in
    front of a reader.  Copy that interpolates these must stay true at zero.
    """
    blank = {"query_head_count": 0, "kv_head_count": 0, "head_dim": 0,
             "key_position_count": 0, "q_projection_width": 0,
             "kv_projection_width": 0, "kv_group_size": 0}
    layers = getattr(capture, "layers", None) or {}
    if not layers:
        return blank
    tensors = getattr(layers[min(layers)], "tensors", {}) or {}
    prepared, keys = tensors.get("q_attended"), tensors.get("k_attended")
    raw_keys, raw_queries = tensors.get("k"), tensors.get("q")
    if prepared is None or getattr(prepared, "ndim", 0) < 4:
        return blank
    head_dim = int(prepared.shape[-1])
    query_heads = int(prepared.shape[-3])
    # The projection widths are read off the arrays the projections produced, which is the
    # only place a saved run records them -- there is no model config here.
    q_width = int(raw_queries.shape[-1]) if getattr(raw_queries, "ndim", 0) >= 1 else 0
    kv_width = int(raw_keys.shape[-1]) if getattr(raw_keys, "ndim", 0) >= 1 else 0
    # A derived division, not a stored field: GQA's KV-head count is the raw key
    # projection's width divided by the head width.
    kv_heads = kv_width // head_dim if head_dim and kv_width else 0
    return {
        "query_head_count": query_heads,
        "head_dim": head_dim,
        "key_position_count": int(keys.shape[-2]) if getattr(keys, "ndim", 0) >= 2 else 0,
        "q_projection_width": q_width,
        "kv_projection_width": kv_width,
        "kv_head_count": kv_heads,
        "kv_group_size": query_heads // kv_heads if kv_heads else 0,
    }


def learn_facts(capture: Any) -> dict[str, Any]:
    """Read structural facts from a saved capture without loading a model.

    Every field any stage's copy interpolates must appear here, or a reader would meet
    a KeyError instead of a sentence.  self_test() formats every stage against this.
    """
    embedding = getattr(capture, "embedding", None)
    logits = getattr(capture, "logits", None)
    vocabulary = int(logits.shape[-1]) if logits is not None else 0
    prompt_tokens = list(getattr(capture, "prompt_tokens", None) or [])
    bands = prompt_token_bands(getattr(capture, "prompt", ""), prompt_tokens)
    typed = sum(end - start for start, end, name in (bands or []) if name == TYPED_BAND)
    trace = list(getattr(capture, "generation_trace", None) or [])
    candidates = (len(trace[0].top_candidates) if trace
                  else len(capture.metadata.get("final_logits_top", []) or []))
    prompt_count = len(capture.prompt_token_ids)
    geometry = _head_geometry(capture)
    metadata = capture.metadata
    generated_count = len(capture.token_ids)

    # How many keys the first generated-token pass attended to.  Read off that phase's own
    # k_attended, because the cached prompt keys live *inside* the captured tensor; it is
    # prompt_count + 1 as a matter of fact, not as a matter of assumption.
    generated_layers = getattr(capture, "generated_layers", None) or {}
    cached_keys = 0
    if generated_layers:
        prepared_keys = (getattr(generated_layers[min(generated_layers)], "tensors", {})
                         or {}).get("k_attended")
        if getattr(prepared_keys, "ndim", 0) >= 2:
            cached_keys = int(prepared_keys.shape[-2])

    # The stopping marker is scored but never emitted, so it has no entry in `tokens` and
    # no tokenizer is available to decode it.  Its spelling is recoverable only when the
    # same ID also appears in the prompt -- which it does for a chat template's end-of-turn
    # marker -- and is left as the bare ID when it does not, rather than guessed at.
    stop_id = metadata.get("stop_token_id")
    stop_spelling = ""
    if stop_id is not None:
        for identifier, spelling in zip(capture.prompt_token_ids, prompt_tokens):
            if int(identifier) == int(stop_id):
                stop_spelling = spelling
                break
    stop_logit = metadata.get("stop_token_logit")

    return {
        "model": capture.metadata.get("model", "an open-weight model"),
        "layer_count": len(capture.layers),
        "prompt_token_count": prompt_count,
        "generated_count": len(capture.token_ids),
        "hidden_size": int(embedding.shape[-1]) if embedding is not None else 0,
        # "full" read as a claim that the whole vocabulary was scored, which is the
        # opposite of what a missing logits vector means.
        "vocab_size": f"{vocabulary:,}" if vocabulary else "unrecorded",
        # Every sentence interpolates the phrase, not the bare count, so a run with no
        # retained logits reads as unknown instead of as "repeated full times".
        "vocab_size_phrase": (f"{vocabulary:,} of them" if vocabulary else
                              "a count this run did not record"),
        # bands is None when the typed text could not be aligned with the stored
        # spellings -- which happens for any prompt BYTE_BPE_REPLACEMENTS does not cover,
        # so routinely, not exceptionally.  The counts are 0 and prompt_count there, which
        # is indistinguishable from "the reader typed nothing"; copy must interpolate
        # typed_token_summary rather than the bare counts, and anything else must branch on
        # prompt_span_known first.
        "prompt_span_known": bands is not None,
        "typed_token_count": typed,
        "template_token_count": prompt_count - typed,
        "typed_token_summary": ((f"Only {typed} of them came from your own text."
                                 if typed < prompt_count else
                                 "Every one of them came from your own text.")
                                if bands is not None else
                                "TensorScope could not line your typed text up with these "
                                "stored spellings unambiguously, so it does not claim which "
                                "of them were yours."),
        "phase_count": sum(1 for group in (getattr(capture, "layers", None),
                                           getattr(capture, "generated_layers", None)) if group),
        "candidate_count": candidates,
        # Reads as a formula, not a claim about a number, when head_dim is unknown.
        "attention_scale_text": f"1/√{geometry['head_dim']}" if geometry["head_dim"] else "1/√d_head",

        # ── The worked notebook's own facts ───────────────────────────────────
        # Thousands-separated variants, because each of these appears inside a shape a
        # person reads, and "[151,936 × 2560]" mixes two conventions in one bracket.
        "hidden_size_text": f"{int(embedding.shape[-1]):,}" if embedding is not None else "0",
        "head_dim_text": f"{geometry['head_dim']:,}",
        "query_head_count_text": f"{geometry['query_head_count']:,}",
        "kv_head_count_text": f"{geometry['kv_head_count']:,}",
        "q_projection_width_text": f"{geometry['q_projection_width']:,}",
        "kv_projection_width_text": f"{geometry['kv_projection_width']:,}",
        "position_count": prompt_count,
        "position_count_text": f"{prompt_count:,}",
        "batch_size": int(embedding.shape[0]) if getattr(embedding, "ndim", 0) >= 1 else 1,
        "cached_key_count": cached_keys,
        "remaining_layer_count": max(len(capture.layers) - 1, 0),

        # An 81-token answer continues for 80 more tokens after the first.  An off-by-one
        # here would be a false claim about the model, so it is computed once, here.
        "continuation_count": max(generated_count - 1, 0),
        "continuation_summary": (
            "This answer was one token long, so there is nothing after it."
            if generated_count <= 1 else
            "One more token followed it, chosen the same way."
            if generated_count == 2 else
            f"{generated_count - 1} more tokens followed it, each chosen the same way."),

        # Decoding and candidate telemetry.  Every one of these has a "not recorded" form,
        # because the pre-schema-4 runs genuinely lack them.
        "decoding_rule": metadata.get("decoding", "not recorded"),
        "candidate_top_k": metadata.get("candidate_top_k", 0) or 0,
        "candidate_source_phrase": (
            f"the {candidates} best-scoring entries this run recorded" if candidates else
            "entries this run did not record"),
        "vocab_size_known": bool(vocabulary),
        # The integer behind vocab_size, for the diagrams: they lay out geometry
        # and cannot parse "151,936".
        "vocab_size_raw": vocabulary,

        # Why generation stopped.  Copy branches on `stop_token_known`; the three _text
        # fields are safe to interpolate either way.
        "stop_reason_text": STOP_REASONS.get(metadata.get("stop_reason", "unknown"),
                                             STOP_REASONS["unknown"]),
        "stop_token_known": stop_id is not None,
        "stop_token_id_text": "not recorded" if stop_id is None else f"{int(stop_id):,}",
        "stop_token_spelling": stop_spelling or "not recoverable from this capture",
        "stop_token_logit_text": ("not recorded" if stop_logit is None
                                  else f"{float(stop_logit):.6g}"),

        # The gated feed-forward path.  Its interior width is not in the capture at all --
        # no activation from inside the MLP was retained -- so it stays words, never a
        # plausible-looking number.
        "ffn_description": "down(silu(gate(x)) ⊙ up(x))",
        "ffn_interior_note": ("The feed-forward interior width is not recorded in this "
                              "capture: no activation from inside the MLP was retained."),
        **geometry,
    }
