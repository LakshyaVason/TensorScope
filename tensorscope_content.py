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


def display_spelling(token: str) -> str:
    """How a token is written on a chip: the text it stands for, with its invisible parts
    made visible.  A leading space is '␣', a newline '↵', a tab '⇥'.

    For display only.  The stored BPE spelling ('Ġcolor') is still shown, verbatim, in the
    selected token's detail line, so the reader can always see what was actually stored.
    """
    text = readable_spelling(token).replace("\n", "↵").replace("\t", "⇥")
    body = text.lstrip(" ")
    return "␣" * (len(text) - len(body)) + body


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


def typed_focus_position(prompt: str, prompt_tokens: list[str]) -> int:
    """Which prompt position the notebook follows first.

    A token the reader typed is more interesting to follow than a chat-template marker, so
    this is the last typed token that contains a letter or a digit -- the last word of the
    question rather than its question mark.  It is only chosen when the typed text lines up
    with the stored spellings exactly (`prompt_token_bands`); otherwise, and for a prompt of
    pure punctuation, it falls back to something that needs no alignment at all: the last
    typed token, then the last position.  Never a guess about which token "matters".
    """
    tokens = list(prompt_tokens or [])
    if not tokens:
        return 0
    bands = prompt_token_bands(prompt, tokens)
    typed = [index for start, end, name in (bands or []) if name == TYPED_BAND
             for index in range(start, end)]
    for index in reversed(typed):
        if any(character.isalnum() for character in readable_spelling(tokens[index])):
            return index
    return typed[-1] if typed else len(tokens) - 1


# The default followed position.  True means the notebook follows the deciding (final)
# prompt position from the start -- where the first token is actually chosen -- rather than
# the last word the reader typed.  This is the one switch designed to be reverted: the
# redesign chose "follow the decision" over "follow the typed word" so the reader watches
# the position that mattered, and op 12 then reads as `focus_is_final` by default.
FOLLOW_DECIDING_POSITION = True


# ── What can and cannot be concluded ─────────────────────────────────────────────

# Each line carries the capability it depends on, because three of them are false for the
# runs saved before that capability existed: a schema-2 run has no score vector, and
# neither schema 2 nor schema 3 records its decoding rule.  `what_we_know(facts)` filters.
WHAT_WE_KNOW = [
    (None, "The exact token IDs the model was given, after any chat template was applied."),
    (None, "The exact embedding vectors those IDs produced in this run."),
    (None, "The exact queries, keys, values, attention scores and attention weights for "
           "every head of every layer, in both captured passes."),
    (None, "The exact output of every decoder layer."),
    ("logits", "The full vocabulary score vector that chose the first generated token."),
    ("trace", "The leading candidate scores at every later generation step, and which "
              "token won."),
    ("decoding", "The decoding rule: greedy, so the highest score is selected with no "
                 "sampling."),
    (None, "That capture did not change the computation — the prompt pass was re-run "
           "under stock eager attention and the logits matched bit for bit."),
]


def what_we_know(facts: dict) -> list[str]:
    """The lines of WHAT_WE_KNOW that are true of the run `facts` describes."""
    have = {"logits": facts.get("vocab_size_known"),
            "trace": facts.get("trace_known"),
            "decoding": facts.get("decoding_known")}
    return [text for requirement, text in WHAT_WE_KNOW
            if requirement is None or have.get(requirement)]


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

# ── How provenance is drawn ──────────────────────────────────────────────────────
# Only the two tiers a reader must stop and weigh carry words inline: DERIVED (TensorScope
# worked it out) and UNATTESTED (real, but no gate re-derived it).  OBSERVED, which is most
# numbers on the page, is a solid coloured left edge on its value box; CONCEPTUAL, whose
# own text already says the value is absent, is a dashed grey edge.  Every edge also sets
# a property, a tooltip and an accessible description, so the tier never lives in colour
# alone.  The one key to all four is in the variable panel, beside the colour key.
EVIDENCE_INLINE_TIERS = (EVIDENCE_DERIVED, EVIDENCE_UNATTESTED)

# The tiers Level 0 and Level 1 may show in the input section's preface: the outcome's
# scores and the first token are OBSERVED, the ranking and the margin are DERIVED, and the
# pipeline's shape is CONCEPTUAL.  The tier test checks the preface against this.
OVERVIEW_TIERS = (EVIDENCE_OBSERVED, EVIDENCE_DERIVED, EVIDENCE_CONCEPTUAL)

EVIDENCE_MARKS = {
    EVIDENCE_OBSERVED: "solid green left edge",
    EVIDENCE_CONCEPTUAL: "dashed grey left edge",
    EVIDENCE_DERIVED: "always named in words, like this",
    EVIDENCE_UNATTESTED: "always named in words, like this",
}

EVIDENCE_LEGEND_TITLE = "where each number came from"

# ── The attention sink ───────────────────────────────────────────────────────────
# A shown weight row that gives position 0 at least this share of its weight gets a
# callout.  Read off the captured row on screen and nothing else; the threshold is stated
# in the copy so the reader knows exactly when the note appears and when it does not.
ATTENTION_SINK_THRESHOLD = 0.4

# Beside every attention-weight percentage, wherever one appears.  The test that enforces
# it looks for this exact phrase.
WEIGHTS_NOT_PROBABILITIES = (
    "Attention weights decide how information is mixed; they are not next-token "
    "probabilities."
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


# What each semantic colour stands for, in the order the computation reaches it.  The
# hues themselves live in tensorscope_common.COLORS, which both the diagrams and the
# variable chips read, so there is exactly one mapping from role to colour and this table
# only names it.  The legend is redundancy, never a substitute: every coloured element on
# a page still carries its own text label.
ROLE_LEGEND = (
    ("q", "Q queries"),
    ("k", "K keys"),
    ("v", "V values"),
    ("scores", "S scores"),
    ("weights", "A weights"),
    ("output", "O output"),
    ("norm", "normalized"),
    ("layer", "layer in/out"),
)


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
    """One anchored step of the worked notebook.

    Every operation is taught in the same order, and the fields are that order:
    `lede` is the purpose in one sentence; `idea` is what the step is *for* in words a
    reader can picture without the math; `terms` is the clickable formula; `inputs`,
    `action` and `outputs` say what entered, what happened and what came out; `note` is the
    one misunderstanding worth heading off at this step, if there is one; and `takeaway` is
    how the result is used next.  The full explanation lives on the reference page, behind
    `topic`, so none of it is repeated here.
    """

    key: str
    number: int
    nav: str
    heading: str
    lede: str
    idea: str = ""
    formula: str = ""
    terms: tuple = ()
    variables: tuple = ()
    tiers: tuple = (EVIDENCE_OBSERVED,)
    check: str = ""
    topic: str = ""
    fields: tuple = ()
    action: str = ""
    inputs: tuple = ()
    outputs: tuple = ()
    takeaway: str = ""
    note: str = ""


NOTEBOOK_OPERATIONS = [
    Operation(
        "input", 1, "The model's input",
        "What this run actually received",
        "Your prompt reached the model as {position_count} token positions, one vocabulary "
        "ID each. {typed_token_summary}",
        idea="A model does not read your words: it reads a list of ID numbers, one per "
        "token. This step is that list, with the tokens you typed set apart from the ones "
        "the chat template added.",
        terms=("text", "→", "[id₀, id₁, …]"),
        tiers=(EVIDENCE_OBSERVED,),
        topic="ids",
        action="split into tokens",
        inputs=("your text",),
        outputs=("token IDs",),
        takeaway="Each ID is used exactly once, as a row number into the embedding table.",
    ),
    Operation(
        "embed", 2, "Embedding lookup",
        "From a token ID to a vector",
        "Each token ID picks one row of a learned table, giving the vector of "
        "{hidden_size_text} numbers the layers will compute with.",
        idea="Every token starts as a single point in a {hidden_size_text}-dimensional "
        "space. This step is the lookup that turns an ID into that starting vector.",
        terms=("E", "[", "selected token ID", "]", "→", "embedding"),
        variables=("embedding", "E"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED),
        check="embed",
        topic="shapes",
        action="look up one row",
        inputs=("token ID",),
        outputs=("embedding",),
        takeaway="Stacked one row per position, these vectors become X, the first layer's "
                 "input.",
        note="No single coordinate has a known meaning. Meaning is spread across many "
             "coordinates at once, so the values below are worth watching for how they "
             "change, not for what any one of them says.",
    ),
    Operation(
        "shape", 3, "The input tensor",
        "One tensor for the whole sequence",
        "Stacking the {position_count_text} rows gives X, the single tensor the first layer "
        "reads: token positions down, features across.",
        idea="One point per token is not enough to compute with: the whole sequence is "
        "stacked into a single tensor so one operation can read every position at once.",
        formula="shape(X) = [{batch_size}, {position_count}, {hidden_size}]",
        variables=("layer_input",),
        tiers=(EVIDENCE_OBSERVED,),
        check="shape",
        topic="shapes",
        action="stack the rows",
        inputs=("embedding",),
        outputs=("layer_input",),
        takeaway="Every layer reads a tensor of exactly this shape and writes one back, so "
                 "this arrangement holds all the way to the last layer.",
    ),
    Operation(
        "norm", 4, "Input normalization",
        "Putting each row on a predictable scale",
        "RMSNorm rescales each position's row on its own, so the layer's arithmetic starts "
        "from vectors of a predictable size.",
        idea="Numbers that pass through many layers can drift to very different sizes, and "
        "a projection fed wildly scaled inputs produces wildly scaled outputs. This step "
        "puts each row on a predictable scale, on its own, before anything is compared.",
        formula="X_norm[i] = X[i] / √(mean(X[i]²) + ε)  ⊙  g",
        terms=("normalized_input", "=", "RMSNorm", "(", "layer_input", ")"),
        variables=("layer_input", "g", "epsilon", "rms", "normalized_input"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED),
        topic="norm",
        action="rescale each row on its own",
        inputs=("layer_input",),
        outputs=("normalized_input",),
        takeaway="X_norm feeds the three projections next. X itself is kept aside: it is "
                 "added back in operation 10.",
        note="Same shape, same rows: each value changed, but no number moved between "
             "positions.",
    ),
    Operation(
        "project", 5, "Raw projections",
        "Three learned views of the same row",
        "Three learned matrices read the same normalized row and produce a query (what this "
        "position looks for), a key (what it can be matched on) and a value (what it hands "
        "over).",
        idea="The same normalized row is read by three learned matrices at once, producing "
        "a query, a key and a value — three separate views of one token, which is what lets "
        "attention be asymmetric.",
        formula="Q_raw = X_norm W_Qᵀ        K_raw = X_norm W_Kᵀ        "
                "V_raw = X_norm W_Vᵀ",
        terms=("q", "=", "normalized_input", "@", "W_Q"),
        variables=("normalized_input", "W_Q", "q", "W_K", "k", "W_V", "v"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL),
        topic="qkv",
        action="multiply by three learned matrices",
        inputs=("normalized_input",),
        outputs=("q", "k", "v"),
        takeaway="These are not yet what attention compares: next they are split into heads "
                 "and Q and K are rotated by position.",
        note="This is the step that changes the width — {hidden_size_text} features in, "
             "{q_projection_width_text} out for Q and {kv_projection_width_text} for K and "
             "V — because a learned matrix computed new numbers.",
    ),
    Operation(
        "prepare", 6, "Preparing for attention",
        "Splitting into heads and marking position",
        "The projections are regrouped into heads, Q and K are rotated by position, and the "
        "shared key/value heads are copied out to the query heads that use them.",
        idea="The projected features are cut into heads that each run the whole "
        "compare-and-mix on their own, at the same time, and the query and key are rotated "
        "so the dot product can depend on where each token sits.",
        formula="Q_raw → reshape to [{query_head_count} × {head_dim}] → "
                "rotate by position → Q",
        terms=("q", "→", "q_attended"),
        variables=("q", "q_attended", "k", "k_attended", "v", "v_attended"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL),
        topic="heads",
        action="regroup into heads · rotate Q and K · share K and V",
        inputs=("q", "k", "v"),
        outputs=("q_attended", "k_attended", "v_attended"),
        takeaway="Q and K are compared next. V waits for operation 8, where the weights "
                 "decide how much of each value is collected.",
        note="Regrouping does no arithmetic: the same {q_projection_width_text} query numbers "
             "are read as {query_head_count_text} heads of {head_dim_text}. The rotation "
             "does change values, which is why Q differs from Q_raw.",
    ),
    Operation(
        "scores", 7, "Scores and weights",
        "Deciding what to read",
        "Each head compares one position's query with the key of every position it may see, "
        "and softmax turns those scores into weights that add up to 1.",
        idea="Attention decides what to read in two steps: first every query is compared "
        "with every key to get a score, then those scores become weights that say how much "
        "of each position to take.",
        formula="S = s·(Q Kᵀ) + M            A = softmax(S)",
        terms=("attention_scores", "→ softmax →", "attention_weights"),
        variables=("q_attended", "k_attended", "QK_raw", "M", "attention_scores",
                   "attention_weights"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED),
        check="scores",
        topic="softmax",
        action="compare · scale · mask · softmax",
        inputs=("q_attended", "k_attended"),
        outputs=("attention_scores", "attention_weights"),
        takeaway="The weights are used straight away, to decide how much of each "
                 "position's value this head collects.",
        note="These attention weights control how information is mixed within this head. "
             "They are not next-token probabilities.",
    ),
    Operation(
        "mix", 8, "Mixing the values",
        "Collecting what was decided",
        "Each head takes a weighted average of the value vectors, so information from the "
        "positions it attended to flows into this one.",
        idea="The weights from the step before are used to take a weighted average of the "
        "values: a position with a heavy weight contributes most of its vector, one with a "
        "weight near zero contributes almost nothing.",
        formula="Z_h = A_h V_h",
        terms=("Z_head", "=", "attention_weights", "@", "v_attended"),
        variables=("attention_weights", "v_attended", "Z_head"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED),
        topic="matching",
        action="weighted sum of value rows",
        inputs=("attention_weights", "v_attended"),
        outputs=("Z_head",),
        takeaway="Every head's result goes on to operation 9, where the heads are joined and "
                 "projected back to the model's width.",
        note="A weight near 1 means this head copies almost all of one position's value; "
             "weights spread evenly mean it blends many positions together.",
    ),
    Operation(
        "join", 9, "Joining the heads and projecting",
        "Combining the heads",
        "Every head's result is laid side by side and one learned matrix maps them back to "
        "{hidden_size_text} features — the only point at which heads combine.",
        idea="Each head worked out its own mix of the values independently; this is the one "
        "step that lays those separate results side by side and maps them back to the "
        "model's own width so they can be added to what came in.",
        formula="Z_joined = concat(Z_h)            O = Z_joined W_Oᵀ",
        terms=("attention_output", "=", "Z_joined", "@", "W_O"),
        variables=("Z_joined", "W_O", "attention_output"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED),
        topic="block",
        action="join the heads · multiply by W_O",
        inputs=("Z_joined",),
        outputs=("attention_output",),
        takeaway="O does not replace X: it is added to it, which is the next operation.",
        note="O is captured after this projection. It is not A @ V, and it is not the "
             "joined heads — neither of those was kept.",
    ),
    Operation(
        "block", 10, "Residual and feed-forward",
        "Add, refine, add again",
        "The attention result is added to the layer's input, then a feed-forward network "
        "refines each position on its own and its result is added in turn.",
        idea="Two additions around a refinement: the attention result is added to what came "
        "in, then a feed-forward network refines each position on its own and its result is "
        "added back — so information both travels around each branch and is changed by it.",
        formula="H = X + O            F = {ffn_description}            X_next = H + F",
        terms=("layer_output", "=", "H", "+", "F"),
        variables=("layer_input", "attention_output", "H", "H_norm", "W_gate", "W_up",
                   "W_down", "F", "layer_output"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED),
        check="block",
        topic="block",
        action="add O · feed-forward · add F",
        inputs=("layer_input", "attention_output"),
        outputs=("layer_output",),
        takeaway="X_next is the next layer's X: operations 4 to 10 run again with that "
                 "layer's own learned parameters.",
        note="Attention moved information between positions. The feed-forward network works "
             "on each position separately, and its gate is a nonlinearity, which lets it "
             "compute combinations of features that no single matrix multiplication can.",
    ),
    Operation(
        "layers", 11, "The remaining layers",
        "That completed one layer of {layer_count}",
        "Operations 4 to 10 repeat once per layer, {layer_count} times in all, each with its "
        "own learned parameters, and the token positions stay separate rows throughout.",
        idea="One layer of {layer_count} is not the whole story: the same block runs again "
        "on its own output, {layer_count} times, each with its own learned parameters, and "
        "the positions stay separate rows the whole way.",
        formula="X → layer 0 → layer 1 → … → "
                "layer {remaining_layer_count} → X_final",
        variables=("layer_output",),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_DERIVED),
        topic="heads",
        action="repeat operations 4–10 once per layer",
        inputs=("layer_input",),
        outputs=("layer_output",),
        takeaway="After the last layer, one position's row is turned into a score for every "
                 "vocabulary entry.",
    ),
    Operation(
        "decide", 12, "The token decision",
        "Choosing the first generated token",
        "After the last layer, the final prompt position's row is normalized once more and "
        "scored against every vocabulary entry, and the first token is picked from those "
        "scores.",
        idea="After the last layer, the final position's row is scored against every "
        "vocabulary entry at once, and the highest score wins — the model produces the "
        "scores, and a simple rule picks from them.",
        formula="logits = RMSNorm(X_final[{position_count}−1]) W_Uᵀ        "
                "selected_token_id = argmax(logits)",
        terms=("final_logits", "→ argmax →", "selected token ID"),
        variables=("layer_output", "X_final_norm", "W_U", "final_logits",
                   "prefill_logits_full"),
        tiers=(EVIDENCE_OBSERVED, EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED),
        check="decide",
        topic="vocabulary",
        action="final norm · score every entry · pick the highest",
        inputs=("layer_output",),
        outputs=("final_logits",),
        takeaway="That token is appended to the text, and the same calculation runs again "
                 "for the next one.",
        note="A logit is a score, not a probability. Nothing here was passed through "
             "softmax, and the highest score wins without one.",
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
    "{continuation_summary} Each later step processed only the newest token, reusing the "
    "same learned parameters and the keys and values already computed for earlier "
    "positions.",
    variables=("final_logits",),
    tiers=(EVIDENCE_OBSERVED, EVIDENCE_UNATTESTED, EVIDENCE_DERIVED, EVIDENCE_CONCEPTUAL),
    topic="stopping",
)

assert len(NOTEBOOK_OPERATIONS) == 12, len(NOTEBOOK_OPERATIONS)
assert len(OPERATION_INDEX) == 12, "duplicate operation key"
assert [operation.number for operation in NOTEBOOK_OPERATIONS] == list(range(1, 13))


# ── Chapters ────────────────────────────────────────────────────────────────────
# Four chapters over the same thirteen anchors.  They are a way of reading the page, not
# a way of moving around it: the shell shows where the reader is in them and the page
# marks where each begins, but nothing navigates by chapter -- the contents menu groups
# its sections under these titles and that is all.  A second navigation system is exactly
# what the redesign set out to remove.

@dataclass(frozen=True)
class Chapter:
    key: str
    title: str
    summary: str
    sections: tuple


CHAPTERS = [
    Chapter("represent", "Represent the input",
            "Text becomes token IDs, IDs become vectors, and the vectors are stacked into "
            "one tensor.",
            ("input", "embed", "shape")),
    Chapter("exchange", "Exchange information",
            "Inside one layer, each position gathers information from the positions before "
            "it.",
            ("norm", "project", "prepare", "scores", "mix", "join")),
    Chapter("refine", "Refine the representation",
            "Each position is refined on its own, and the layer repeats with new "
            "parameters.",
            ("block", "layers")),
    Chapter("select", "Select a token",
            "The final position's row becomes a score for every vocabulary entry, and the "
            "answer is built one token at a time.",
            ("decide", "continuation")),
]

CHAPTER_OF = {key: chapter for chapter in CHAPTERS for key in chapter.sections}

assert [key for chapter in CHAPTERS for key in chapter.sections] == \
    OPERATION_ORDER + [CONTINUATION.key], "chapters must cover the page in order"


# ── Captions used by the notebook's operation bodies ───────────────────────────────
# Kept here so no widget carries copy of its own.  Most format against learn_facts();
# the few that need a value only the builder has (a weight, a count of keys) declare those
# extra fields in NOTEBOOK_TEXT_FIELDS, and the self-test fills them with zeros so every
# line is checked against every run anyway.

NOTEBOOK_TEXT = {
    # The destination card at the top of the page.
    "destination_asked": "YOU ASKED",
    "destination_answered": "THE MODEL ANSWERED",
    "destination_follow":
        "This page follows the calculation that chose the first token of that answer, "
        "{first_token_text}. To watch one row change from step to step it follows one "
        "prompt position, {focus_token_text} at position {focus_position}, among the "
        "{position_count} the model processed together. Choose another with the "
        "Following control.",
    "destination_unaligned":
        "To watch one row change from step to step, this page follows the last prompt "
        "position, {focus_position}, among the {position_count} the model processed "
        "together. Choose another with the Following control.",

    # The three-part line every operation opens with.
    "io_in": "IN",
    "io_does": "DOES",
    "io_out": "OUT",
    "takeaway_prefix": "NEXT",
    "kept_captured": "Captured here: {names}",
    "kept_described": "Described, not shown: {names}",

    # Op 1.
    "control_markers":
        "Spellings written <|like this|> are control markers added by the chat template.",
    "template_position":
        "You did not type this one. The chat template adds markers like it to say whose "
        "turn it is.",
    "focus_hint": "Choose a token to follow it through the operations below.",
    "template_tokens_why":
        "{template_token_count} of these {position_count} positions are tokens you did not "
        "type. A chat model is trained on conversations, so before your text reaches it a "
        "chat template wraps it in markers that say whose turn it is and where each turn "
        "ends. The model reads those markers exactly like your words; they are shown faded "
        "here only so your own tokens stand out.",

    # Level 0 · Outcome, in the destination card.
    "outcome_heading":
        "The model scored every one of {vocab_size_phrase} and the highest was "
        "{first_token_text}.",
    "outcome_caption":
        "The scores below are the logits at the final prompt position, position "
        "{final_position}. The marked dot is the token the run emitted; the margin is the "
        "difference between the two highest scores.",
    "outcome_margin":
        "The top score beats the next by {margin:.1f}.",
    "outcome_spellings_unrecorded":
        "This run did not record the spellings of these tokens, so they are labelled by "
        "token ID.",
    "outcome_no_logits":
        "This run did not keep the score vector, so the ranking below is not shown. The "
        "token it emitted was {first_token_text}.",

    # Level 1 · Map, below the outcome.
    "map_heading":
        "How this run turned {position_count_text} tokens into one answer, and where the "
        "attention settled.",
    "map_pipeline_caption":
        "The shape the data took. The projection to {vocab_size_phrase} and the choice of "
        "one token are computed and discarded, so they are dashed.",
    "map_depth_caption":
        "For the deciding position, {final_token_text} at position {final_position}, how "
        "much of its attention weight each layer put on each key, averaged over heads. "
        "Typed positions are emphasised.",
    "map_depth_pick":
        "The layer outlined put the most total head-mean weight on the typed positions: "
        "{pick_layer} at {pick_weight:.3f}.",
    "map_depth_no_bands":
        "The typed positions could not be lined up with the stored spellings, so no layer "
        "is outlined.",
    "map_depth_legend":
        "Darker means more of this position's attention weight went to that key in that layer.",

    # The attention sink, wherever a shown weight row trips ATTENTION_SINK_THRESHOLD.
    # Op 7's own note already says attention weights are not next-token probabilities,
    # so the per-row form does not repeat it; op 11's form has no such neighbour and does.
    "attention_sink":
        "Position 0, {spelling}, takes {share}. This is a widely reported pattern in "
        "trained transformers, often called an attention sink: heads with nothing useful "
        "to collect park their leftover weight on the first token. It is not evidence that "
        "the model is reading what that token says.",
    "attention_sink_layers":
        "In {count} of {layer_count} layers, the head-averaged weight row for position "
        "{query} gives position 0, {spelling}, at least {threshold} of its weight — from "
        "{low} to {high}. This is a widely reported pattern in trained transformers, often "
        "called an attention sink: heads with nothing useful to collect park their leftover "
        "weight on the first token. It is not evidence that the model is reading what that "
        "token says. " + WEIGHTS_NOT_PROBABILITIES,
    # The rule, as a smaller secondary line under either form.
    "attention_sink_rule":
        "TensorScope adds this note whenever a weight row on screen gives position 0 at "
        "least {threshold} of its weight.",
    "mean_weight_share": "≈{percent} of the weight, averaged over the heads",

    # Followed-feature samples, ops 2, 4, 9 and 10.
    "tracked_features_caption":
        "the {count} features of position {position}'s layer input with the largest "
        "magnitude, followed through this layer",
    "tracked_features_note":
        "Chosen for display by size, at this layer's input, so the same coordinates can be "
        "watched from operation to operation. TensorScope picked them; the model did not.",
    "rounded_values":
        "Values are rounded for reading. Hover over one for the exact stored number, or "
        "open the variable for the whole array.",

    # Op 3.
    "row_highlight":
        "The highlighted row is the position you are following. The other rows are the "
        "other positions, computed alongside it.",

    # Op 5 / 6.
    "projection_widths":
        "K_raw = X_norm W_Kᵀ and V_raw = X_norm W_Vᵀ work the same way with their own "
        "learned matrices, and come out narrower — {kv_projection_width_text} features — "
        "because each of their heads is shared by several query heads.",
    "head_width_decoupled":
        "In this model {query_head_count_text} heads × {head_dim_text} features is "
        "{q_projection_width_text}, not the model width of {hidden_size_text}: a head's "
        "width is its own setting, read here off the captured tensors.",
    "kv_sharing":
        "This model has {query_head_count_text} query heads but only {kv_head_count_text} "
        "key/value heads, so each key/value head is copied out to {kv_group_size} query "
        "heads. The captured K and V already have {query_head_count_text} heads.",
    "rotation_unrecorded":
        "The captured Q and K are after the rotation by position, and after any per-head "
        "normalization this architecture applies. A saved run holds no model "
        "configuration, so the reference page describes those steps rather than this one.",
    "head_slice_caption": "first {count} features of head {head}, position {position}",

    # Op 7.
    "mask_caption":
        "Click a cell to choose a query row and a key column. Blocked cells had a large "
        "negative number added to their score before softmax.",
    "compare_caption":
        "The query row and key row behind the selected cell. Their dot product, scaled, is "
        "that cell's score.",
    "cached_keys":
        "This pass has one query and {cached_key_count} keys: the earlier positions' keys "
        "were reused from the cache, which is why the matrix is one row tall.",
    "bars_caption":
        "Head {head}, query position {query}: the {count} keys with the most weight, in "
        "position order. Score above, the weight softmax gave it below.",
    "bars_caption_all":
        "Head {head}, query position {query}: every key it may read. Score above, the "
        "weight softmax gave it below.",
    "weight_share": "≈{percent} of this head's mixing weight for this query",
    "heaviest_key":
        "The heaviest key is position {key} ({spelling}): weight {weight} — approximately "
        "{percent} of this head's mixing weight for this query.",
    "weight_row_total": "the whole weight row added up",
    "scores_scaled_masked":
        "The scale {attention_scale_text} and the mask are already inside the captured "
        "scores: they are exactly what softmax received.",
    "head_grid_title": "Show every query and key for head {head}",
    "explorer_title": "Open the full attention explorer for this layer",

    # Op 8.
    "mix_weights_caption":
        "the {count} largest weights in head {head}'s row for query position {query}",
    "mix_values_caption":
        "feature 0 of each of those positions' value vectors, head {head}",
    "mix_products_caption": "each weight times its value",
    "mix_remainder": "the other {count} positions together",
    "mix_total_caption": "every product added up: feature 0 of head {head}'s output",
    "mix_step_weights": "The weights this head gave",
    "mix_step_weights_body":
        "One weight per position this query may read; they add up to 1, so what follows "
        "is an average. Only the largest are shown separately. "
        + WEIGHTS_NOT_PROBABILITIES,
    "mix_step_values": "The values it is choosing between",
    "mix_step_values_body":
        "Each position contributes a whole vector of {head_dim} features. One feature is "
        "followed here; the same arithmetic runs independently for every one of them.",
    "mix_step_products": "Multiply each weight by its value",
    "mix_step_products_body":
        "A position with a large weight contributes most of its value; one with a weight "
        "near zero contributes almost nothing.",
    "mix_step_total": "Add them up",
    "mix_step_total_body":
        "The shown products plus the remainder are the whole sum. The result is one "
        "number of the head's output for this position.",

    # Op 9.
    "join_shape_caption":
        "The joined heads were not kept, so their width here comes from the captured "
        "tensors on either side: {q_projection_width_text} features in, "
        "{hidden_size_text} out.",
    "join_side_by_side":
        "X (what entered the layer) and O (what attention proposes adding), at the same "
        "features. The next operation adds them; that sum was not captured.",

    # Op 10.
    "residual_caption":
        "Each branch proposes a change; the input travels around it on the bypass and is "
        "added back. The first sum, H, was not captured. The second is X_next, the "
        "layer's captured output.",
    "scope_caption":
        "Attention reads across positions. The feed-forward network reads one position at "
        "a time.",
    "layer_out_caption":
        "X_next at the same features: this layer's output, and the next layer's X.",

    # Op 11.
    "layer_table_caption":
        "One row per layer, for the position you are following. Click a row to point the "
        "operations above at that layer.",
    "layer_table_note":
        "Most-attended position is where this position's attention weight went, averaged "
        "over the {query_head_count_text} heads. It is picked out for display — a ranking "
        "of captured values, not a value the model reported.",
    "layer_feature_title": "The largest coordinate of each layer's output, X_next",
    "layer_feature_dominance":
        "Feature {feature} is the largest-magnitude coordinate of X_next in {count} of "
        "{layer_count} layers for this position, so this column mostly shows that one "
        "coordinate's size growing, not a change in which feature leads. Picked out for "
        "display — a ranking of captured values, not a value the model reported.",
    "layer_feature_note":
        "The largest-magnitude coordinate of each layer's output for this position. Picked "
        "out for display — a ranking of captured values, not a value the model reported.",
    "layer_shapes_title": "Stored tensors and their shapes (the same for every layer)",
    "layer_stack_caption":
        "The highlighted layer is the one the operations above are describing.",

    # Op 12.
    "decision_chain_caption":
        "Solid steps were captured. Dashed steps happened in the model and were not kept.",
    "projection_caption": "Only the final position's score row was stored.",
    "focus_shift":
        "Until now this page followed position {focus_position}, {focus_token_text}. The "
        "next token is chosen at the final prompt position, {final_position} "
        "({final_token_text}), because it is the last one: its row after the final layer "
        "is the one turned into vocabulary scores. Position {focus_position} was computed "
        "through every layer too, but its row did not make this choice.",
    "focus_is_final":
        "The position this page has been following, {final_position} "
        "({final_token_text}), is the final prompt position: its row after the final "
        "layer is the one turned into vocabulary scores.",
    "follow_final": "Follow position {final_position} instead",
    "score_count": "The final position's score vector holds {count} numbers, one per "
                   "vocabulary entry.",
    "selected_line":
        "selected_token_id = argmax(logits) = {token_id}, which decodes to {token}.",
    "margin_line":
        "It leads the runner-up by ≈{margin} — a difference in score, not in probability.",

    # The single concrete thing this run shows at each step — one highlighted line per
    # operation card's "Your run" pane.  Each reads a number off a captured array and says
    # what it means; where the run lacks the tensor the builder degrades to a neutral form.
    "finding_input":
        "You typed {typed_token_count} of these {position_count} positions; the rest are "
        "markers the chat template added.",
    "finding_embed":
        "The {hidden_size_text} numbers of this token's embedding are its starting point; "
        "no single coordinate means anything on its own.",
    "finding_shape":
        "The whole sequence is one tensor, {position_count_text} rows of {hidden_size_text} "
        "features — the shape every layer reads and writes.",
    "finding_norm":
        "Every value changed, but no number moved between positions: normalization rescales "
        "each row on its own.",
    "finding_project":
        "The width grows from {hidden_size_text} to {q_projection_width_text} for Q — a "
        "learned matrix computed new numbers, so the projection is wider than the model.",
    "finding_prepare":
        "The {q_projection_width_text} query features are read as {query_head_count_text} "
        "heads of {head_dim_text}; the same numbers, regrouped, not recomputed.",
    "finding_scores":
        "For the followed query, the heaviest key is position {heaviest_key} "
        "({heaviest_spelling}), at {heaviest_percent} of this head's mixing weight.",
    "finding_mix":
        "This head's output is a weighted average of {mix_shown} positions, the rest "
        "blended in; the weights add up to 1, so it is an average, not a sum.",
    "finding_join":
        "The {query_head_count_text} heads are laid side by side and mapped back to "
        "{hidden_size_text} features — the only step at which the heads combine.",
    "finding_block":
        "Attention read across positions; the feed-forward network now reads this one "
        "position on its own, through a gate that lets it express combinations, not sums.",
    "finding_layers":
        "Across the {layer_count} layers, the followed position's attention put the most "
        "head-mean weight on position {most_attended} in layer {most_attended_layer}.",
    "finding_decide":
        "The winning score was {winner} and the runner-up {runner_up}; a difference in "
        "score, and nothing turned either into a probability.",

    # The continuation.
    "answer_heading": "THE ANSWER — SELECT ANY TOKEN",
    "timeline_heading": "ONE PASS PER TOKEN",
    "timeline_first": "Process the prompt → select token 1, {token}",
    "timeline_next": "Process token {previous} with cached context → select token {number}, "
                     "{token}",
    "timeline_stop": "Process token {previous} with cached context → the stopping marker "
                     "won, and was not emitted",
    "timeline_layers_saved": "layer tensors saved",
    "timeline_layers_not_saved": "no layer tensors",
    "timeline_full_vector": "full vocabulary scores saved",
    "timeline_leading": "leading candidate scores saved",
    "timeline_not_recorded": "candidate scores not recorded",
    "timeline_show_all": "Show all {count} passes",
    "timeline_show_fewer": "Show fewer passes",
    "kv_reuse":
        "Every pass uses the same learned parameters. After the first, each pass computes "
        "only the newest token's row and reuses the keys and values already computed for "
        "earlier positions; it does not recompute the prompt.",
    "phase_toggle_on": "Show the first generated-token pass in operations 4–11",
    "phase_toggle_off": "Back to the prompt pass",
    "limits_marks_heading": "HOW TO READ THE PROVENANCE MARKS",
    "phase_toggle_note":
        "Only two passes kept their layer tensors: the prompt pass, and the pass that "
        "processed the first generated token.",
}

# Fields a NOTEBOOK_TEXT entry needs beyond learn_facts(), supplied by its builder.
NOTEBOOK_TEXT_FIELDS = {
    "attention_sink": ("spelling", "share", "threshold"),
    "attention_sink_layers": ("count", "query", "spelling", "threshold", "low", "high"),
    "attention_sink_rule": ("threshold",),
    "layer_feature_dominance": ("feature", "count"),
    "mean_weight_share": ("percent",),
    "outcome_margin": ("margin",),
    "map_depth_pick": ("pick_layer", "pick_weight"),
    "kept_captured": ("names",),
    "kept_described": ("names",),
    "tracked_features_caption": ("count", "position"),
    "head_slice_caption": ("count", "head", "position"),
    "bars_caption": ("head", "query", "count"),
    "heaviest_key": ("key", "spelling", "weight", "percent"),
    "bars_caption_all": ("head", "query"),
    "weight_share": ("percent",),
    "head_grid_title": ("head",),
    "mix_weights_caption": ("count", "head", "query"),
    "mix_values_caption": ("head",),
    "mix_remainder": ("count",),
    "mix_total_caption": ("head",),
    "score_count": ("count",),
    "selected_line": ("token_id", "token"),
    "margin_line": ("margin",),
    "timeline_first": ("token",),
    "timeline_next": ("previous", "number", "token"),
    "timeline_stop": ("previous",),
    "timeline_show_all": ("count",),
    # The per-operation finding line: the fields each one interpolates.  All are read off
    # captured arrays or `learn_facts()`, so the self-test can fill them with zeros.
    "finding_input": ("typed_token_count", "position_count"),
    "finding_embed": ("hidden_size_text",),
    "finding_shape": ("position_count_text", "hidden_size_text"),
    "finding_norm": (),
    "finding_project": ("hidden_size_text", "q_projection_width_text"),
    "finding_prepare": ("q_projection_width_text", "query_head_count_text", "head_dim_text"),
    "finding_scores": ("heaviest_key", "heaviest_spelling", "heaviest_percent"),
    "finding_mix": ("mix_shown",),
    "finding_join": ("query_head_count_text", "hidden_size_text"),
    "finding_block": (),
    "finding_layers": ("layer_count", "most_attended", "most_attended_layer"),
    "finding_decide": ("winner", "runner_up"),
}


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

    # The position the notebook follows, and the one that actually chose the token.  The
    # decision section has to name both, and say which is which, whenever they differ.  By
    # default the notebook follows the deciding (final) position; `FOLLOW_DECIDING_POSITION`
    # is the one switch that reverts it to the last typed word.
    follow = typed_focus_position(getattr(capture, "prompt", ""), prompt_tokens)
    final = max(prompt_count - 1, 0)
    focus = final if FOLLOW_DECIDING_POSITION else follow

    def spelled(index: int) -> str:
        return repr(readable_spelling(prompt_tokens[index])) if 0 <= index < len(prompt_tokens) else "''"

    emitted = list(getattr(capture, "tokens", None) or [])
    response = str(getattr(capture, "response", "") or "").strip()
    # The first sentence of the answer, so the page can open on where it is going.  Cut at
    # a word boundary and marked as cut, never silently shortened.
    excerpt = response
    for stop in (". ", "! ", "? ", "\n"):
        cut = excerpt.find(stop)
        if 0 < cut < len(excerpt) - 1:
            excerpt = excerpt[:cut + 1].strip()
    if len(excerpt) > 160:
        excerpt = excerpt[:160].rsplit(" ", 1)[0] + "…"
    elif len(excerpt) < len(response):
        excerpt = excerpt + " …"

    decoding = metadata.get("decoding")
    if decoding == "greedy":
        rule = ("This run used greedy decoding: the entry with the highest score is "
                "selected, with no sampling.")
    elif decoding:
        rule = f"This run recorded its selection rule as {decoding!r}."
    elif logits is not None:
        rule = ("This run does not record its selection rule, but its stored score vector "
                "peaks at exactly the token it emitted.")
    else:
        rule = ("This run records neither its selection rule nor its score vector; only "
                "the emitted token is known.")

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

        # ── The reader's destination and the position they follow ───────────────
        "prompt_text": repr(str(getattr(capture, "prompt", "") or "")),
        "response_excerpt": excerpt or "(an empty response)",
        "first_token_text": repr(emitted[0]) if emitted else "no token",
        "first_token_id": int(capture.token_ids[0]) if capture.token_ids else 0,
        "focus_position": focus,
        "focus_token_text": spelled(focus),
        "final_position": final,
        "final_token_text": spelled(final),
        "focus_is_final": focus == final,
        # Qwen3 sets the head width independently of hidden_size / heads, so copy that
        # explains the head split must not assume they agree.
        "head_width_decoupled": bool(geometry["q_projection_width"]
                                     and embedding is not None
                                     and geometry["q_projection_width"] != int(embedding.shape[-1])),
        "selection_rule_sentence": rule,
        "trace_known": bool(trace),
        "decoding_known": bool(decoding),
        **geometry,
    }
