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
        "At every step the model scores every vocabulary entry ({vocab_size_phrase}) and the "
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


# ── The default lesson: eight screens from prompt to first token ─────────────────
# One real captured run, in the order the calculations happened.  Each screen names the
# question it answers; the arithmetic is revealed step by step beside real operands, and
# every screen says which of its numbers the model produced and which it did not.

@dataclass(frozen=True)
class LessonStage:
    """One screen of the lesson; format fields come from learn_facts()."""

    key: str
    nav: str        # short label for the contents list and the pipeline chips
    question: str   # the question this screen answers, shown as its heading
    plain: str      # two or three sentences; the screen itself carries the detail


LESSON_STAGES = [
    LessonStage(
        "received", "1  Received",
        "What did the model actually receive?",
        "You typed one line. The model was handed {prompt_token_count} numbered vocabulary "
        "entries: your words split into pieces, wrapped in the chat markers this model was "
        "trained to expect. It never sees letters — only those ID numbers, in this order."
    ),
    LessonStage(
        "vectors", "2  Vectors",
        "How does an ID become something you can do arithmetic on?",
        "An ID is a row number. The model looks that row up in a learned table and gets "
        "{hidden_size} numbers back — one vector per position. Everything that follows in "
        "this lesson is arithmetic on rows of numbers like these."
    ),
    LessonStage(
        "transform", "3  Transformed",
        "What does it mean to transform a vector?",
        "One operation appears everywhere in this model: take a row of numbers and a column "
        "of numbers, multiply them entry by entry, and add the products. Do that with many "
        "columns and one vector becomes another. This layer does it three times over, for "
        "three different purposes."
    ),
    LessonStage(
        "compare", "4  Compared",
        "How does the model compare two positions?",
        "Attention compares one position's query with another position's key using that same "
        "operation: {head_dim} pairs multiplied, summed, then scaled. One pair of positions "
        "produces exactly one number, which is why the result is a grid of positions against "
        "positions."
    ),
    LessonStage(
        "weights", "5  Weights",
        "How do those scores become mixing weights?",
        "A row of raw scores is not yet usable: the numbers are unbounded, and some of them "
        "refer to positions this token is not allowed to read. Masking and softmax turn the "
        "row into nonnegative weights that say how much of each position to mix in."
    ),
    LessonStage(
        "combine", "6  Combined",
        "How is the information actually combined?",
        "The weights blend the value vectors, the heads are joined and projected back to the "
        "model's own width, and the result is added onto what the position already carried. "
        "Then the whole block repeats, {layer_count} times."
    ),
    LessonStage(
        "select", "7  Selected",
        "Which number chose the first token?",
        "After the last layer, one position's vector is turned into one score per vocabulary "
        "entry — {vocab_size_phrase}. The largest score wins. Nothing was sampled, and "
        "nothing was normalized into a probability."
    ),
    LessonStage(
        "continue", "8  Continues",
        "How does the rest of the answer appear?",
        "The chosen token is appended to the input and the model runs again, which is why the "
        "answer arrived one token at a time — {generated_count} of them here. TensorScope "
        "captured the first of those passes in full; later ones are recorded as the scores "
        "that decided them."
    ),
]

LESSON_STAGE_INDEX = {stage.key: stage for stage in LESSON_STAGES}


# Reveal copy for the screens that walk a calculation.  Each entry is one deliberate
# step: `caption` heads it, `body` explains it, `tier` says what kind of number it puts
# on screen.  Bodies are formatted against learn_facts(), so a step may name a real
# dimension but must never name a value -- values come from the captured arrays.
LESSON_STEPS = {
    "transform": [
        {
            "caption": "Put the vector on a predictable scale",
            "body": "RMSNorm divides a position's vector by a measure of its own magnitude, then "
                    "scales it feature by feature with learned values. Nothing moves between "
                    "positions here. The captured result is real; the magnitude statistic and the "
                    "learned scales were not saved.",
            "tier": EVIDENCE_OBSERVED,
        },
        {
            "caption": "Pair the entries",
            "body": "A row times a column works entry by entry: the row's first number with the "
                    "column's first number, the second with the second, on to the end. The row "
                    "below holds real captured values. The column is written as symbols because "
                    "the learned weights were never part of this capture.",
            "tier": EVIDENCE_CONCEPTUAL,
        },
        {
            "caption": "Multiply, then add",
            "body": "Each pair is multiplied, and all {hidden_size} products are added into a "
                    "single number. That one number is one feature of the output vector. A "
                    "different column gives the next output feature, and so on.",
            "tier": EVIDENCE_CONCEPTUAL,
        },
        {
            "caption": "Three columns' worth of purpose",
            "body": "The same normalized vector is read by three different learned projections. "
                    "One asks what this position is looking for, one advertises what it can be "
                    "matched on, one carries what it will contribute if matched. Only now is it "
                    "useful to name them: query, key, value.",
            "tier": EVIDENCE_CONCEPTUAL,
        },
        {
            "caption": "What this run actually produced",
            "body": "These are the real projection outputs for this layer. They are raw: the split "
                    "into heads, any per-head normalization the architecture applies, and the rotary "
                    "position transformation have not happened yet. The prepared tensors on the next "
                    "screen are the ones that enter the comparison.",
            "tier": EVIDENCE_OBSERVED,
        },
    ],
    "compare": [
        {
            "caption": "One query, one key",
            "body": "Both vectors hold {head_dim} numbers, because both were cut to the same head "
                    "width. That is the only reason the comparison below is defined at all.",
            "tier": EVIDENCE_OBSERVED,
        },
        {
            "caption": "Multiply the pairs",
            "body": "Entry one with entry one, entry two with entry two. A large positive product "
                    "means the two vectors agree along that feature; a negative one means they "
                    "disagree. These products are calculated here for the display — the model kept "
                    "only their total.",
            "tier": EVIDENCE_DERIVED,
        },
        {
            "caption": "Add all {head_dim} products",
            "body": "The sum is a single number measuring how well this query matches this key. "
                    "Larger means more relevant. This is the dot product, and it is the whole of "
                    "how attention decides what to read.",
            "tier": EVIDENCE_DERIVED,
        },
        {
            "caption": "Scale, then compare with what the model stored",
            "body": "The total is multiplied by {attention_scale_text} to keep the numbers in a "
                    "range softmax handles well. Beside it is the score the model actually saved "
                    "for this pair. They agree closely rather than exactly: the model computed in "
                    "bfloat16, and this display re-added the products in double precision.",
            "tier": EVIDENCE_OBSERVED,
        },
    ],
    "weights": [
        {
            "caption": "The scale is already inside these numbers",
            "body": "The captured score row is exactly what was handed to softmax: already "
                    "multiplied by {attention_scale_text} and already masked. The unscaled product "
                    "and the mask were not saved separately.",
            "tier": EVIDENCE_OBSERVED,
        },
        {
            "caption": "The future is masked out",
            "body": "A position may only read itself and what came before it. Any later position "
                    "in this row was set to a very large negative number before softmax — not a "
                    "placeholder this display invented, but the actual value stored above. A row "
                    "with nothing after it is masked nowhere, which is why the last prompt "
                    "position and the generated pass's single row look different.",
            "tier": EVIDENCE_OBSERVED,
        },
        {
            "caption": "Softmax across the row",
            "body": "Softmax makes every entry nonnegative and larger scores dominate. Wherever "
                    "this row was masked the weight comes out at exactly zero, so nothing from a "
                    "later position contributes to what this position reads.",
            "tier": EVIDENCE_OBSERVED,
        },
        {
            "caption": "What the row adds up to",
            "body": "In mathematics a softmax row sums to one. Added up as stored it lands slightly "
                    "off, because the weights were rounded to the model's 16-bit format. The sum "
                    "below was calculated here; the weights themselves are the model's.",
            "tier": EVIDENCE_DERIVED,
        },
    ],
    "select": [
        {
            "caption": "Start from one position's finished vector",
            "body": "The deciding position is the last one in the prompt: it is the only position "
                    "that has read everything to its left. Its vector after the final layer is "
                    "captured.",
            "tier": EVIDENCE_OBSERVED,
        },
        {
            "caption": "One last normalization",
            "body": "The model applies a final RMSNorm before scoring. This capture does not retain "
                    "its output, so the step is described rather than shown.",
            "tier": EVIDENCE_CONCEPTUAL,
        },
        {
            "caption": "One column per vocabulary entry",
            "body": "The vocabulary projection is the same row-times-column operation as before, "
                    "repeated once per entry the model could emit: {vocab_size_phrase}. Its learned "
                    "matrix is not part of this capture.",
            "tier": EVIDENCE_CONCEPTUAL,
        },
        # These last three steps -- this one and the two below, shown as steps 4, 5 and 6 --
        # assert that the score vector was saved and that the stock-eager comparison covers
        # it, which is false for a run with no retained logits.  When `capture.logits is
        # None` LessonView must replace all three with a single LOGITS_UNAVAILABLE step, and
        # must also skip LESSON_CHECKS["logits"]: its question interpolates scores that only
        # a retained vector can supply.
        {
            "caption": "The captured score vector",
            "body": "The result is one number per vocabulary entry, and this run saved the whole "
                    "vector. The prompt pass was re-run under stock attention and these values "
                    "matched bit for bit, which is what makes them trustworthy.",
            "tier": EVIDENCE_OBSERVED,
        },
        {
            "caption": "A few real candidates",
            "body": "These are actual entries from that vector, highest first. They are scores, not "
                    "probabilities: they are unbounded, can be negative, and nothing here was "
                    "divided by a total.",
            "tier": EVIDENCE_OBSERVED,
        },
        {
            "caption": "The largest score selects the token",
            "body": "Greedy decoding takes the highest-scoring entry and emits it. The position of "
                    "that maximum is found here for the display; the number it points at is the "
                    "model's, and the token it names is the one this run actually produced.",
            "tier": EVIDENCE_DERIVED,
        },
    ],
}


# ── Optional predict-and-reveal checks ───────────────────────────────────────────
# Four, all skippable, none gating anything.  `fields` names the extra placeholders a
# check's copy uses beyond learn_facts(), so the self-test can format every one of them.

LESSON_CHECKS = {
    # How *many* extra entries there are depends on the prompt's length, so this asks what
    # they are instead.  The answer names the chat markers, so it is only honest for a run
    # that has some: LessonView must skip this check when template_token_count is 0, which
    # is what a tokenizer with no chat template produces.
    "tokens": {
        "screen": "received",
        "question": "Before you look: what do you think the model was handed?",
        "options": ["Exactly the characters I typed",
                    "My text split into pieces, plus markers the template added",
                    "One entry per word I typed"],
        "answer": 1,
        "explanation": "This prompt became {prompt_token_count} numbered entries: your text "
                       "split into pieces this tokenizer knows, wrapped in markers the chat "
                       "template adds so the model knows who is speaking. {typed_token_summary}",
        "fields": (),
    },
    "dot": {
        "screen": "compare",
        "question": "Two vectors of {head_dim} numbers are compared. How many numbers come out?",
        "options": ["{head_dim}, one per feature", "One", "{head_dim} × {head_dim}"],
        "answer": 1,
        "explanation": "One. Multiplying the pairs and adding them collapses both vectors into a "
                       "single score for that pair of positions — which is why the captured score "
                       "tensor is a grid of positions against positions, not of features.",
        "fields": (),
    },
    # The explanation points at columns that are only there when the selected row has
    # later positions, so LessonView must skip this check when masked_columns() is empty.
    "mask": {
        "screen": "weights",
        "question": "What weight will softmax give the positions after this one?",
        "options": ["A small but non-zero share", "Exactly zero", "It depends on the scores"],
        "answer": 1,
        "explanation": "Exactly zero, as stored. The scores for later positions were set to a very "
                       "large negative number before softmax, and the captured weights at those "
                       "columns are 0.0 — so nothing from the future is mixed in.",
        "fields": (),
    },
    "logits": {
        "screen": "select",
        "question": "The winner scored {winner} and the runner-up {runner_up}. Was the winner "
                    "{ratio} times as likely?",
        "options": ["Yes, the scores are proportional to likelihood",
                    "No — these are unnormalized scores",
                    "Only after dividing by the vocabulary size"],
        "answer": 1,
        "explanation": "No. These are raw scores: nothing divided them by a total, so their ratio "
                       "means nothing on its own. Converting them to probabilities needs a softmax "
                       "over the whole score vector, and this run did not compute one — it "
                       "took the largest.",
        "fields": ("winner", "runner_up", "ratio"),
    },
}

LESSON_CHECK_ORDER = ["tokens", "dot", "mask", "logits"]

assert set(LESSON_CHECK_ORDER) == set(LESSON_CHECKS), "LESSON_CHECKS and its order disagree"


# ── Notes the lesson reuses in several places ───────────────────────────────────

LESSON_TIMING_CAVEAT = (
    "Revealed one step at a time for reading. The model did not run them in sequence at "
    "this speed: a layer's arithmetic happens at once, in parallel, on the GPU."
)

LESSON_DISPLAY_CALC = (
    "Calculated here from captured operands, for this display only. It is not a tensor the "
    "model saved, and it is not required to match the captured result exactly — the model "
    "computed in bfloat16, whose steps between representable numbers are coarse."
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
    "The second pass processes one new token, but compares it against every earlier position: "
    "the prompt's keys and values were kept from the first pass rather than recomputed. Those "
    "cached rows are inside the captured prepared tensors; the cache object itself is not saved."
)


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
        "projection turn it into one score per vocabulary entry ({vocab_size_phrase}). The "
        "normalization output and the projection matrix are not saved; the resulting score "
        "vector is."
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


def _head_geometry(capture: Any) -> dict[str, int]:
    """Head geometry read off the captured arrays, never off a model config.

    Degrades to zeros instead of raising.  learn_facts() is formatted against synthetic
    captures in the self-test and against runs saved by older schema versions, and a
    tensor this function cannot find must become a number rather than a KeyError in
    front of a reader.  Copy that interpolates these must stay true at zero.
    """
    blank = {"query_head_count": 0, "kv_head_count": 0, "head_dim": 0, "key_position_count": 0}
    layers = getattr(capture, "layers", None) or {}
    if not layers:
        return blank
    tensors = getattr(layers[min(layers)], "tensors", {}) or {}
    prepared, keys, raw_keys = tensors.get("q_attended"), tensors.get("k_attended"), tensors.get("k")
    if prepared is None or getattr(prepared, "ndim", 0) < 4:
        return blank
    head_dim = int(prepared.shape[-1])
    return {
        "query_head_count": int(prepared.shape[-3]),
        "head_dim": head_dim,
        "key_position_count": int(keys.shape[-2]) if getattr(keys, "ndim", 0) >= 2 else 0,
        # A derived division, not a stored field: GQA's KV-head count is the raw key
        # projection's width divided by the head width.
        "kv_head_count": int(raw_keys.shape[-1]) // head_dim
        if head_dim and getattr(raw_keys, "ndim", 0) >= 1 else 0,
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
        "layer_step_count": len(LAYER_STEP_ORDER),
        "candidate_count": candidates,
        # Reads as a formula, not a claim about a number, when head_dim is unknown.
        "attention_scale_text": f"1/√{geometry['head_dim']}" if geometry["head_dim"] else "1/√d_head",
        **geometry,
    }
