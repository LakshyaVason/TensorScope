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


# ── The guided lesson ────────────────────────────────────────────────────────────
# The default mode is a linear lesson: one card at a time, Back/Next, a stable
# ribbon carrying the real prompt.  It replaces the old five-question Understand
# journey, whose weakness was structural rather than editorial -- it explained
# tokens, attention weights and candidate scores on separate screens without ever
# showing the operation that connects one to the next.
#
# Two rules for every string below:
#
#   * Introduce a term only after the reader has met the problem it solves.  Plain
#     role first ("what this position is looking for"), technical name second
#     ("query, Q").  TERMS holds that mapping in one place.
#   * Never name a coordinate.  A feature is a learned number, not "plural" or
#     "France"; the caveat in TERMS says so once, centrally, so no scene has to.
#
# Copy may interpolate any field learn_facts() returns and nothing else.  Literal
# braces are therefore forbidden here: self_test() formats every string below.

@dataclass(frozen=True)
class LessonScene:
    """One card of the lesson.  Scene widgets are built by tensorscope_views."""

    key: str
    nav: str      # short label for the contents menu and the breadcrumb
    title: str    # the card heading: what this card is for
    plain: str    # two or three sentences; the visual carries the rest


LESSON_SCENES = [
    LessonScene(
        "opening", "The question",
        "How did this prompt become an answer?",
        "This is one real run of {model}: a prompt went in, and {generated_count} tokens "
        "came out one at a time. Nothing ahead is an illustration — every number was read "
        "back from the arrays this run stored while it ran. We will follow the prompt from "
        "text to the model's very first choice."
    ),
    LessonScene(
        "pieces", "Pieces of text",
        "The model never sees your sentence",
        "Text is first cut into vocabulary entries — whole words, word fragments, "
        "punctuation, even line breaks — and each entry is replaced by its ID number. This "
        "prompt became {prompt_token_count} pieces, including the scaffolding a chat model "
        "expects around your words. Select any piece to see what the model was handed in "
        "its place."
    ),
    LessonScene(
        "rows", "Rows of numbers",
        "Each piece becomes a row of numbers",
        "An ID number is only an address. The model looks that address up in a stored table "
        "and takes out a row of {hidden_size} numbers — the same row every time that piece "
        "appears anywhere. From here on the piece's identity stays fixed, and this row of "
        "numbers is the thing that gets changed."
    ),
    LessonScene(
        "multiply", "Multiplying",
        "What does multiplying by a matrix actually do?",
        "Every transformation ahead is one operation. To produce a single output number, "
        "multiply each entry of the input row by its own weight and add the results "
        "together; to produce a whole output row, do that again with a different set of "
        "weights per output. A matrix is just the collection of those weight sets — "
        "{hidden_size} multiplications and additions per output number."
    ),
    LessonScene(
        "three", "Three jobs",
        "The same row is transformed for three different jobs",
        "One row cannot serve three purposes at once, and a position needs all three: say "
        "what it is looking for, say what it can be found by, and carry what it will "
        "contribute if it is found. So the same normalized row is multiplied by three "
        "separate weight sets, giving three different rows. Their conventional names are "
        "query, key and value."
    ),
    LessonScene(
        "prepare", "Getting ready",
        "Preparing those rows to be compared",
        "Two things happen before any comparison. The wide query and key rows are cut into "
        "{head_count} independent slices of {head_dim} numbers each — called heads — so "
        "several comparisons can run side by side; then every slice is rotated by an amount "
        "that depends on which position it sits at. The lookup table knows nothing about "
        "position, so this rotation is where word order enters the model at all."
    ),
    LessonScene(
        "compare", "Comparing",
        "Comparing one position against the ones it may read",
        "Within one head, a position's query slice is compared against a key slice by "
        "multiplying matching entries and adding them up — the same multiply-and-add as "
        "before, so a large result means that position is relevant to this one. Every pair "
        "at once is a single matrix multiplication over the {head_dim} numbers the two "
        "slices share, producing exactly one number per pair."
    ),
    LessonScene(
        "weights", "Mixing amounts",
        "Turning comparisons into mixing amounts",
        "Comparison numbers can be any size, so each row goes through softmax: raise e to "
        "each number, then divide by the row's total. What comes out is non-negative and "
        "sums to one, which is what makes it usable as an amount — how much of each "
        "position to take. Any position later in the text is forced to a hugely negative "
        "number first, so it emerges at essentially zero."
    ),
    LessonScene(
        "combine", "Combining",
        "Using those amounts to combine information",
        "A position's output is the mix: scale every value row by that position's amount "
        "and add them all together. Each head does this with its own amounts, the head "
        "results are laid side by side, and one more matrix multiplication brings the width "
        "back to {hidden_size}. That final result is the array this capture stored."
    ),
    LessonScene(
        "repeat", "Again",
        "Update the representation, then do it all again",
        "The attention result is added onto the row it started from rather than replacing "
        "it, and a small per-position network refines it — so a row keeps its identity "
        "while accumulating context. The layer hands out a row of the same width it "
        "received, and the next layer starts again from there. This model repeats that "
        "{layer_count} times."
    ),
    LessonScene(
        "scores", "Scoring the vocabulary",
        "The last position becomes one score per vocabulary entry",
        "Only the final position has been able to read the whole prompt, so it is the one "
        "that chooses what comes next. Its row is multiplied one last time into "
        "{vocab_size} numbers — a score for every entry in the vocabulary. This run decodes "
        "greedily, which means the highest score simply wins, with no sampling."
    ),
    LessonScene(
        "append", "And again",
        "Append the token and run the whole thing again",
        "The chosen token is added to the text and the model runs again to score the token "
        "after it, which is why an answer arrives piece by piece. The new position needs "
        "keys and values for everything before it, and those were already computed, so they "
        "are kept and reused instead of recalculated. TensorScope captured this second pass "
        "in full; the remaining {remaining_count} passes ran with capture off so you can "
        "read a complete answer."
    ),
    LessonScene(
        "limits", "What we can say",
        "What we can and cannot conclude from this",
        "Everything you just walked through was read out of arrays this run produced, and "
        "every screen said which numbers those were. That is a strong claim about values "
        "and a deliberately weak one about meaning: knowing every number a model computed "
        "is not the same as knowing why it learned to compute them."
    ),
]

LESSON_SCENE_INDEX = {scene.key: scene for scene in LESSON_SCENES}
LESSON_SCENE_KEYS = [scene.key for scene in LESSON_SCENES]


@dataclass(frozen=True)
class Reveal:
    """One optional expansion inside a scene.

    `key` is stable and is what LessonState records, so a reveal the reader opened is
    still open after Back/Next rebuilds the scene.  `body` may be empty when the
    expansion is a widget (a heatmap, an inspector, a token picker) rather than prose.
    """

    key: str
    title: str
    body: str = ""


REVEALS: dict[str, tuple[Reveal, ...]] = {
    "opening": (
        Reveal("response", "Show the model's whole response"),
        Reveal(
            "provenance", "How do we know these numbers are real?",
            "The tensors were taken out of the running model by forward hooks, not "
            "reproduced afterwards. Three checks had to pass before this run could be "
            "saved: the arrays came from the capture path, the attention arithmetic still "
            "matched the library's own implementation on every captured call, and the "
            "prompt pass was re-run with capture switched off — the resulting scores had to "
            "agree bit for bit, a difference of exactly zero rather than a tolerance. Had "
            "any of the three failed there would be no run here to open."
        ),
    ),
    "pieces": (
        Reveal(
            "template", "Why are there pieces I did not write?",
            "An instruction-tuned model is trained on conversations laid out in a fixed "
            "format, so the tokenizer wraps your words in the markers that format uses: who "
            "is speaking, where a turn begins, where it ends. Those markers are ordinary "
            "vocabulary entries with ordinary IDs, and the model attends to them exactly as "
            "it attends to your words."
        ),
        Reveal(
            "ids", "Is the ID number itself meaningful?",
            "No. An ID is a position in a list — entry 1894 is simply the 1,895th string "
            "the tokenizer knows. Nothing is added, compared or ordered using the ID; it is "
            "only used to pick a row out of the lookup table on the next screen. Two "
            "adjacent IDs have no more in common than two adjacent entries in an index."
        ),
    ),
    "rows": (
        Reveal("inspect", "Inspect the whole row exactly"),
        Reveal(
            "table", "Where does the row come from?",
            "From a learned table with one row per vocabulary entry, trained along with "
            "everything else. This capture stores the rows this run looked up, not the "
            "table — the full table for this model would be far larger than the run. "
            "Nothing on screen is a lookup performed by TensorScope; these are the rows the "
            "model itself produced."
        ),
        Reveal(
            "features", "What does one of these numbers mean?",
            "Individually, nothing you can name. The row is a learned coordinate vector, "
            "and whatever it encodes is spread across many entries at once rather than "
            "filed one idea per slot. Labelling a single coordinate with a human concept "
            "would be an invention, so TensorScope does not do it."
        ),
    ),
    "multiply": (
        Reveal(
            "arithmetic", "Show the arithmetic for one output number",
            "Written out, one output number is x₀·w₀ + x₁·w₁ + x₂·w₂ + … summed over the "
            "whole input row, plus a bias where the architecture uses one. The x values on "
            "this screen are captured. The w values are learned parameters, which this "
            "capture does not store — so the individual products cannot be shown, only the "
            "input row, the shape of the weight set, and the captured result."
        ),
        Reveal(
            "normalize", "Why is the row normalized first?",
            "Because the same operation runs dozens of times in a row. Each pass through a "
            "layer can make a row's numbers systematically larger or smaller, and over "
            "{layer_count} layers that drift would swamp the arithmetic. Normalization "
            "rescales every row by its own magnitude, then applies learned per-entry "
            "scales, so each layer starts from a predictable size. It changes no widths and "
            "mixes no positions together."
        ),
    ),
    "three": (
        Reveal(
            "names", "Query, key and value — and what they are not",
            "The names come from database lookup and are a loose analogy, not a mechanism: "
            "nothing is searched and nothing matches exactly. All three are rows of learned "
            "numbers produced by multiplication, and the only thing that makes one a query "
            "and another a key is which side of the upcoming comparison it sits on."
        ),
        Reveal(
            "widths", "Why are the three widths different?",
            "The query rows are {projection_width} numbers wide and the key and value rows "
            "{key_width}. That is grouped-query attention: this model has {head_count} "
            "query heads but only {kv_head_count} sets of keys and values, and several query "
            "heads share one set. Sharing cuts how much has to be kept and reused while "
            "decoding. The shared sets are copied out to match the query heads before any "
            "comparison, so the arrays that reach the multiplication do line up."
        ),
    ),
    "prepare": (
        Reveal("same_token", "The same piece of text at two different positions"),
        Reveal(
            "rope", "The names for these two steps",
            "Splitting the row into {head_count} slices of {head_dim} is what gives a model "
            "multiple attention heads. Rotating each slice by its position is rotary "
            "position embedding, RoPE: pairs of entries are turned through an angle "
            "proportional to the position index, so a later comparison between two slices "
            "depends on how far apart they are. Some models additionally normalize each "
            "query and key slice before rotating. Only the finished rows are captured, "
            "never the rotation's intermediate values."
        ),
    ),
    "compare": (
        Reveal("dot", "Recompute one comparison from the stored rows"),
        Reveal(
            "scale", "Why divide by the square root of {head_dim}?",
            "A sum of {head_dim} products grows with how many terms it has, so wider slices "
            "would give systematically larger comparison numbers, and the softmax on the "
            "next screen would then push nearly all of the amount onto a single position. "
            "Dividing by √{head_dim} keeps the numbers in a usable range. The stored "
            "comparison array already has this division and the masking applied — it is "
            "exactly what was handed to softmax."
        ),
        Reveal("heads", "Look at a different head"),
    ),
    "weights": (
        Reveal("softmax", "Show the softmax arithmetic for this row"),
        Reveal("mask", "Show what a masked position holds"),
        Reveal("heatmap", "See the whole head at once"),
        Reveal(
            "sum", "Do the amounts add up to exactly one?",
            "Mathematically yes; in the stored array, not quite. The model computes in a "
            "16-bit format whose smallest relative step is about 0.4%, so a row that sums "
            "to one before rounding sums to slightly more or less after it. The amounts "
            "shown are the ones the model actually used, so TensorScope displays them as "
            "stored rather than re-normalizing them to look tidy."
        ),
    ),
    "combine": (
        Reveal("value_row", "Look at the value row this amount scales"),
        Reveal(
            "not_saved", "What this capture does not keep here",
            "Two intermediates are missing by design. The per-head mix — amounts times "
            "value rows, before the heads are joined — was never written out by the model, "
            "and neither was the joined-head row that goes into the last multiplication. "
            "The learned weights of that multiplication are not stored either. What is "
            "stored is its output, which is why the diagram shows those steps as described "
            "operations with captured endpoints either side."
        ),
    ),
    "repeat": (
        Reveal(
            "residual", "The addition and the per-position network",
            "After attention the layer adds its result onto the row that went in, so "
            "information already present is carried forward rather than overwritten — that "
            "running row is usually called the residual stream. Then the row is normalized "
            "again and passed through a small two-stage network that works on each position "
            "on its own, and that result is added on as well. Attention is the only part "
            "that moves information between positions. The sum and the network's interior "
            "are not stored; the layer's finished output is."
        ),
        Reveal("compare_layers", "Compare an early layer with a later one"),
    ),
    "scores": (
        Reveal("inspect_logits", "Inspect the full score vector"),
        Reveal("runner_up", "What came second?"),
        Reveal(
            "percentages", "Turning scores into percentages",
            "Scores are not probabilities, but the same softmax from earlier converts them "
            "into one: raise e to each score and divide by the total. TensorScope computes "
            "that for the leading candidates only, and labels it as its own calculation, "
            "because the model never produced it — greedy decoding compares the raw scores "
            "and takes the largest. The percentages also depend on how many candidates are "
            "included, which is another reason they are presentation rather than data."
        ),
    ),
    "append": (
        Reveal("token_pick", "Pick any token of the response"),
        Reveal(
            "cache", "Keys and values are kept, not recomputed",
            "Every earlier position's key and value rows depend only on that position, so "
            "they do not change when a token is appended. Keeping them is the key-value "
            "cache, and it is why the second pass processes one new position instead of "
            "redoing all {prompt_token_count}. The saved keys and values for the second "
            "pass contain both the reused rows and the new one, because that is the array "
            "the comparison actually used."
        ),
        Reveal(
            "shapes", "Why the second pass has a different shape",
            "In the prompt pass every position is a query, so the comparisons form a square "
            "grid. In the second pass there is a single new query and it may read every "
            "earlier position plus itself, so its comparisons are one row, one entry longer "
            "than the prompt. Same operation, different rectangle — the arrays are stored "
            "in that shape and the inspector shows them that way."
        ),
    ),
    "limits": (
        Reveal("tiers", "What the four labels on every number mean"),
        Reveal("know", "What this capture does establish"),
        Reveal("cannot", "What it cannot establish"),
    ),
}

REVEAL_INDEX = {
    scene: {reveal.key: reveal for reveal in reveals} for scene, reveals in REVEALS.items()
}

assert set(REVEALS) <= set(LESSON_SCENE_INDEX), "REVEALS names a scene that does not exist"


# ── Plain language first, technical name second ──────────────────────────────────
# The lesson introduces each of these by its role and only then by its name.  The
# caveat lives here once so that no individual scene has to repeat it, and so that it
# cannot be lost by editing a scene.

@dataclass(frozen=True)
class Term:
    plain: str    # the role, in words a reader already has
    name: str     # the conventional name
    short: str    # the usual symbol, or "" where there is none
    gloss: str


TERMS = (
    Term("what this position is looking for", "query", "Q",
         "A row of learned numbers that will be compared against other positions."),
    Term("what each position can be matched against", "key", "K",
         "A second row of learned numbers: the other side of that comparison."),
    Term("the information available to be combined", "value", "V",
         "A third row — what a position contributes when others take from it."),
    Term("one of several comparisons running side by side", "head", "",
         "A fixed slice of the query, key and value rows with its own mixing pattern."),
    Term("rotating a slice by its position", "rotary position embedding", "RoPE",
         "Turns pairs of entries through an angle set by the position index, so a "
         "comparison can depend on how far apart two positions are."),
    Term("several query heads sharing one set of keys and values",
         "grouped-query attention", "GQA",
         "Fewer stored key and value sets, copied out to match the query heads."),
    Term("turning any list of numbers into amounts that sum to one", "softmax", "",
         "Raise e to each number, then divide by the total."),
    Term("hiding positions that come later in the text", "causal mask", "",
         "Forces those comparisons to a hugely negative number, so softmax gives them "
         "essentially no amount."),
    Term("one score per vocabulary entry", "logits", "",
         "The final row multiplied out to vocabulary width; the largest wins under "
         "greedy decoding."),
    Term("reusing earlier keys and values while decoding", "key-value cache", "KV cache",
         "Earlier positions' keys and values do not change, so they are kept."),
)

TERM_CAVEAT = (
    "These are useful descriptions of numerical operations, not sentences the model "
    "reads. A query row is not a question and a value row is not a fact; they are rows "
    "of learned numbers whose only meaning is what the arithmetic does with them."
)


# ── Optional understanding checks ────────────────────────────────────────────────
# Every option carries its own explanation, so answering wrongly teaches rather than
# scores.  Nothing here gates navigation: Next always works, answered or not.

@dataclass(frozen=True)
class Checkpoint:
    key: str
    scene: str
    question: str
    options: tuple[str, ...]
    correct: int
    explain: tuple[str, ...]      # one per option, in the same order

    def __post_init__(self) -> None:
        assert len(self.options) == len(self.explain), self.key
        assert 0 <= self.correct < len(self.options), self.key


CHECKPOINTS = (
    Checkpoint(
        "id_meaning", "pieces",
        "Does a token ID measure how meaningful a word is?",
        (
            "No — it is only a position in the vocabulary list",
            "Yes — a larger ID means a more significant word",
            "Yes — IDs are ordered by how often a word appears",
        ),
        0,
        (
            "Right. The ID selects a row from the lookup table and is never added, compared "
            "or ranked. The numbers that carry information are the ones inside that row, "
            "which is the next screen.",
            "Not quite — the ID is only an index. Nothing in the model does arithmetic with "
            "it; it picks out a row, and that row is what everything afterwards works on.",
            "Frequency does affect how a vocabulary gets built, but the resulting ID is "
            "still just a position in a list. The model never compares IDs — it looks up "
            "the row stored at that position.",
        ),
    ),
    Checkpoint(
        "which_dims", "compare",
        "To compare the queries against the keys, which sizes have to match?",
        (
            "The {head_dim} numbers in each slice — the axis being summed over",
            "The number of positions on each side",
            "Nothing has to match; a matrix multiplication accepts any two shapes",
        ),
        0,
        (
            "Right. Each comparison multiplies matching entries and adds them up, so the "
            "two slices must be the same length — {head_dim} here. The position counts need "
            "not agree, which is exactly why the second pass works.",
            "They happen to be equal during the prompt pass, but they need not be: in the "
            "second pass one query is compared against every cached position. What must "
            "match is the slice length being summed over, {head_dim}.",
            "It does not. Every product pairs one entry of the query slice with the same "
            "entry of the key slice, so the two have to be the same length — {head_dim} in "
            "this model.",
        ),
    ),
    Checkpoint(
        "causal", "weights",
        "During the prompt pass, can a position take information from a later position?",
        (
            "No — later positions are masked before softmax",
            "Yes — the whole prompt is present, so all positions can see each other",
            "Only in the first layer, before positions are rotated",
        ),
        0,
        (
            "Right. Those comparisons are forced to a hugely negative number first, so "
            "softmax leaves them at essentially zero amount. The stored value is visible in "
            "this screen's mask expansion.",
            "The whole prompt is present in memory, but the mask still applies. Training "
            "and generation have to agree: while generating, later text does not exist yet, "
            "so it is hidden during the prompt pass too.",
            "The mask applies in every layer, not just the first. You can check that — the "
            "upper-right cells hold the same stored value at every layer.",
        ),
    ),
    Checkpoint(
        "who_chose", "append",
        "Did the first generated token's own forward pass select that token?",
        (
            "No — the prompt's last position selected it; this pass scores what comes next",
            "Yes — the token was scored by the pass that processed it",
            "Both passes contributed to the choice",
        ),
        0,
        (
            "Right. The prompt pass produced the scores that chose it. Running the chosen "
            "token through the model is how the position after it gets scored — one pass "
            "behind, always.",
            "The other way round. A position's scores are for the token that follows it, so "
            "a token has already been chosen before its own pass begins.",
            "Each choice comes from exactly one pass: the one ending at the position before "
            "it. The second pass chooses the second token, not the first.",
        ),
    ),
)

CHECKPOINT_INDEX = {check.key: check for check in CHECKPOINTS}
CHECKPOINTS_BY_SCENE: dict[str, tuple[Checkpoint, ...]] = {
    scene.key: tuple(c for c in CHECKPOINTS if c.scene == scene.key) for scene in LESSON_SCENES
}

assert all(check.scene in LESSON_SCENE_INDEX for check in CHECKPOINTS), "checkpoint scene missing"


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


def visible_text(text: str) -> str:
    """A printable stand-in for a token spelling, for use inside a sentence.

    Token spellings legitimately contain newlines, tabs and leading spaces.  Copy that
    interpolates one needs it to stay on one line without pretending the whitespace
    is not there, so control characters become visible glyphs.  The token's exact
    stored spelling is always available through the chips and the inspector; this is
    only for prose.
    """
    if not text:
        return "∅"
    shown = text.replace("\n", "⏎").replace("\r", "⏎").replace("\t", "⇥")
    return shown.replace(" ", "␣") if not shown.strip() else shown


def _width(layer: Any, name: str) -> int:
    """Last-axis size of a captured tensor, or 0 when this run has no such tensor."""
    tensors = getattr(layer, "tensors", None) or {}
    array = tensors.get(name)
    shape = getattr(array, "shape", ())
    return int(shape[-1]) if shape else 0


def learn_facts(capture: Any) -> dict[str, Any]:
    """Read structural facts from a saved capture without loading a model.

    Every field any scene, reveal or checkpoint interpolates must appear here, or a
    reader would meet a KeyError instead of a sentence.  self_test() formats all of
    that copy against this, including against a tiny synthetic capture whose tensors
    are three-dimensional and whose embedding is one-dimensional -- so every shape
    read below is guarded rather than assumed.
    """
    embedding = getattr(capture, "embedding", None)
    logits = getattr(capture, "logits", None)
    vocabulary = int(logits.shape[-1]) if logits is not None else 0
    layers = getattr(capture, "layers", {}) or {}
    first = layers[min(layers)] if layers else None

    prepared = getattr(getattr(first, "tensors", None) or {}, "get", lambda _: None)("q_attended")
    prepared_shape = tuple(getattr(prepared, "shape", ()) or ())
    head_dim = int(prepared_shape[-1]) if prepared_shape else 0
    head_count = int(prepared_shape[-3]) if len(prepared_shape) >= 4 else 1

    projection_width = _width(first, "q")
    key_width = _width(first, "k")
    if head_dim and key_width and key_width % head_dim == 0:
        kv_head_count = key_width // head_dim
    else:
        kv_head_count = head_count

    generated = list(getattr(capture, "tokens", []) or [])
    prompt_count = len(getattr(capture, "prompt_token_ids", []) or [])
    return {
        "model": capture.metadata.get("model", "an open-weight model"),
        "layer_count": len(layers),
        "prompt_token_count": prompt_count,
        "generated_count": len(getattr(capture, "token_ids", []) or []),
        "remaining_count": max(len(getattr(capture, "token_ids", []) or []) - 1, 0),
        "hidden_size": int(embedding.shape[-1]) if embedding is not None else 0,
        "vocab_size": f"{vocabulary:,}" if vocabulary else "full",
        "head_count": head_count,
        "head_dim": head_dim,
        "kv_head_count": kv_head_count,
        "projection_width": projection_width,
        "key_width": key_width,
        "first_token": visible_text(generated[0]) if generated else "—",
        "final_position": max(prompt_count - 1, 0),
    }
