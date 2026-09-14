"""Educational copy for the saved-capture UI; no model or GUI imports.

Equations describe the architecture. Numerical displays must use the saved arrays,
including when a mathematically intermediate result was not retained by capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
        "These are attention mixing weights, not probabilities of next output tokens. "
        "They describe this head's computation and are not a complete explanation of "
        "why the model answered as it did."
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


@dataclass(frozen=True)
class StoryStage:
    """One journey stage; format fields come from story_facts()."""

    key: str
    heading: str
    plain: str


STORY_STAGES = [
    StoryStage(
        "prompt", "A real prompt becomes a next-token decision",
        "Follow this model's actual captured activations from input text to its first "
        "output token. TensorScope retains selected tensors at named capture points, "
        "not every intermediate operation. The journey distinguishes captured data "
        "from conceptual steps whose intermediate values were not saved. There are two "
        "captured phases: prompt prefill, then a forward pass for the first selected token. "
        "Later forward passes are not captured; the remaining generated text is included "
        "to put that first token in context."
    ),
    StoryStage(
        "tokenization", "1 · Text becomes token IDs",
        "A tokenizer maps text into vocabulary entries: words, fragments, punctuation, "
        "whitespace, or special markers. This run supplied {token_count} token IDs to "
        "the model. The saved token strings are the tokenizer's pieces, so they can "
        "include visible spellings for whitespace rather than ordinary prose. When a "
        "chat template is available, TensorScope applies it before tokenization; this "
        "can add role markers and an assistant prefix that you did not type. Token IDs "
        "are lookup indices, not numerical measurements of meaning."
    ),
    StoryStage(
        "embedding", "2 · IDs become feature vectors",
        "The embedding module looks up a learned vector for each token ID. In this run, "
        "each vector has {hidden_size} model features. The captured embedding output "
        "has axes [batch, token position, model feature]; it supplies the first decoder "
        "layer. Individual features are learned coordinates, not named concepts. The "
        "complete embedding lookup table is not saved, only the rows produced for "
        "this pass. In the generated-token phase, just the new token receives a new "
        "embedding; prompt information is available through cached keys and values."
    ),
    StoryStage(
        "layers", "3 · Follow one of {layer_count} decoder layers",
        "Each decoder layer repeats the same general structure with different learned "
        "parameters and activations. Normalization prepares the input, Q/K/V projections "
        "create attention features, and attention mixes information across token positions. "
        "Residual connections and a feed-forward network complete the layer. Select one "
        "layer and follow its steps; its output feeds the next layer. Shapes are taken "
        "from the captured arrays. A preview or selected head is only a display view; "
        "the complete saved tensors remain available in Raw / Detail mode."
    ),
    StoryStage(
        "logits", "4 · The final prompt position chooses a token",
        "After the last decoder layer, Qwen applies a final normalization and a learned "
        "vocabulary projection. The final normalization output and projection weights "
        "are not saved. The retained logits, when available, are the vocabulary score "
        "vector at the final prompt position. A logit is a score, not a probability. "
        "TensorScope uses greedy decoding: argmax chooses the vocabulary ID with the "
        "highest score as the first output token. No vocabulary-softmax probability "
        "tensor is persisted. Older runs can lack the logits vector while retaining "
        "the selected token and the rest of the capture."
    ),
    StoryStage(
        "coda", "5 · The first selected token starts the next pass",
        "The first token was chosen by the prompt-prefill logits. That token then enters "
        "a second forward pass through all {layer_count} decoder layers. It contributes "
        "one new query position while each layer reuses the prompt's cached keys and "
        "values and adds the new token's K/V. TensorScope captures this first-token "
        "pass too. It computes scores for choosing the second output token, but those "
        "logits are not persisted. Further forward passes are uncaptured. The answer "
        "is generated one token at a time, which may be a word fragment rather than "
        "a whole word."
    ),
]

STORY_STAGE_INDEX = {stage.key: stage for stage in STORY_STAGES}

LOGITS_UNAVAILABLE = (
    "This saved run has no retained logits vector. Earlier capture versions did not "
    "persist it. The selected first token is available, but its vocabulary scores "
    "cannot be recovered from this capture. A new run can capture those scores."
)


def story_facts(capture: Any) -> dict[str, Any]:
    """Read structural facts from a saved capture without loading a model."""
    return {
        "layer_count": len(capture.layers),
        "token_count": len(capture.prompt_token_ids),
        "hidden_size": int(capture.embedding.shape[-1]) if capture.embedding is not None else 0,
    }
