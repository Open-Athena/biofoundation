from jaxtyping import Float, Int
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import cast
from einops import rearrange, reduce

from biofoundation.model.base import (
    CausalLM,
    CausalLMWithEmbeddings,
    EmbeddingModel,
    MaskedLM,
)


def compute_llr_mlm(
    model: MaskedLM,
    input_ids: Int[Tensor, "B L"],
    pos: Int[Tensor, " B"],
    ref: Int[Tensor, " B"],
    alt: Int[Tensor, " B"],
) -> Float[Tensor, " B"]:
    B, _ = input_ids.shape
    batch_indices = torch.arange(B)
    logits = model(input_ids)
    logits_pos = logits[batch_indices, pos]
    logits_ref = logits_pos[batch_indices, ref]
    logits_alt = logits_pos[batch_indices, alt]
    return cast(Float[Tensor, " B"], logits_alt - logits_ref)


def compute_llr_clm(
    model: CausalLM,
    input_ids: Int[Tensor, "B 2 L"],
) -> Float[Tensor, " B"]:
    """Compute log-likelihood ratio for causal language models.

    Args:
        model: Causal language model
        input_ids: Input sequences with shape [B, 2, L] where the 2 sequences are [ref, alt]

    Returns:
        Log-likelihood ratio (alt_logprob - ref_logprob) with shape [B]
    """
    B = input_ids.shape[0]
    # Reshape to process both sequences together: [B*2, L]
    input_ids = rearrange(input_ids, "B V L -> (B V) L")

    # Get logits from the model
    logits = model(input_ids)

    # Compute sequence-level log probabilities
    log_prob = _clm_seq_logprob(logits, input_ids)

    # Reshape back to [B, 2] where dim 1 is [ref_logprob, alt_logprob]
    log_prob = rearrange(log_prob, "(B V) -> B V", B=B)

    # Compute LLR as alt_logprob - ref_logprob
    llr = log_prob[:, 1] - log_prob[:, 0]  # alt - ref

    return llr


def compute_reflogprob_mlm(
    model: MaskedLM,
    input_ids: Int[Tensor, "B L"],
    pos: Int[Tensor, " B"],
    ref: Int[Tensor, " B"],
) -> Float[Tensor, " B"]:
    """Forward pass to compute reference log probabilities.

    This method:
    1. Runs the input through the language model to get logits
    2. Extracts logits at the specified positions
    3. Computes log probabilities using softmax
    4. Returns the log probability of the reference tokens

    Args:
        model: Language model to use for inference
        input_ids: Batch of token ID sequences with shape [B, L] where
            B is batch size and L is sequence length
        pos: Positions to evaluate for each sequence in the batch with
            shape [B] where each element is the position index
        ref: Reference token IDs to compute log probabilities for with
            shape [B] where each element is the token ID

    Returns:
        Log probabilities of the reference tokens at the specified positions
        with shape [B] where B is the batch size
    """
    B, _ = input_ids.shape
    batch_indices = torch.arange(B)
    logits = model(input_ids)
    logits = logits[batch_indices, pos]
    logprobs = F.log_softmax(logits, dim=-1)
    res = logprobs[batch_indices, ref]
    return res


def compute_reflogprob_clm(
    model: CausalLM,
    input_ids: Int[Tensor, "B 4 L"],
    ref: Int[Tensor, " B"],
) -> Float[Tensor, " B"]:
    B = input_ids.shape[0]
    batch_indices = torch.arange(B)
    input_ids = rearrange(input_ids, "B V L -> (B V) L")
    logits = model(input_ids)
    log_prob = _clm_seq_logprob(logits, input_ids)
    log_prob = rearrange(log_prob, "(B V) -> B V", B=B)
    # marginal log-probability of each of the 4 alleles
    marginal_log_prob = torch.log_softmax(log_prob, dim=-1)
    ref_log_prob = marginal_log_prob[batch_indices, ref]
    return ref_log_prob


# https://github.com/ArcInstitute/evo2/blob/4c3c8522dc99d2dc14b5b5a07cd65f2b67e6f457/evo2/scoring.py#L37
def _logits_to_logprobs(
    logits: Float[Tensor, "B L V"],
    input_ids: Int[Tensor, "B L"],
) -> Float[Tensor, "B L-1"]:
    """
    Takes in a tensor of logits of dimension (batch, length, vocab).
    Computes the log-likelihoods using a softmax along the vocab dimension.
    Uses the `input_ids` to index into the log-likelihoods and returns the likelihood
    of the provided sequence at each position with dimension (batch, length-1).
    """
    softmax_logprobs = torch.log_softmax(logits, dim=-1)
    softmax_logprobs = softmax_logprobs[:, :-1]
    input_ids = input_ids[:, 1:]
    assert softmax_logprobs.shape[1] == input_ids.shape[1]

    logprobs = torch.gather(
        softmax_logprobs,  # Gather likelihoods...
        2,  # along the vocab dimension...
        input_ids.unsqueeze(-1),  # using the token ids to index.
    ).squeeze(-1)

    return logprobs


def _clm_seq_logprob(
    logits: Float[Tensor, "B L V"],
    input_ids: Int[Tensor, "B L"],
) -> Float[Tensor, " B"]:
    # token-level log-probability
    log_probs = _logits_to_logprobs(logits, input_ids)
    # seq-level log-probability
    return reduce(log_probs.float(), "B L -> B", "sum")


def compute_ll_clm(
    model: CausalLM,
    input_ids: Int[Tensor, "B L"],
    is_upper: Int[Tensor, "B L"] | None = None,
) -> Float[Tensor, "B 2"] | Float[Tensor, "B 4"]:
    """Per-sequence log-likelihood sums and target counts under a CLM.

    Returns the *raw ingredients* for a dataset-wide token-weighted mean
    LL — sums and counts, not means. Aggregation happens outside (the
    caller does ``pred.sum(axis=0)`` then divides), which is what
    Marin/levanter eval does and what makes the result correct in the
    presence of all-upper / all-lower sequences.

    The "case of a token" is the loss weight applied when *predicting*
    that token. ``_logits_to_logprobs`` returns ``[B, L-1]`` where entry
    ``[b, i]`` is ``log p(input_ids[b, i+1] | input_ids[b, :i+1])``, so
    the relevant case is the case of the *target* ``input_ids[i+1]``. We
    therefore slice ``is_upper[:, 1:]`` to align with the L-1 log-probs.

    Output:

    - Without ``is_upper``: ``[B, 2]`` of ``(ll_sum, n)`` per row.
      ``n = L - 1`` for every row.
    - With ``is_upper``: ``[B, 4]`` of
      ``(ll_sum_upper, ll_sum_lower, n_upper, n_lower)`` per row.
      Invariants: ``ll_sum_upper + ll_sum_lower`` is the total sum,
      ``n_upper + n_lower = L - 1``. Special-token target positions
      (``is_upper = False``) fall into the "lower" bucket.

    Aggregation pattern::

        pred = ...                                         # [N, 4]
        S_u, S_l, n_u, n_l = pred.astype(np.float64).sum(axis=0)
        LL_all   = (S_u + S_l) / (n_u + n_l)
        LL_upper = S_u / n_u
        LL_lower = S_l / n_l

    Per-row sums are fp32; cast to fp64 before summing across rows for a
    realistic dataset (fp32 accumulates ~0.5 absolute error at totals of
    ~10^6, which is reachable on a 16k-row eval set).
    """
    logits = model(input_ids)
    logp = _logits_to_logprobs(logits, input_ids).float()  # [B, L-1]
    L_minus_1 = logp.shape[-1]
    if is_upper is None:
        ll_sum = logp.sum(dim=-1)
        n = torch.full_like(ll_sum, float(L_minus_1))
        return torch.stack([ll_sum, n], dim=-1)
    upper_t = is_upper[:, 1:].float()
    lower_t = 1.0 - upper_t
    n_upper = upper_t.sum(dim=-1)
    n_lower = lower_t.sum(dim=-1)
    ll_sum_upper = (logp * upper_t).sum(dim=-1)
    ll_sum_lower = (logp * lower_t).sum(dim=-1)
    return torch.stack([ll_sum_upper, ll_sum_lower, n_upper, n_lower], dim=-1)


def compute_euclidean_distance(
    model: EmbeddingModel,
    input_ids: Int[Tensor, "B 2 L"],
) -> Float[Tensor, " B"]:
    """Compute Euclidean distance between reference and alternate embeddings.

    Args:
        model: Embedding model
        input_ids: Input sequences with shape [B, 2, L] where the 2 sequences are [ref, alt]

    Returns:
        Euclidean distance with shape [B]
    """
    B = input_ids.shape[0]
    # Reshape to process both sequences together: [B*2, L]
    input_ids = rearrange(input_ids, "B V L -> (B V) L")

    # Get embeddings from the model: [B*2, L, D]
    embeddings = model(input_ids)

    # Reshape to [B, 2, L*D] and flatten L and D dimensions
    embeddings = rearrange(embeddings, "(B V) L D -> B V (L D)", B=B)

    # Extract ref and alt embeddings: [B, L*D] each
    ref_emb = embeddings[:, 0, :]
    alt_emb = embeddings[:, 1, :]

    # Compute pairwise Euclidean distance
    return F.pairwise_distance(ref_emb, alt_emb)  # [B]


def compute_llr_and_embedding_distance(
    model: CausalLMWithEmbeddings,
    input_ids: Int[Tensor, "B 2 L"],
) -> Float[Tensor, "B 3"]:
    """Compute LLR, last-layer distance, and middle-layer distance in one pass.

    Args:
        model: CausalLM model that returns logits and embeddings
        input_ids: Input sequences with shape [B, 2, L] where the 2 sequences are [ref, alt]

    Returns:
        Tensor with shape [B, 3] where columns are:
            - [:, 0]: LLR (log-likelihood ratio: alt_logprob - ref_logprob)
            - [:, 1]: Euclidean distance between last-layer embeddings
            - [:, 2]: Euclidean distance between middle-layer embeddings
    """
    B = input_ids.shape[0]
    # Reshape to process both sequences together: [B*2, L]
    input_ids_flat = rearrange(input_ids, "B V L -> (B V) L")

    # Single forward pass gets everything
    logits, last_emb, middle_emb = model(input_ids_flat)

    # Compute LLR from logits
    log_prob = _clm_seq_logprob(logits, input_ids_flat)
    log_prob = rearrange(log_prob, "(B V) -> B V", B=B)
    llr = log_prob[:, 1] - log_prob[:, 0]  # alt - ref

    # Compute last-layer embedding distance
    last_emb = rearrange(last_emb, "(B V) L D -> B V (L D)", B=B)
    last_distance = F.pairwise_distance(last_emb[:, 0], last_emb[:, 1])

    # Compute middle-layer embedding distance
    middle_emb = rearrange(middle_emb, "(B V) L D -> B V (L D)", B=B)
    middle_distance = F.pairwise_distance(middle_emb[:, 0], middle_emb[:, 1])

    # Stack into [B, 3] array
    return torch.stack([llr, last_distance, middle_distance], dim=1)
