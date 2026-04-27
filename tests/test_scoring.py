"""Tests for biofoundation.model.scoring.

Focus: off-by-one alignment between logits and target tokens in
``compute_ll_clm``. We cross-check against HuggingFace's
``labels=input_ids`` cross-entropy loss, which is the gold standard for
the standard CLM shift convention.

``compute_ll_clm`` returns sums + counts (per row), not means. The
caller does ``pred.sum(axis=0)`` then divides to get a token-weighted
dataset-wide LL — matching how Marin/levanter computes ``eval/loss``.
"""

import math

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM

from biofoundation.model.adapters.hf import HFCausalLM
from biofoundation.model.base import CausalLM
from biofoundation.model.scoring import compute_ll_clm


TINY_CLM = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"


def _load_tiny_clm():
    return HFCausalLM(AutoModelForCausalLM.from_pretrained(TINY_CLM))


class _DeterministicCLM(CausalLM):
    """Test double whose forward returns a fixed logits tensor."""

    def __init__(self, logits: Tensor):
        super().__init__()
        # Register as buffer so .to(device) works, but value is fixed.
        self.register_buffer("_logits", logits)

    def forward(self, input_ids):  # type: ignore[override]
        # Returns a *copy* sliced to the input batch/length so callers can
        # use any input_ids shape that matches.
        B, L = input_ids.shape
        assert self._logits.shape[0] >= B and self._logits.shape[1] >= L
        return self._logits[:B, :L].clone()


def test_compute_ll_clm_matches_hf_cross_entropy():
    """ll_sum / n  ==  -model(input_ids, labels=input_ids).loss.

    HF's CausalLM models compute loss as mean cross-entropy over the L-1
    shifted targets. Dividing our per-row ll_sum by n recovers the same
    quantity, with the standard sign flip.
    """
    torch.manual_seed(0)
    model = _load_tiny_clm()
    raw = AutoModelForCausalLM.from_pretrained(TINY_CLM)
    raw.eval()
    model.eval()

    vocab_size = raw.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (3, 17))

    with torch.no_grad():
        out = compute_ll_clm(model, input_ids)  # [B, 2]
        assert out.shape == (3, 2)
        ll_mean = out[:, 0] / out[:, 1]
        for i in range(input_ids.shape[0]):
            hf_loss = raw(input_ids[i : i + 1], labels=input_ids[i : i + 1]).loss
            assert math.isclose(
                ll_mean[i].item(), -hf_loss.item(), rel_tol=1e-5, abs_tol=1e-5
            ), f"row {i}: ours={ll_mean[i].item()} hf={-hf_loss.item()}"


def test_compute_ll_clm_hand_computed_two_token():
    """Smallest non-trivial off-by-one check with a known logits tensor."""
    # Vocab size 4, batch 1, length 3
    # logits[0, 0] predicts input_ids[0, 1]
    # logits[0, 1] predicts input_ids[0, 2]
    # logits[0, 2] is unused (last position)
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, 0.0],  # softmax: index 1 favoured
                [2.0, 0.0, 0.0, 0.0],  # softmax: index 0 favoured
                [9.0, 9.0, 9.0, 9.0],  # ignored
            ]
        ]
    )
    input_ids = torch.tensor([[3, 1, 0]])
    model = _DeterministicCLM(logits)

    log_softmax_0 = torch.log_softmax(
        logits[0, 0], dim=-1
    )  # for target input_ids[0,1]=1
    log_softmax_1 = torch.log_softmax(
        logits[0, 1], dim=-1
    )  # for target input_ids[0,2]=0
    expected_sum = (log_softmax_0[1] + log_softmax_1[0]).item()

    out = compute_ll_clm(model, input_ids)
    assert out.shape == (1, 2)
    assert math.isclose(out[0, 0].item(), expected_sum, rel_tol=1e-6, abs_tol=1e-7)
    assert out[0, 1].item() == 2.0  # n = L - 1 = 2


def test_compute_ll_clm_target_side_shift():
    """is_upper applies to the *target* token (input_ids[i+1]), not the source."""
    torch.manual_seed(1)
    B, L, V = 1, 8, 5
    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))
    model = _DeterministicCLM(logits)

    # Source-aligned mask: positions [0..3] uppercase, [4..7] lowercase.
    is_upper = torch.tensor([[True, True, True, True, False, False, False, False]])

    out = compute_ll_clm(model, input_ids, is_upper)  # [B, 4]
    assert out.shape == (1, 4)
    ll_sum_upper, ll_sum_lower, n_upper, n_lower = out[0].tolist()

    # Manual computation
    log_softmax = torch.log_softmax(logits[0, :-1], dim=-1)
    targets = input_ids[0, 1:]
    per_target_logp = log_softmax[torch.arange(L - 1), targets]
    # Target frame: is_upper[1:] = [T, T, T, F, F, F, F]  → 3 upper, 4 lower
    upper_mask = is_upper[0, 1:]
    lower_mask = ~upper_mask

    expected_sum_upper = per_target_logp[upper_mask].sum().item()
    expected_sum_lower = per_target_logp[lower_mask].sum().item()

    assert math.isclose(ll_sum_upper, expected_sum_upper, rel_tol=1e-6, abs_tol=1e-7)
    assert math.isclose(ll_sum_lower, expected_sum_lower, rel_tol=1e-6, abs_tol=1e-7)
    assert n_upper == 3.0
    assert n_lower == 4.0

    # Sanity: had we mistakenly used the SOURCE frame [:-1] — would give
    # n_upper=4, n_lower=3 (different counts) and a different sum.
    assert n_upper != is_upper[0, :-1].sum().item()


def test_compute_ll_clm_invariants():
    """Per-row invariants of the [B, 4] output."""
    torch.manual_seed(2)
    B, L, V = 4, 11, 7
    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))
    model = _DeterministicCLM(logits)

    is_upper = torch.zeros(B, L, dtype=torch.bool)
    is_upper[0, :3] = True
    is_upper[1, :7] = True
    is_upper[2, :2] = True
    is_upper[3, :9] = True

    out = compute_ll_clm(model, input_ids, is_upper)  # [B, 4]
    ll_sum_upper, ll_sum_lower, n_upper, n_lower = out.unbind(-1)
    out_no_mask = compute_ll_clm(model, input_ids)  # [B, 2]
    ll_sum_total, n_total = out_no_mask.unbind(-1)

    # Sums partition the total
    assert torch.allclose(ll_sum_upper + ll_sum_lower, ll_sum_total, atol=1e-5)
    # Counts partition L-1
    assert torch.equal(n_upper + n_lower, n_total)
    assert torch.all(n_total == float(L - 1))


def test_compute_ll_clm_dataset_wide_token_weighted_mean():
    """The intended aggregation pattern works and beats avg-of-means
    when n_upper / n_lower vary across rows."""
    torch.manual_seed(5)
    B, L, V = 4, 11, 7
    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))
    model = _DeterministicCLM(logits)

    is_upper = torch.zeros(B, L, dtype=torch.bool)
    is_upper[0, :3] = True
    is_upper[1, :7] = True
    is_upper[2, :2] = True
    is_upper[3, :9] = True

    out = compute_ll_clm(model, input_ids, is_upper).double()  # cast for fp64 accumulate
    S_u, S_l, n_u, n_l = out.sum(dim=0).unbind(-1)
    LL_all = ((S_u + S_l) / (n_u + n_l)).item()
    LL_upper = (S_u / n_u).item()
    LL_lower = (S_l / n_l).item()

    # Brute-force: gather *every* target logp across the whole batch and
    # split by mask.
    log_softmax = torch.log_softmax(logits[:, :-1], dim=-1)
    targets = input_ids[:, 1:]
    per_target_logp = torch.gather(log_softmax, 2, targets.unsqueeze(-1)).squeeze(-1)
    target_upper = is_upper[:, 1:]
    expected_LL_all = per_target_logp.mean().item()
    expected_LL_upper = per_target_logp[target_upper].mean().item()
    expected_LL_lower = per_target_logp[~target_upper].mean().item()

    assert math.isclose(LL_all, expected_LL_all, rel_tol=1e-6, abs_tol=1e-7)
    assert math.isclose(LL_upper, expected_LL_upper, rel_tol=1e-6, abs_tol=1e-7)
    assert math.isclose(LL_lower, expected_LL_lower, rel_tol=1e-6, abs_tol=1e-7)

    # Sanity that this differs from avg-of-per-sequence-means with
    # heterogeneous counts (the wrong way to aggregate).
    per_seq_upper = (out[:, 0] / out[:, 2]).double()
    naive = per_seq_upper.mean().item()
    assert not math.isclose(LL_upper, naive, rel_tol=1e-3, abs_tol=1e-3)


def test_compute_ll_clm_all_upper_or_all_lower_rows_aggregate_correctly():
    """All-upper / all-lower rows have n=0 in one bucket, ll_sum=0 there.
    They still contribute correctly when summing across the dataset (no
    NaN in the per-row tensor, no NaN gymnastics needed at aggregation)."""
    torch.manual_seed(8)
    B, L, V = 3, 6, 4
    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))
    model = _DeterministicCLM(logits)

    is_upper = torch.zeros(B, L, dtype=torch.bool)
    is_upper[0, :] = True   # row 0: all upper (target frame too)
    # row 1: all lower (default)
    is_upper[2, :3] = True  # row 2: mixed

    out = compute_ll_clm(model, input_ids, is_upper)
    assert out.shape == (B, 4)
    # Per-row primitive output is finite everywhere — no NaN to manage.
    assert torch.isfinite(out).all()
    # Row 0: n_lower == 0, ll_sum_lower == 0
    assert out[0, 1].item() == 0.0
    assert out[0, 3].item() == 0.0
    # Row 1: n_upper == 0, ll_sum_upper == 0
    assert out[1, 0].item() == 0.0
    assert out[1, 2].item() == 0.0

    # Aggregating across all 3 rows still gives meaningful global LLs
    S_u, S_l, n_u, n_l = out.double().sum(dim=0).tolist()
    assert n_u > 0 and n_l > 0  # because row 2 contributes to both
    LL_upper = S_u / n_u
    LL_lower = S_l / n_l
    assert math.isfinite(LL_upper) and math.isfinite(LL_lower)


def test_compute_ll_clm_shape_without_mask():
    torch.manual_seed(4)
    B, L, V = 2, 6, 4
    logits = torch.randn(B, L, V)
    input_ids = torch.randint(0, V, (B, L))
    model = _DeterministicCLM(logits)
    out = compute_ll_clm(model, input_ids)
    assert out.shape == (B, 2)
    assert torch.all(out[:, 1] == float(L - 1))
