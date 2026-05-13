import textwrap

import pandas as pd
import pytest
import torch
from Bio.Seq import Seq
from transformers import AutoTokenizer

from biofoundation.data import (
    NUCLEOTIDES,
    Genome,
    GenomicSet,
    _complement_base,
    _get_special_token_counts,
    transform_llr_mlm,
    transform_llr_clm,
    transform_reflogprob_mlm,
    transform_reflogprob_clm,
    transform_ll_clm,
)
from biofoundation.model.adapters.hf import HFTokenizer
from biofoundation.model.base import Tokenizer


class _SpecialTokensTokenizer(Tokenizer):
    """Wrap an HF tokenizer to optionally prepend BOS / append EOS.

    Uses synthetic IDs that don't collide with real DNA tokens.
    """

    def __init__(self, base, *, bos_id=None, eos_id=None):
        self._base = base
        self._bos = bos_id
        self._eos = eos_id

    def encode(self, text):
        ids = list(self._base.encode(text))
        if self._bos is not None:
            ids = [self._bos] + ids
        if self._eos is not None:
            ids = ids + [self._eos]
        return ids

    @property
    def mask_token_id(self):
        return self._base.mask_token_id

    @property
    def bos_token_id(self) -> int:
        if self._bos is None:
            raise AttributeError("no BOS configured")
        return self._bos

    @property
    def eos_token_id(self) -> int:
        if self._eos is None:
            raise AttributeError("no EOS configured")
        return self._eos


_BOS_ID = 100
_EOS_ID = 101


def _write_test_fasta(tmp_path):
    fasta = textwrap.dedent(">chr1\nACGTACGTAC\n")
    path = tmp_path / "llr_test.fa"
    path.write_text(fasta)
    return path


def _write_genome_fasta(tmp_path):
    fasta = textwrap.dedent(">chr1\nACGTACGTAC\n>chr2\nGGGCCCAGTA\n")
    path = tmp_path / "genome.fa"
    path.write_text(fasta)
    return path


def test_transform_reflogprob_mlm_basic():
    """Test basic functionality of transform_reflogprob_mlm with real DNA tokenizer"""
    # Setup - use real DNA tokenizer
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    pos = 2
    example = {"seq": "ATCGATCG", "pos": pos}

    # Execute
    result = transform_reflogprob_mlm(example, tokenizer)

    # Assert
    assert isinstance(result, dict)
    assert "input_ids" in result
    assert "pos" in result
    assert "ref" in result

    # Check types
    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["pos"], int)
    assert isinstance(result["ref"], int)

    # Check values
    assert result["pos"] == pos

    # Verify the sequence was properly tokenized
    input_ids = result["input_ids"]
    assert len(input_ids) > 0  # Should have tokens

    # Check that the position is valid
    assert pos < len(input_ids)


def test_transform_reflogprob_mlm_mask_position():
    """Test that the specified position is masked"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    pos = 1
    example = {"seq": "GCTAGCTA", "pos": pos}

    result = transform_reflogprob_mlm(example, tokenizer)

    # The input_ids should have the mask token at the specified position
    input_ids = result["input_ids"]
    assert input_ids[pos] == tokenizer.mask_token_id


def test_transform_reflogprob_mlm_reference_value():
    """Test that ref contains the original token value before masking"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    pos = 3
    example = {"seq": "TATATATA", "pos": pos}

    result = transform_reflogprob_mlm(example, tokenizer)

    # ref should contain the original token value before masking
    # This should be the token ID for the nucleotide at position 3
    assert result["ref"] >= 0  # Should be a valid token ID
    assert result["ref"] != tokenizer.mask_token_id  # Should not be the mask token


def test_transform_reflogprob_mlm_different_sequences():
    """Test with different DNA sequences"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")

    # Test with different sequences
    sequences = ["ACGT", "GCTA", "TTAA", "CGCG"]

    for seq in sequences:
        pos = 1
        example = {"seq": seq, "pos": pos}

        result = transform_reflogprob_mlm(example, tokenizer)

        # Basic structure checks
        assert "input_ids" in result
        assert "pos" in result
        assert "ref" in result

        # Position should be masked
        input_ids = result["input_ids"]
        assert input_ids[pos] == tokenizer.mask_token_id

        # Reference should be the original token
        assert result["ref"] != tokenizer.mask_token_id


def test_transform_llr_mlm_masks_reference_position(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))
    window_size = 4
    example = {"chrom": "chr1", "pos": 6, "ref": "C", "alt": "A"}

    result = transform_llr_mlm(example, tokenizer, genome, window_size)

    assert result["pos"] == window_size // 2
    assert isinstance(result["input_ids"], torch.Tensor)
    assert result["input_ids"].shape[0] == window_size
    assert result["input_ids"][result["pos"]].item() == tokenizer.mask_token_id
    expected_ref_id = tokenizer(example["ref"])["input_ids"][0]
    assert result["ref"] == expected_ref_id


def test_transform_llr_mlm_returns_ref_and_alt_tokens(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))
    window_size = 6
    example = {"chrom": "chr1", "pos": 5, "ref": "A", "alt": "G"}

    result = transform_llr_mlm(example, tokenizer, genome, window_size)

    expected_ref_id = tokenizer(example["ref"])["input_ids"][0]
    expected_alt_id = tokenizer(example["alt"])["input_ids"][0]

    assert result["ref"] == expected_ref_id
    assert result["alt"] == expected_alt_id
    assert result["ref"] != tokenizer.mask_token_id
    assert result["alt"] != tokenizer.mask_token_id
    assert result["input_ids"][result["pos"]].item() == tokenizer.mask_token_id


def test_transform_llr_clm_basic_functionality(tmp_path):
    """Test basic functionality of transform_llr_clm"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))
    window_size = 16
    example = {"chrom": "chr1", "pos": 6, "ref": "C", "alt": "A"}

    result = transform_llr_clm(example, tokenizer, genome, window_size)

    # Check return structure
    assert isinstance(result, dict)
    assert "input_ids" in result
    assert isinstance(result["input_ids"], torch.Tensor)

    # Check shape: should be [2, L] for ref and alt sequences
    assert result["input_ids"].shape[0] == 2
    assert result["input_ids"].shape[1] == window_size


def test_transform_llr_clm_creates_ref_and_alt_sequences(tmp_path):
    """Test that transform_llr_clm creates both ref and alt sequences correctly"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))
    window_size = 16
    example = {"chrom": "chr1", "pos": 6, "ref": "C", "alt": "G"}

    result = transform_llr_clm(example, tokenizer, genome, window_size)

    input_ids = result["input_ids"]

    # Check that we have 2 sequences
    assert input_ids.shape[0] == 2

    # The two sequences should differ at exactly one position
    diff_mask = input_ids[0] != input_ids[1]
    num_diffs = diff_mask.sum().item()

    # They should differ at exactly 1 position (the variant)
    assert num_diffs == 1

    # The position where they differ should be at or near window_size // 2
    diff_pos = diff_mask.nonzero()[0].item()

    # Verify the tokens at the different position correspond to ref and alt
    ref_token_id = tokenizer.encode(example["ref"])[0]
    alt_token_id = tokenizer.encode(example["alt"])[0]

    assert input_ids[0, diff_pos].item() == ref_token_id
    assert input_ids[1, diff_pos].item() == alt_token_id


def test_transform_llr_clm_tokenizes_both_sequences(tmp_path):
    """Test that both sequences are properly tokenized and stacked"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))
    window_size = 16
    example = {"chrom": "chr1", "pos": 6, "ref": "C", "alt": "T"}

    result = transform_llr_clm(example, tokenizer, genome, window_size)

    input_ids = result["input_ids"]

    # Check that we have exactly 2 sequences
    assert input_ids.shape[0] == 2

    # Check that both sequences have the same length
    assert input_ids[0].shape == input_ids[1].shape

    # Check that all token IDs are valid (non-negative)
    assert (input_ids >= 0).all()


def test_transform_llr_clm_first_tokens_match(tmp_path):
    """Test that first 8 tokens match between ref and alt as asserted in code"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))
    window_size = 16
    # Use position that's far enough from the start
    example = {"chrom": "chr1", "pos": 6, "ref": "C", "alt": "A"}

    result = transform_llr_clm(example, tokenizer, genome, window_size)

    input_ids = result["input_ids"]

    # The code asserts that first 8 tokens should match
    # This is because the variant is at position window_size//2 = 8
    # So the first 8 positions should definitely match
    assert (input_ids[0, :8] == input_ids[1, :8]).all()


def test_transform_llr_clm_different_window_sizes(tmp_path):
    """Test with various window sizes"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))

    # Use window sizes >= 16 to ensure variant is at position 8 or later
    # so that the first 8 tokens assertion in the code works
    for window_size in [16, 18, 20]:
        example = {"chrom": "chr1", "pos": 6, "ref": "C", "alt": "T"}

        result = transform_llr_clm(example, tokenizer, genome, window_size)

        # Check shape matches window size
        assert result["input_ids"].shape == (2, window_size)

        # Check first 8 tokens match as asserted in the actual code
        assert (result["input_ids"][0, :8] == result["input_ids"][1, :8]).all()


@pytest.mark.parametrize(
    "bos_id,eos_id,counts",
    [
        (None, None, (0, 0)),
        (_BOS_ID, None, (1, 0)),
        (None, _EOS_ID, (0, 1)),
        (_BOS_ID, _EOS_ID, (1, 1)),
    ],
)
def test_get_special_token_counts(bos_id, eos_id, counts):
    base = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    tokenizer = _SpecialTokensTokenizer(base, bos_id=bos_id, eos_id=eos_id)
    assert _get_special_token_counts(tokenizer) == counts


@pytest.mark.parametrize(
    "tokenizer_name,counts",
    [
        ("songlab/tokenizer-dna-mlm", (0, 0)),
        ("songlab/tokenizer-dna-clm", (0, 0)),
        ("bolinas-dna/tokenizer-char-bos", (1, 0)),
        ("bolinas-dna/tokenizer-char-bos-eos", (1, 1)),
    ],
)
def test_get_special_token_counts_real_tokenizers(tokenizer_name, counts):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    assert _get_special_token_counts(tokenizer) == counts


def test_transform_llr_clm_exp136_recipe(tmp_path):
    """Regression for issue #19: window_size=255 + bolinas BOS tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("bolinas-dna/tokenizer-char-bos")
    fasta_path = tmp_path / "g.fa"
    fasta_path.write_text(">chr1\n" + ("ACGT" * 100) + "\n")
    genome = Genome(fasta_path)
    example = {"chrom": "chr1", "pos": 200, "ref": "T", "alt": "G"}

    result = transform_llr_clm(example, tokenizer, genome, window_size=255)

    assert result["input_ids"].shape == (2, 256)
    diff_mask = result["input_ids"][0] != result["input_ids"][1]
    assert diff_mask.sum().item() == 1
    assert diff_mask.nonzero()[0].item() == 128


def test_transform_llr_clm_window_size_one(tmp_path):
    """Smallest window: just the variant base itself."""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))
    example = {"chrom": "chr1", "pos": 6, "ref": "C", "alt": "A"}

    result = transform_llr_clm(example, tokenizer, genome, window_size=1)

    assert result["input_ids"].shape == (2, 1)
    assert result["input_ids"][0, 0] != result["input_ids"][1, 0]


@pytest.mark.parametrize(
    "bos_id,eos_id",
    [(None, None), (_BOS_ID, None), (None, _EOS_ID), (_BOS_ID, _EOS_ID)],
)
@pytest.mark.parametrize("window_size", [15, 16])
def test_transform_llr_clm_handles_bos_eos(tmp_path, bos_id, eos_id, window_size):
    base = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    tokenizer = _SpecialTokensTokenizer(base, bos_id=bos_id, eos_id=eos_id)
    genome = Genome(_write_test_fasta(tmp_path))
    example = {"chrom": "chr1", "pos": 6, "ref": "C", "alt": "A"}

    result = transform_llr_clm(example, tokenizer, genome, window_size)

    has_bos = bos_id is not None
    has_eos = eos_id is not None
    expected_len = window_size + int(has_bos) + int(has_eos)
    assert result["input_ids"].shape == (2, expected_len)
    diff_mask = result["input_ids"][0] != result["input_ids"][1]
    assert diff_mask.sum().item() == 1
    assert diff_mask.nonzero()[0].item() == window_size // 2 + int(has_bos)


@pytest.mark.parametrize(
    "bos_id,eos_id",
    [(None, None), (_BOS_ID, None), (None, _EOS_ID), (_BOS_ID, _EOS_ID)],
)
@pytest.mark.parametrize("window_size", [5, 6])
def test_transform_llr_mlm_handles_bos_eos(tmp_path, bos_id, eos_id, window_size):
    base = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    tokenizer = _SpecialTokensTokenizer(base, bos_id=bos_id, eos_id=eos_id)
    genome = Genome(_write_test_fasta(tmp_path))
    example = {"chrom": "chr1", "pos": 5, "ref": "A", "alt": "G"}

    result = transform_llr_mlm(example, tokenizer, genome, window_size)

    has_bos = bos_id is not None
    has_eos = eos_id is not None
    expected_len = window_size + int(has_bos) + int(has_eos)
    assert result["input_ids"].shape[0] == expected_len
    assert result["pos"] == window_size // 2 + int(has_bos)
    assert result["input_ids"][result["pos"]].item() == base.mask_token_id
    assert result["ref"] == base.encode(example["ref"])[0]
    assert result["alt"] == base.encode(example["alt"])[0]
    if has_bos:
        assert result["ref"] != bos_id
        assert result["alt"] != bos_id


@pytest.mark.parametrize(
    "bos_id,eos_id",
    [(None, None), (_BOS_ID, None), (None, _EOS_ID), (_BOS_ID, _EOS_ID)],
)
def test_transform_reflogprob_mlm_handles_bos_eos(bos_id, eos_id):
    base = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    tokenizer = _SpecialTokensTokenizer(base, bos_id=bos_id, eos_id=eos_id)
    pos = 3
    example = {"seq": "ATCGATCG", "pos": pos}

    result = transform_reflogprob_mlm(example, tokenizer)

    has_bos = bos_id is not None
    has_eos = eos_id is not None
    expected_len = len(example["seq"]) + int(has_bos) + int(has_eos)
    assert result["input_ids"].shape[0] == expected_len
    assert result["pos"] == pos + int(has_bos)
    assert result["input_ids"][result["pos"]].item() == base.mask_token_id
    assert result["ref"] == base.encode(example["seq"][pos])[0]
    if has_bos:
        assert result["ref"] != bos_id


@pytest.mark.parametrize(
    "bos_id,eos_id",
    [(None, None), (_BOS_ID, None), (None, _EOS_ID), (_BOS_ID, _EOS_ID)],
)
def test_transform_reflogprob_clm_handles_bos_eos(bos_id, eos_id):
    base = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    tokenizer = _SpecialTokensTokenizer(base, bos_id=bos_id, eos_id=eos_id)
    pos = 3
    example = {"seq": "ATCGATCG", "pos": pos}

    result = transform_reflogprob_clm(example, tokenizer)

    has_bos = bos_id is not None
    has_eos = eos_id is not None
    expected_len = len(example["seq"]) + int(has_bos) + int(has_eos)
    input_ids = result["input_ids"]
    assert input_ids.shape == (4, expected_len)

    tokenized_pos = pos + int(has_bos)
    nucleotides = ["A", "C", "G", "T"]
    for i, nuc in enumerate(nucleotides):
        assert input_ids[i, tokenized_pos].item() == base.encode(nuc)[0]
        for j in range(expected_len):
            if j == tokenized_pos:
                continue
            assert input_ids[i, j].item() == input_ids[0, j].item()
    assert nucleotides[result["ref"]] == example["seq"][pos]


def _write_long_test_fasta(tmp_path):
    """400-bp FASTA — long enough to extract any reasonable window without N-padding."""
    fasta = ">chr1\n" + ("ACGT" * 100) + "\n"
    path = tmp_path / "long.fa"
    path.write_text(fasta)
    return path


def test_complement_base():
    assert _complement_base("A") == "T"
    assert _complement_base("C") == "G"
    assert _complement_base("G") == "C"
    assert _complement_base("T") == "A"
    # Non-ACGT round-trips unchanged
    assert _complement_base("N") == "N"
    assert _complement_base("M") == "M"
    assert _complement_base("R") == "R"


@pytest.mark.parametrize(
    "bos_id,eos_id",
    [(None, None), (_BOS_ID, None), (None, _EOS_ID), (_BOS_ID, _EOS_ID)],
)
@pytest.mark.parametrize("window_size", [5, 6, 15, 16])
def test_transform_llr_clm_strand_rc_matrix(tmp_path, bos_id, eos_id, window_size):
    """Full {even/odd window_size} × {BOS y/n} × {EOS y/n} coverage for the
    RC-strand path of transform_llr_clm: shape, variant index, complemented
    tokens, and that the ref sequence is the revcomp of the FWD ref sequence."""
    base = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    tokenizer = _SpecialTokensTokenizer(base, bos_id=bos_id, eos_id=eos_id)
    genome = Genome(_write_long_test_fasta(tmp_path))
    example = {"chrom": "chr1", "pos": 200, "ref": "T", "alt": "G"}

    fwd = transform_llr_clm(example, tokenizer, genome, window_size, strand="+")
    rc = transform_llr_clm(example, tokenizer, genome, window_size, strand="-")

    has_bos = bos_id is not None
    has_eos = eos_id is not None
    expected_len = window_size + int(has_bos) + int(has_eos)

    assert rc["input_ids"].shape == (2, expected_len)

    diff_mask = rc["input_ids"][0] != rc["input_ids"][1]
    assert diff_mask.sum().item() == 1
    rc_dna_pos = (
        window_size - 1 - window_size // 2
    )  # asymmetric when window_size is even
    assert diff_mask.nonzero()[0].item() == rc_dna_pos + int(has_bos)

    nuc_ids = {n: base.encode(n)[0] for n in "ACGT"}
    rc_token_idx = rc_dna_pos + int(has_bos)
    assert rc["input_ids"][0, rc_token_idx].item() == nuc_ids["A"]  # complement("T")
    assert rc["input_ids"][1, rc_token_idx].item() == nuc_ids["C"]  # complement("G")

    # The body of rc[0] equals revcomp of the body of fwd[0]
    body_slice = slice(int(has_bos), expected_len - int(has_eos))
    id_to_nuc = {v: k for k, v in nuc_ids.items()}
    fwd_body_dna = "".join(id_to_nuc[t.item()] for t in fwd["input_ids"][0, body_slice])
    rc_body_dna = "".join(id_to_nuc[t.item()] for t in rc["input_ids"][0, body_slice])
    assert str(Seq(fwd_body_dna).reverse_complement()) == rc_body_dna


@pytest.mark.parametrize(
    "bos_id,eos_id",
    [(None, None), (_BOS_ID, None), (None, _EOS_ID), (_BOS_ID, _EOS_ID)],
)
@pytest.mark.parametrize("window_size", [5, 6, 15, 16])
def test_transform_llr_mlm_strand_rc_matrix(tmp_path, bos_id, eos_id, window_size):
    """Full matrix for the RC-strand path of transform_llr_mlm."""
    base = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    tokenizer = _SpecialTokensTokenizer(base, bos_id=bos_id, eos_id=eos_id)
    genome = Genome(_write_long_test_fasta(tmp_path))
    example = {"chrom": "chr1", "pos": 200, "ref": "T", "alt": "G"}

    fwd = transform_llr_mlm(example, tokenizer, genome, window_size, strand="+")
    rc = transform_llr_mlm(example, tokenizer, genome, window_size, strand="-")

    has_bos = bos_id is not None
    has_eos = eos_id is not None
    expected_len = window_size + int(has_bos) + int(has_eos)
    rc_dna_pos = window_size - 1 - window_size // 2

    assert rc["input_ids"].shape[0] == expected_len
    assert rc["pos"] == rc_dna_pos + int(has_bos)
    assert rc["input_ids"][rc["pos"]].item() == base.mask_token_id
    assert rc["ref"] == base.encode("A")[0]  # complement("T")
    assert rc["alt"] == base.encode("C")[0]  # complement("G")

    # Body (excluding mask + BOS/EOS) on RC equals revcomp of FWD body
    nuc_ids = {n: base.encode(n)[0] for n in "ACGT"}
    id_to_nuc = {v: k for k, v in nuc_ids.items()}
    fwd_dna = []
    rc_dna = []
    for i in range(int(has_bos), expected_len - int(has_eos)):
        fwd_t = fwd["input_ids"][i].item()
        rc_t = rc["input_ids"][i].item()
        if fwd_t == base.mask_token_id:
            fwd_dna.append("N")
        else:
            fwd_dna.append(id_to_nuc[fwd_t])
        if rc_t == base.mask_token_id:
            rc_dna.append("N")
        else:
            rc_dna.append(id_to_nuc[rc_t])
    assert str(Seq("".join(fwd_dna)).reverse_complement()) == "".join(rc_dna)


@pytest.mark.parametrize("window_size", [1, 5, 15, 255])
def test_transform_llr_clm_odd_window_strand_symmetric(tmp_path, window_size):
    """For odd window_size, the variant DNA index is identical on both strands."""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_long_test_fasta(tmp_path))
    example = {"chrom": "chr1", "pos": 200, "ref": "T", "alt": "G"}

    fwd = transform_llr_clm(example, tokenizer, genome, window_size, strand="+")
    rc = transform_llr_clm(example, tokenizer, genome, window_size, strand="-")

    fwd_diff = (fwd["input_ids"][0] != fwd["input_ids"][1]).nonzero()[0].item()
    rc_diff = (rc["input_ids"][0] != rc["input_ids"][1]).nonzero()[0].item()
    assert fwd_diff == rc_diff == window_size // 2


def test_transform_llr_clm_strand_rc_n_padding(tmp_path):
    """Variant near chrom start — FWD window N-pads on the left, RC on the right."""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))  # 10-bp chrom: ACGTACGTAC
    window_size = 8
    example = {"chrom": "chr1", "pos": 2, "ref": "C", "alt": "T"}

    fwd = transform_llr_clm(example, tokenizer, genome, window_size, strand="+")
    rc = transform_llr_clm(example, tokenizer, genome, window_size, strand="-")

    assert fwd["input_ids"].shape == (2, window_size)
    assert rc["input_ids"].shape == (2, window_size)

    # On both strands, ref and alt differ at exactly one position
    fwd_diff = (fwd["input_ids"][0] != fwd["input_ids"][1]).nonzero()[0].item()
    rc_diff = (rc["input_ids"][0] != rc["input_ids"][1]).nonzero()[0].item()
    assert fwd_diff == window_size // 2
    assert rc_diff == window_size - 1 - window_size // 2

    # The N-padded body on FWD reverse-complements to the N-padded body on RC
    # (N maps to N under reverse_complement)
    base = tokenizer
    nuc_ids = {n: base.encode(n)[0] for n in "ACGTN"}
    id_to_nuc = {v: k for k, v in nuc_ids.items()}
    fwd_dna = "".join(id_to_nuc.get(t.item(), "?") for t in fwd["input_ids"][0])
    rc_dna = "".join(id_to_nuc.get(t.item(), "?") for t in rc["input_ids"][0])
    # Sanity: at least one N appears on each strand (chrom boundary)
    assert "N" in fwd_dna
    assert "N" in rc_dna


@pytest.mark.parametrize(
    "bos_id,eos_id",
    [(None, None), (_BOS_ID, None), (None, _EOS_ID), (_BOS_ID, _EOS_ID)],
)
@pytest.mark.parametrize("seq", ["ATCGATCG", "ACGTAC"])
def test_transform_reflogprob_mlm_strand_rc_matrix(bos_id, eos_id, seq):
    """Full matrix for the RC-strand path of transform_reflogprob_mlm."""
    base = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    tokenizer = _SpecialTokensTokenizer(base, bos_id=bos_id, eos_id=eos_id)
    pos = 2
    example = {"seq": seq, "pos": pos}

    rc = transform_reflogprob_mlm(example, tokenizer, strand="-")

    has_bos = bos_id is not None
    has_eos = eos_id is not None
    expected_len = len(seq) + int(has_bos) + int(has_eos)
    rc_pos_dna = len(seq) - 1 - pos

    assert rc["input_ids"].shape[0] == expected_len
    assert rc["pos"] == rc_pos_dna + int(has_bos)
    assert rc["input_ids"][rc["pos"]].item() == base.mask_token_id
    assert rc["ref"] == base.encode(_complement_base(seq[pos]))[0]


@pytest.mark.parametrize(
    "bos_id,eos_id",
    [(None, None), (_BOS_ID, None), (None, _EOS_ID), (_BOS_ID, _EOS_ID)],
)
@pytest.mark.parametrize("seq", ["ATCGATCG", "ACGTAC"])
def test_transform_reflogprob_clm_strand_rc_matrix(bos_id, eos_id, seq):
    """Full matrix for the RC-strand path of transform_reflogprob_clm."""
    base = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    tokenizer = _SpecialTokensTokenizer(base, bos_id=bos_id, eos_id=eos_id)
    pos = 2
    example = {"seq": seq, "pos": pos}

    rc = transform_reflogprob_clm(example, tokenizer, strand="-")

    has_bos = bos_id is not None
    has_eos = eos_id is not None
    expected_len = len(seq) + int(has_bos) + int(has_eos)
    rc_pos_dna = len(seq) - 1 - pos
    rc_token_idx = rc_pos_dna + int(has_bos)

    assert rc["input_ids"].shape == (4, expected_len)
    # Each of the 4 sequences has the corresponding nucleotide at the RC index
    for i, nuc in enumerate(NUCLEOTIDES):
        assert rc["input_ids"][i, rc_token_idx].item() == base.encode(nuc)[0]
    # ref is the index of the complement of the original ref base
    assert NUCLEOTIDES[rc["ref"]] == _complement_base(seq[pos])


def test_transform_reflogprob_clm_basic_functionality():
    """Test basic functionality of transform_reflogprob_clm"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    pos = 2
    example = {"seq": "ATCGATCG", "pos": pos}

    result = transform_reflogprob_clm(example, tokenizer)

    # Check return structure
    assert isinstance(result, dict)
    assert "input_ids" in result
    assert "ref" in result

    # Check types
    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["ref"], int)

    # Check shape: should be [4, L] for four nucleotide variants
    assert result["input_ids"].shape[0] == 4
    assert result["input_ids"].shape[1] == len(example["seq"])


def test_transform_reflogprob_clm_creates_four_sequences():
    """Test that transform_reflogprob_clm creates 4 sequences (one per nucleotide)"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    example = {"seq": "AAACCCGGG", "pos": 4}

    result = transform_reflogprob_clm(example, tokenizer)

    input_ids = result["input_ids"]

    # Should have exactly 4 sequences (A, C, G, T)
    assert input_ids.shape[0] == 4

    # All sequences should have the same length
    for i in range(1, 4):
        assert input_ids[i].shape == input_ids[0].shape


def test_transform_reflogprob_clm_correct_nucleotides_at_position():
    """Test that each sequence has the correct nucleotide at the specified position"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    pos = 3
    example = {"seq": "ATCGATCG", "pos": pos}

    result = transform_reflogprob_clm(example, tokenizer)

    input_ids = result["input_ids"]

    # Each of the 4 sequences should have a different nucleotide at pos
    nucleotides = ["A", "C", "G", "T"]
    for i, nuc in enumerate(nucleotides):
        # Get the token ID for this nucleotide
        expected_token_id = tokenizer.encode(nuc)[0]
        actual_token_id = input_ids[i, pos].item()
        assert actual_token_id == expected_token_id


def test_transform_reflogprob_clm_ref_index_mapping():
    """Test that ref index correctly maps to nucleotide (0=A, 1=C, 2=G, 3=T)"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")

    nucleotides = ["A", "C", "G", "T"]

    for expected_idx, nuc in enumerate(nucleotides):
        pos = 2
        example = {"seq": f"NN{nuc}NNNNN", "pos": pos}

        result = transform_reflogprob_clm(example, tokenizer)

        # The ref should be the index corresponding to the nucleotide
        assert result["ref"] == expected_idx


def test_transform_reflogprob_clm_different_nucleotides():
    """Test with each nucleotide (A, C, G, T) as the reference"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")

    test_cases = [
        ("AAATTTCCC", 3, "T"),  # T at position 3
        ("GCGCGCGC", 2, "G"),  # G at position 2
        ("TTCCGGAA", 1, "T"),  # T at position 1
        ("ACGTACGT", 4, "A"),  # A at position 4
    ]

    for seq, pos, expected_nuc in test_cases:
        example = {"seq": seq, "pos": pos}

        result = transform_reflogprob_clm(example, tokenizer)

        # Check that ref maps to the correct nucleotide
        nucleotides = ["A", "C", "G", "T"]
        assert nucleotides[result["ref"]] == expected_nuc


def test_transform_reflogprob_clm_different_positions():
    """Test with different positions in the sequence"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    seq = "ACGTACGT"

    for pos in range(len(seq)):
        example = {"seq": seq, "pos": pos}

        result = transform_reflogprob_clm(example, tokenizer)

        # Should always create 4 sequences
        assert result["input_ids"].shape[0] == 4

        # ref should be valid index (0-3)
        assert 0 <= result["ref"] < 4

        # The nucleotide at pos should match what's in the sequence
        nucleotides = ["A", "C", "G", "T"]
        assert nucleotides[result["ref"]] == seq[pos]


def test_genome_returns_subsequence(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path)

    seq = genome("chr1", start=2, end=7)

    assert seq == "GTACG"


def test_genome_reverse_complement(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path)

    seq = genome("chr2", start=1, end=6, strand="-")

    assert seq == "GGGCC"


def test_genome_left_padding(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path)

    seq = genome("chr1", start=-2, end=3)

    assert seq == "NNACG"


def test_genome_right_padding(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path)

    seq = genome("chr1", start=7, end=12)

    assert seq == "TACNN"


def test_genome_padding_both_sides(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path)

    seq = genome("chr1", start=-3, end=12)

    assert seq == "NNNACGTACGTACNN"


def test_genome_requires_known_chromosome(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path)

    with pytest.raises(ValueError, match="chromosome chr3 not found"):
        genome("chr3", start=0, end=1)


def test_genome_validates_span(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path)

    assert genome("chr1", start=5, end=5) == ""
    with pytest.raises(ValueError, match="start 6 must be less than or equal to end 4"):
        genome("chr1", start=6, end=4)

    with pytest.raises(ValueError, match="end -1 must be non-negative"):
        genome("chr1", start=-5, end=-1)

    with pytest.raises(ValueError, match="start 11 is out of range"):
        genome("chr1", start=11, end=11)

    with pytest.raises(ValueError, match="start 10 is out of range"):
        genome("chr1", start=10, end=12)


def test_genome_respects_subset(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path, subset_chroms={"chr2"})

    assert genome("chr2", start=0, end=4) == "GGGC"

    with pytest.raises(ValueError, match="chromosome chr1 not found"):
        genome("chr1", start=0, end=4)


# GenomicSet tests
def test_genomic_set_initialization_non_overlapping():
    """Test GenomicSet initialization with non-overlapping intervals.

    Input: chr1:0-50, chr1:100-150, chr2:0-200
    Output: chr1:0-50, chr1:100-150, chr2:0-200 (sorted, non-overlapping maintained)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [0, 100, 0],
            "end": [50, 150, 200],
        }
    )
    gs = GenomicSet(data)

    # Should maintain the same intervals since they're non-overlapping, sorted
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr2"],
                "start": [0, 100, 0],
                "end": [50, 150, 200],
            }
        )
    )
    assert gs == expected


def test_genomic_set_merges_overlapping_intervals():
    """Test that GenomicSet merges overlapping intervals on initialization.

    Input: chr1:0-50, chr1:40-60, chr1:90-150
    Output: chr1:0-60, chr1:90-150 (overlapping intervals merged)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 40, 90],
            "end": [50, 60, 150],
        }
    )
    gs = GenomicSet(data)

    # Overlapping intervals [0,50] and [40,60] should merge to [0,60]
    # [90,150] should remain separate
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [0, 90],
                "end": [60, 150],
            }
        )
    )
    assert gs == expected


def test_genomic_set_merges_adjacent_intervals():
    """Test that GenomicSet merges adjacent intervals on initialization.

    Input: chr1:0-100, chr1:100-200 (adjacent: end of first equals start of second)
    Output: chr1:0-200 (adjacent intervals merged into single interval)

    Note: Adjacent intervals (where end of one equals start of the next) are merged
    because they form a continuous region.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [100, 200],
        }
    )
    gs = GenomicSet(data)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [0],
                "end": [200],
            }
        )
    )
    assert gs == expected


def test_genomic_set_union():
    """Test union operation (|) between two GenomicSets.

    Input set 1: chr1:0-50, chr1:100-150 (two separate intervals with gap 50-100)
    Input set 2: chr1:40-80, chr2:0-200
    Output: chr1:0-80, chr1:100-150, chr2:0-200 (overlaps merged)

    Note: chr1:0-50 and chr1:40-80 merge to chr1:0-80. chr1:100-150 remains separate.
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [50, 150],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [40, 0],
            "end": [80, 200],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 | gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr2"],
                "start": [0, 100, 0],
                "end": [80, 150, 200],
            }
        )
    )
    assert result == expected


def test_genomic_set_intersection():
    """Test intersection operation (&) between two GenomicSets.

    Input set 1: chr1:0-50, chr1:100-150
    Input set 2: chr1:25-75, chr1:120-200
    Output: chr1:25-50, chr1:120-150 (only overlapping regions)

    Note: chr1:0-50 and chr1:100-150 are separate intervals (not merged due to gap).
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [50, 150],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [25, 120],
            "end": [75, 200],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 & gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [25, 120],
                "end": [50, 150],
            }
        )
    )
    assert result == expected


def test_genomic_set_subtraction():
    """Test subtraction operation (-) between two GenomicSets.

    Input set 1: chr1:0-200 (single merged interval)
    Input set 2: chr1:40-60
    Output: chr1:0-40, chr1:60-200 (subtracted region removed, creating two intervals)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [200],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [40],
            "end": [60],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 - gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [0, 60],
                "end": [40, 200],
            }
        )
    )
    assert result == expected


def test_genomic_set_union_no_overlap():
    """Test union with completely non-overlapping intervals.

    Input set 1: chr1:0-50
    Input set 2: chr2:100-200
    Output: chr1:0-50, chr2:100-200 (all intervals included, no merging)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [50],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr2"],
            "start": [100],
            "end": [200],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 | gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr2"],
                "start": [0, 100],
                "end": [50, 200],
            }
        )
    )
    assert result == expected


def test_genomic_set_intersection_no_overlap():
    """Test intersection with no overlapping intervals.

    Input set 1: chr1:0-50
    Input set 2: chr1:100-200
    Output: (empty set - no overlap)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [50],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 & gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    assert result == expected


def test_genomic_set_subtraction_no_overlap():
    """Test subtraction when there's no overlap.

    Input set 1: chr1:0-50
    Input set 2: chr1:100-200
    Output: chr1:0-50 (unchanged, no overlap to subtract)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [50],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    result = gs1 - gs2

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [0],
                "end": [50],
            }
        )
    )
    assert result == expected


def test_genomic_set_different_chromosomes():
    """Test operations with intervals on different chromosomes.

    Input set 1: chr1:0-100, chr2:0-100
    Input set 2: chr1:50-150, chr3:50-150

    Union output: chr1:0-150, chr2:0-100, chr3:50-150 (all chromosomes)
    Intersection output: chr1:50-100 (only chr1 overlaps)
    Subtraction output: chr1:0-50, chr2:0-100 (chr1 overlap removed, chr2 unchanged)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 0],
            "end": [100, 100],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr3"],
            "start": [50, 50],
            "end": [150, 150],
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)

    union_result = gs1 | gs2
    expected_union = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr2", "chr3"],
                "start": [0, 0, 50],
                "end": [150, 100, 150],
            }
        )
    )
    assert union_result == expected_union

    intersection_result = gs1 & gs2
    expected_intersection = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [50],
                "end": [100],
            }
        )
    )
    assert intersection_result == expected_intersection

    subtraction_result = gs1 - gs2
    expected_subtraction = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1", "chr2"],
                "start": [0, 0],
                "end": [50, 100],
            }
        )
    )
    assert subtraction_result == expected_subtraction


def test_genomic_set_to_pandas():
    """Test conversion to pandas DataFrame.

    Input: chr1:0-50, chr2:100-200
    Output: Same intervals as pandas DataFrame
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    gs = GenomicSet(data)
    df = gs.to_pandas()

    expected = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    pd.testing.assert_frame_equal(df, expected)
    # Also verify equality works
    assert gs == GenomicSet(expected)


def test_genomic_set_empty():
    """Test GenomicSet with empty DataFrame.

    Input: (empty set)
    Output: (empty set)
    """
    data = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    gs = GenomicSet(data)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    assert gs == expected


def test_genomic_set_repr():
    """Test string representation of GenomicSet.

    Input: chr1:0-100
    Output: String representation containing "GenomicSet"
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    repr_str = repr(gs)

    assert "GenomicSet" in repr_str


def test_genomic_set_adjacent_intervals():
    """Test that adjacent intervals (touching but not overlapping) are handled correctly.

    Input: chr1:0-50, chr1:50-100 (adjacent/touching intervals)
    Output: chr1:0-100 (adjacent intervals merged)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 50],
            "end": [50, 100],
        }
    )
    gs = GenomicSet(data)

    # Adjacent intervals should be merged by bioframe
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [0],
                "end": [100],
            }
        )
    )
    assert gs == expected


def test_genomic_set_single_interval():
    """Test GenomicSet with a single interval.

    Input: chr1:100-200
    Output: chr1:100-200 (unchanged)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
        }
    )
    gs = GenomicSet(data)

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [100],
                "end": [200],
            }
        )
    )
    assert gs == expected


def test_genomic_set_self_union():
    """Test union of a GenomicSet with itself.

    Input: chr1:0-100 | chr1:0-100
    Output: chr1:0-100 (union with itself equals original)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    result = gs | gs

    assert result == gs


def test_genomic_set_self_intersection():
    """Test intersection of a GenomicSet with itself.

    Input: chr1:0-100 & chr1:0-100
    Output: chr1:0-100 (intersection with itself equals original)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    result = gs & gs

    assert result == gs


def test_genomic_set_self_subtraction():
    """Test subtraction of a GenomicSet from itself.

    Input: chr1:0-100 - chr1:0-100
    Output: (empty set - subtracting itself removes everything)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    result = gs - gs

    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    assert result == expected


def test_genomic_set_equality():
    """Test equality comparison between GenomicSets.

    Input set 1: chr1:0-50, chr2:100-200
    Input set 2: chr1:0-50, chr2:100-200 (same as set 1)
    Input set 3: chr1:0-51, chr2:100-200 (different end)

    Tests:
    - Set 1 == Set 2: True (identical intervals)
    - Set 1 != Set 3: True (different intervals)
    - Set 1 != non-GenomicSet: True (different type)
    """
    data1 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    data2 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50, 200],
        }
    )
    data3 = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [51, 200],  # Different end value
        }
    )

    gs1 = GenomicSet(data1)
    gs2 = GenomicSet(data2)
    gs3 = GenomicSet(data3)

    assert gs1 == gs2
    assert gs2 == gs1
    assert gs1 != gs3
    assert gs3 != gs1
    assert gs1 != "not a GenomicSet"
    assert gs1 != None  # noqa: E711
    assert gs1 != 123


def test_genomic_set_equality_empty():
    """Test equality with empty GenomicSets.

    Input: (empty set) == (empty set)
    Output: True (empty sets are equal)
    """
    empty1 = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )
    empty2 = GenomicSet(
        pd.DataFrame(
            {
                "chrom": [],
                "start": [],
                "end": [],
            }
        )
    )

    assert empty1 == empty2
    assert empty2 == empty1


# Invalid input tests
def test_genomic_set_zero_length_interval():
    """Test that GenomicSet accepts zero-length intervals (start == end).

    Input: chr1:50-50 (start equals end, zero-length interval)
    Output: chr1:50-50 (bioframe allows zero-length intervals)

    Note: While conceptually zero-length intervals may seem invalid,
    bioframe's validation only checks that start <= end (not start < end).
    Zero-length intervals are technically valid in bedFrame format.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [50],
            "end": [50],  # start == end, zero-length interval
        }
    )

    # This should work - bioframe allows start == end
    gs = GenomicSet(data)
    result_df = gs.to_pandas()
    assert len(result_df) == 1
    assert result_df.iloc[0]["chrom"] == "chr1"
    assert result_df.iloc[0]["start"] == 50
    assert result_df.iloc[0]["end"] == 50


def test_genomic_set_invalid_start_greater_than_end():
    """Test that GenomicSet rejects intervals where start > end.

    Input: chr1:100-50 (start > end, invalid)
    Expected: ValueError (invalid bedFrame - starts exceed ends)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [100],
            "end": [50],  # start > end, invalid
        }
    )

    with pytest.raises(ValueError, match="starts exceed ends"):
        GenomicSet(data)


def test_genomic_set_negative_start():
    """Test that GenomicSet accepts negative start (bioframe allows it).

    Input: chr1:-10-50 (negative start)
    Output: chr1:-10-50 (bioframe doesn't validate non-negative starts)

    Note: While bioframe validates start < end, it doesn't enforce start >= 0.
    Negative starts are technically valid in bioframe's bedFrame format,
    even though the constraint "0 <= start < end" may be conceptually desired.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [-10],  # negative start, but valid for bioframe
            "end": [50],
        }
    )

    # This should work - bioframe doesn't reject negative starts
    gs = GenomicSet(data)
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [-10],
                "end": [50],
            }
        )
    )
    assert gs == expected


def test_genomic_set_invalid_chrom_dtype():
    """Test that GenomicSet rejects invalid chrom dtypes.

    Input: chrom column with int dtype (should be object/string/categorical)
    Expected: TypeError (invalid bedFrame - invalid column dtypes)
    """
    data = pd.DataFrame(
        {
            "chrom": [1, 2],  # int dtype instead of string
            "start": [0, 100],
            "end": [50, 200],
        }
    )

    with pytest.raises(TypeError, match="Invalid bedFrame"):
        GenomicSet(data)


def test_genomic_set_invalid_start_dtype():
    """Test that GenomicSet rejects invalid start dtype.

    Input: start column with float dtype (should be int/Int64Dtype)
    Expected: TypeError (invalid bedFrame - invalid column dtypes)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0.5, 100.0],  # float dtype instead of int
            "end": [50, 200],
        }
    )

    with pytest.raises(TypeError, match="Invalid bedFrame"):
        GenomicSet(data)


def test_genomic_set_invalid_end_dtype():
    """Test that GenomicSet rejects invalid end dtype.

    Input: end column with float dtype (should be int/Int64Dtype)
    Expected: TypeError (invalid bedFrame - invalid column dtypes)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [0, 100],
            "end": [50.5, 200.0],  # float dtype instead of int
        }
    )

    with pytest.raises(TypeError, match="Invalid bedFrame"):
        GenomicSet(data)


def test_genomic_set_invalid_string_start():
    """Test that GenomicSet rejects string values in start column.

    Input: start column with string values (should be int/Int64Dtype)
    Expected: TypeError (invalid bedFrame - invalid column dtypes)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": ["0"],  # string instead of int
            "end": [50],
        }
    )

    with pytest.raises(TypeError, match="Invalid bedFrame"):
        GenomicSet(data)


def test_genomic_set_valid_categorical_chrom():
    """Test that GenomicSet accepts categorical chrom dtype (valid).

    Input: chr1:0-50 with categorical chrom
    Output: chr1:0-50 (categorical is valid dtype for chrom)

    Note: After processing, categorical may be converted to object dtype,
    but the content should be equivalent.
    """
    data = pd.DataFrame(
        {
            "chrom": pd.Categorical(["chr1"]),
            "start": [0],
            "end": [50],
        }
    )

    gs = GenomicSet(data)
    # After merge and processing, check that we have the same intervals
    result_df = gs.to_pandas()
    assert len(result_df) == 1
    assert result_df.iloc[0]["chrom"] == "chr1"
    assert result_df.iloc[0]["start"] == 0
    assert result_df.iloc[0]["end"] == 50


# expand_min_size and add_random_shift tests
def test_genomic_set_expand_min_size_smaller_intervals():
    """Test expand_min_size with intervals smaller than min_size.

    Input: chr1:10-40 (size=30), min_size=50
    Output: chr1:0-50 (expanded equally on both sides to reach min_size=50)

    Note: Interval is expanded equally on both sides to reach min_size.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [10],
            "end": [40],  # size = 30
        }
    )
    gs = GenomicSet(data)
    result = gs.expand_min_size(min_size=50)

    # Should be expanded: pad = ceil((50-30)/2) = ceil(10) = 10
    # So start = 10-10 = 0, end = 40+10 = 50
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [0],
                "end": [50],
            }
        )
    )
    assert result == expected


def test_genomic_set_expand_min_size_larger_intervals():
    """Test expand_min_size with intervals already larger than min_size.

    Input: chr1:0-100 (size=100), min_size=50
    Output: chr1:0-100 (unchanged, already larger than min_size)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],  # size = 100, already larger than min_size
        }
    )
    gs = GenomicSet(data)
    result = gs.expand_min_size(min_size=50)

    # Should be unchanged (pad = max(ceil((50-100)/2), 0) = max(ceil(-25), 0) = 0)
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [0],
                "end": [100],
            }
        )
    )
    assert result == expected


def test_genomic_set_expand_min_size_causes_overlaps():
    """Test expand_min_size causing overlaps that get merged.

    Input: chr1:10-30, chr1:35-50 (separate, gap 30-35), min_size=30
    Output: chr1:5-58 (expanded intervals overlap and merge)

    Note: After expansion, intervals overlap and should be merged.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [10, 35],
            "end": [30, 50],  # sizes: 20, 15; gap between them
        }
    )
    gs = GenomicSet(data)
    result = gs.expand_min_size(min_size=30)

    # First: size=20, pad = ceil((30-20)/2) = 5, becomes 5-35
    # Second: size=15, pad = ceil((30-15)/2) = ceil(7.5) = 8, becomes 27-58
    # Now they overlap (35 > 27), so should merge to chr1:5-58
    expected = GenomicSet(
        pd.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [5],
                "end": [58],
            }
        )
    )
    assert result == expected


def test_genomic_set_add_random_shift_different_seeds():
    """Test add_random_shift with different seeds produces different results.

    Input: chr1:50-100, seed=42 vs seed=123
    Output: Different shifted positions for each seed
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [50],
            "end": [100],
        }
    )
    gs = GenomicSet(data)

    result1 = gs.add_random_shift(max_shift=10, seed=42)
    result2 = gs.add_random_shift(max_shift=10, seed=123)

    # Results should be different (different random shifts)
    assert result1 != result2

    # Both should have the same interval size (shift preserves size)
    df1 = result1.to_pandas()
    df2 = result2.to_pandas()
    assert len(df1) == 1
    assert len(df2) == 1
    assert (df1.iloc[0]["end"] - df1.iloc[0]["start"]) == (
        df2.iloc[0]["end"] - df2.iloc[0]["start"]
    )
    assert (df1.iloc[0]["end"] - df1.iloc[0]["start"]) == 50  # Original size preserved


def test_genomic_set_add_random_shift_same_seed():
    """Test add_random_shift with same seed produces same results.

    Input: chr1:50-100, seed=42 (twice)
    Output: Same shifted positions for both calls
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [50],
            "end": [100],
        }
    )
    gs = GenomicSet(data)

    result1 = gs.add_random_shift(max_shift=10, seed=42)
    result2 = gs.add_random_shift(max_shift=10, seed=42)

    # Results should be identical (same seed)
    assert result1 == result2


def test_genomic_set_add_random_shift_causes_overlaps():
    """Test add_random_shift causing overlaps that get merged.

    Input: chr1:50-60, chr1:61-70 (adjacent, separate), max_shift=5, seed=42
    Output: Overlapping intervals after shift get merged

    Note: Random shifts may cause intervals to overlap, which should be merged.
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [50, 61],
            "end": [60, 70],  # Adjacent intervals
        }
    )
    gs = GenomicSet(data)
    result = gs.add_random_shift(max_shift=5, seed=42)

    # With max_shift=5, intervals could shift and overlap
    # The result should be a valid GenomicSet (merged if overlapping)
    result_df = result.to_pandas()
    assert len(result_df) >= 1  # May be 1 or 2 depending on shifts
    assert all(result_df["chrom"] == "chr1")


def test_genomic_set_add_random_shift_negative_start():
    """Test add_random_shift with negative start values (should be allowed).

    Input: chr1:5-15, max_shift=10, seed that causes negative shift
    Output: chr1 with potentially negative start (allowed)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [5],
            "end": [15],
        }
    )
    gs = GenomicSet(data)
    result = gs.add_random_shift(max_shift=10, seed=999)

    # Should work even if start becomes negative
    result_df = result.to_pandas()
    assert len(result_df) == 1
    assert result_df.iloc[0]["chrom"] == "chr1"
    # Size should be preserved
    assert (result_df.iloc[0]["end"] - result_df.iloc[0]["start"]) == 10


def test_genomic_set_expand_min_size_returns_new_instance():
    """Test that expand_min_size returns a new GenomicSet instance.

    Input: chr1:10-40, expand_min_size(50)
    Output: New GenomicSet, original unchanged
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [10],
            "end": [40],
        }
    )
    gs = GenomicSet(data)
    original_df = gs.to_pandas()

    result = gs.expand_min_size(min_size=50)

    # Original should be unchanged
    assert gs.to_pandas().equals(original_df)
    # Result should be different
    assert result != gs
    # Result should be a new GenomicSet
    assert isinstance(result, GenomicSet)


def test_genomic_set_add_random_shift_returns_new_instance():
    """Test that add_random_shift returns a new GenomicSet instance.

    Input: chr1:50-100, add_random_shift(10, 42)
    Output: New GenomicSet, original unchanged
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [50],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    original_df = gs.to_pandas()

    result = gs.add_random_shift(max_shift=10, seed=42)

    # Original should be unchanged
    assert gs.to_pandas().equals(original_df)
    # Result should be different (unless shift happened to be 0)
    # Result should be a new GenomicSet
    assert isinstance(result, GenomicSet)


# n_intervals and total_size tests
def test_genomic_set_n_intervals_empty():
    """Test n_intervals with empty set.

    Input: (empty set)
    Output: 0
    """
    data = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    gs = GenomicSet(data)
    assert gs.n_intervals() == 0


def test_genomic_set_n_intervals_single():
    """Test n_intervals with single interval.

    Input: chr1:0-100
    Output: 1
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    assert gs.n_intervals() == 1


def test_genomic_set_n_intervals_multiple():
    """Test n_intervals with multiple intervals.

    Input: chr1:0-50, chr1:100-150, chr2:0-200
    Output: 3
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [0, 100, 0],
            "end": [50, 150, 200],
        }
    )
    gs = GenomicSet(data)
    assert gs.n_intervals() == 3


def test_genomic_set_n_intervals_after_merge():
    """Test n_intervals after operations that merge intervals.

    Input: chr1:0-50, chr1:40-60 (overlapping, merge to 1 interval)
    Output: 1 (merged from 2)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 40],
            "end": [50, 60],
        }
    )
    gs = GenomicSet(data)
    # Overlapping intervals should merge to 1
    assert gs.n_intervals() == 1


def test_genomic_set_total_size_empty():
    """Test total_size with empty set.

    Input: (empty set)
    Output: 0
    """
    data = pd.DataFrame(
        {
            "chrom": [],
            "start": [],
            "end": [],
        }
    )
    gs = GenomicSet(data)
    assert gs.total_size() == 0


def test_genomic_set_total_size_single():
    """Test total_size with single interval.

    Input: chr1:0-100 (size=100)
    Output: 100
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
        }
    )
    gs = GenomicSet(data)
    assert gs.total_size() == 100


def test_genomic_set_total_size_multiple():
    """Test total_size with multiple intervals.

    Input: chr1:0-50 (size=50), chr1:100-150 (size=50), chr2:0-200 (size=200)
    Output: 300 (sum of all sizes)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [0, 100, 0],
            "end": [50, 150, 200],
        }
    )
    gs = GenomicSet(data)
    assert gs.total_size() == 300


def test_genomic_set_total_size_after_merge():
    """Test total_size after operations that merge intervals.

    Input: chr1:0-50 (size=50), chr1:40-60 (size=20, overlaps with first)
    Output: 60 (merged to chr1:0-60, size=60, less than 50+20=70 due to overlap removal)
    """
    data = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 40],
            "end": [50, 60],
        }
    )
    gs = GenomicSet(data)
    # Overlapping intervals merge to chr1:0-60, size=60
    # Original sizes were 50+20=70, but overlap reduces total to 60
    assert gs.total_size() == 60


# --- transform_ll_clm tests -------------------------------------------------


class _StubCharTokenizer(Tokenizer):
    """Char-level Tokenizer stub with configurable BOS/EOS for testing."""

    def __init__(self, bos: int | None = None, eos: int | None = None):
        # Map A/C/G/T (case-insensitive) to ids 10..13; lowercase too.
        self._vocab = {"a": 10, "c": 11, "g": 12, "t": 13, "n": 14}
        self._bos = bos
        self._eos = eos

    def encode(self, text: str) -> list[int]:
        body = [self._vocab[c.lower()] for c in text]
        out = []
        if self._bos is not None:
            out.append(self._bos)
        out.extend(body)
        if self._eos is not None:
            out.append(self._eos)
        return out

    @property
    def bos_token_id(self) -> int:
        if self._bos is None:
            raise AttributeError("no BOS")
        return self._bos

    @property
    def eos_token_id(self) -> int:
        if self._eos is None:
            raise AttributeError("no EOS")
        return self._eos


def test_transform_ll_clm_no_specials():
    tokenizer = HFTokenizer(AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm"))
    seq = "ACgtAC"
    out = transform_ll_clm({"seq": seq}, tokenizer)

    assert out["input_ids"].dtype == torch.long
    assert out["is_upper"].dtype == torch.bool
    # tokenizer is char-level with no BOS/EOS, so length == len(seq)
    assert out["input_ids"].shape == (len(seq),)
    assert out["is_upper"].shape == (len(seq),)
    expected_upper = torch.tensor([True, True, False, False, True, True])
    assert torch.equal(out["is_upper"], expected_upper)


@pytest.mark.parametrize(
    "bos,eos",
    [(None, None), (101, None), (None, 102), (101, 102)],
)
def test_transform_ll_clm_special_tokens(bos, eos):
    tokenizer = _StubCharTokenizer(bos=bos, eos=eos)
    seq = "ACgt"
    out = transform_ll_clm({"seq": seq}, tokenizer)

    body_upper = [True, True, False, False]
    expected_upper = []
    if bos is not None:
        expected_upper.append(False)
    expected_upper.extend(body_upper)
    if eos is not None:
        expected_upper.append(False)
    assert torch.equal(out["is_upper"], torch.tensor(expected_upper))

    expected_ids = []
    if bos is not None:
        expected_ids.append(bos)
    expected_ids.extend([10, 11, 12, 13])  # A C g t
    if eos is not None:
        expected_ids.append(eos)
    assert out["input_ids"].tolist() == expected_ids


def test_transform_ll_clm_rejects_non_char_level():
    """A tokenizer that splits a single character into multiple tokens should fail."""

    class _BPELikeTokenizer(Tokenizer):
        def encode(self, text: str) -> list[int]:
            # Pretend each character maps to two tokens — char-level assertion must fire.
            return [ord(c) for c in text for _ in range(2)]

    with pytest.raises(AssertionError, match="Char-level"):
        transform_ll_clm({"seq": "ACGT"}, _BPELikeTokenizer())


def test_transform_ll_clm_all_lower_and_all_upper():
    tokenizer = HFTokenizer(AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm"))
    out_upper = transform_ll_clm({"seq": "ACGTAC"}, tokenizer)
    out_lower = transform_ll_clm({"seq": "acgtac"}, tokenizer)
    assert out_upper["is_upper"].all()
    assert (~out_lower["is_upper"]).all()
    # Tokenizer is case-insensitive, so input_ids should match.
    assert torch.equal(out_upper["input_ids"], out_lower["input_ids"])


def test_transform_ll_clm_honors_disabled_auto_insertion():
    """A tokenizer with bos/eos defined but auto-insertion disabled (a
    common HF setup, e.g. GPT-2-style) must not gain extra special-token
    targets. We honor whatever ``add_special_tokens=True`` returns; if
    the tokenizer chose not to insert, neither do we."""

    class _NoAutoInsertTokenizer(Tokenizer):
        # bos/eos IDs are defined, but encode never inserts them.
        def encode(self, text: str) -> list[int]:
            return [{"a": 10, "c": 11, "g": 12, "t": 13}[c.lower()] for c in text]

        @property
        def bos_token_id(self) -> int:
            return 99

        @property
        def eos_token_id(self) -> int:
            return 98

    out = transform_ll_clm({"seq": "ACgt"}, _NoAutoInsertTokenizer())
    # No specials in the encoding → no specials in input_ids; is_upper is
    # purely the per-char case. Crucially, n_total = len(seq), not len(seq)+2.
    assert out["input_ids"].tolist() == [10, 11, 12, 13]
    assert out["is_upper"].tolist() == [True, True, False, False]


def test_transform_ll_clm_byte_level_tokenizer_uppercases():
    """Case-sensitive byte-level tokenizer (e.g. Evo2's vortex
    CharLevelTokenizer) must still produce correct, identical input_ids
    for upper / lower / mixed sequences — transform_ll_clm uppercases
    before tokenizing so the model only ever sees uppercase bytes.
    """

    class _ByteLevelTokenizer(Tokenizer):
        def encode(self, text: str) -> list[int]:
            return list(text.encode("utf-8"))

    tokenizer = _ByteLevelTokenizer()
    seqs = ["ACGTAC", "acgtac", "AcGtAc"]
    outs = [transform_ll_clm({"seq": s}, tokenizer) for s in seqs]
    # All three sequences must produce identical input_ids — that's what
    # makes Evo2 happy. The is_upper masks differ as expected.
    for o in outs[1:]:
        assert torch.equal(o["input_ids"], outs[0]["input_ids"])
    # Sanity: input_ids correspond to ASCII codes for uppercase ACGTAC.
    assert outs[0]["input_ids"].tolist() == [65, 67, 71, 84, 65, 67]
    assert outs[0]["is_upper"].tolist() == [True, True, True, True, True, True]
    assert outs[1]["is_upper"].tolist() == [False, False, False, False, False, False]
    assert outs[2]["is_upper"].tolist() == [True, False, True, False, True, False]
