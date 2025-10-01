import textwrap

import pytest
import torch
from transformers import AutoTokenizer

from biofoundation.data import Genome, transform_llr_mlm, transform_reflogprob_mlm


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
    assert result["ref"].ids[0] == expected_ref_id


def test_transform_llr_mlm_returns_ref_and_alt_tokens(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    genome = Genome(_write_test_fasta(tmp_path))
    window_size = 6
    example = {"chrom": "chr1", "pos": 5, "ref": "A", "alt": "G"}

    result = transform_llr_mlm(example, tokenizer, genome, window_size)

    expected_ref_id = tokenizer(example["ref"])["input_ids"][0]
    expected_alt_id = tokenizer(example["alt"])["input_ids"][0]

    assert result["ref"].ids[0] == expected_ref_id
    assert result["alt"].ids[0] == expected_alt_id
    assert result["ref"].ids[0] != tokenizer.mask_token_id
    assert result["alt"].ids[0] != tokenizer.mask_token_id
    assert result["input_ids"][result["pos"]].item() == tokenizer.mask_token_id


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
