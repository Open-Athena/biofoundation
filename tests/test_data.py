import textwrap

import pytest
import torch
from transformers import AutoTokenizer

from biofoundation.data import (
    Genome,
    transform_llr_mlm,
    transform_llr_clm,
    transform_reflogprob_mlm,
    transform_reflogprob_clm,
)


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
