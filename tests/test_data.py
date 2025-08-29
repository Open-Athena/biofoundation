import torch
from transformers import AutoTokenizer
from biofoundation.data import transform_reflogprob_mlm


def test_transform_reflogprob_mlm_basic():
    """Test basic functionality of transform_reflogprob_mlm with real DNA tokenizer"""
    # Setup - use real DNA tokenizer
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    example = {"sequences": "ATCGATCG"}
    pos = 2
    seq_col = "sequences"

    # Execute
    result = transform_reflogprob_mlm(example, tokenizer, pos, seq_col)

    # Assert
    assert isinstance(result, dict)
    assert "input_ids_BL" in result
    assert "pos_B" in result
    assert "ref_B" in result

    # Check types
    assert isinstance(result["input_ids_BL"], torch.Tensor)
    assert isinstance(result["pos_B"], int)
    assert isinstance(result["ref_B"], int)

    # Check values
    assert result["pos_B"] == pos

    # Verify the sequence was properly tokenized
    input_ids = result["input_ids_BL"]
    assert len(input_ids) > 0  # Should have tokens

    # Check that the position is valid
    assert pos < len(input_ids)


def test_transform_reflogprob_mlm_mask_position():
    """Test that the specified position is masked"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    example = {"sequences": "GCTAGCTA"}
    pos = 1
    seq_col = "sequences"

    result = transform_reflogprob_mlm(example, tokenizer, pos, seq_col)

    # The input_ids should have the mask token at the specified position
    input_ids = result["input_ids_BL"]
    assert input_ids[pos] == tokenizer.mask_token_id


def test_transform_reflogprob_mlm_reference_value():
    """Test that ref_B contains the original token value before masking"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")
    example = {"sequences": "TATATATA"}
    pos = 3
    seq_col = "sequences"

    result = transform_reflogprob_mlm(example, tokenizer, pos, seq_col)

    # ref_B should contain the original token value before masking
    # This should be the token ID for the nucleotide at position 3
    assert result["ref_B"] >= 0  # Should be a valid token ID
    assert result["ref_B"] != tokenizer.mask_token_id  # Should not be the mask token


def test_transform_reflogprob_mlm_different_sequences():
    """Test with different DNA sequences"""
    tokenizer = AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm")

    # Test with different sequences
    sequences = ["ACGT", "GCTA", "TTAA", "CGCG"]

    for seq in sequences:
        example = {"sequences": seq}
        pos = 1
        seq_col = "sequences"

        result = transform_reflogprob_mlm(example, tokenizer, pos, seq_col)

        # Basic structure checks
        assert "input_ids_BL" in result
        assert "pos_B" in result
        assert "ref_B" in result

        # Position should be masked
        input_ids = result["input_ids_BL"]
        assert input_ids[pos] == tokenizer.mask_token_id

        # Reference should be the original token
        assert result["ref_B"] != tokenizer.mask_token_id
