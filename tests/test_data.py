import torch
from transformers import AutoTokenizer
from biofoundation.data import transform_reflogprob_mlm


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
