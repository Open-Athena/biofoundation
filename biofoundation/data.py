from transformers import PreTrainedTokenizerBase
from typing import Any


NUCLEOTIDES = list("ACGT")


def transform_reflogprob_mlm(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    """Transform a sequence example for reference log probability MLM inference.

    This function prepares a sequence example for masked language modeling by:
    1. Tokenizing the input sequence
    2. Masking a specific position in the sequence
    3. Recording the reference token at that position

    Args:
        example: Dictionary containing the sequence data. Must have a key matching
            `seq_col` that contains the input sequence.
        tokenizer: HuggingFace tokenizer for converting text to token IDs.

    Returns:
        Dictionary with three keys:
        - input_ids: Token IDs with the specified position masked
        - pos: The masked position
        - ref: The reference token ID that was at the masked position

    Example:
        >>> example = {"seq": "ATCG"}
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> result = transform_reflogprob_mlm(example, tokenizer, 1)
        >>> print(result)
        {'input_ids': tensor([...]), 'pos': 1, 'ref': 3}
    """
    pos = example["pos"]
    assert example["seq"][pos] in NUCLEOTIDES
    input_ids = tokenizer(example["seq"], return_tensors="pt")["input_ids"][0]
    ref = input_ids[pos].item()
    input_ids[pos] = tokenizer.mask_token_id
    return dict(input_ids=input_ids, pos=pos, ref=ref)


def transform_reflogprob_clm(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    pos = example["pos"]
    assert example["seq"][pos] in NUCLEOTIDES
    input_ids = tokenizer(example["seq"], return_tensors="pt")["input_ids"][0]
    ref = input_ids[pos].item()
    # Create 4 copies of the input sequence
    new_input_ids = input_ids.unsqueeze(0).repeat(len(NUCLEOTIDES), 1)
    for i, nuc in enumerate(NUCLEOTIDES):
        new_input_ids[i, pos] = tokenizer.encode(nuc)[0]
    ref = NUCLEOTIDES.index(example["seq"][pos])
    return dict(input_ids=new_input_ids, ref=ref)
