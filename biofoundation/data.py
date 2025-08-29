from transformers import PreTrainedTokenizerBase
from typing import Any


def transform_reflogprob_mlm(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    pos: int,
    seq_col: str = "seq",
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
        pos: Position in the sequence to mask (0-indexed).
        seq_col: Key in the example dictionary that contains the sequence.
            Defaults to "seq".

    Returns:
        Dictionary with three keys:
        - input_ids_BL: Token IDs with the specified position masked
        - pos_B: The masked position
        - ref_B: The reference token ID that was at the masked position

    Example:
        >>> example = {"seq": "ATCG"}
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> result = transform_reflogprob_mlm(example, tokenizer, 1)
        >>> print(result)
        {'input_ids_BL': tensor([...]), 'pos_B': 1, 'ref_B': 3}
    """
    input_ids = tokenizer(example[seq_col], return_tensors="pt")["input_ids"][0]
    ref = input_ids[pos].item()
    input_ids[pos] = tokenizer.mask_token_id
    return dict(input_ids_BL=input_ids, pos_B=pos, ref_B=ref)
