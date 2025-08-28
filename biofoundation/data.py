from transformers import PreTrainedTokenizerBase
from typing import Any


def transform_reflogprob_mlm(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    pos: int,
    seq_col: str = "seq",
) -> dict[str, Any]:
    input_ids = tokenizer(example[seq_col], return_tensors="pt")["input_ids"][0]
    ref = input_ids[pos].item()
    input_ids[pos] = tokenizer.mask_token_id
    return dict(input_ids_BL=input_ids, pos_B=pos, ref_B=ref)