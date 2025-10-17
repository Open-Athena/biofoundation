from jaxtyping import Float, Int
from torch import Tensor
from typing import cast
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..base import CausalLM, MaskedLM, Tokenizer


class HFMaskedLM(MaskedLM):
    def __init__(self, model: PreTrainedModel):
        assert model.__class__.__name__.endswith("MaskedLM")
        super().__init__()
        self._model = model

    def forward(self, input_ids: Int[Tensor, "B L"]) -> Float[Tensor, "B L V"]:
        return cast(Float[Tensor, "B L V"], self._model(input_ids).logits)


class HFCausalLM(CausalLM):
    def __init__(self, model: PreTrainedModel):
        assert model.__class__.__name__.endswith("CausalLM")
        super().__init__()
        self._model = model

    def forward(self, input_ids: Int[Tensor, "B L"]) -> Float[Tensor, "B L V"]:
        return cast(Float[Tensor, "B L V"], self._model(input_ids).logits)


class HFTokenizer(Tokenizer):
    """Adapter for HuggingFace tokenizers."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return cast(list[int], self._tokenizer.encode(text))

    @property
    def mask_token_id(self) -> int:
        return cast(int, self._tokenizer.mask_token_id)
