from jaxtyping import Float, Int
from torch import Tensor
from typing import cast
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..base import CausalLM, MaskedLM, Tokenizer


class GPNMaskedLM(MaskedLM):
    def __init__(self, model: PreTrainedModel):
        assert model.__class__.__name__.endswith("MaskedLM")
        super().__init__()
        self._model = model

    def forward(self, input_ids: Int[Tensor, "B L"]) -> Float[Tensor, "B L V"]:
        return cast(Float[Tensor, "B L V"], self._model(input_ids=input_ids).logits)
