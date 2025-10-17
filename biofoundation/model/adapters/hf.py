from jaxtyping import Float, Int
from torch import Tensor
from typing import cast
from transformers import PreTrainedModel

from ..base import CausalLM, MaskedLM


class HFMaskedLM(MaskedLM):
    def __init__(self, model: PreTrainedModel):
        assert model.__class__.__name__.endswith("MaskedLM")
        super().__init__()
        self.model = model

    def forward(self, input_ids: Int[Tensor, "B L"]) -> Float[Tensor, "B L V"]:
        return cast(Float[Tensor, "B L V"], self.model(input_ids).logits)


class HFCausalLM(CausalLM):
    def __init__(self, model: PreTrainedModel):
        assert model.__class__.__name__.endswith("CausalLM")
        super().__init__()
        self.model = model

    def forward(self, input_ids: Int[Tensor, "B L"]) -> Float[Tensor, "B L V"]:
        return cast(Float[Tensor, "B L V"], self.model(input_ids).logits)
