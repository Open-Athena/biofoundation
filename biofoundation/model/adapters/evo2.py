from jaxtyping import Float, Int
from torch import Tensor
from typing import cast

from evo2 import Evo2
from ..base import CausalLM


class Evo2CausalLM(CausalLM):
    def __init__(self, model: Evo2):
        super().__init__()
        self.model = model

    def forward(self, input_ids: Int[Tensor, "B L"]) -> Float[Tensor, "B L V"]:
        # Evo2 returns (outputs, embeddings) tuple
        # outputs[0] contains the logits
        outputs, _ = self.model(input_ids)
        return cast(Float[Tensor, "B L V"], outputs[0])
