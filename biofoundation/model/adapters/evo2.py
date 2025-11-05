from jaxtyping import Float, Int
from torch import Tensor
from typing import cast

from evo2 import Evo2
from vortex.model.tokenizer import CharLevelTokenizer
from biofoundation.model.base import CausalLM, Tokenizer


class Evo2CausalLM(CausalLM):
    def __init__(self, model: Evo2):
        super().__init__()
        self._model = model

    def forward(self, input_ids: Int[Tensor, "B L"]) -> Float[Tensor, "B L V"]:
        # Evo2 returns (outputs, embeddings) tuple
        # outputs[0] contains the logits
        outputs, _ = self._model(input_ids)
        return cast(Float[Tensor, "B L V"], outputs[0])


class Evo2Tokenizer(Tokenizer):
    """Adapter for evo2 CharLevelTokenizer."""

    def __init__(self, tokenizer: CharLevelTokenizer):
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(map(int, self._tokenizer.tokenize(text)))
