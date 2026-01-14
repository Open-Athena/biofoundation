from jaxtyping import Float, Int
from torch import Tensor
from typing import cast
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from biofoundation.model.base import CausalLM, EmbeddingModel, MaskedLM, Tokenizer


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


class HFEmbeddingModel(EmbeddingModel):
    def __init__(self, model: PreTrainedModel, layer: str = "last"):
        assert model.__class__.__name__.endswith("Model")
        super().__init__()
        self._model = model
        self._layer = layer
        assert layer in ["last", "middle"]

    def forward(self, input_ids: Int[Tensor, "B L"]) -> Float[Tensor, "B L D"]:
        if self._layer == "last":
            output = self._model(input_ids=input_ids)
            hidden_state = output.last_hidden_state
        elif self._layer == "middle":
            output = self._model(input_ids=input_ids, output_hidden_states=True)
            hidden_states = output.hidden_states  # Tuple of all layer outputs
            middle_idx = len(hidden_states) // 2
            hidden_state = hidden_states[middle_idx]
        return cast(Float[Tensor, "B L D"], hidden_state)


class HFTokenizer(Tokenizer):
    """Adapter for HuggingFace tokenizers."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return cast(list[int], self._tokenizer.encode(text))

    @property
    def mask_token_id(self) -> int:
        return cast(int, self._tokenizer.mask_token_id)
