from jaxtyping import Float, Int
from torch import Tensor
from typing import cast
from transformers import PreTrainedModel

from biofoundation.model.base import EmbeddingModel, MaskedLM


class GPNMaskedLM(MaskedLM):
    def __init__(self, model: PreTrainedModel):
        assert model.__class__.__name__.endswith("MaskedLM")
        super().__init__()
        self._model = model

    def forward(self, input_ids: Int[Tensor, "B L"]) -> Float[Tensor, "B L V"]:
        return cast(Float[Tensor, "B L V"], self._model(input_ids=input_ids).logits)


class GPNEmbeddingModel(EmbeddingModel):
    def __init__(self, model: PreTrainedModel, layer: str = "last"):
        assert model.__class__.__name__.endswith("Model")
        super().__init__()
        self._model = model
        self._layer = layer
        assert layer in ["last", "middle"]

    def forward(self, input_ids: Int[Tensor, "B L"]) -> Float[Tensor, "B L D"]:
        if self._layer == "last":
            res = self._model(input_ids=input_ids).last_hidden_state
        elif self._layer == "middle":
            # hacky due to poor GPN implementation
            n_layers = len(self._model.encoder.layer)  # type: ignore[union-attr, arg-type]
            x = self._model.embeddings(input_ids=input_ids)  # type: ignore[operator]
            middle_layers = self._model.encoder.layer[: n_layers // 2]  # type: ignore[union-attr, index]
            res = middle_layers(x)  # type: ignore[operator]
        return cast(Float[Tensor, "B L D"], res)
