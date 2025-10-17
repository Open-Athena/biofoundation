from abc import ABC, abstractmethod
from jaxtyping import Float, Int
import torch.nn as nn
from torch import Tensor


class CausalLM(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input_ids: Int[Tensor, "B L"],
    ) -> Float[Tensor, "B L V"]:
        pass


class MaskedLM(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input_ids: Int[Tensor, "B L"],
    ) -> Float[Tensor, "B L V"]:
        pass
