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


class Tokenizer(ABC):
    """Abstract base class for tokenizers in biofoundation.

    Provides a minimal interface for single-sequence tokenization.
    All methods return Python lists of token IDs (not tensors).
    """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode a single text string to token IDs.

        Args:
            text: Input string to encode

        Returns:
            List of token IDs
        """
        pass

    @property
    def mask_token_id(self) -> int:
        """Return the mask token ID (for MLM models).

        Raises:
            AttributeError: If mask token is not supported by this tokenizer
        """
        raise AttributeError(f"Mask token not supported by {self.__class__.__name__}")
