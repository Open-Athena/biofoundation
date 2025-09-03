from abc import ABC, abstractmethod
from jaxtyping import Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel
from typing import cast


class LanguageModel(ABC, nn.Module):
    @abstractmethod
    def forward(
        self,
        input_ids: Int[Tensor, "B L"],
    ) -> Float[Tensor, "B L V"]:
        pass


class HFLanguageModel(LanguageModel):
    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: Int[Tensor, "B L"]) -> Float[Tensor, "B L V"]:
        return cast(Float[Tensor, "B L V"], self.model(input_ids).logits)


def compute_reflogprob_mlm(
    model: LanguageModel,
    input_ids: Int[Tensor, "B L"],
    pos: Int[Tensor, " B"],
    ref: Int[Tensor, " B"],
) -> Float[Tensor, " B"]:
    """Forward pass to compute reference log probabilities.

    This method:
    1. Runs the input through the language model to get logits
    2. Extracts logits at the specified positions
    3. Computes log probabilities using softmax
    4. Returns the log probability of the reference tokens

    Args:
        model: Language model to use for inference
        input_ids: Batch of token ID sequences with shape [B, L] where
            B is batch size and L is sequence length
        pos: Positions to evaluate for each sequence in the batch with
            shape [B] where each element is the position index
        ref: Reference token IDs to compute log probabilities for with
            shape [B] where each element is the token ID

    Returns:
        Log probabilities of the reference tokens at the specified positions
        with shape [B] where B is the batch size
    """
    B, _ = input_ids.shape
    batch_indices = torch.arange(B)
    logits = model(input_ids)
    logits = logits[batch_indices, pos]
    logprobs = F.log_softmax(logits, dim=-1)
    res = logprobs[batch_indices, ref]
    return res
