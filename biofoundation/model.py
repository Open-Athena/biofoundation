import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float, Int


class RefLogProbMLM(nn.Module):
    """Reference Log Probability Masked Language Model wrapper.

    This class wraps a pre-trained language model to compute reference log probabilities
    for masked language modeling tasks. It takes a sequence with a masked position
    and returns the log probability of the reference token at that position.

    The model expects input tensors with specific shapes:
    - input_ids_BL: Batch of token ID sequences [batch_size, sequence_length]
    - pos_B: Positions to evaluate for each sequence in the batch
    - ref_B: Reference token IDs to compute log probabilities for

    Attributes:
        model: The underlying language model that provides logits
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids_BL: Int[Tensor, "B L"],
        pos_B: Int[Tensor, " B"],
        ref_B: Int[Tensor, " B"],
    ) -> Float[Tensor, " B"]:
        """Forward pass to compute reference log probabilities.

        This method:
        1. Runs the input through the language model to get logits
        2. Extracts logits at the specified positions
        3. Computes log probabilities using softmax
        4. Returns the log probability of the reference tokens

        Args:
            input_ids_BL: Batch of token ID sequences with shape [B, L] where
                B is batch size and L is sequence length
            pos_B: Positions to evaluate for each sequence in the batch with
                shape [B] where each element is the position index
            ref_B: Reference token IDs to compute log probabilities for with
                shape [B] where each element is the token ID

        Returns:
            Log probabilities of the reference tokens at the specified positions
            with shape [B] where B is the batch size
        """
        B, L = input_ids_BL.shape
        batch_indices_B = torch.arange(B)
        logits_BLV = self.model(input_ids_BL).logits
        logits_BV = logits_BLV[batch_indices_B, pos_B]
        logprobs_BV = F.log_softmax(logits_BV, dim=-1)
        res_B = logprobs_BV[batch_indices_B, ref_B]
        return res_B
