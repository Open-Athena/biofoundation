import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float, Int


class RefLogProbMLM(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids_BL: Int[Tensor, "B L"],
        pos_B: Int[Tensor, "B"],
        ref_B: Int[Tensor, "B"],
    ) -> Float[Tensor, "B"]:
        B, L = input_ids_BL.shape
        batch_indices_B = torch.arange(B)
        logits_BLV = self.model(input_ids_BL).logits
        V = logits_BLV.shape[2]
        logits_BV = logits_BLV[batch_indices_B, pos_B]
        logprobs_BV = F.log_softmax(logits_BV, dim=-1)
        res_B = logprobs_BV[batch_indices_B, ref_B]
        return res_B