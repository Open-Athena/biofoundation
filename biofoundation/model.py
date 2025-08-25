import torch
import torch.nn as nn
import torch.nn.functional as F


class EvolutionaryConstraintMLM(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids,
        pos,
        ref,
    ):
        logits = self.model(input_ids).logits[torch.arange(len(pos)), pos]
        logprobs = F.log_softmax(logits, dim=-1)
        return logprobs[torch.arange(len(ref)), ref]