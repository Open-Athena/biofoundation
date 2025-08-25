from dataclasses import dataclass
from datasets import load_dataset
import torch
from typing import Callable, Any, Optional


# Ideas for transforms:
# - get seq, pos given chrom, pos
# - tokenizer transform?
# - prepare mlm given seq, pos
# - prepare clm given seq, pos

# TODO: should handle optional start and end tokens
# TODO: should handle optional reverse complement averaging


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, transforms: Optional[list[Callable]] = None, **kwargs):
        """
        Initialize HuggingFace dataset wrapper.
        
        Args:
            transforms: List of transform functions to apply in order
            **kwargs: All arguments passed directly to load_dataset
                     (e.g., path, split, config_name, data_dir, data_files, etc.)
        """
        self.dataset = load_dataset(**kwargs)
        self.transforms = transforms or []

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        for transform in self.transforms:
            sample = transform(sample)
        return sample


@dataclass
class EvolutionaryConstraintMLMTransform:
    tokenizer: Any
    pos: int = 255
    seq_col: str = "sequences"

    def __call__(self, sample):
        input_ids = self.tokenizer(
            sample[self.seq_col],
            return_tensors="pt",
        )["input_ids"][0]
        ref = input_ids[self.pos].item()
        input_ids[self.pos] = self.tokenizer.mask_token_id
        return dict(input_ids=input_ids, pos=self.pos, ref=ref)
