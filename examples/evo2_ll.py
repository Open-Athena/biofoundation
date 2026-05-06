# Token-weighted dataset-wide log-likelihood with case-based breakdown for Evo2.
#
# transform_ll_clm uppercases before tokenization so Evo2 (case-sensitive
# byte-level tokenizer) sees the bytes it was trained on; the original
# case is preserved in the loss-weight mask.
#
# Same GPU-visibility shim as examples/evo2_llr.py — set
# CUDA_VISIBLE_DEVICES per LOCAL_RANK before importing torch / vortex /
# evo2, e.g.:
# OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 examples/evo2_ll.py
import os

local_rank = os.environ.get("LOCAL_RANK")
if local_rank is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(local_rank))

import numpy as np  # noqa: E402
from datasets import load_dataset  # noqa: E402
from evo2 import Evo2  # noqa: E402

from biofoundation.inference import run_ll_clm  # noqa: E402
from biofoundation.model.adapters.evo2 import Evo2CausalLM, Evo2Tokenizer  # noqa: E402


# model_name = "evo2_1b_base"
model_name = "evo2_7b"
_model = Evo2(model_name)
model = Evo2CausalLM(_model)
tokenizer = Evo2Tokenizer(_model.tokenizer)

# Replace with any HF dataset that has a `seq` column of fixed-length
# mixed-case DNA (uppercase = functional, lowercase = non-functional).
dataset = load_dataset(
    "bolinas-dna/genomes-v5-validation-intervals-v1_255_255",
    split="validation",
)

pred = run_ll_clm(
    model,
    tokenizer,
    dataset,
    data_transform_on_the_fly=True,
    inference_kwargs=dict(
        per_device_eval_batch_size=8,  # evo2_7b
        dataloader_num_workers=4,
        remove_unused_columns=False,
    ),
)
pred = np.asarray(pred)  # [N, 4]: ll_sum_upper, ll_sum_lower, n_upper, n_lower

# Sum sums and counts across the dataset, then divide. fp64 cast avoids
# fp32 accumulation drift on large datasets.
S_u, S_l, n_u, n_l = pred.astype(np.float64).sum(axis=0)
LL_all = (S_u + S_l) / (n_u + n_l)
LL_upper = S_u / n_u
LL_lower = S_l / n_l
gap = LL_upper - LL_lower

print(f"  rows scored:             {len(pred)}")
print(
    f"  target tokens:           {int(n_u + n_l)} (upper={int(n_u)}, lower={int(n_l)})"
)
print(f"  LL(all):                 {LL_all:+.4f}")
print(f"  LL(functional):          {LL_upper:+.4f}")
print(f"  LL(non-functional):      {LL_lower:+.4f}")
print(f"  LL(func) - LL(nonfunc):  {gap:+.4f}")
