"""Reproduce bolinas-dna issue #8 LL metrics offline.

Loads a Marin ``eda-functional-pos`` checkpoint + the same validation
dataset Marin uses during training, computes per-token cross-entropy
broken down by case (uppercase = phyloP-functional, lowercase =
non-functional), and compares to the W&B numbers.

Default target: ``eda-functional-pos-primates-6m-409ae3-v2`` step-9999
(see https://wandb.ai/gonzalobenegas/marin/runs/eda-functional-pos-primates-6m-409ae3-v2).

W&B reports per-token *token-weighted global averages* (sum of
per-target losses / total weight). ``run_ll_clm`` returns per-sequence
(ll_sum_upper, ll_sum_lower, n_upper, n_lower); we sum the columns over
the dataset (in fp64 to avoid fp32 accumulation error) and divide.

Download the checkpoint first:
    gsutil -m cp -r \\
      gs://marin-dna-us-central1/checkpoints/eda-functional-pos-primates-6m-409ae3/hf/step-9999 \\
      data/marin_eda_functional_pos/primates-6m-step-9999
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from biofoundation.inference import run_ll_clm
from biofoundation.model.adapters.hf import HFCausalLM, HFTokenizer


DEFAULT_MODEL_PATH = "data/marin_eda_functional_pos/primates-6m-step-9999"
DEFAULT_DATASET = "bolinas-dna/genomes-v5-validation-intervals-v1_255_255"
DEFAULT_SPLIT = "validation"

# W&B targets for primates-6m step-9999.
WANDB_TARGETS = {
    "eval/loss": 1.2184784412384033,
    "eval/val_functional/loss": 1.0384986400604248,
    "eval/val_nonfunctional/loss": 1.259624719619751,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(args.model_path))
    model = HFCausalLM(AutoModelForCausalLM.from_pretrained(args.model_path))

    print(f"Loading dataset {args.dataset}:{args.split}")
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_rows is not None:
        ds = ds.select(range(min(args.max_rows, len(ds))))
    print(f"  {len(ds)} sequences")

    pred = run_ll_clm(
        model,
        tokenizer,
        ds,
        data_transform_on_the_fly=True,
        inference_kwargs=dict(
            per_device_eval_batch_size=args.batch_size,
            bf16_full_eval=torch.cuda.is_available(),
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="none",
        ),
    )
    pred = np.asarray(pred)  # [N, 4]: ll_sum_upper, ll_sum_lower, n_upper, n_lower

    S_u, S_l, n_u, n_l = pred.astype(np.float64).sum(axis=0)
    LL_all = (S_u + S_l) / (n_u + n_l)
    LL_upper = S_u / n_u
    LL_lower = S_l / n_l

    print()
    print(f"  rows scored:           {len(pred)}")
    print(
        f"  target tokens:         {int(n_u + n_l)} "
        f"(upper={int(n_u)}, lower={int(n_l)})"
    )
    print()
    print(
        f"  {'metric':<28s} {'ours (loss = -LL)':>20s}  {'W&B target':>14s}  {'|diff|':>10s}"
    )
    rows = [
        (LL_all, "eval/loss"),
        (LL_upper, "eval/val_functional/loss"),
        (LL_lower, "eval/val_nonfunctional/loss"),
    ]
    diffs = []
    for ll, key in rows:
        target = WANDB_TARGETS[key]
        loss = -ll
        diff = abs(loss - target)
        diffs.append(diff)
        print(f"  {key:<28s} {loss:>20.6f}  {target:>14.6f}  {diff:>10.6f}")
    max_diff = max(diffs)
    print()
    print(f"  max |diff|: {max_diff:.6f}", "  PASS" if max_diff < 1e-3 else "  CHECK")


if __name__ == "__main__":
    main()
