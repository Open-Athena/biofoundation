from biofoundation.data import Genome
from biofoundation.model.adapters.hf import HFCausalLM, HFTokenizer
from biofoundation.inference import run_llr_clm
from datasets import Dataset, load_dataset
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer, AutoModelForCausalLM


# model_name = "data/marin_checkpoints/animal-promoters-standard-r08-eaa762/step-6000"
# model_name = "data/marin_checkpoints/animal-promoters-repeat-weight-0.01-r01-da6be1/step-6000"
model_name = "data/marin_checkpoints/animal-promoters-yolo-r01-213a8e/step-22000"

tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(model_name))
model = HFCausalLM(AutoModelForCausalLM.from_pretrained(model_name))

# Downloaded from https://ftp.ensembl.org/pub/release-115/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.toplevel.fa.gz
genome_path = "data/Homo_sapiens.GRCh38.dna_sm.toplevel.fa.gz"
genome = Genome(genome_path)
window_size = 512

# Load TraitGym dataset with promoter subset
dataset = load_dataset(
    "songlab/TraitGym",
    "mendelian_traits",
    split="test",
)
V = dataset.to_pandas()
subset = pd.read_parquet(
    "https://huggingface.co/datasets/songlab/TraitGym/resolve/main/mendelian_traits_matched_9/subset/nonexonic_AND_proximal.parquet"
)
V = V.merge(subset, on=["chrom", "pos", "ref", "alt"], how="inner")
label = np.array(V["label"])
dataset = Dataset.from_pandas(V, preserve_index=False)

# Run LLR inference
llr = run_llr_clm(
    model,
    tokenizer,
    dataset,
    genome,
    window_size,
    data_transform_on_the_fly=True,
    inference_kwargs=dict(
        # per_device_eval_batch_size=32,
        # per_device_eval_batch_size=256,
        per_device_eval_batch_size=128,
        bf16_full_eval=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    ),
)

# Compute metrics
AUPRC = average_precision_score(label, -llr)
print(f"{AUPRC=:.3f}")

# | model                                          | step  | AUPRC |
# |------------------------------------------------|-------|-------|
# | animal-promoters-standard-r08-eaa762           |  2000 | 0.137 |
# | animal-promoters-standard-r08-eaa762           |  4000 | 0.173 |
# | animal-promoters-standard-r08-eaa762           |  6000 | 0.250 |
# | animal-promoters-standard-r08-eaa762           |  8000 | 0.301 |
# | animal-promoters-standard-r08-eaa762           | 10000 | 0.329 |
# | animal-promoters-standard-r08-eaa762           | 12000 | 0.373 |
# | animal-promoters-standard-r08-eaa762           | 14000 | 0.496 |
# | animal-promoters-standard-r08-eaa762           | 16000 | 0.605 |
# | animal-promoters-standard-r08-eaa762           | 18000 | 0.620 |
# | animal-promoters-standard-r08-eaa762           | 19999 | 0.633 |
# | animal-promoters-repeat-weight-0.01-r01-da6be1 |  2000 | 0.147 |
# | animal-promoters-repeat-weight-0.01-r01-da6be1 |  4000 | 0.178 |
# | animal-promoters-repeat-weight-0.01-r01-da6be1 |  6000 | 0.205 |
# | animal-promoters-repeat-weight-0.01-r01-da6be1 |  8000 | 0.314 |
# | animal-promoters-repeat-weight-0.01-r01-da6be1 | 10000 | 0.364 |
# | animal-promoters-repeat-weight-0.01-r01-da6be1 | 12000 | 0.370 |
# | animal-promoters-repeat-weight-0.01-r01-da6be1 | 14000 | 0.414 |
# | animal-promoters-repeat-weight-0.01-r01-da6be1 | 16000 | 0.445 |
# | animal-promoters-repeat-weight-0.01-r01-da6be1 | 18000 | 0.487 |
# | animal-promoters-repeat-weight-0.01-r01-da6be1 | 19999 | 0.564 |
# | animal-promoters-yolo-r01-213a8e               |  2000 | 0.170 |
# | animal-promoters-yolo-r01-213a8e               |  6000 | 0.257 |
# | animal-promoters-yolo-r01-213a8e               |  8000 | 0.394 |
# | animal-promoters-yolo-r01-213a8e               | 10000 | 0.481 |
# | animal-promoters-yolo-r01-213a8e               | 14000 | 0.617 |
# | animal-promoters-yolo-r01-213a8e               | 16000 | 0.714 |
# | animal-promoters-yolo-r01-213a8e               | 18000 | 0.702 |
# | animal-promoters-yolo-r01-213a8e               | 20000 | 0.694 |
# | animal-promoters-yolo-r01-213a8e               | 22000 | 0.701 |
