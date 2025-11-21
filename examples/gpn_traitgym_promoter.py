from biofoundation.data import Genome
from biofoundation.model.adapters.hf import HFTokenizer
from biofoundation.model.adapters.gpn import GPNMaskedLM
from biofoundation.inference import run_llr_mlm
from datasets import Dataset, load_dataset
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer, AutoModelForMaskedLM


# step = 60000
# model_name = f"data/gpn_checkpoints/checkpoint-{step}"

# step = 10_000
# model_name = f"data/gpn_animal_promoter_checkpoints/checkpoint-{step}"

# step = 9_000
# model_name = f"data/gpn_animal_promoter_early_checkpoints/checkpoint-{step}"

# model_name = "../gpn/analysis/gpn_animal_promoter/second_part/checkpoints/checkpoint-100000"

model_name = "songlab/gpn-animal-promoter"

tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(model_name))
model = GPNMaskedLM(AutoModelForMaskedLM.from_pretrained(model_name))

# Downloaded from https://ftp.ensembl.org/pub/release-115/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.toplevel.fa.gz
genome_path = "data/Homo_sapiens.GRCh38.dna_sm.toplevel.fa.gz"
genome = Genome(genome_path)
window_size = 512

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

llr = run_llr_mlm(
    model,
    tokenizer,
    dataset,
    genome,
    window_size,
    data_transform_on_the_fly=True,
    inference_kwargs=dict(
        per_device_eval_batch_size=128,
        # torch_compile=True,  # so fast it's not worth the compile time
        bf16_full_eval=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    ),
)

AUPRC = average_precision_score(label, -llr)
print(f"{AUPRC=:.3f}")

# | Step   | AUPRC |
# |--------|-------|
# | gpn-animal-promoter (10k)         | 0.157 |
# | gpn-animal-promoter (370k)        | 0.566 |
# | gpn-animal-promoter (370k + 100k) | 0.621 |
# | gpn-animal-promoter-early (1k)    | 0.107 |
# | gpn-animal-promoter-early (2k)    | 0.114 |
# | gpn-animal-promoter-early (3k)    | 0.120 |
# | gpn-animal-promoter-early (4k)    | 0.136 |
# | gpn-animal-promoter-early (5k)    | 0.131 |
# | gpn-animal-promoter-early (6k)    | 0.136 |
# | gpn-animal-promoter-early (7k)    | 0.138 |
# | gpn-animal-promoter-early (8k)    | 0.138 |
# | gpn-animal-promoter-early (9k)    | 0.145 |
# | gpn-animal-promoter-early (10k)   | 0.146 |
