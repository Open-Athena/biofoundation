from biofoundation.data import Genome
from biofoundation.model.adapters.hf import HFTokenizer
from biofoundation.model.adapters.gpn import GPNMaskedLM
from biofoundation.inference import run_llr_mlm
from datasets import Dataset
import gpn.model  # noqa: F401  # Registers the GPN architecture
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForMaskedLM


# step = 60000
# model_name = f"data/gpn_checkpoints/checkpoint-{step}"

model_name = "data/gpn_animal_promoter_checkpoints/checkpoint-10000"

# step = 9_000
# model_name = f"data/gpn_animal_promoter_early_checkpoints/checkpoint-{step}"

# uv run hf download gonzalobenegas/gpn-animal-promoter-checkpoints-second-part --repo-type dataset --local-dir data/gpn_animal_promoter_checkpoints_second_part
# model_name = "data/gpn_animal_promoter_checkpoints_second_part/checkpoint-130000"

# model_name = "songlab/gpn-animal-promoter"

tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(model_name))
model = GPNMaskedLM(AutoModelForMaskedLM.from_pretrained(model_name))

# Downloaded from https://ftp.ensembl.org/pub/release-115/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.toplevel.fa.gz
genome_path = "data/Homo_sapiens.GRCh38.dna_sm.toplevel.fa.gz"
genome = Genome(genome_path)
window_size = 512

promoters = [
    "F9",
    "GP1BA",
    "HBB",
    "HBG1",
    "HNF4A",
    "LDLR",
    "MSMB",
    "PKLR",
    "TERT",
]

V = pd.read_parquet("hf://datasets/gonzalobenegas/sat_mut_mpra/test.parquet")
V = V[V["element"].isin(promoters)]
print(V.element.value_counts())
V["label"] = V["label"].abs()  # abs(LFC)
dataset = Dataset.from_pandas(V[["chrom", "pos", "ref", "alt"]], preserve_index=False)


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

V["pred"] = np.abs(llr)

res = V.groupby("element").apply(lambda x: spearmanr(x["label"], x["pred"])[0].round(3))
print(res)
print(res.mean(axis=0))

# seems like some of these are kinda random?

# | Step   | Spearman |
# |--------|-------|
# | gpn-animal-promoter (10k) | 0.149|
# | gpn-animal-promoter (370k) | 0.069 |
# | gpn-animal-promoter (370k + 100k) | 0.140 |
