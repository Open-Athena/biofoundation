from biofoundation.data import Genome
from biofoundation.model.adapters.hf import HFTokenizer, HFCausalLMWithEmbeddings
from biofoundation.inference import run_llr_and_embedding_distance
from datasets import Dataset, load_dataset
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "data/marin_checkpoints/animal-promoters-yolo-r01-213a8e/step-18000"

tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(model_name))
model = HFCausalLMWithEmbeddings(AutoModelForCausalLM.from_pretrained(model_name))

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

# Run combined inference - computes LLR and embedding distances in single forward pass
# Returns array with shape [B, 3] where columns are [llr, last_distance, middle_distance]
results = run_llr_and_embedding_distance(
    model,
    tokenizer,
    dataset,
    genome,
    window_size,
    data_transform_on_the_fly=True,
    inference_kwargs=dict(
        per_device_eval_batch_size=128,
        bf16_full_eval=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    ),
)

# Extract each metric from the results array
llr = results[:, 0]
last_distance = results[:, 1]
middle_distance = results[:, 2]

# Compute metrics for each score
# LLR: Higher -llr indicates more pathogenic
AUPRC_llr = average_precision_score(label, -llr)

# Embedding distances: Higher distance indicates more pathogenic
AUPRC_last = average_precision_score(label, last_distance)
AUPRC_middle = average_precision_score(label, middle_distance)

print(f"Results for {model_name}:")
print(f"  LLR AUPRC:              {AUPRC_llr:.3f}")
print(f"  Last-layer AUPRC:       {AUPRC_last:.3f}")
print(f"  Middle-layer AUPRC:     {AUPRC_middle:.3f}")

# | model                                          | step  | LLR   | Last  | Middle |
# |------------------------------------------------|-------|-------|-------|--------|
# |                                                |       |       |       |        |
