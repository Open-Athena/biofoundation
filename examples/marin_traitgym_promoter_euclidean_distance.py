from biofoundation.data import Genome
from biofoundation.model.adapters.hf import HFTokenizer, HFEmbeddingModel
from biofoundation.inference import run_euclidean_distance
from datasets import Dataset, load_dataset
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer, AutoModel


model_name = "data/marin_checkpoints/animal-promoters-yolo-r01-213a8e/step-18000"
layer = "last"
# layer = "middle"

tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(model_name))
model = HFEmbeddingModel(AutoModel.from_pretrained(model_name), layer=layer)

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

# Run Euclidean distance inference
distance = run_euclidean_distance(
    model,
    tokenizer,
    dataset,
    genome,
    window_size,
    data_transform_on_the_fly=True,
    inference_kwargs=dict(
        per_device_eval_batch_size=128,
        torch_compile=False,
        bf16_full_eval=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    ),
)

# Higher distance indicates more pathogenic (different from reference)
AUPRC = average_precision_score(label, distance)
print(f"{AUPRC=:.3f}")

# | model                            | step  | layer  | AUPRC |
# |----------------------------------|-------|--------|-------|
# | animal-promoters-yolo-r01-213a8e | 18000 | last   | 0.696 |
# | animal-promoters-yolo-r01-213a8e | 18000 | middle | 0.277 |
