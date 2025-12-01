from biofoundation.data import Genome
from biofoundation.model.adapters.hf import HFTokenizer
from biofoundation.model.adapters.glm_experiments import GLMExperimentsMaskedLM
from biofoundation.inference import run_llr_mlm
from datasets import Dataset, load_dataset
from glm_experiments.models.lm_lit_module import MLMLitModule
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer


# uv run hf download gonzalobenegas/dev-glm-experiments-logs --repo-type dataset --local-dir data/gonzalobenegas/dev-glm-experiments-logs

# Load GLM checkpoint
checkpoint_path = "TODO"
lit_module = MLMLitModule.load_from_checkpoint(checkpoint_path, weights_only=False)
model = GLMExperimentsMaskedLM(lit_module.net)

# Load tokenizer (use appropriate tokenizer for your model)
tokenizer = HFTokenizer(AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm"))

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
llr = run_llr_mlm(
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

# Compute metrics
AUPRC = average_precision_score(label, -llr)
print(f"{AUPRC=:.3f}")
