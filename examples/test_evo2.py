from biofoundation.data import Genome, _get_variant_window
import numpy as np
from datasets import load_dataset
from evo2 import Evo2


# Load the Evo2 model using the native class
model_name = "evo2_7b"
model = Evo2(model_name)

# Load genome and dataset (same as evo2_llr.py)
genome_path = "data/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_sm.toplevel.fa.gz"
genome = Genome(genome_path)
window_size = 8192

print("Loading maize allele frequency dataset...")
dataset = load_dataset(
    "plantcad/maize-allele-frequency",
    split="test",
)
dataset = dataset.select(range(10))  # First 10 variants

print(f"Processing {len(dataset)} variants with window size {window_size}")

# Extract reference and variant sequences
ref_seqs = []
var_seqs = []

for i, example in enumerate(dataset):
    print(
        f"Processing variant {i + 1}/{len(dataset)}: {example['chrom']}:{example['pos']} {example['ref']}->{example['alt']}"
    )

    # Extract centered window around variant
    seq, pos = _get_variant_window(example, genome, window_size)

    # Create reference sequence (original)
    ref_seq = seq

    # Create variant sequence (with alt allele)
    var_seq = seq[:pos] + example["alt"] + seq[pos + 1 :]

    ref_seqs.append(ref_seq)
    var_seqs.append(var_seq)

# Score sequences using official Evo2 method
print(f"Scoring likelihoods of {len(ref_seqs)} reference sequences with Evo2...")
ref_scores = np.array(model.score_sequences(ref_seqs, reduce_method="sum"))
print(f"{ref_scores=}")

print(f"Scoring likelihoods of {len(var_seqs)} variant sequences with Evo2...")
var_scores = np.array(model.score_sequences(var_seqs, reduce_method="sum"))
print(f"{var_scores=}")

# Compute LLR as variant_score - reference_score
llr = var_scores - ref_scores
print(f"{llr=}")
