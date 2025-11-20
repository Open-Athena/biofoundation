from biofoundation.data import Genome
from biofoundation.model.adapters.hf import HFTokenizer
from biofoundation.model.adapters.gpn import GPNMaskedLM
from biofoundation.inference import run_llr_mlm
from datasets import load_dataset
import numpy as np
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer, AutoModelForMaskedLM


# step = 60000
# model_name = f"data/gpn_checkpoints/checkpoint-{step}"

# step = 300_000
# model_name = f"data/gpn_animal_promoter_checkpoints/checkpoint-{step}"

# step = 7_000
# model_name = f"data/gpn_animal_promoter_early_checkpoints/checkpoint-{step}"

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
label = np.array(dataset["label"])

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
# | new run 10k    | 0.096 |
# | new run 30k    | 0.104 |
# | new run 60k    | 0.111 |
# | new run 70k    | 0.119 |
# | gpn-animal-promoter (10k)  | 0.149 |
# | gpn-animal-promoter (100k) | 0.313 |
# | gpn-animal-promoter (200k) | 0.379 |
# | gpn-animal-promoter (300k) | 0.392 |
# | gpn-animal-promoter (370k) | 0.397 |
# | gpn-animal-promoter-early (1k)  | 0.106 |
# | gpn-animal-promoter-early (2k)  | 0.117 |
# | gpn-animal-promoter-early (3k)  | 0.137 |
# | gpn-animal-promoter-early (4k)  | 0.138 |
# | gpn-animal-promoter-early (5k)  | 0.137 |
# | gpn-animal-promoter-early (6k)  | 0.138 |
# | gpn-animal-promoter-early (7k)  | 0.143
# | gpn-animal-promoter-early (8k)  | 0.144 |
# | gpn-animal-promoter-early (9k)  | 0.150 |
# | gpn-animal-promoter-early (10k) | 0.158 |
