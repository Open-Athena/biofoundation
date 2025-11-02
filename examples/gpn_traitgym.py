from biofoundation.data import Genome
from biofoundation.model.adapters.hf import HFTokenizer
from biofoundation.model.adapters.gpn import GPNMaskedLM
from biofoundation.inference import run_llr_mlm
from datasets import load_dataset
import gpn.model
import numpy as np
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer, AutoModelForMaskedLM


step = 60000
model_name = f"data/gpn_checkpoints/checkpoint-{step}"
tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(model_name))
model = GPNMaskedLM(
    AutoModelForMaskedLM.from_pretrained(model_name)
)

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
        per_device_eval_batch_size=512,
        # torch_compile=True,  # so fast it's not worth the compile time
        bf16_full_eval=True,
        dataloader_num_workers=8,
        remove_unused_columns=False,
        report_to="none",
    ),
)

AUPRC = average_precision_score(label, -llr)
print(f"{AUPRC=}")

# | Step   | AUPRC |
# |--------|-------|
# | 10k    | 0.096 |
# | 30k    | 0.104 |
# | 60k    | 0.111 |
# | 70k    | 0.119 |
