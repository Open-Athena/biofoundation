from biofoundation.data import Genome
from biofoundation.model import HFMaskedLM
from biofoundation.inference import run_llr_mlm
from datasets import load_dataset
import numpy as np
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, AutoModelForMaskedLM


# model_name = "kuleshov-group/PlantCaduceus_l20" # 0.140, 0.108
model_name = "kuleshov-group/PlantCaduceus_l32"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = HFMaskedLM(
    AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
)

# Downloaded from https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-62/fasta/zea_mays/dna/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_sm.toplevel.fa.gz
genome_path = "data/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_sm.toplevel.fa.gz"
genome = Genome(genome_path)
window_size = 512

dataset = load_dataset(
    "plantcad/maize-allele-frequency",
    split="test",
)
AF = np.array(dataset["AF"])

llr = run_llr_mlm(
    model,
    tokenizer,
    dataset,
    genome,
    window_size,
    data_transform_on_the_fly=True,
    inference_kwargs=dict(
        per_device_eval_batch_size=64,
        torch_compile=False,
        bf16_full_eval=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    ),
)
print(f"{pearsonr(AF, llr)=}")  # TODO: print stat rounded to 3 decimal places
print(f"{spearmanr(AF, llr)=}")
