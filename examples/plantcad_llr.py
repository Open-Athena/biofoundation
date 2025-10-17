from biofoundation.data import Genome
from biofoundation.model.adapters.hf import HFMaskedLM, HFTokenizer
from biofoundation.inference import run_llr_mlm
from datasets import load_dataset
import numpy as np
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, AutoModelForMaskedLM


model_name = "kuleshov-group/PlantCAD2-Large-l48-d1536"
tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(model_name))
model = HFMaskedLM(
    AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
)

# Downloaded from https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-62/fasta/zea_mays/dna/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_sm.toplevel.fa.gz
genome_path = "data/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_sm.toplevel.fa.gz"
genome = Genome(genome_path)
window_size = 8192

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
        per_device_eval_batch_size=32,
        torch_compile=True,
        bf16_full_eval=True,
        dataloader_num_workers=8,
        remove_unused_columns=False,
    ),
)
pearson = pearsonr(AF, llr)[0].round(3)
spearman = spearmanr(AF, llr)[0].round(3)
print(f"{pearson=} {spearman=}")

# | Model                                      | Window Size | Pearson | Spearman |
# |--------------------------------------------|-------------|---------|----------|
# | kuleshov-group/PlantCaduceus_l20           |  512        | 0.140   | 0.108    |
# | kuleshov-group/PlantCaduceus_l24           |  512        | 0.144   | 0.109    |
# | kuleshov-group/PlantCaduceus_l28           |  512        | 0.155   | 0.119    |
# | kuleshov-group/PlantCaduceus_l32           |  512        | 0.167   | 0.126    |
# | kuleshov-group/PlantCAD2-Small-l24-d0768   | 8192        | 0.164   | 0.127    |
# | kuleshov-group/PlantCAD2-Medium-l48-d1024  | 8192        | 0.186   | 0.147    |
# | kuleshov-group/PlantCAD2-Large-l48-d1536   | 8192        | 0.202   | 0.158    |
