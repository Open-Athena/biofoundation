from biofoundation.data import Genome
from biofoundation.model import HFCausalLM
from biofoundation.inference import run_llr_clm
from datasets import load_dataset
import numpy as np
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM


# download checkpoint
# hf download plantcad/_dev_marin_plantcad1_v3_train --local-dir data/_dev_marin_plantcad1_v3_train --include local_store/checkpoints/plantcad-train-600m-r16-a1bc43/hf/step-26782/*

model_name = "data/_dev_marin_plantcad1_v3_train/local_store/checkpoints/plantcad-train-600m-r16-a1bc43/hf/step-26782"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = HFCausalLM(AutoModelForCausalLM.from_pretrained(model_name))

# Downloaded from https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-62/fasta/zea_mays/dna/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_sm.toplevel.fa.gz
genome_path = "data/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_sm.toplevel.fa.gz"
genome = Genome(genome_path)
window_size = 512

dataset = load_dataset(
    "plantcad/maize-allele-frequency",
    split="test",
)
AF = np.array(dataset["AF"])

llr = run_llr_clm(
    model,
    tokenizer,
    dataset,
    genome,
    window_size,
    data_transform_on_the_fly=True,
    inference_kwargs=dict(
        per_device_eval_batch_size=512,
        torch_compile=True,
        bf16_full_eval=True,
        dataloader_num_workers=8,
        remove_unused_columns=False,
    ),
)
print(f"{llr.min()=} {llr.max()=} {llr.mean()=}")
pearson = pearsonr(AF, llr)[0].round(3)
spearman = spearmanr(AF, llr)[0].round(3)
print(f"{pearson=} {spearman=}")

# | Model                          | Window Size | Pearson | Spearman |
# |--------------------------------|-------------|---------|----------|
# | _dev_marin_plantcad_v1         |   512       | 0.104   | 0.075    |
# | step-2678                      |   512       | 0.117   | 0.106    |
# | step-26782                     |   512       | 0.119   | 0.106    |
