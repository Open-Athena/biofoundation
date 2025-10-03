from biofoundation.data import Genome
from biofoundation.model import HFMaskedLM
from biofoundation.inference import run_llr_mlm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM


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
dataset = dataset.select(range(10))

pred = run_llr_mlm(
    model,
    tokenizer,
    dataset,
    genome,
    window_size,
    data_transform_on_the_fly=True,
    inference_kwargs=dict(
        per_device_eval_batch_size=32,
        torch_compile=False,
        bf16_full_eval=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    ),
)
print(pred)
