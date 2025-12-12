from biofoundation.data import Genome
from biofoundation.model.adapters.hf import HFTokenizer
from biofoundation.model.adapters.gpn import GPNEmbeddingModel
from biofoundation.inference import run_euclidean_distance
from datasets import load_dataset
import gpn.model  # noqa: F401  # Registers the GPN architecture
import numpy as np
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer, AutoModel

model_name = "songlab/gpn-animal-promoter"
# layer = "last"
layer = "middle"

tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(model_name))
model = GPNEmbeddingModel(AutoModel.from_pretrained(model_name), layer=layer)

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

distance = run_euclidean_distance(
    model,
    tokenizer,
    dataset,
    genome,
    window_size,
    data_transform_on_the_fly=True,
    inference_kwargs=dict(
        per_device_eval_batch_size=256,
        torch_compile=True,
        bf16_full_eval=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    ),
)

# Higher distance indicates more pathogenic (different from reference)
AUPRC = average_precision_score(label, distance)
print(f"{AUPRC=:.3f}")

# | Step     | Layer  | AUPRC |
# |----------|--------|-------|
# | 370k     | last   | 0.307 |
# | 370k     | middle | 0.167 |
