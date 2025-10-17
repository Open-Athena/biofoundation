# This makes only the assigned local GPU visible to this process, so that
# Evo2's Vortex initialization only uses the assigned GPU.
# when running e.g.
# OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 biofoundation/examples/evo2_llr.py
import os

local_rank = os.environ.get("LOCAL_RANK") or os.environ.get("LOCAL_RANK", None)
if local_rank is not None:
    # Make only the assigned local GPU visible to this process.
    # Note: set this BEFORE importing torch or any CUDA libraries.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(local_rank))

from biofoundation.data import Genome  # noqa: E402
from biofoundation.model.adapters.evo2 import Evo2CausalLM, Evo2Tokenizer  # noqa: E402
from biofoundation.inference import run_llr_clm  # noqa: E402
from datasets import load_dataset  # noqa: E402
from evo2 import Evo2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402


model_name = "evo2_1b_base"
_model = Evo2(model_name)
model = Evo2CausalLM(_model)
tokenizer = Evo2Tokenizer(_model.tokenizer)

# Downloaded from https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-62/fasta/zea_mays/dna/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_sm.toplevel.fa.gz
genome_path = "data/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_sm.toplevel.fa.gz"
genome = Genome(genome_path)
window_size = 8192

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
        per_device_eval_batch_size=16,
        # torch_compile=True,
        # bf16_full_eval=True,
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
# | evo2_1b_base                   |   8192      | 0.093     | 0.081      |
