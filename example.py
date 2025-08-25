from biofoundation.data import HFDataset, EvolutionaryConstraintMLMTransform
from biofoundation.model import EvolutionaryConstraintMLM
from biofoundation.inference import run_inference
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForMaskedLM


model_name = "kuleshov-group/PlantCaduceus_l20"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

dataset = HFDataset(
    path="plantcad/evolutionary-constraint-example",
    split="validation",
    transforms=[EvolutionaryConstraintMLMTransform(tokenizer)],
)

pred = run_inference(
    EvolutionaryConstraintMLM(model),
    dataset,
    per_device_eval_batch_size=256,
    torch_compile=True,
    bf16_full_eval=True,
    dataloader_num_workers=8,
)

# Not expected to be handled by inference logic?
label = np.array(dataset.dataset["label"])
print(f"{roc_auc_score(label, pred)=}")