from biofoundation.model.adapters.hf import HFMaskedLM
from biofoundation.inference import run_reflogprob_mlm
from datasets import load_dataset
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForMaskedLM


model_name = "kuleshov-group/PlantCaduceus_l20"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = HFMaskedLM(
    AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
)

dataset = load_dataset(
    "plantcad/evolutionary-constraint-example",
    "10k",
    split="validation",
)
label = np.array(dataset["label"])

pred = run_reflogprob_mlm(
    model,
    tokenizer,
    dataset,
    data_transform_kwargs=dict(
        remove_columns=dataset.column_names,
        num_proc=4,
    ),
    inference_kwargs=dict(
        per_device_eval_batch_size=32,
        torch_compile=False,
        bf16_full_eval=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    ),
)

print(f"{roc_auc_score(label, pred)=}")
