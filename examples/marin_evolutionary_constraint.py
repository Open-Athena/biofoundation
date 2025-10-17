from biofoundation.model.adapters.hf import HFCausalLM
from biofoundation.inference import run_reflogprob_clm
from datasets import load_dataset
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "plantcad/_dev_marin_plantcad_v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = HFCausalLM(AutoModelForCausalLM.from_pretrained(model_name))

dataset = load_dataset(
    "plantcad/evolutionary-constraint-example",
    "10k",
    split="validation",
)
label = np.array(dataset["label"])

pred = run_reflogprob_clm(
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
