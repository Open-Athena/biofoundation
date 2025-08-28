from biofoundation.data import transform_reflogprob_mlm
from biofoundation.model import RefLogProbMLM
from biofoundation.inference import run_inference
from datasets import load_dataset
from functools import partial
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForMaskedLM


model_name = "kuleshov-group/PlantCaduceus_l20"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

dataset = load_dataset(
    "plantcad/evolutionary-constraint-example",
    "10k",
    split="validation",
)
label = np.array(dataset["label"])

dataset = dataset.map(
    partial(
        transform_reflogprob_mlm,
        tokenizer=tokenizer,
        pos=255,
        seq_col="sequences",
    ),
    remove_columns=dataset.column_names,
    num_proc=8,
)

pred = run_inference(
    RefLogProbMLM(model),
    dataset,
    per_device_eval_batch_size=256,
    torch_compile=False,
    bf16_full_eval=True,
    dataloader_num_workers=8,
)

print(f"{roc_auc_score(label, pred)=}")