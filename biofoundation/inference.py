import datasets
from jaxtyping import Shaped
import numpy as np
import tempfile
import torch.nn as nn
from transformers import Trainer, TrainingArguments


def run_inference(
    model: nn.Module,
    dataset: datasets.Dataset,
    **kwargs,
) -> Shaped[np.ndarray, "batch ..."]:
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        **kwargs,
    )
    trainer = Trainer(model=model, args=training_args)
    return trainer.predict(test_dataset=dataset).predictions
