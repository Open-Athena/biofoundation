import tempfile
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn


def run_inference(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    **kwargs,
):
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        **kwargs,
    )
    trainer = Trainer(model=model, args=training_args)
    return trainer.predict(test_dataset=dataset).predictions
