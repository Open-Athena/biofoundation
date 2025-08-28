import datasets
import tempfile
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from typing import Any


def run_inference(
    model: nn.Module,
    dataset: datasets.Dataset,
    **kwargs: Any,
) -> Any:
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        **kwargs,
    )
    trainer = Trainer(model=model, args=training_args)
    return trainer.predict(test_dataset=dataset).predictions
