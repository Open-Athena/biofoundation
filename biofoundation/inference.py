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
    """Run inference on a dataset using a trained model.

    Args:
        model: A trained PyTorch model that can be used with the HuggingFace Trainer.
        dataset: HuggingFace dataset to run inference on. The dataset should be
            compatible with the model's expected input format.
        **kwargs: Additional keyword arguments to pass to TrainingArguments.
            Common options include:
            - per_device_eval_batch_size: Batch size for evaluation
            - dataloader_num_workers: Number of workers for data loading
            - torch_compile: Whether to use torch.compile for faster inference
            - bf16_full_eval: Whether to use bf16 for evaluation

    Returns:
        The model's predictions on the dataset. The exact format depends on the
        model and dataset, but typically includes probabilities or embeddings.
    """
    training_args = TrainingArguments(
        output_dir=tempfile.TemporaryDirectory().name,
        **kwargs,
    )
    trainer = Trainer(model=model, args=training_args)
    return trainer.predict(test_dataset=dataset).predictions
