import datasets
import tempfile
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from typing import Any, Callable
from functools import partial

from biofoundation.model.base import Tokenizer

from biofoundation.data import (
    Genome,
    transform_reflogprob_mlm,
    transform_reflogprob_clm,
    transform_llr_mlm,
    transform_llr_clm,
)
from biofoundation.model.scoring import (
    compute_reflogprob_mlm,
    compute_reflogprob_clm,
    compute_llr_mlm,
    compute_llr_clm,
)


def run_inference(
    model: nn.Module,
    tokenizer: Tokenizer,
    dataset: datasets.Dataset,
    compute_fn: Callable[..., Any],
    data_transform_fn: Callable[..., dict[str, Any]] | None = None,
    data_transform_on_the_fly: bool = False,
    data_transform_kwargs: dict[str, Any] | None = None,
    inference_kwargs: dict[str, Any] | None = None,
) -> Any:
    processed_dataset = _process_dataset(
        dataset,
        tokenizer,
        data_transform_fn,
        data_transform_on_the_fly,
        data_transform_kwargs,
    )
    return _run_inference(
        _ModelComputeFnWrapper(model, compute_fn),
        processed_dataset,
        **(inference_kwargs or {}),
    )


run_reflogprob_mlm = partial(
    run_inference,
    compute_fn=compute_reflogprob_mlm,
    data_transform_fn=transform_reflogprob_mlm,
)

run_reflogprob_clm = partial(
    run_inference,
    compute_fn=compute_reflogprob_clm,
    data_transform_fn=transform_reflogprob_clm,
)


def run_llr_mlm(
    model: nn.Module,
    tokenizer: Tokenizer,
    dataset: datasets.Dataset,
    genome: Genome,
    window_size: int,
    **kwargs: Any,
) -> Any:
    return run_inference(
        model,
        tokenizer,
        dataset,
        compute_fn=compute_llr_mlm,
        data_transform_fn=partial(
            transform_llr_mlm, genome=genome, window_size=window_size
        ),
        **kwargs,
    )


def run_llr_clm(
    model: nn.Module,
    tokenizer: Tokenizer,
    dataset: datasets.Dataset,
    genome: Genome,
    window_size: int,
    **kwargs: Any,
) -> Any:
    return run_inference(
        model,
        tokenizer,
        dataset,
        compute_fn=compute_llr_clm,
        data_transform_fn=partial(
            transform_llr_clm, genome=genome, window_size=window_size
        ),
        **kwargs,
    )


def _run_inference(
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
        **(kwargs or {}),
    )
    trainer = Trainer(model=model, args=training_args)
    return trainer.predict(test_dataset=dataset).predictions


class _ModelComputeFnWrapper(nn.Module):
    def __init__(self, model: nn.Module, compute_fn: Callable[..., Any]):
        super().__init__()
        self.model = model
        self.compute_fn = compute_fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.compute_fn(self.model, *args, **kwargs)


def _process_dataset(
    dataset: datasets.Dataset,
    tokenizer: Tokenizer,
    data_transform_fn: Callable[..., dict[str, Any]] | None = None,
    data_transform_on_the_fly: bool = False,
    data_transform_kwargs: dict[str, Any] | None = None,
) -> datasets.Dataset:
    if data_transform_fn is None:
        return dataset
    if data_transform_kwargs is None:
        data_transform_kwargs = {}
    data_transform_fn = partial(data_transform_fn, tokenizer=tokenizer)
    if data_transform_on_the_fly:
        return dataset.with_transform(
            _make_batch_transform(data_transform_fn),
            **data_transform_kwargs,
        )
    return dataset.map(
        data_transform_fn,
        **data_transform_kwargs,
    )


def _make_batch_transform(
    transform_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, list[Any]]], dict[str, list[Any]]]:
    def batch_transform_fn(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # Convert batch format to list of examples
        examples = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        # Apply transform to each example
        transformed_examples = [transform_fn(example) for example in examples]
        # Convert back to batch format
        return {
            key: [ex[key] for ex in transformed_examples]
            for key in transformed_examples[0].keys()
        }

    return batch_transform_fn
