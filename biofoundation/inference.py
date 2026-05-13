import datasets
import numpy as np
import tempfile
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from typing import Any, Callable, Literal
from functools import partial

from biofoundation.model.base import Tokenizer

from biofoundation.data import (
    Genome,
    transform_reflogprob_mlm,
    transform_reflogprob_clm,
    transform_llr_mlm,
    transform_llr_clm,
    transform_ll_clm,
)
from biofoundation.model.scoring import (
    compute_reflogprob_mlm,
    compute_reflogprob_clm,
    compute_llr_mlm,
    compute_llr_clm,
    compute_ll_clm,
    compute_euclidean_distance,
    compute_llr_and_embedding_distance,
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


def _run_strand_aware(
    model: nn.Module,
    tokenizer: Tokenizer,
    dataset: datasets.Dataset,
    *,
    compute_fn: Callable[..., Any],
    transform_fn: Callable[..., dict[str, Any]],
    transform_kwargs: dict[str, Any] | None = None,
    rc_avg: bool = False,
    **kwargs: Any,
) -> Any:
    """Run inference once (FWD) or twice (FWD + RC) and average.

    When ``rc_avg=True``, calls ``run_inference`` with ``strand="+"`` and then
    ``strand="-"`` (bound into ``transform_fn`` via ``partial``) and returns
    the element-wise mean of the two predictions arrays — element-wise so it
    works for shape ``[N]`` (e.g. ``run_llr_clm``) and ``[N, 3]`` (e.g.
    ``run_llr_and_embedding_distance``).
    """

    def _one(strand: Literal["+", "-"]) -> Any:
        return run_inference(
            model,
            tokenizer,
            dataset,
            compute_fn=compute_fn,
            data_transform_fn=partial(
                transform_fn, strand=strand, **(transform_kwargs or {})
            ),
            **kwargs,
        )

    fwd = _one("+")
    if not rc_avg:
        return fwd
    rc = _one("-")
    return (np.asarray(fwd) + np.asarray(rc)) / 2


run_ll_clm = partial(
    run_inference,
    compute_fn=compute_ll_clm,
    data_transform_fn=transform_ll_clm,
)


def run_reflogprob_mlm(
    model: nn.Module,
    tokenizer: Tokenizer,
    dataset: datasets.Dataset,
    rc_avg: bool = False,
    **kwargs: Any,
) -> Any:
    return _run_strand_aware(
        model,
        tokenizer,
        dataset,
        compute_fn=compute_reflogprob_mlm,
        transform_fn=transform_reflogprob_mlm,
        rc_avg=rc_avg,
        **kwargs,
    )


def run_reflogprob_clm(
    model: nn.Module,
    tokenizer: Tokenizer,
    dataset: datasets.Dataset,
    rc_avg: bool = False,
    **kwargs: Any,
) -> Any:
    return _run_strand_aware(
        model,
        tokenizer,
        dataset,
        compute_fn=compute_reflogprob_clm,
        transform_fn=transform_reflogprob_clm,
        rc_avg=rc_avg,
        **kwargs,
    )


def run_llr_mlm(
    model: nn.Module,
    tokenizer: Tokenizer,
    dataset: datasets.Dataset,
    genome: Genome,
    window_size: int,
    rc_avg: bool = False,
    **kwargs: Any,
) -> Any:
    return _run_strand_aware(
        model,
        tokenizer,
        dataset,
        compute_fn=compute_llr_mlm,
        transform_fn=transform_llr_mlm,
        transform_kwargs=dict(genome=genome, window_size=window_size),
        rc_avg=rc_avg,
        **kwargs,
    )


def run_llr_clm(
    model: nn.Module,
    tokenizer: Tokenizer,
    dataset: datasets.Dataset,
    genome: Genome,
    window_size: int,
    rc_avg: bool = False,
    **kwargs: Any,
) -> Any:
    return _run_strand_aware(
        model,
        tokenizer,
        dataset,
        compute_fn=compute_llr_clm,
        transform_fn=transform_llr_clm,
        transform_kwargs=dict(genome=genome, window_size=window_size),
        rc_avg=rc_avg,
        **kwargs,
    )


def run_euclidean_distance(
    model: nn.Module,
    tokenizer: Tokenizer,
    dataset: datasets.Dataset,
    genome: Genome,
    window_size: int,
    rc_avg: bool = False,
    **kwargs: Any,
) -> Any:
    return _run_strand_aware(
        model,
        tokenizer,
        dataset,
        compute_fn=compute_euclidean_distance,
        transform_fn=transform_llr_clm,
        transform_kwargs=dict(genome=genome, window_size=window_size),
        rc_avg=rc_avg,
        **kwargs,
    )


def run_llr_and_embedding_distance(
    model: nn.Module,
    tokenizer: Tokenizer,
    dataset: datasets.Dataset,
    genome: Genome,
    window_size: int,
    rc_avg: bool = False,
    **kwargs: Any,
) -> Any:
    """Run combined LLR and embedding distance inference.

    Computes LLR, last-layer embedding distance, and middle-layer embedding distance
    in a single forward pass.

    Args:
        model: CausalLMWithEmbeddings model
        tokenizer: Tokenizer for the model
        dataset: Dataset with variant information (chrom, pos, ref, alt)
        genome: Genome object for sequence extraction
        window_size: Window size for sequence context
        rc_avg: If True, also score the reverse-complemented window for
            each variant and return the element-wise average of FWD and
            RC predictions (shape ``[N, 3]``). Doubles inference cost.
        **kwargs: Additional arguments passed to run_inference

    Returns:
        Numpy array with shape [B, 3] where columns are:
            - [:, 0]: LLR (log-likelihood ratio)
            - [:, 1]: Last-layer embedding distance
            - [:, 2]: Middle-layer embedding distance
    """
    return _run_strand_aware(
        model,
        tokenizer,
        dataset,
        compute_fn=compute_llr_and_embedding_distance,
        transform_fn=transform_llr_clm,
        transform_kwargs=dict(genome=genome, window_size=window_size),
        rc_avg=rc_avg,
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
