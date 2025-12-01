# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

BioFoundation is an ecosystem for biological foundation models. It provides a unified interface for working with various genomic foundation models (HuggingFace models, GPN, Evo2, PlantCaduceus, etc.) through adapters, along with utilities for genomic data transformation and inference.

## Development Commands

### Environment Setup

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install PyTorch first (required)
uv pip install torch

# Install base package
uv pip install -e .

# Install optional dependencies for specific models
uv pip install -e .[mamba]  # For PlantCaduceus
uv pip install -e .[gpn]    # For GPN
uv pip install -e .[glm-experiments]  # For GLM-Experiments (installs Lightning/Hydra)
# Then manually install glm-experiments to avoid circular dependency:
uv pip install git+https://github.com/Open-Athena/glm-experiments

# Install development dependencies
uv pip install --group dev

# Install pre-commit hooks
pre-commit install
```

### Testing and Quality Checks

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_data.py

# Run a specific test function
pytest tests/test_data.py::test_function_name

# Run pre-commit hooks (linting, formatting, type checking)
pre-commit run

# Run pre-commit on all files
pre-commit run --all-files

# Manual quality checks
ruff check .                    # Linting
ruff format .                   # Formatting
mypy biofoundation              # Type checking (excludes tests/ and examples/)
```

### Running Examples

```bash
# Light job example
python examples/marin_evolutionary_constraint.py

# Heavy job with distributed training (8 GPUs, 8 CPU threads per GPU)
OMP_NUM_THREADS=8 torchrun --nproc_per_node=8 examples/plantcad_llr.py
```

## Architecture

### Core Abstractions

The package is built around three key base classes in `biofoundation/model/base.py`:

1. **CausalLM**: Abstract base class for causal language models (autoregressive)
2. **MaskedLM**: Abstract base class for masked language models (BERT-style)
3. **Tokenizer**: Minimal tokenizer interface that returns Python lists of token IDs

All models are wrapped through adapters that implement these interfaces.

### Model Adapters

Located in `biofoundation/model/adapters/`, adapters wrap different model types:

- **HuggingFace models** (`hf.py`): Wraps any HF `PreTrainedModel` ending in `MaskedLM` or `CausalLM`
  - `HFMaskedLM`, `HFCausalLM`, `HFTokenizer`
- **GPN models** (`gpn.py`): Specialized adapter for GPN (Genomic Pre-trained Network)
  - `GPNMaskedLM`
- **Evo2 models** (`evo2.py`): Adapter for Evo2 models (requires Docker, see `docker/evo2/README.md`)
  - `Evo2CausalLM`, `Evo2Tokenizer`
- **GLM-Experiments models** (`glm_experiments.py`): Adapter for glm-experiments Lightning modules
  - `GLMExperimentsMaskedLM`, `GLMExperimentsCausalLM`
  - Requires loading checkpoint with PyTorch Lightning first
  - Uses HF tokenizers via `HFTokenizer`

### Data Layer

`biofoundation/data.py` provides genomic data utilities:

- **Genome**: FASTA file reader with random access to genomic sequences
  - Supports N-padding for out-of-bounds coordinates
  - Handles reverse complement for negative strand
  - Usage: `genome(chrom, start, end, strand="+")`

- **GenomicSet**: Non-overlapping genomic intervals with set operations
  - Automatically merges overlapping intervals
  - Supports union (`|`), intersection (`&`), subtraction (`-`)
  - Methods: `expand_min_size()`, `add_random_shift()`, `to_pandas()`

- **Transform functions**: Prepare data for different scoring tasks
  - `transform_llr_mlm/clm`: Log-likelihood ratio for masked/causal LMs
  - `transform_reflogprob_mlm/clm`: Reference log probability scoring
  - All transforms follow VCF semantics (1-based positions)

### Scoring and Inference

`biofoundation/model/scoring.py` implements scoring functions:

- **LLR (Log-Likelihood Ratio)**: Compare ref vs alt allele probabilities
  - `compute_llr_mlm()`: For masked language models
  - `compute_llr_clm()`: For causal language models

- **Reference Log Probability**: Evolutionary constraint scoring
  - `compute_reflogprob_mlm()`: For masked language models
  - `compute_reflogprob_clm()`: For causal language models (marginal probability over 4 nucleotides)

`biofoundation/inference.py` provides high-level inference functions:

- Uses HuggingFace Trainer for batched inference
- Convenience functions: `run_llr_mlm()`, `run_llr_clm()`, `run_reflogprob_mlm()`, `run_reflogprob_clm()`
- Supports on-the-fly data transformation via `data_transform_on_the_fly=True`

### Coordinate Systems

- **VCF semantics**: 1-based, inclusive positions (used in transform functions)
- **BED/Python semantics**: 0-based, half-open intervals [start, end) (used in Genome and GenomicSet)

Example: A variant at VCF position 1000 corresponds to Python index 999 (0-based)

## Type Checking with jaxtyping

The codebase uses `jaxtyping` for tensor shape annotations:
- `Int[Tensor, "B L"]`: Integer tensor with shape (batch, length)
- `Float[Tensor, "B L V"]`: Float tensor with shape (batch, length, vocab)

Ruff is configured to ignore F722 (syntax error in forward annotation) for jaxtyping compatibility.

## Example Workflow Pattern

See `examples/` for complete examples. Typical workflow:

1. Load model and tokenizer via adapter
2. Load genome reference (if needed for LLR scoring)
3. Load dataset from HuggingFace datasets
4. Run inference with appropriate scoring function
5. Compute metrics (pearson/spearman correlation, ROC-AUC, etc.)

Example for MLM LLR scoring:
```python
from biofoundation.model.adapters.hf import HFMaskedLM, HFTokenizer
from biofoundation.data import Genome
from biofoundation.inference import run_llr_mlm

tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(model_name))
model = HFMaskedLM(AutoModelForMaskedLM.from_pretrained(model_name))
genome = Genome(genome_path)

llr = run_llr_mlm(model, tokenizer, dataset, genome, window_size=8192)
```

## Distributed Training

For multi-GPU inference (e.g., `plantcad_llr.py`):
- Use `torchrun --nproc_per_node=N` to launch across N GPUs
- Set `OMP_NUM_THREADS` to control CPU threads per GPU
- Evo2 models require special GPU visibility handling (see `examples/evo2_llr.py`)
