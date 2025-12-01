![BioFoundation Logo](docs/source/_static/logo.png)

# BioFoundation

An ecosystem for biological foundation models.

## Installation

Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Pre-requisites:
- Pytorch (e.g. `uv pip install torch`)

Install the package:

```bash
uv pip install -e .
```

## Additional Dependencies

```bash
# PlantCaduceus
uv pip install -e .[mamba]
# GPN
uv pip install -e .[gpn]
# GLM-Experiments
uv pip install -e .[glm-experiments]
# Then manually install glm-experiments to avoid circular dependency:
uv pip install git+https://github.com/Open-Athena/glm-experiments
```

### Docker Installation (For Evo2)

For evo2 support, Docker is recommended as it handles all complex CUDA/cuDNN dependencies automatically.

See [`docker/evo2/README.md`](docker/evo2/README.md) for detailed instructions.

**Quick start:**

```bash
cd docker/evo2
./run.sh local  # or ./run.sh github
```

## Usage

Run the example scripts:

```bash
source .venv/bin/activate
# light job
python examples/marin_evolutionary_constraint.py
# heavier job, split across 8 GPUs with 8 CPU threads per GPU
OMP_NUM_THREADS=8 torchrun --nproc_per_node=8 examples/plantcad_llr.py
```

## Development Setup

To set up the development environment with linting, formatting, type checking, and testing:

```bash
# Install development dependencies
uv pip install --group dev
# Install pre-commit hooks
pre-commit install
```

## Development Tools

### Running Quality Checks

```bash
# Run all pre-commit hooks (linting, formatting, type checking)
pre-commit run
```

### Running Tests

```bash
# Run all tests
pytest
```
