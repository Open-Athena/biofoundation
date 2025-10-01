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
# PlantCad1
uv pip install -e .[mamba]
```

## Usage

Run the example script:

```bash
source .venv/bin/activate
python examples/marin_evolutionary_constraint.py
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
