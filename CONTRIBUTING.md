# Contributing to art-interp

Thank you for your interest in contributing to art-interp! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment and install dependencies:

```bash
cd art-interp
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

## Development Workflow

### Running Tests

```bash
make test
```

### Code Quality

Before submitting a PR, ensure your code passes all checks:

```bash
make check  # Runs lint, typecheck, and test
```

To auto-format code:

```bash
make format
```

### Making Changes

1. Create a new branch for your feature/fix
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation as needed
5. Submit a pull request

## Code Style

- We use `ruff` for linting and formatting
- We use `mypy` for type checking
- All public functions should have docstrings
- Type hints are required for all function signatures

## Project Structure

```
art-interp/
├── src/interpretability/    # Core Python package
│   ├── core/                # SAE feature extraction
│   ├── analysis/            # Analysis algorithms
│   ├── storage/             # Data access protocols
│   └── labeling/            # VLM-based labeling
├── server/                  # FastAPI server
├── web/                     # React frontend
├── scripts/                 # Analysis pipeline
└── tests/                   # Test suite
```

## Key Patterns

### Protocol-Based Design

Use protocols for data access to keep code modular:

```python
from interpretability.storage.protocols import ImageProvider

def my_analysis(provider: ImageProvider):
    for item in provider.iter_images():
        process(item)
```

### Independent Embeddings for Monosemanticity

When computing monosemanticity scores, always use an embedding model independent of the one used for SAE training (e.g., DINOv2 for CLIP SAEs).

## Reporting Issues

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or error messages

## Pull Request Process

1. Update the README.md if needed
2. Update ARCHITECTURE.md for architectural changes
3. Add tests for new functionality
4. Ensure CI passes
5. Request review from maintainers

## Questions?

Feel free to open an issue for questions or discussions about the project.
