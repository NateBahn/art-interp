# CLAUDE.md - art-interp

Instructions for AI assistants working on this repository.

## Project Overview

art-interp is an open-source toolkit for understanding what vision models learn about art, using Sparse Autoencoders (SAEs) to extract interpretable features from CLIP ViT-B-32.

## Repository Structure

```
art-interp/
├── src/interpretability/    # Core Python package
│   ├── core/                # SAEFeatureExtractor, configs, types
│   ├── analysis/            # Monosemanticity scoring, correlations
│   ├── storage/             # Protocol-based data access
│   └── labeling/            # Gemini VLM labeling
├── web/                     # React frontend (connects directly to Supabase)
│   └── src/                 # TypeScript source
├── scripts/                 # Analysis pipeline scripts
├── database/                # SQL schema for Supabase
├── data/                    # Sample data and outputs
└── tests/                   # Test suite
```

## Key Commands

```bash
# Install in development mode
pip install -e ".[all]"

# Run tests
make test

# Format and lint
make format
make lint

# Start web frontend
cd web && npm run dev
```

## Architecture

The web frontend connects **directly to Supabase** - there is no backend server. Data flow:

```
Web Frontend (React) → Supabase JS Client → Supabase PostgreSQL
```

Python scripts also connect directly to Supabase for data population.

## Code Patterns

### Protocol-Based Data Access

Never access data directly. Use protocols from `storage/protocols.py`:

```python
from interpretability.storage.protocols import ImageProvider, FeatureStore

def my_function(provider: ImageProvider):
    for item in provider.iter_images():
        process(item)
```

### Feature Extraction

```python
from interpretability.core import SAEFeatureExtractor

extractor = SAEFeatureExtractor(layer=8)
result = extractor.extract_from_image(pil_image)
```

### Monosemanticity Scoring

Always use an INDEPENDENT embedding model (DINOv2, not CLIP) to avoid circular reasoning:

```python
from interpretability.analysis import MonosemanticityScorer

# embedding_provider should provide DINOv2 embeddings, NOT CLIP
scorer = MonosemanticityScorer(embedding_provider)
```

## Testing Guidelines

- Tests use mock providers from `tests/mocks/`
- Don't load real ML models in tests - mock the extractor
- Use `SampleImageProvider` for integration tests

## Working on Components

### interpretability package (src/interpretability/)
Core algorithms. Changes here affect everything. Test thoroughly.

### web (web/)
React/TypeScript frontend. Connects directly to Supabase.
Run `npm run dev` to start development server.

### scripts (scripts/)
Analysis pipeline. Run these to generate analysis outputs and populate Supabase.

## File Dependencies

| File | Impact |
|------|--------|
| `src/interpretability/core/sae_extractor.py` | Core extraction - test thoroughly |
| `src/interpretability/storage/protocols.py` | All data access goes through here |
| `database/schema.sql` | Supabase database schema |

## Adding New Features

1. Follow existing patterns in the relevant module
2. Add type hints to all public functions
3. Write docstrings with examples
4. Update `__init__.py` exports if adding public APIs
5. Add tests in `tests/`

## Common Issues

### Import Errors
The package uses `src/` layout. Install with `pip install -e .` before importing.

### GPU Memory
SAE extraction loads ~2GB of weights. Use `device="cpu"` for testing if needed.

### Monosemanticity Requires Independent Embeddings
Using CLIP embeddings for monosemanticity scoring is methodologically invalid. Always use DINOv2 or another independent model.
