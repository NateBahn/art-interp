# art-interp: What Does AI See in Art?

An interactive tool for exploring what vision models learn, using Sparse Autoencoders (SAEs) to decompose CLIP's neural activations into interpretable features.

**[Live Demo →](https://art-interp.example.com)** *(coming soon)*

![Screenshot of the web explorer](docs/screenshot.png)

## Features

- **Interactive Web Explorer** - Browse artworks and see which neural features activate
- **Feature Visualization** - Heatmaps showing WHERE in an image each feature fires
- **Correlation Analysis** - Discover which features predict aesthetic ratings
- **VLM-Generated Labels** - Human-readable descriptions of what each feature detects

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/art-interp.git
cd art-interp

# Install Python package
pip install -e .

# Install web dependencies
cd web && npm install && cd ..
```

### 2. Set Up Supabase

The web frontend connects directly to Supabase for data:

1. Create a free project at [supabase.com](https://supabase.com)
2. Run `database/schema.sql` in the Supabase SQL editor
3. Configure the frontend:

```bash
cp web/.env.example web/.env.local
# Edit web/.env.local with your VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY
```

### 3. Run the Web Explorer

```bash
cd web && npm run dev
```

Open http://localhost:5173 to explore!

---

## How It Works

### Sparse Autoencoders (SAEs)

Vision models like CLIP learn rich representations, but their internal features are entangled. SAEs decompose these into sparse, interpretable features:

```
Image → CLIP → 768-dim activation → SAE → 49,152 sparse features
                                              ↓
                                    Most are 0, a few are active
                                    Each active feature ≈ one visual concept
```

### What You Can Discover

| Feature | Correlation | What It Detects |
|---------|-------------|-----------------|
| #586 | +0.31 with "emotional_impact" | Crowd scenes, busy compositions |
| #1825 | +0.28 with "wholeness" | Centered subjects, balanced composition |
| #41202 | +0.25 with "inner_light" | Luminous skies, atmospheric light |

---

## Populating Your Database

Use the scripts to analyze your own artwork collection:

```bash
# Extract SAE features from images and upload to Supabase
python scripts/extract_sae_features.py --images ./my_paintings/

# Compute monosemanticity scores
python scripts/compute_monosemanticity.py

# Generate labels with Gemini (requires GOOGLE_API_KEY in .env)
python scripts/label_features_gemini.py
```

---

## Python Library

Use the interpretability package directly in your code:

```python
from PIL import Image
from src.interpretability.core import SAEFeatureExtractor

# Extract features from an image
extractor = SAEFeatureExtractor(layer=8)
image = Image.open("painting.jpg")
result = extractor.extract_from_image(image)

print(f"Active features: {result.num_active}")
print(f"Top feature: #{result.top_indices[0]} = {result.top_values[0]:.2f}")
```

### Multi-Layer Analysis

Different CLIP layers capture different abstractions:

| Layer | What It Captures | Example Features |
|-------|------------------|------------------|
| 7 | Textures, patterns | "brushstroke texture", "geometric grid" |
| 8 | Composition, shapes | "centered subject", "diagonal lines" |
| 11 | Semantics, concepts | "portrait", "landscape", "abstract" |

---

## Project Structure

```
art-interp/
├── src/interpretability/     # Core Python library
│   ├── core/                 # SAE feature extraction
│   ├── analysis/             # Monosemanticity, correlations
│   └── labeling/             # Gemini VLM integration
├── web/                      # React frontend (connects to Supabase)
│   └── src/
├── scripts/                  # Analysis pipelines
├── database/                 # SQL schema for Supabase
└── data/                     # Sample heatmaps
```

---

## Requirements

- Python 3.11+
- Node.js 18+ (for web frontend)
- Supabase account (free tier works)
- ~2GB disk for SAE weights (downloaded on first use)
- GPU recommended for feature extraction (works on CPU, but slower)

---

## Research Background

This toolkit implements methods from:

- **Sparse Autoencoders**: [Cunningham et al., 2023](https://arxiv.org/abs/2309.08600)
- **SAE-for-VLM**: [NeurIPS 2024](https://arxiv.org/abs/2402.00838)
- **Pre-trained SAEs**: [Prisma-Multimodal](https://huggingface.co/Prisma-Multimodal) on HuggingFace

---

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
