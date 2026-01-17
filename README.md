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

# Install Python package with server dependencies
pip install -e ".[server]"

# Install web dependencies
cd web && npm install && cd ..
```

### 2. Add Your Artworks

```bash
# Option A: From a folder of images
python scripts/populate_database.py --images ./my_paintings/

# Option B: From a CSV with metadata
python scripts/populate_database.py --csv artworks.csv
```

This creates a local SQLite database and extracts SAE features from each image.

<details>
<summary>CSV format example</summary>

```csv
image_path,title,artist,year
./images/starry_night.jpg,The Starry Night,Vincent van Gogh,1889
./images/mona_lisa.jpg,Mona Lisa,Leonardo da Vinci,1503
./images/guernica.jpg,Guernica,Pablo Picasso,1937
```

Only `image_path` is required. Other columns are optional.
</details>

### 3. Run the Web Explorer

```bash
# Start the API server
uvicorn server.main:app --port 8001 &

# Start the web frontend
cd web && npm run dev
```

Open http://localhost:5173 to explore your artworks!

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

## Configuration

### Local SQLite (Default)

No configuration needed! The database is created automatically at `./data/art_interp.db`.

### Supabase (Production)

For a hosted database with the web frontend connecting directly:

1. Create a project at [supabase.com](https://supabase.com)
2. Run `server/database/schema.sql` in the SQL editor
3. Create config files:

```bash
# Root .env (for Python server)
cp .env.example .env
# Edit with your DATABASE_URL

# Web .env.local (for frontend)
cp web/.env.example web/.env.local
# Edit with your VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY
```

---

## Advanced Usage

### Python Library

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

### Analysis Pipeline

For deeper analysis, use the scripts in `scripts/`:

```bash
# Compute monosemanticity scores (which features are interpretable?)
python scripts/compute_monosemanticity.py

# Correlate with ratings (requires rating data)
python scripts/compute_spatial_correlations.py

# Generate labels with Gemini (requires GOOGLE_API_KEY in .env)
python scripts/label_features_gemini.py

# Import labels to database
python scripts/import_feature_labels_to_db.py
```

### Multi-Layer Analysis

Different CLIP layers capture different abstractions:

| Layer | What It Captures | Example Features |
|-------|------------------|------------------|
| 7 | Textures, patterns | "brushstroke texture", "geometric grid" |
| 8 | Composition, shapes | "centered subject", "diagonal lines" |
| 11 | Semantics, concepts | "portrait", "landscape", "abstract" |

```bash
# Extract features from layer 11 instead of default layer 8
python scripts/populate_database.py --images ./art/ --layer 11
```

---

## Project Structure

```
art-interp/
├── src/interpretability/     # Core Python library
│   ├── core/                 # SAE feature extraction
│   ├── analysis/             # Monosemanticity, correlations
│   └── labeling/             # Gemini VLM integration
├── server/                   # FastAPI server
│   ├── main.py               # API endpoints
│   └── database/             # SQLAlchemy models
├── web/                      # React frontend
│   └── src/
├── scripts/                  # Analysis pipelines
│   ├── populate_database.py  # Main setup script
│   ├── compute_monosemanticity.py
│   └── label_features_gemini.py
└── data/                     # Local data storage
```

---

## Requirements

- Python 3.11+
- Node.js 18+ (for web frontend)
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
