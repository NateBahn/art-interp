#!/usr/bin/env python3
"""
Text2Concept SAE Feature Probing

Uses CLIP's text encoder to probe which SAE features correspond to specific
visual/aesthetic concepts. This is a zero-shot approach that requires no training.

The method:
1. Get CLIP text embedding for a concept (e.g., "baroque painting with dramatic lighting")
2. For each artwork, compute similarity between its CLIP embedding and the concept
3. Correlate these "concept scores" with each SAE feature activation across artworks
4. High correlation = that SAE feature detects that concept

Usage:
    python scripts/text2concept_probe.py
"""

import asyncio
import json
import logging
import sys
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import open_clip
import torch
from PIL import Image
from scipy import stats
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import get_db_context
from database.models import Artwork
from services.sae_features import load_features_batch, SAE_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
FEATURES_DIR = Path(__file__).parent.parent / "data" / "sae_features"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "sae_analysis"
CLIP_CACHE_FILE = Path(__file__).parent.parent / "data" / "clip_b32_embeddings.npz"

# Art concepts to probe, organized by category
CONCEPTS = {
    "style": [
        "baroque painting with dramatic lighting and rich colors",
        "impressionist painting with loose brushwork and soft colors",
        "renaissance painting with linear perspective and classical composition",
        "abstract geometric painting with bold shapes",
        "romantic landscape painting with emotional atmosphere",
        "medieval religious painting with gold leaf",
        "dutch golden age painting with realistic detail",
        "expressionist painting with distorted forms and intense colors",
    ],
    "color": [
        "painting with warm orange red and golden colors",
        "painting with cool blue green and silver tones",
        "high contrast black and white monochrome image",
        "painting with vibrant saturated colors",
        "painting with muted earthy brown tones",
        "painting with pastel soft colors",
    ],
    "composition": [
        "symmetrically composed painting with centered subject",
        "painting with dynamic diagonal composition",
        "painting with rule of thirds composition",
        "crowded multi-figure composition",
        "minimalist composition with empty space",
        "painting with strong geometric structure",
    ],
    "subject": [
        "portrait painting of a person face",
        "landscape painting of nature scenery",
        "religious painting with biblical scene",
        "still life painting with flowers fruit or objects",
        "mythological painting with gods and heroes",
        "genre scene of everyday life",
        "historical painting depicting an event",
        "animal painting",
    ],
    "technique": [
        "painting with fine detailed brushwork",
        "painting with thick impasto texture",
        "painting with soft sfumato edges",
        "painting with strong chiaroscuro lighting",
        "painting with visible bold brushstrokes",
        "highly polished smooth painting surface",
    ],
    "emotion_mood": [
        "serene peaceful calm painting",
        "dramatic intense emotional painting",
        "melancholic sad somber painting",
        "joyful happy celebratory painting",
        "mysterious enigmatic atmospheric painting",
        "violent turbulent chaotic painting",
    ],
    "alexander_qualities": [
        "painting with sense of wholeness and unity",
        "painting with inner light and luminosity",
        "painting that feels alive and organic",
        "painting with deep structural coherence",
        "painting with peaceful grounded presence",
    ],
}


def load_clip_model(device: str = None):
    """Load CLIP model and tokenizer."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info(f"Loading CLIP ViT-B-32 on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="datacomp_xl_s13b_b90k"
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return model, tokenizer, preprocess, device


def get_text_embedding(text: str, model, tokenizer, device: str) -> np.ndarray:
    """Get CLIP text embedding for a concept."""
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
    return embedding.cpu().numpy().squeeze()


def download_image(url: str, timeout: float = 30.0) -> Image.Image | None:
    """Download an image from URL."""
    try:
        headers = {"User-Agent": "ArtRecommender/1.0 (Research)"}
        with httpx.Client(headers=headers, follow_redirects=True) as client:
            response = client.get(url, timeout=timeout)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return None


def compute_clip_embeddings(
    artwork_ids: list[str],
    model,
    preprocess,
    device: str,
) -> dict[str, np.ndarray]:
    """Compute CLIP ViT-B-32 embeddings for artworks (with caching)."""

    # Check cache first
    if CLIP_CACHE_FILE.exists():
        logger.info(f"Loading cached embeddings from {CLIP_CACHE_FILE}")
        cached = np.load(CLIP_CACHE_FILE, allow_pickle=True)
        embeddings = {k: cached[k] for k in cached.files}

        # Check if we have all needed embeddings
        missing_ids = set(artwork_ids) - set(embeddings.keys())
        if not missing_ids:
            logger.info(f"All {len(embeddings)} embeddings found in cache")
            return embeddings
        logger.info(f"Cache has {len(embeddings)} embeddings, need {len(missing_ids)} more")
    else:
        embeddings = {}
        missing_ids = set(artwork_ids)

    # Get artwork URLs for missing IDs
    artwork_urls = {}
    with get_db_context() as db:
        artworks = db.query(Artwork).filter(Artwork.id.in_(list(missing_ids))).all()
        for artwork in artworks:
            if artwork.image_url:
                artwork_urls[artwork.id] = artwork.image_url

    logger.info(f"Computing embeddings for {len(artwork_urls)} artworks...")

    # Compute embeddings for missing artworks
    for artwork_id in tqdm(list(artwork_urls.keys()), desc="Computing CLIP embeddings"):
        url = artwork_urls[artwork_id]
        image = download_image(url)
        if image is None:
            continue

        try:
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embeddings[artwork_id] = embedding.cpu().numpy().squeeze()
        except Exception as e:
            logger.warning(f"Failed to encode {artwork_id}: {e}")

    # Save updated cache
    CLIP_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CLIP_CACHE_FILE, **embeddings)
    logger.info(f"Saved {len(embeddings)} embeddings to cache")

    return embeddings


def load_artwork_embeddings(model, preprocess, device: str, sae_artwork_ids: set[str]) -> dict[str, np.ndarray]:
    """Load or compute CLIP ViT-B-32 embeddings for artworks with SAE features."""
    return compute_clip_embeddings(list(sae_artwork_ids), model, preprocess, device)


def load_all_sae_features() -> dict[str, dict]:
    """Load SAE features from batch files as sparse dicts."""
    all_features = {}
    for batch_file in sorted(FEATURES_DIR.glob("batch_*.json")):
        with open(batch_file) as f:
            batch_data = json.load(f)
            all_features.update(batch_data)

    logger.info(f"Loaded SAE features for {len(all_features)} artworks")
    return all_features


def compute_concept_artwork_scores(
    concept_embedding: np.ndarray,
    artwork_embeddings: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute similarity between concept and each artwork."""
    scores = {}
    for artwork_id, artwork_emb in artwork_embeddings.items():
        # Cosine similarity (both are normalized)
        score = float(np.dot(concept_embedding, artwork_emb))
        scores[artwork_id] = score
    return scores


def correlate_concept_with_sae_features(
    concept_scores: dict[str, float],
    sae_features: dict[str, dict],
    top_k: int = 20,
) -> list[dict]:
    """
    Find SAE features that correlate with concept scores.

    For each SAE feature, compute Spearman correlation between:
    - Feature activations across artworks
    - Concept scores across artworks

    Returns top-K features with highest absolute correlation.
    """
    # Get common artwork IDs
    common_ids = set(concept_scores.keys()) & set(sae_features.keys())
    if len(common_ids) < 100:
        logger.warning(f"Only {len(common_ids)} common artworks, need more data")
        return []

    artwork_ids = sorted(common_ids)

    # Build concept score array
    concept_arr = np.array([concept_scores[aid] for aid in artwork_ids])

    # Track correlations for each feature
    feature_correlations = []

    # Check each feature
    for feat_idx in range(SAE_DIM):
        # Get feature activations (sparse, so many will be 0)
        feat_arr = np.array([
            sae_features[aid].get(str(feat_idx), 0)
            for aid in artwork_ids
        ])

        # Skip features that are nearly constant (all zeros or same value)
        if np.std(feat_arr) < 1e-6:
            continue

        # Compute Spearman correlation
        corr, p_value = stats.spearmanr(concept_arr, feat_arr)

        if not np.isnan(corr):
            feature_correlations.append({
                "feature_idx": feat_idx,
                "correlation": float(corr),
                "p_value": float(p_value),
                "num_nonzero": int(np.sum(feat_arr > 0)),
            })

    # Sort by absolute correlation
    feature_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return feature_correlations[:top_k]


def get_top_artworks_for_concept(
    concept_scores: dict[str, float],
    top_k: int = 10,
) -> list[str]:
    """Get artwork IDs with highest concept scores."""
    sorted_artworks = sorted(
        concept_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return [aid for aid, _ in sorted_artworks[:top_k]]


def main():
    """Run Text2Concept probing on all defined concepts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load models and data
    model, tokenizer, preprocess, device = load_clip_model()
    sae_features = load_all_sae_features()

    # Get artwork embeddings (using same CLIP model for consistency)
    sae_artwork_ids = set(sae_features.keys())
    artwork_embeddings = load_artwork_embeddings(model, preprocess, device, sae_artwork_ids)

    # Results storage
    all_results = {
        "metadata": {
            "num_artworks": len(artwork_embeddings),
            "num_artworks_with_sae": len(sae_features),
            "sae_dim": SAE_DIM,
        },
        "concepts": {},
    }

    # Process each concept
    total_concepts = sum(len(v) for v in CONCEPTS.values())
    processed = 0

    for category, concepts in CONCEPTS.items():
        logger.info(f"\n=== Processing category: {category} ===")
        all_results["concepts"][category] = {}

        for concept in concepts:
            processed += 1
            logger.info(f"[{processed}/{total_concepts}] {concept[:50]}...")

            # Get concept embedding
            concept_emb = get_text_embedding(concept, model, tokenizer, device)

            # Score artworks by concept similarity
            concept_scores = compute_concept_artwork_scores(concept_emb, artwork_embeddings)

            # Find correlating SAE features
            top_features = correlate_concept_with_sae_features(
                concept_scores, sae_features, top_k=20
            )

            # Get top artworks for this concept
            top_artworks = get_top_artworks_for_concept(concept_scores, top_k=10)

            # Store results
            result = {
                "concept": concept,
                "top_features": top_features[:10],  # Top 10 for storage
                "top_artworks": top_artworks,
                "feature_summary": {
                    "positive": [f for f in top_features if f["correlation"] > 0][:5],
                    "negative": [f for f in top_features if f["correlation"] < 0][:5],
                },
            }
            all_results["concepts"][category][concept] = result

            # Print summary
            if top_features:
                top = top_features[0]
                print(f"  Top feature: {top['feature_idx']} (r={top['correlation']:.3f})")

    # Save results
    output_file = OUTPUT_DIR / "concept_feature_mapping.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")

    # Generate summary report
    generate_summary_report(all_results, OUTPUT_DIR / "concept_probe_report.md")

    return all_results


def generate_summary_report(results: dict, output_path: Path):
    """Generate markdown summary of concept probing results."""
    lines = [
        "# Text2Concept SAE Feature Probing Results\n",
        f"Analyzed {results['metadata']['num_artworks']} artworks with SAE features.\n",
        "## Top Features by Concept\n",
    ]

    for category, concepts in results["concepts"].items():
        lines.append(f"### {category.replace('_', ' ').title()}\n")
        lines.append("| Concept | Top Feature | Correlation | # Active |")
        lines.append("|---------|-------------|-------------|----------|")

        for concept_text, data in concepts.items():
            top_feats = data.get("top_features", [])
            if top_feats:
                top = top_feats[0]
                short_concept = concept_text[:40] + "..." if len(concept_text) > 40 else concept_text
                lines.append(
                    f"| {short_concept} | {top['feature_idx']} | "
                    f"{top['correlation']:.3f} | {top['num_nonzero']} |"
                )

        lines.append("")

    # Feature frequency analysis
    lines.append("## Most Frequently Top-Correlated Features\n")
    feature_counts = {}
    for category, concepts in results["concepts"].items():
        for concept_text, data in concepts.items():
            for feat in data.get("top_features", [])[:5]:
                feat_idx = feat["feature_idx"]
                if feat_idx not in feature_counts:
                    feature_counts[feat_idx] = {"count": 0, "concepts": []}
                feature_counts[feat_idx]["count"] += 1
                feature_counts[feat_idx]["concepts"].append(concept_text[:30])

    # Top features by frequency
    top_by_freq = sorted(
        feature_counts.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )[:20]

    lines.append("| Feature | # Concepts | Example Concepts |")
    lines.append("|---------|------------|------------------|")
    for feat_idx, data in top_by_freq:
        examples = ", ".join(data["concepts"][:3])
        lines.append(f"| {feat_idx} | {data['count']} | {examples} |")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary report saved to {output_path}")


if __name__ == "__main__":
    main()
