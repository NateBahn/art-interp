#!/usr/bin/env python3
"""
Compute SAE feature correlations with aesthetic ratings - FINE ART ONLY.

Excludes:
- NYPL (library, not art museum)
- Carriage/vehicle design drawings from Met

This focuses the analysis on actual fine art to avoid dataset biases.
"""

import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import get_db_context
from database.models import Artwork
from services.sae_features import load_features_batch


# Paths
FEATURES_DIR = Path(__file__).parent.parent / "data" / "sae_features"
ANALYSIS_DIR = Path(__file__).parent.parent / "output" / "sae_analysis"
FINE_ART_IDS_FILE = Path(__file__).parent.parent / "data" / "fine_art_artwork_ids.json"
OUTPUT_FILE = ANALYSIS_DIR / "correlations_fine_art_layer11.json"

RATINGS = [
    "mirror_self", "wholeness", "inner_light",
    "deepest_honest", "drawn_to", "choose_to_look",
    "technical_skill", "emotional_impact", "want_to_understand"
]

# Minimum absolute correlation to report
MIN_CORRELATION = 0.05


def load_fine_art_ids() -> set[str]:
    """Load the fine art artwork IDs."""
    with open(FINE_ART_IDS_FILE) as f:
        data = json.load(f)
    return set(data["artwork_ids"])


def main():
    print("=" * 70)
    print("Computing SAE Correlations - FINE ART ONLY")
    print("=" * 70)

    # Load fine art IDs
    print("\nLoading fine art filter...")
    fine_art_ids = load_fine_art_ids()
    print(f"  Fine art artworks: {len(fine_art_ids)}")

    # Load SAE features (Layer 11)
    print("\nLoading SAE features (Layer 11)...")
    layer_dir = FEATURES_DIR / "layer_11"
    if not layer_dir.exists():
        print(f"  ERROR: {layer_dir} not found")
        return

    start = time.time()
    features = {}
    for batch_file in sorted(layer_dir.glob("batch_*.json")):
        batch_features = load_features_batch(batch_file, to_dense=True)
        # Filter to fine art only
        for aid, feat in batch_features.items():
            if aid in fine_art_ids:
                features[aid] = feat

    print(f"  Loaded {len(features)} fine art features in {time.time()-start:.1f}s")

    # Load artwork labels
    print("\nLoading artwork labels...")
    with get_db_context() as db:
        artworks = db.query(Artwork).filter(
            Artwork.labels.isnot(None),
            Artwork.id.in_(list(fine_art_ids))
        ).all()
        artwork_labels = {}
        for a in artworks:
            if a.labels:
                labels = json.loads(a.labels) if isinstance(a.labels, str) else a.labels
                artwork_labels[a.id] = labels

    print(f"  Loaded labels for {len(artwork_labels)} fine art works")

    # Get common artworks (have both features and labels)
    common_ids = [aid for aid in features.keys() if aid in artwork_labels]
    print(f"  Common artworks with features + labels: {len(common_ids)}")

    # Pre-compute rating arrays
    print("\nPre-computing rating arrays...")
    rating_arrays = {}
    for rating in RATINGS:
        values = [artwork_labels[aid].get(rating) for aid in common_ids]
        valid_mask = [v is not None for v in values]
        valid_values = np.array([v for v, m in zip(values, valid_mask) if m])
        valid_ids = [aid for aid, m in zip(common_ids, valid_mask) if m]

        rating_arrays[rating] = {
            "values": valid_values,
            "valid_ids": valid_ids,
        }
        print(f"  {rating}: {len(valid_values)} valid ratings")

    # Build feature matrix for valid artworks
    print("\nBuilding feature matrix...")
    # Use first rating's valid IDs (they should all be similar)
    primary_rating = "mirror_self"
    valid_ids = rating_arrays[primary_rating]["valid_ids"]

    feature_matrix = np.zeros((len(valid_ids), 49152), dtype=np.float32)
    for i, aid in enumerate(valid_ids):
        feature_matrix[i] = features[aid]

    print(f"  Feature matrix: {feature_matrix.shape}")

    # Find active features (non-zero in at least 1% of artworks)
    min_active = int(0.01 * len(valid_ids))
    active_per_feature = (feature_matrix > 0).sum(axis=0)
    active_features = np.where(active_per_feature >= min_active)[0]
    print(f"  Active features (>={min_active} artworks): {len(active_features)}")

    # Compute correlations
    print(f"\nComputing correlations for {len(active_features)} features...")

    results = {
        "metadata": {
            "dataset": "fine_art_only",
            "total_artworks": len(fine_art_ids),
            "artworks_with_features": len(features),
            "artworks_with_labels": len(artwork_labels),
            "artworks_analyzed": len(valid_ids),
            "features_analyzed": len(active_features),
        },
        "by_rating": {},
        "by_feature": {},
    }

    for rating in tqdm(RATINGS, desc="Ratings"):
        rating_data = rating_arrays[rating]
        # Get rating values for the common valid IDs
        rating_values = []
        for aid in valid_ids:
            val = artwork_labels[aid].get(rating)
            rating_values.append(val if val is not None else np.nan)
        rating_values = np.array(rating_values)

        # Skip if too many missing
        valid_mask = ~np.isnan(rating_values)
        if valid_mask.sum() < 100:
            print(f"  Skipping {rating}: only {valid_mask.sum()} valid ratings")
            continue

        correlations = []
        for feat_idx in active_features:
            feat_values = feature_matrix[:, feat_idx]

            # Only use rows where both rating and feature are valid
            mask = valid_mask & (feat_values > 0)
            if mask.sum() < 50:
                continue

            r, p = stats.pearsonr(feat_values[mask], rating_values[mask])

            if abs(r) >= MIN_CORRELATION:
                correlations.append({
                    "feature_idx": int(feat_idx),
                    "correlation": round(r, 4),
                    "p_value": float(p),
                    "n_samples": int(mask.sum()),
                })

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        results["by_rating"][rating] = {
            "total_correlated": len(correlations),
            "positive": len([c for c in correlations if c["correlation"] > 0]),
            "negative": len([c for c in correlations if c["correlation"] < 0]),
            "top_positive": [c for c in correlations if c["correlation"] > 0][:50],
            "top_negative": [c for c in correlations if c["correlation"] < 0][:50],
        }

        print(f"  {rating}: {len(correlations)} correlated features "
              f"(+{results['by_rating'][rating]['positive']}/-{results['by_rating'][rating]['negative']})")

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_FILE}")

    # Print top findings for mirror_self
    if "mirror_self" in results["by_rating"]:
        print("\n" + "=" * 70)
        print("TOP mirror_self CORRELATIONS (Fine Art Only)")
        print("=" * 70)

        ms = results["by_rating"]["mirror_self"]
        print(f"\nPositive (increase mirror_self): {ms['positive']} features")
        for c in ms["top_positive"][:10]:
            print(f"  Feature #{c['feature_idx']}: r = +{c['correlation']:.3f} (n={c['n_samples']})")

        print(f"\nNegative (decrease mirror_self): {ms['negative']} features")
        for c in ms["top_negative"][:10]:
            print(f"  Feature #{c['feature_idx']}: r = {c['correlation']:.3f} (n={c['n_samples']})")


if __name__ == "__main__":
    main()
