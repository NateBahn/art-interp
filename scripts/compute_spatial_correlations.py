#!/usr/bin/env python3
"""
Compute correlations between Spatial SAE features and aesthetic ratings.

Uses max-pooled spatial features (same features shown in heatmaps)
so that correlations and visualizations are consistent.
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

# Paths
SPATIAL_FEATURES_DIR = Path(__file__).parent.parent / "data" / "spatial_sae_features"
ANALYSIS_DIR = Path(__file__).parent.parent / "output" / "sae_analysis"
OUTPUT_FILE = ANALYSIS_DIR / "spatial_correlations.json"

SAE_DIM = 49152

RATINGS = [
    "mirror_self", "wholeness", "inner_light",
    "deepest_honest", "drawn_to", "choose_to_look",
    "technical_skill", "emotional_impact"
]


def load_spatial_features() -> dict[str, np.ndarray]:
    """Load all spatial features from batch files."""
    features = {}

    for batch_file in sorted(SPATIAL_FEATURES_DIR.glob("spatial_batch_*.json")):
        with open(batch_file) as f:
            batch_data = json.load(f)

        for artwork_id, data in batch_data.items():
            # Convert sparse format to dense
            dense = np.zeros(SAE_DIM, dtype=np.float32)
            for idx_str, value in data["sparse_features"].items():
                dense[int(idx_str)] = value
            features[artwork_id] = dense

    return features


def main():
    print("=" * 60)
    print("Computing Spatial SAE Feature Correlations")
    print("=" * 60)

    # Load spatial features
    print("\nLoading spatial SAE features...")
    start = time.time()
    features = load_spatial_features()
    print(f"  Loaded {len(features)} artworks in {time.time()-start:.1f}s")

    if len(features) == 0:
        print("\nNo spatial features found! Run extract_spatial_features.py first.")
        print(f"  Expected directory: {SPATIAL_FEATURES_DIR}")
        return

    # Load artwork labels
    print("\nLoading artwork labels...")
    with get_db_context() as db:
        artworks = db.query(Artwork).filter(Artwork.labels.isnot(None)).all()
        artwork_labels = {}
        for a in artworks:
            if a.labels:
                artwork_labels[a.id] = json.loads(a.labels)
    print(f"  Loaded labels for {len(artwork_labels)} artworks")

    # Get common artworks
    common_ids = [aid for aid in features.keys() if aid in artwork_labels]
    print(f"  Common artworks with both features and labels: {len(common_ids)}")

    if len(common_ids) < 10:
        print("\nNot enough common artworks for meaningful correlations!")
        return

    # Pre-compute rating arrays
    print("\nPre-computing rating arrays...")
    rating_arrays = {}
    for rating in RATINGS:
        values = [artwork_labels[aid].get(rating) for aid in common_ids]
        valid_mask = [v is not None for v in values]
        rating_arrays[rating] = {
            "values": np.array([v for v, m in zip(values, valid_mask) if m]),
            "mask": valid_mask,
            "valid_ids": [aid for aid, m in zip(common_ids, valid_mask) if m]
        }
        print(f"  {rating}: {len(rating_arrays[rating]['values'])} valid ratings")

    # Find features with any activation
    print("\nIdentifying active features...")
    all_features_matrix = np.stack([features[aid] for aid in common_ids])
    max_per_feature = all_features_matrix.max(axis=0)
    active_features = np.where(max_per_feature > 0)[0]
    print(f"  Found {len(active_features):,} features with activation > 0")

    # Compute correlations
    print(f"\nComputing correlations for {len(active_features):,} active features...")
    start = time.time()

    all_correlations = {}
    top_by_rating = defaultdict(list)

    for feat_idx in tqdm(active_features):
        feat_correlations = {}

        for rating in RATINGS:
            rating_data = rating_arrays[rating]
            valid_ids = rating_data["valid_ids"]
            rating_values = rating_data["values"]

            # Get feature values for valid artworks
            feat_values = np.array([features[aid][feat_idx] for aid in valid_ids])

            # Skip if no variance
            if np.std(feat_values) < 1e-6 or np.std(rating_values) < 1e-6:
                continue

            corr, pval = stats.pearsonr(feat_values, rating_values)

            if not np.isnan(corr):
                feat_correlations[rating] = {
                    "correlation": float(corr),
                    "p_value": float(pval),
                    "abs_correlation": abs(float(corr))
                }

                top_by_rating[rating].append({
                    "feature_idx": int(feat_idx),
                    "correlation": float(corr),
                    "p_value": float(pval),
                })

        if feat_correlations:
            max_corr_rating = max(feat_correlations.items(),
                                   key=lambda x: x[1]["abs_correlation"])
            all_correlations[int(feat_idx)] = {
                "correlations": feat_correlations,
                "max_correlation": max_corr_rating[1]["correlation"],
                "max_correlation_rating": max_corr_rating[0],
            }

    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")

    # Sort top lists by absolute correlation
    for rating in RATINGS:
        top_by_rating[rating].sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # Compute statistics
    all_max_corrs = [abs(v["max_correlation"]) for v in all_correlations.values()]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nFeatures with valid correlations: {len(all_correlations):,}")

    if all_max_corrs:
        print(f"\nMax |correlation| distribution:")
        print(f"  Mean: {np.mean(all_max_corrs):.4f}")
        print(f"  Median: {np.median(all_max_corrs):.4f}")
        print(f"  Max: {np.max(all_max_corrs):.4f}")

        # Count by threshold
        thresholds = [0.30, 0.25, 0.20, 0.15, 0.10]
        print(f"\nFeatures by |correlation| threshold:")
        for thresh in thresholds:
            count = sum(1 for c in all_max_corrs if c >= thresh)
            print(f"  |corr| >= {thresh}: {count:,} features")

        # Top features per rating
        print(f"\nTop 5 features per rating:")
        for rating in RATINGS:
            if top_by_rating[rating]:
                print(f"\n  {rating}:")
                for item in top_by_rating[rating][:5]:
                    print(f"    F{item['feature_idx']}: r={item['correlation']:.3f}")

    # Save results
    output = {
        "metadata": {
            "sae_type": "spatial",
            "num_features_analyzed": len(active_features),
            "num_features_with_correlations": len(all_correlations),
            "num_artworks": len(common_ids),
        },
        "statistics": {
            "mean_max_correlation": float(np.mean(all_max_corrs)) if all_max_corrs else 0,
            "median_max_correlation": float(np.median(all_max_corrs)) if all_max_corrs else 0,
            "max_max_correlation": float(np.max(all_max_corrs)) if all_max_corrs else 0,
        },
        "top_features_by_rating": {
            rating: items[:100] for rating, items in top_by_rating.items()
        },
        "all_correlations": {str(k): v for k, v in all_correlations.items()},
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
