#!/usr/bin/env python3
"""
Compute correlations for ALL monosemantic features with aesthetic ratings.

This expands beyond the original top-N per rating to find ALL features
that correlate with aesthetic judgments.
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
MONO_FILE = ANALYSIS_DIR / "monosemanticity_scores.json"
OUTPUT_FILE = ANALYSIS_DIR / "all_correlations.json"

RATINGS = [
    "mirror_self", "wholeness", "inner_light",
    "deepest_honest", "drawn_to", "choose_to_look",
    "technical_skill", "emotional_impact"
]


def main():
    print("=" * 60)
    print("Computing correlations for ALL monosemantic features")
    print("=" * 60)

    # Load monosemanticity scores
    print("\nLoading monosemanticity scores...")
    with open(MONO_FILE) as f:
        mono_data = json.load(f)

    mono_features = {
        int(k): v["monosemanticity_score"]
        for k, v in mono_data["scores"].items()
        if v.get("monosemanticity_score", 0) >= 0.6
    }
    print(f"  Found {len(mono_features):,} monosemantic features (>=0.6)")

    # Load SAE features
    print("\nLoading SAE features...")
    start = time.time()
    features = {}
    for batch_file in sorted(FEATURES_DIR.glob("batch_*.json")):
        batch_features = load_features_batch(batch_file, to_dense=True)
        features.update(batch_features)
    print(f"  Loaded {len(features)} artworks in {time.time()-start:.1f}s")

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

    # Pre-compute rating arrays
    print("\nPre-computing rating arrays...")
    rating_arrays = {}
    for rating in RATINGS:
        values = [artwork_labels[aid].get(rating) for aid in common_ids]
        # Filter out None values
        valid_mask = [v is not None for v in values]
        rating_arrays[rating] = {
            "values": np.array([v for v, m in zip(values, valid_mask) if m]),
            "mask": valid_mask,
            "valid_ids": [aid for aid, m in zip(common_ids, valid_mask) if m]
        }
        print(f"  {rating}: {len(rating_arrays[rating]['values'])} valid ratings")

    # Compute correlations for all monosemantic features
    print(f"\nComputing correlations for {len(mono_features):,} features...")
    start = time.time()

    all_correlations = {}
    top_by_rating = defaultdict(list)

    for feat_idx in tqdm(mono_features.keys()):
        feat_correlations = {}

        for rating in RATINGS:
            rating_data = rating_arrays[rating]
            valid_ids = rating_data["valid_ids"]
            rating_values = rating_data["values"]

            # Get feature values for valid artworks
            feat_values = np.array([features[aid][feat_idx] for aid in valid_ids])

            # Skip if no variance
            if np.std(feat_values) == 0 or np.std(rating_values) == 0:
                continue

            corr, pval = stats.pearsonr(feat_values, rating_values)

            if not np.isnan(corr):
                feat_correlations[rating] = {
                    "correlation": float(corr),
                    "p_value": float(pval),
                    "abs_correlation": abs(float(corr))
                }

                # Track for top lists
                top_by_rating[rating].append({
                    "feature_idx": feat_idx,
                    "correlation": float(corr),
                    "p_value": float(pval),
                    "monosemanticity": mono_features[feat_idx]
                })

        if feat_correlations:
            # Find max correlation across ratings
            max_corr_rating = max(feat_correlations.items(),
                                   key=lambda x: x[1]["abs_correlation"])

            all_correlations[feat_idx] = {
                "correlations": feat_correlations,
                "max_correlation": max_corr_rating[1]["correlation"],
                "max_correlation_rating": max_corr_rating[0],
                "monosemanticity": mono_features[feat_idx]
            }

    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")

    # Sort top lists
    for rating in RATINGS:
        top_by_rating[rating].sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # Compute statistics
    all_max_corrs = [v["max_correlation"] for v in all_correlations.values()]
    abs_max_corrs = [abs(c) for c in all_max_corrs]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nFeatures with valid correlations: {len(all_correlations):,}")
    print(f"\nMax |correlation| distribution:")
    print(f"  Mean: {np.mean(abs_max_corrs):.4f}")
    print(f"  Median: {np.median(abs_max_corrs):.4f}")
    print(f"  Max: {np.max(abs_max_corrs):.4f}")

    # Count by correlation threshold
    thresholds = [0.30, 0.25, 0.20, 0.15, 0.10]
    print(f"\nFeatures by |correlation| threshold:")
    for thresh in thresholds:
        count = sum(1 for c in abs_max_corrs if c >= thresh)
        print(f"  |corr| >= {thresh}: {count:,} features")

    # Top features per rating
    print(f"\nTop 5 features per rating:")
    for rating in RATINGS:
        print(f"\n  {rating}:")
        for item in top_by_rating[rating][:5]:
            print(f"    F{item['feature_idx']}: r={item['correlation']:.3f}, mono={item['monosemanticity']:.3f}")

    # Save results
    output = {
        "metadata": {
            "num_features_analyzed": len(mono_features),
            "num_features_with_correlations": len(all_correlations),
            "num_artworks": len(common_ids),
            "monosemanticity_threshold": 0.6,
        },
        "statistics": {
            "mean_max_correlation": float(np.mean(abs_max_corrs)),
            "median_max_correlation": float(np.median(abs_max_corrs)),
            "max_max_correlation": float(np.max(abs_max_corrs)),
        },
        "top_features_by_rating": {
            rating: items[:100] for rating, items in top_by_rating.items()
        },
        "all_correlations": all_correlations,
    }

    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
