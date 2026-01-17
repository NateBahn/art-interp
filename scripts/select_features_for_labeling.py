#!/usr/bin/env python3
"""
Select high-value SAE features for VLM labeling.

Combines monosemanticity scores with aesthetic rating correlations to identify
features that are both interpretable and predictive of aesthetic preferences.

Tiered selection:
- Tier 1: monosemanticity >= 0.65 AND |correlation| >= 0.20
- Tier 2: monosemanticity >= 0.55 AND |correlation| >= 0.15
- Tier 3: Top 50 by monosemanticity (regardless of correlation)

Updated to support multi-layer analysis (layers 7, 8, 11).
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "sae_analysis"


def load_monosemanticity_scores(path: Path) -> dict:
    """Load monosemanticity scores from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["scores"]


def load_correlations(path: Path) -> dict:
    """Load correlation results and build feature -> rating correlation mapping."""
    with open(path) as f:
        data = json.load(f)

    # Build mapping: feature_idx -> {rating: correlation, ...}
    feature_correlations = defaultdict(dict)

    # Support both old format (top_features_by_rating) and new format (top_features_by_label)
    top_features_key = "top_features_by_label" if "top_features_by_label" in data else "top_features_by_rating"

    for rating, features in data[top_features_key].items():
        for item in features:
            feature_idx = str(item["feature_idx"])
            feature_correlations[feature_idx][rating] = item["correlation"]

    return dict(feature_correlations)


def select_features(
    mono_scores: dict,
    correlations: dict,
    tier1_mono: float = 0.65,
    tier1_corr: float = 0.20,
    tier2_mono: float = 0.55,
    tier2_corr: float = 0.15,
    tier3_count: int = 50,
) -> list[dict]:
    """
    Select features based on tiered criteria.

    Returns list of feature dicts with selection metadata.
    """
    selected = {}

    # Process all features with monosemanticity scores
    for feature_idx, mono_data in mono_scores.items():
        mono_score = mono_data["monosemanticity_score"]
        top_artworks = mono_data.get("top_artworks", [])[:5]

        # Get correlations for this feature
        feature_corrs = correlations.get(feature_idx, {})

        if feature_corrs:
            # Find strongest correlation
            strongest_rating = max(feature_corrs.items(), key=lambda x: abs(x[1]))
            max_corr = abs(strongest_rating[1])
            strongest_rating_name = strongest_rating[0]
            strongest_corr_value = strongest_rating[1]
        else:
            max_corr = 0
            strongest_rating_name = None
            strongest_corr_value = None

        # Determine tier
        tier = None
        if mono_score >= tier1_mono and max_corr >= tier1_corr:
            tier = 1
        elif mono_score >= tier2_mono and max_corr >= tier2_corr:
            tier = 2

        if tier is not None:
            selected[feature_idx] = {
                "feature_idx": int(feature_idx),
                "monosemanticity_score": mono_score,
                "strongest_rating": strongest_rating_name,
                "strongest_correlation": strongest_corr_value,
                "all_correlations": feature_corrs,
                "top_5_artwork_ids": top_artworks,
                "tier": tier,
            }

    # Tier 3: Top N by monosemanticity regardless of correlation
    sorted_by_mono = sorted(
        mono_scores.items(),
        key=lambda x: x[1]["monosemanticity_score"],
        reverse=True
    )

    tier3_added = 0
    for feature_idx, mono_data in sorted_by_mono:
        if tier3_added >= tier3_count:
            break

        if feature_idx not in selected:
            feature_corrs = correlations.get(feature_idx, {})

            if feature_corrs:
                strongest_rating = max(feature_corrs.items(), key=lambda x: abs(x[1]))
                strongest_rating_name = strongest_rating[0]
                strongest_corr_value = strongest_rating[1]
            else:
                strongest_rating_name = None
                strongest_corr_value = None

            selected[feature_idx] = {
                "feature_idx": int(feature_idx),
                "monosemanticity_score": mono_data["monosemanticity_score"],
                "strongest_rating": strongest_rating_name,
                "strongest_correlation": strongest_corr_value,
                "all_correlations": feature_corrs,
                "top_5_artwork_ids": mono_data.get("top_artworks", [])[:5],
                "tier": 3,
            }
            tier3_added += 1

    # Sort by tier, then by combined score
    result = sorted(
        selected.values(),
        key=lambda x: (
            x["tier"],
            -(x["monosemanticity_score"] * abs(x["strongest_correlation"] or 0))
        )
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Select features for VLM labeling")
    parser.add_argument(
        "--layer",
        type=int,
        default=11,
        choices=[7, 8, 11],
        help="SAE layer to select features from (default: 11)"
    )
    parser.add_argument(
        "--tier1-mono",
        type=float,
        default=0.65,
        help="Tier 1 monosemanticity threshold"
    )
    parser.add_argument(
        "--tier1-corr",
        type=float,
        default=0.20,
        help="Tier 1 correlation threshold"
    )
    parser.add_argument(
        "--tier2-mono",
        type=float,
        default=0.55,
        help="Tier 2 monosemanticity threshold"
    )
    parser.add_argument(
        "--tier2-corr",
        type=float,
        default=0.15,
        help="Tier 2 correlation threshold"
    )
    parser.add_argument(
        "--tier3-count",
        type=int,
        default=50,
        help="Number of Tier 3 features (top by monosemanticity)"
    )
    args = parser.parse_args()

    layer = args.layer

    # Use layer-specific files
    mono_path = OUTPUT_DIR / f"monosemanticity_scores_layer{layer}.json"
    corr_path = OUTPUT_DIR / f"correlation_results_layer{layer}.json"
    output_path = OUTPUT_DIR / f"features_to_label_layer{layer}.json"

    print(f"=" * 60)
    print(f"Selecting Features for VLM Labeling (Layer {layer})")
    print(f"=" * 60)

    print(f"\nLoading monosemanticity scores from {mono_path}")
    mono_scores = load_monosemanticity_scores(mono_path)
    print(f"  Loaded {len(mono_scores)} features")

    print(f"Loading correlations from {corr_path}")
    correlations = load_correlations(corr_path)
    print(f"  Loaded correlations for {len(correlations)} features")

    print("\nSelecting features with criteria:")
    print(f"  Tier 1: mono >= {args.tier1_mono}, |corr| >= {args.tier1_corr}")
    print(f"  Tier 2: mono >= {args.tier2_mono}, |corr| >= {args.tier2_corr}")
    print(f"  Tier 3: Top {args.tier3_count} by monosemanticity")

    selected = select_features(
        mono_scores,
        correlations,
        tier1_mono=args.tier1_mono,
        tier1_corr=args.tier1_corr,
        tier2_mono=args.tier2_mono,
        tier2_corr=args.tier2_corr,
        tier3_count=args.tier3_count,
    )

    # Count by tier
    tier_counts = defaultdict(int)
    for f in selected:
        tier_counts[f["tier"]] += 1

    print(f"\nSelected {len(selected)} features:")
    print(f"  Tier 1 (high priority): {tier_counts[1]}")
    print(f"  Tier 2 (medium priority): {tier_counts[2]}")
    print(f"  Tier 3 (exploration): {tier_counts[3]}")

    # Show some examples
    print("\nExample Tier 1 features:")
    for f in selected[:5]:
        if f["tier"] == 1:
            print(f"  Feature {f['feature_idx']}: mono={f['monosemanticity_score']:.3f}, "
                  f"{f['strongest_rating']}={f['strongest_correlation']:.3f}")

    # Save results
    output = {
        "metadata": {
            "layer": layer,
            "total_features": len(selected),
            "tier_counts": dict(tier_counts),
            "selection_criteria": {
                "tier1": {"monosemanticity": args.tier1_mono, "correlation": args.tier1_corr},
                "tier2": {"monosemanticity": args.tier2_mono, "correlation": args.tier2_corr},
                "tier3": {"count": args.tier3_count},
            }
        },
        "features": selected,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
