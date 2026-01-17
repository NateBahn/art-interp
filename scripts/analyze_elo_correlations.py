"""
Analyze correlations between Elo scores and artwork features.

Usage:
    python scripts/analyze_elo_correlations.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import get_db_context
from database.models import Artwork

# Constants
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output" / "elo_comparison"


def load_elo_ratings() -> dict[str, float]:
    """Load Elo ratings from latest checkpoint."""
    checkpoints = list(OUTPUT_DIR.glob("elo_experiment_*.json"))
    checkpoints = [c for c in checkpoints if "_backup" not in c.name and "_final" not in c.name]
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found")

    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"Loading Elo ratings from: {latest.name}")

    with open(latest) as f:
        data = json.load(f)

    return {
        aid: r["rating"]
        for aid, r in data["ratings"].items()
        if r["matches_played"] > 0
    }


def load_artwork_labels() -> dict[str, dict]:
    """Load artwork labels from database."""
    print("Loading artwork labels from database...")

    with get_db_context() as db:
        artworks = db.query(Artwork.id, Artwork.labels, Artwork.title, Artwork.artist_name).all()

        labels_map = {}
        for art_id, labels_json, title, artist in artworks:
            if labels_json:
                try:
                    labels = json.loads(labels_json)
                    labels["_title"] = title
                    labels["_artist"] = artist
                    labels_map[art_id] = labels
                except json.JSONDecodeError:
                    pass

        return labels_map


def pearson_correlation(x: list, y: list) -> tuple[float, int]:
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 3:
        return 0.0, len(x)

    x = np.array(x)
    y = np.array(y)

    # Remove NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return 0.0, len(x)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))

    if denominator == 0:
        return 0.0, len(x)

    return numerator / denominator, len(x)


def analyze_numeric_correlations(elo_ratings: dict, labels_map: dict) -> dict:
    """Correlate Elo with numeric features (1-10 ratings)."""
    numeric_features = [
        "mirror_self", "wholeness", "inner_light",
        "deepest_honest", "drawn_to", "choose_to_look",
        "technical_skill", "emotional_impact"
    ]

    results = {}

    for feature in numeric_features:
        elo_values = []
        feature_values = []

        for aid, elo in elo_ratings.items():
            if aid in labels_map and feature in labels_map[aid]:
                val = labels_map[aid][feature]
                if isinstance(val, (int, float)) and 1 <= val <= 10:
                    elo_values.append(elo)
                    feature_values.append(val)

        if len(elo_values) >= 10:
            corr, n = pearson_correlation(elo_values, feature_values)
            results[feature] = {"correlation": corr, "n": n}

    return results


def analyze_categorical_features(elo_ratings: dict, labels_map: dict) -> dict:
    """Analyze average Elo by categorical features."""
    categorical_features = [
        "color_palette", "subject_matter", "mood", "suitable_for_home",
        "artwork_format", "contains_nudity", "contains_violence"
    ]

    results = {}

    for feature in categorical_features:
        category_elos = defaultdict(list)

        for aid, elo in elo_ratings.items():
            if aid in labels_map and feature in labels_map[aid]:
                val = labels_map[aid][feature]
                if val:
                    category_elos[str(val)].append(elo)

        if category_elos:
            category_stats = {}
            for cat, elos in category_elos.items():
                if len(elos) >= 5:  # Minimum sample size
                    category_stats[cat] = {
                        "mean_elo": np.mean(elos),
                        "std_elo": np.std(elos),
                        "count": len(elos)
                    }

            if category_stats:
                # Sort by mean Elo
                sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]["mean_elo"], reverse=True)
                results[feature] = dict(sorted_cats)

    return results


def print_results(numeric_corr: dict, categorical_stats: dict):
    """Print analysis results."""
    print("\n" + "=" * 70)
    print("ELO CORRELATION ANALYSIS")
    print("=" * 70)

    # Numeric correlations
    print("\n### NUMERIC FEATURE CORRELATIONS (with Elo)")
    print("-" * 50)

    sorted_numeric = sorted(numeric_corr.items(), key=lambda x: abs(x[1]["correlation"]), reverse=True)

    for feature, stats in sorted_numeric:
        corr = stats["correlation"]
        n = stats["n"]
        direction = "+" if corr > 0 else ""
        strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
        print(f"  {feature:20} r={direction}{corr:.3f}  (n={n:,})  [{strength}]")

    # Categorical features
    print("\n### CATEGORICAL FEATURE ANALYSIS")

    for feature, categories in categorical_stats.items():
        print(f"\n--- {feature.upper()} ---")
        for cat, stats in categories.items():
            mean_elo = stats["mean_elo"]
            std_elo = stats["std_elo"]
            count = stats["count"]
            # Show relative to 1400 baseline
            diff = mean_elo - 1400
            direction = "+" if diff > 0 else ""
            print(f"  {cat:30} Elo={mean_elo:.0f} ({direction}{diff:.0f})  n={count}")


def save_results(numeric_corr: dict, categorical_stats: dict, output_path: Path):
    """Save results to JSON."""
    results = {
        "numeric_correlations": numeric_corr,
        "categorical_analysis": {
            feat: {
                cat: {
                    "mean_elo": round(s["mean_elo"], 1),
                    "std_elo": round(s["std_elo"], 1),
                    "count": s["count"]
                }
                for cat, s in cats.items()
            }
            for feat, cats in categorical_stats.items()
        }
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    # Load data
    elo_ratings = load_elo_ratings()
    print(f"Loaded {len(elo_ratings):,} Elo ratings")

    labels_map = load_artwork_labels()
    print(f"Loaded labels for {len(labels_map):,} artworks")

    # Find overlap
    overlap = set(elo_ratings.keys()) & set(labels_map.keys())
    print(f"Artworks with both Elo and labels: {len(overlap):,}")

    if len(overlap) < 100:
        print("ERROR: Not enough overlapping data for analysis")
        return

    # Run analysis
    numeric_corr = analyze_numeric_correlations(elo_ratings, labels_map)
    categorical_stats = analyze_categorical_features(elo_ratings, labels_map)

    # Print results
    print_results(numeric_corr, categorical_stats)

    # Save results
    output_path = OUTPUT_DIR / "elo_feature_correlations.json"
    save_results(numeric_corr, categorical_stats, output_path)


if __name__ == "__main__":
    main()
