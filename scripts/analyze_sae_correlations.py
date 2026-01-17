#!/usr/bin/env python3
"""
SAE Feature Correlation Analysis

Correlates sparse autoencoder features with aesthetic ratings to discover
which visual features predict different types of aesthetic judgments.

Usage:
    python scripts/analyze_sae_correlations.py [--output-dir PATH]
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from database.connection import get_db_context
from database.models import Artwork
from services.sae_features import SAE_CONFIGS, SAE_DIM, load_features_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
BASE_FEATURES_DIR = Path(__file__).parent.parent / "data" / "sae_features"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "sae_analysis"

# Numeric ratings (1-10 scale) - use Pearson correlation
ALEXANDER_RATINGS = ["mirror_self", "wholeness", "inner_light"]
ASKELL_RATINGS = ["deepest_honest", "drawn_to", "choose_to_look"]
CRITICAL_RATINGS = ["technical_skill", "emotional_impact"]
NUMERIC_RATINGS = ALEXANDER_RATINGS + ASKELL_RATINGS + CRITICAL_RATINGS

# Categorical labels (one-hot encoded) - use point-biserial correlation
SUBJECT_MATTER = [
    "portrait_single", "portrait_group",
    "landscape_natural", "landscape_urban", "seascape_maritime",
    "still_life", "floral_botanical",
    "abstract_geometric", "abstract_organic",
    "religious_mythological", "historical_narrative",
    "domestic_interior", "animals_wildlife",
]

MOOD = [
    "serene_peaceful", "melancholic_somber", "dramatic_intense",
    "joyful_celebratory", "mysterious_enigmatic", "romantic_idealized",
    "stark_austere", "chaotic_turbulent", "intimate_quiet", "grand_majestic",
]

# All label dimensions for correlation
ALL_LABEL_DIMS = NUMERIC_RATINGS + [f"subject:{s}" for s in SUBJECT_MATTER] + [f"mood:{m}" for m in MOOD]


def get_features_dir(layer: int) -> Path:
    """Get layer-specific features directory."""
    return BASE_FEATURES_DIR / f"layer_{layer}"


@dataclass
class CorrelationResult:
    """Result for a single feature-rating correlation."""

    feature_idx: int
    rating_name: str
    correlation: float
    p_value: float
    num_samples: int


@dataclass
class FeatureAnalysis:
    """Complete analysis for a single SAE feature."""

    feature_idx: int
    correlations: dict[str, float]  # rating_name -> correlation
    p_values: dict[str, float]
    max_activation: float
    mean_activation: float
    num_nonzero: int  # How many artworks have this feature active
    top_artworks: list[str]  # Artwork IDs with highest activation


def load_all_features(layer: int = 8) -> dict[str, np.ndarray]:
    """Load all SAE features from batch files for a specific layer."""
    all_features = {}
    features_dir = get_features_dir(layer)

    if not features_dir.exists():
        logger.error(f"Features directory not found: {features_dir}")
        return all_features

    batch_files = sorted(features_dir.glob("batch_*.json"))
    logger.info(f"Loading layer {layer} features from {len(batch_files)} batch files...")

    for batch_file in batch_files:
        batch_features = load_features_batch(batch_file, to_dense=True)
        all_features.update(batch_features)
        logger.debug(f"  Loaded {len(batch_features)} from {batch_file.name}")

    logger.info(f"Total artworks with features: {len(all_features)}")
    return all_features


def load_labels(exclude_sources: list[str] | None = None) -> dict[str, dict[str, float]]:
    """
    Load all label dimensions for artworks from database.

    Args:
        exclude_sources: List of sources to exclude (e.g., ['nypl'])

    Returns dict mapping artwork_id -> {label_dim: value} where:
    - Numeric ratings (1-10): stored as-is
    - subject_matter: one-hot encoded as "subject:category" -> 0/1
    - mood: one-hot encoded as "mood:category" -> 0/1
    """
    labels_dict = {}

    with get_db_context() as db:
        query = db.query(Artwork).filter(Artwork.labels.isnot(None))

        if exclude_sources:
            query = query.filter(~Artwork.source.in_(exclude_sources))

        artworks = query.all()

        for artwork in artworks:
            try:
                labels = json.loads(artwork.labels) if artwork.labels else {}
                artwork_labels = {}

                # Extract numeric ratings (1-10)
                for rating_name in NUMERIC_RATINGS:
                    value = labels.get(rating_name)
                    if value is not None and isinstance(value, (int, float)):
                        artwork_labels[rating_name] = float(value)

                # Extract subject_matter as one-hot
                subject = labels.get("subject_matter")
                if subject and subject in SUBJECT_MATTER:
                    for s in SUBJECT_MATTER:
                        artwork_labels[f"subject:{s}"] = 1.0 if s == subject else 0.0

                # Extract mood as one-hot
                mood = labels.get("mood")
                if mood and mood in MOOD:
                    for m in MOOD:
                        artwork_labels[f"mood:{m}"] = 1.0 if m == mood else 0.0

                if artwork_labels:
                    labels_dict[artwork.id] = artwork_labels

            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded labels for {len(labels_dict)} artworks")
    return labels_dict


# Backwards compatibility alias
def load_ratings(exclude_sources: list[str] | None = None) -> dict[str, dict[str, float]]:
    """Load ratings for all artworks. Alias for load_labels()."""
    return load_labels(exclude_sources=exclude_sources)


def compute_feature_correlations(
    features: dict[str, np.ndarray],
    labels: dict[str, dict[str, float]],
    min_samples: int = 30,
    apply_fdr: bool = True,
    fdr_alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[CorrelationResult]]]:
    """
    Compute correlations between all SAE features and all label dimensions.

    Args:
        features: artwork_id -> feature vector
        labels: artwork_id -> {label_dim: value}
        min_samples: Minimum samples required for correlation
        apply_fdr: Whether to apply FDR correction
        fdr_alpha: FDR significance threshold

    Returns:
        correlation_matrix: (num_features, num_labels) array
        fdr_matrix: (num_features, num_labels) array of FDR-corrected q-values
        results_by_label: {label_name: [CorrelationResult, ...]}
    """
    # Find artworks with both features and labels
    common_ids = set(features.keys()) & set(labels.keys())
    logger.info(f"Artworks with both features and labels: {len(common_ids)}")

    if len(common_ids) < min_samples:
        logger.error(f"Not enough samples ({len(common_ids)} < {min_samples})")
        return np.array([]), np.array([]), {}

    # Build aligned arrays
    artwork_ids = sorted(common_ids)
    feature_matrix = np.stack([features[aid] for aid in artwork_ids])  # (N, 49152)
    logger.info(f"Feature matrix shape: {feature_matrix.shape}")

    # Initialize correlation matrix
    num_labels = len(ALL_LABEL_DIMS)
    correlation_matrix = np.zeros((SAE_DIM, num_labels), dtype=np.float32)
    p_value_matrix = np.ones((SAE_DIM, num_labels), dtype=np.float32)

    results_by_label = {label: [] for label in ALL_LABEL_DIMS}

    # Compute correlations for each label dimension
    for label_idx, label_name in enumerate(ALL_LABEL_DIMS):
        logger.info(f"Computing correlations for {label_name}...")

        # Build label array
        label_values = []
        valid_indices = []
        for i, aid in enumerate(artwork_ids):
            if label_name in labels[aid]:
                label_values.append(labels[aid][label_name])
                valid_indices.append(i)

        if len(label_values) < min_samples:
            logger.warning(f"  Skipping {label_name}: only {len(label_values)} samples")
            continue

        label_array = np.array(label_values, dtype=np.float32)
        feature_subset = feature_matrix[valid_indices]  # (N_valid, 49152)

        # Check label variance
        if np.std(label_array) < 1e-10:
            logger.warning(f"  Skipping {label_name}: no variance in labels")
            continue

        logger.info(f"  Using {len(label_values)} samples")

        # Compute Pearson correlation for each feature
        for feat_idx in range(SAE_DIM):
            feat_values = feature_subset[:, feat_idx]

            # Skip features with zero variance
            if np.std(feat_values) < 1e-10:
                continue

            try:
                corr, pval = stats.pearsonr(feat_values, label_array)
                correlation_matrix[feat_idx, label_idx] = corr
                p_value_matrix[feat_idx, label_idx] = pval

            except Exception as e:
                logger.debug(f"  Error computing correlation for feature {feat_idx}: {e}")

    # Apply FDR correction across all tests
    if apply_fdr:
        logger.info("Applying FDR correction (Benjamini-Hochberg)...")
        all_pvals = p_value_matrix.flatten()
        # Replace NaN with 1.0 for FDR
        all_pvals = np.nan_to_num(all_pvals, nan=1.0)
        _, fdr_pvals, _, _ = multipletests(all_pvals, alpha=fdr_alpha, method='fdr_bh')
        fdr_matrix = fdr_pvals.reshape(p_value_matrix.shape)
        num_significant = (fdr_matrix < fdr_alpha).sum()
        logger.info(f"  FDR-significant correlations (q < {fdr_alpha}): {num_significant}")
    else:
        fdr_matrix = p_value_matrix

    # Build results by label (using FDR-corrected p-values)
    for label_idx, label_name in enumerate(ALL_LABEL_DIMS):
        for feat_idx in range(SAE_DIM):
            corr = correlation_matrix[feat_idx, label_idx]
            q_val = fdr_matrix[feat_idx, label_idx]

            # Only store FDR-significant results with |r| > 0.1
            if q_val < fdr_alpha and abs(corr) > 0.1:
                results_by_label[label_name].append(
                    CorrelationResult(
                        feature_idx=feat_idx,
                        rating_name=label_name,
                        correlation=float(corr),
                        p_value=float(q_val),  # Store FDR q-value
                        num_samples=len([aid for aid in artwork_ids if label_name in labels[aid]]),
                    )
                )

        # Sort by absolute correlation
        results_by_label[label_name].sort(key=lambda r: -abs(r.correlation))
        if results_by_label[label_name]:
            logger.info(f"  {label_name}: {len(results_by_label[label_name])} FDR-significant correlations")

    return correlation_matrix, fdr_matrix, results_by_label


def analyze_feature_patterns(
    features: dict[str, np.ndarray],
    labels: dict[str, dict[str, float]],
    correlation_matrix: np.ndarray,
    top_k: int = 50,
) -> dict[int, FeatureAnalysis]:
    """
    Analyze top features in detail.

    Returns analysis for features with strongest correlations.
    """
    # Find features with strongest correlations (any label)
    max_abs_corr = np.max(np.abs(correlation_matrix), axis=1)
    top_feature_indices = np.argsort(max_abs_corr)[-top_k:][::-1]

    logger.info(f"Analyzing top {top_k} features by max correlation...")

    artwork_ids = list(features.keys())
    feature_matrix = np.stack([features[aid] for aid in artwork_ids])

    analyses = {}
    for feat_idx in top_feature_indices:
        feat_values = feature_matrix[:, feat_idx]

        # Find top activating artworks
        top_artwork_indices = np.argsort(feat_values)[-10:][::-1]
        top_artworks = [artwork_ids[i] for i in top_artwork_indices]

        analyses[int(feat_idx)] = FeatureAnalysis(
            feature_idx=int(feat_idx),
            correlations={
                label: float(correlation_matrix[feat_idx, i]) for i, label in enumerate(ALL_LABEL_DIMS)
            },
            p_values={},  # Could add if needed
            max_activation=float(feat_values.max()),
            mean_activation=float(feat_values.mean()),
            num_nonzero=int((feat_values > 0).sum()),
            top_artworks=top_artworks,
        )

    return analyses


def find_differentiating_features(
    correlation_matrix: np.ndarray,
    top_k: int = 20,
) -> dict[str, list[int]]:
    """
    Find features that differentiate between label categories.

    Returns features where correlations differ across:
    - Alexander vs Askell ratings
    - Subject matter categories
    - Mood categories
    """
    result = {}

    # Get indices for each category in ALL_LABEL_DIMS
    alexander_idx = [ALL_LABEL_DIMS.index(r) for r in ALEXANDER_RATINGS if r in ALL_LABEL_DIMS]
    askell_idx = [ALL_LABEL_DIMS.index(r) for r in ASKELL_RATINGS if r in ALL_LABEL_DIMS]
    critical_idx = [ALL_LABEL_DIMS.index(r) for r in CRITICAL_RATINGS if r in ALL_LABEL_DIMS]
    subject_idx = [ALL_LABEL_DIMS.index(f"subject:{s}") for s in SUBJECT_MATTER]
    mood_idx = [ALL_LABEL_DIMS.index(f"mood:{m}") for m in MOOD]

    # Mean correlation per numeric rating category
    if alexander_idx and askell_idx:
        alexander_corr = np.mean(correlation_matrix[:, alexander_idx], axis=1)
        askell_corr = np.mean(correlation_matrix[:, askell_idx], axis=1)
        critical_corr = np.mean(correlation_matrix[:, critical_idx], axis=1) if critical_idx else np.zeros_like(alexander_corr)

        # Features that predict Alexander but not Askell
        alexander_specific = alexander_corr - askell_corr
        result["alexander_specific"] = np.argsort(alexander_specific)[-top_k:][::-1].tolist()

        # Features that predict Askell but not Alexander
        askell_specific = askell_corr - alexander_corr
        result["askell_specific"] = np.argsort(askell_specific)[-top_k:][::-1].tolist()

        # Features that predict Critical but not personal preference
        if critical_idx:
            critical_specific = critical_corr - (alexander_corr + askell_corr) / 2
            result["critical_specific"] = np.argsort(critical_specific)[-top_k:][::-1].tolist()

    # Features specific to each subject matter
    if subject_idx:
        for i, subject in enumerate(SUBJECT_MATTER):
            subj_corr = correlation_matrix[:, subject_idx[i]]
            other_subj_corr = np.mean(
                correlation_matrix[:, [j for j in subject_idx if j != subject_idx[i]]],
                axis=1
            )
            subject_specific = subj_corr - other_subj_corr
            result[f"subject_{subject}"] = np.argsort(subject_specific)[-top_k:][::-1].tolist()

    # Features specific to each mood
    if mood_idx:
        for i, mood in enumerate(MOOD):
            mood_corr = correlation_matrix[:, mood_idx[i]]
            other_mood_corr = np.mean(
                correlation_matrix[:, [j for j in mood_idx if j != mood_idx[i]]],
                axis=1
            )
            mood_specific = mood_corr - other_mood_corr
            result[f"mood_{mood}"] = np.argsort(mood_specific)[-top_k:][::-1].tolist()

    return result


def generate_summary_report(
    results_by_label: dict[str, list[CorrelationResult]],
    feature_analyses: dict[int, FeatureAnalysis],
    differentiating: dict[str, list[int]],
    output_dir: Path,
    layer: int = 8,
) -> None:
    """Generate summary report as JSON and markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary
    summary = {
        "layer": layer,
        "label_dimensions": ALL_LABEL_DIMS,
        "top_features_by_label": {
            label: [
                {
                    "feature_idx": r.feature_idx,
                    "correlation": round(r.correlation, 4),
                    "q_value": round(r.p_value, 6),  # FDR-corrected
                }
                for r in results[:20]
            ]
            for label, results in results_by_label.items()
            if results  # Only include labels with results
        },
        "feature_analyses": {
            str(idx): {
                "feature_idx": a.feature_idx,
                "correlations": {k: round(v, 4) for k, v in a.correlations.items()},
                "max_activation": round(a.max_activation, 4),
                "mean_activation": round(a.mean_activation, 4),
                "num_nonzero": a.num_nonzero,
                "top_artworks": a.top_artworks,
            }
            for idx, a in feature_analyses.items()
        },
        "differentiating_features": differentiating,
        "label_categories": {
            "numeric_ratings": {
                "alexander": ALEXANDER_RATINGS,
                "askell": ASKELL_RATINGS,
                "critical": CRITICAL_RATINGS,
            },
            "subject_matter": SUBJECT_MATTER,
            "mood": MOOD,
        },
    }

    json_path = output_dir / f"correlation_results_layer{layer}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved JSON summary to {json_path}")

    # Markdown report
    md_lines = [
        f"# SAE Feature Correlation Analysis (Layer {layer})",
        "",
        "All p-values are FDR-corrected (Benjamini-Hochberg).",
        "",
        "## Top Features by Numeric Rating",
        "",
    ]

    for rating in NUMERIC_RATINGS:
        results = results_by_label.get(rating, [])
        md_lines.append(f"### {rating}")
        md_lines.append("")
        if results:
            md_lines.append("| Feature | Correlation | q-value |")
            md_lines.append("|---------|-------------|---------|")
            for r in results[:10]:
                md_lines.append(f"| {r.feature_idx} | {r.correlation:.4f} | {r.p_value:.2e} |")
        else:
            md_lines.append("*No FDR-significant correlations*")
        md_lines.append("")

    md_lines.extend(["## Top Features by Subject Matter", ""])
    for subject in SUBJECT_MATTER:
        label = f"subject:{subject}"
        results = results_by_label.get(label, [])
        if results:
            md_lines.append(f"### {subject}")
            md_lines.append("")
            md_lines.append("| Feature | Correlation | q-value |")
            md_lines.append("|---------|-------------|---------|")
            for r in results[:5]:
                md_lines.append(f"| {r.feature_idx} | {r.correlation:.4f} | {r.p_value:.2e} |")
            md_lines.append("")

    md_lines.extend(["## Top Features by Mood", ""])
    for mood in MOOD:
        label = f"mood:{mood}"
        results = results_by_label.get(label, [])
        if results:
            md_lines.append(f"### {mood}")
            md_lines.append("")
            md_lines.append("| Feature | Correlation | q-value |")
            md_lines.append("|---------|-------------|---------|")
            for r in results[:5]:
                md_lines.append(f"| {r.feature_idx} | {r.correlation:.4f} | {r.p_value:.2e} |")
            md_lines.append("")

    md_lines.extend(
        [
            "## Differentiating Features",
            "",
        ]
    )

    if "alexander_specific" in differentiating:
        md_lines.extend([
            "### Alexander-specific (predict 'wholeness' but not 'drawn_to')",
            f"Features: {differentiating['alexander_specific'][:10]}",
            "",
        ])

    if "askell_specific" in differentiating:
        md_lines.extend([
            "### Askell-specific (predict 'drawn_to' but not 'wholeness')",
            f"Features: {differentiating['askell_specific'][:10]}",
            "",
        ])

    if "critical_specific" in differentiating:
        md_lines.extend([
            "### Critical-specific (predict 'technical_skill' vs personal preference)",
            f"Features: {differentiating['critical_specific'][:10]}",
            "",
        ])

    md_path = output_dir / f"correlation_report_layer{layer}.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    logger.info(f"Saved markdown report to {md_path}")


def main(layer: int = 8, output_dir: Path = OUTPUT_DIR, exclude_sources: list[str] | None = None):
    """Main analysis pipeline for a single layer."""
    logger.info("=" * 60)
    logger.info(f"SAE Feature Correlation Analysis (Layer {layer})")
    if exclude_sources:
        logger.info(f"Excluding sources: {exclude_sources}")
    logger.info("=" * 60)

    # Load data
    features = load_all_features(layer=layer)
    if not features:
        logger.error(f"No features found for layer {layer}. Run extract_sae_features.py --layer {layer} first.")
        return

    labels = load_labels(exclude_sources=exclude_sources)
    if not labels:
        logger.error("No labels found in database.")
        return

    # Compute correlations with FDR correction
    correlation_matrix, fdr_matrix, results_by_label = compute_feature_correlations(
        features, labels, apply_fdr=True
    )

    if correlation_matrix.size == 0:
        logger.error("Failed to compute correlations.")
        return

    # Analyze patterns
    feature_analyses = analyze_feature_patterns(features, labels, correlation_matrix)

    # Find differentiating features
    differentiating = find_differentiating_features(correlation_matrix)

    # Generate reports
    generate_summary_report(results_by_label, feature_analyses, differentiating, output_dir, layer=layer)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info(f"Analysis Complete! (Layer {layer})")
    logger.info("=" * 60)

    logger.info("\nTop features by numeric rating (FDR-significant):")
    for rating in NUMERIC_RATINGS:
        results = results_by_label.get(rating, [])
        if results:
            top = results[0]
            logger.info(f"  {rating}: feature {top.feature_idx} (r={top.correlation:.4f}, q={top.p_value:.2e})")

    # Count FDR-significant correlations per category
    subject_count = sum(len(results_by_label.get(f"subject:{s}", [])) for s in SUBJECT_MATTER)
    mood_count = sum(len(results_by_label.get(f"mood:{m}", [])) for m in MOOD)
    logger.info(f"\nFDR-significant correlations:")
    logger.info(f"  Subject matter: {subject_count}")
    logger.info(f"  Mood: {mood_count}")

    logger.info(f"\nResults saved to: {output_dir}")


def main_multi_layer(layers: list[int] = None, output_dir: Path = OUTPUT_DIR, exclude_sources: list[str] | None = None):
    """Run analysis for multiple layers and generate comparison."""
    if layers is None:
        layers = list(SAE_CONFIGS.keys())

    logger.info("=" * 60)
    logger.info(f"Multi-Layer SAE Analysis: Layers {layers}")
    if exclude_sources:
        logger.info(f"Excluding sources: {exclude_sources}")
    logger.info("=" * 60)

    # Run analysis for each layer
    for layer in layers:
        main(layer=layer, output_dir=output_dir, exclude_sources=exclude_sources)

    logger.info("\n" + "=" * 60)
    logger.info("Multi-Layer Analysis Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SAE feature correlations")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        choices=list(SAE_CONFIGS.keys()),
        help="CLIP transformer layer to analyze (default: all layers)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--exclude-sources",
        type=str,
        nargs="+",
        default=None,
        help="Sources to exclude (e.g., --exclude-sources nypl)",
    )
    args = parser.parse_args()

    if args.layer is not None:
        main(layer=args.layer, output_dir=args.output_dir, exclude_sources=args.exclude_sources)
    else:
        # Analyze all layers
        main_multi_layer(output_dir=args.output_dir, exclude_sources=args.exclude_sources)
