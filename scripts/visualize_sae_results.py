#!/usr/bin/env python3
"""
SAE Feature Visualization

Generates static HTML pages visualizing the correlation analysis results.
Shows top features, their correlations, and top-activating artworks.

Usage:
    python scripts/visualize_sae_results.py [--layer N]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from database.connection import get_db_context
from database.models import Artwork
from services.sae_features import SAE_CONFIGS, load_features_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
BASE_FEATURES_DIR = Path(__file__).parent.parent / "data" / "sae_features"
ANALYSIS_DIR = Path(__file__).parent.parent / "output" / "sae_analysis"
OUTPUT_DIR = ANALYSIS_DIR  # Put HTML in same directory as analysis

# Rating categories (numeric only)
RATING_CATEGORIES = {
    "Alexander (Wholeness)": ["mirror_self", "wholeness", "inner_light"],
    "Askell (Personal)": ["deepest_honest", "drawn_to", "choose_to_look"],
    "Critical (Technical)": ["technical_skill", "emotional_impact"],
}

NUMERIC_RATINGS = [
    "mirror_self", "wholeness", "inner_light",
    "deepest_honest", "drawn_to", "choose_to_look",
    "technical_skill", "emotional_impact"
]

# Subject matter and mood categories
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


def get_features_dir(layer: int) -> Path:
    """Get layer-specific features directory."""
    return BASE_FEATURES_DIR / f"layer_{layer}"


def load_artwork_metadata() -> dict[str, dict]:
    """Load artwork metadata from database."""
    metadata = {}
    with get_db_context() as db:
        artworks = db.query(Artwork).filter(Artwork.labels.isnot(None)).all()
        for artwork in artworks:
            try:
                labels = json.loads(artwork.labels) if artwork.labels else {}
            except json.JSONDecodeError:
                labels = {}

            metadata[artwork.id] = {
                "title": artwork.title or "Untitled",
                "artist": artwork.artist_name or "Unknown",
                "year": artwork.year,
                "image_url": artwork.image_url,
                "thumbnail_url": artwork.thumbnail_url or artwork.image_url,
                "labels": labels,
            }
    return metadata


def generate_correlation_heatmap(results: dict, output_path: Path, layer: int = 8) -> None:
    """Generate a heatmap of top feature correlations with numeric ratings."""
    # Get top features across numeric ratings
    all_features = set()
    top_features_data = results.get("top_features_by_label", results.get("top_features_by_rating", {}))
    for rating in NUMERIC_RATINGS:
        rating_results = top_features_data.get(rating, [])
        for r in rating_results[:10]:
            all_features.add(r["feature_idx"])

    top_features = sorted(all_features)[:30]

    # Build correlation matrix
    feature_analyses = results.get("feature_analyses", {})

    matrix = np.zeros((len(top_features), len(NUMERIC_RATINGS)))
    for i, feat_idx in enumerate(top_features):
        feat_data = feature_analyses.get(str(feat_idx), {})
        correlations = feat_data.get("correlations", {})
        for j, rating in enumerate(NUMERIC_RATINGS):
            matrix[i, j] = correlations.get(rating, 0)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix,
        xticklabels=NUMERIC_RATINGS,
        yticklabels=[f"Feature {f}" for f in top_features],
        cmap="RdBu_r",
        center=0,
        vmin=-0.4,
        vmax=0.4,
        annot=True,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title(f"SAE Feature Correlations with Aesthetic Ratings (Layer {layer})")
    ax.set_xlabel("Rating")
    ax.set_ylabel("SAE Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved heatmap to {output_path}")


def generate_feature_importance_chart(results: dict, output_path: Path, layer: int = 8) -> None:
    """Generate bar chart of feature importance per rating."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    top_features_data = results.get("top_features_by_label", results.get("top_features_by_rating", {}))

    for idx, rating in enumerate(NUMERIC_RATINGS):
        ax = axes[idx]
        rating_results = top_features_data.get(rating, [])

        if rating_results:
            features = [r["feature_idx"] for r in rating_results[:10]]
            correlations = [r["correlation"] for r in rating_results[:10]]
            colors = ["green" if c > 0 else "red" for c in correlations]

            ax.barh(range(len(features)), correlations, color=colors, alpha=0.7)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels([f"F{f}" for f in features])
            ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
            ax.set_xlim(-0.5, 0.5)

        ax.set_title(rating, fontsize=10)
        ax.set_xlabel("Correlation")

    plt.suptitle(f"Top 10 SAE Features per Rating (Layer {layer})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved importance chart to {output_path}")


def generate_feature_page(
    feat_idx: int,
    feat_data: dict,
    features: dict[str, np.ndarray],
    metadata: dict[str, dict],
    output_dir: Path,
    layer: int = 8,
) -> str:
    """Generate HTML page for a single feature."""
    # Get top artworks for this feature
    top_artworks = feat_data.get("top_artworks", [])
    correlations = feat_data.get("correlations", {})

    # Build artwork cards HTML
    artwork_cards = []
    for artwork_id in top_artworks[:10]:
        if artwork_id not in metadata:
            continue

        art_meta = metadata[artwork_id]
        labels = art_meta.get("labels", {})

        # Get ratings
        ratings_html = []
        for rating in NUMERIC_RATINGS:
            value = labels.get(rating)
            if value is not None:
                ratings_html.append(f"<span class='rating'>{rating}: {value}</span>")

        # Get feature activation
        activation = 0
        if artwork_id in features:
            activation = features[artwork_id][feat_idx]

        card = f"""
        <div class="artwork-card">
            <img src="{art_meta.get('thumbnail_url', '')}" alt="{art_meta.get('title', '')}" loading="lazy">
            <div class="artwork-info">
                <h4>{art_meta.get('title', 'Untitled')[:50]}</h4>
                <p class="artist">{art_meta.get('artist', 'Unknown')}</p>
                <p class="activation">Activation: {activation:.3f}</p>
                <div class="ratings">{' '.join(ratings_html[:4])}</div>
            </div>
        </div>
        """
        artwork_cards.append(card)

    # Build correlation table for numeric ratings
    numeric_corr_rows = []
    for rating in NUMERIC_RATINGS:
        corr = correlations.get(rating, 0)
        color = "green" if corr > 0 else "red" if corr < 0 else "gray"
        numeric_corr_rows.append(f"<tr><td>{rating}</td><td style='color:{color}'>{corr:.4f}</td></tr>")

    # Build correlation table for subject matter
    subject_corr_rows = []
    for subject in SUBJECT_MATTER:
        key = f"subject:{subject}"
        corr = correlations.get(key, 0)
        if abs(corr) > 0.05:  # Only show non-trivial correlations
            color = "green" if corr > 0 else "red"
            subject_corr_rows.append(f"<tr><td>{subject}</td><td style='color:{color}'>{corr:.4f}</td></tr>")

    # Build correlation table for mood
    mood_corr_rows = []
    for mood in MOOD:
        key = f"mood:{mood}"
        corr = correlations.get(key, 0)
        if abs(corr) > 0.05:  # Only show non-trivial correlations
            color = "green" if corr > 0 else "red"
            mood_corr_rows.append(f"<tr><td>{mood}</td><td style='color:{color}'>{corr:.4f}</td></tr>")

    # Backwards compatibility: use old format if new format not present
    corr_rows = numeric_corr_rows

    # Build additional sections for subject/mood if available
    subject_section = ""
    if subject_corr_rows:
        subject_section = f"""
        <h4>Subject Matter Correlations</h4>
        <table>
            <tr><th>Subject</th><th>Correlation</th></tr>
            {''.join(subject_corr_rows)}
        </table>
        """

    mood_section = ""
    if mood_corr_rows:
        mood_section = f"""
        <h4>Mood Correlations</h4>
        <table>
            <tr><th>Mood</th><th>Correlation</th></tr>
            {''.join(mood_corr_rows)}
        </table>
        """

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SAE Feature {feat_idx} (Layer {layer})</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; }}
        .stats {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .corr-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        .artwork-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }}
        .artwork-card {{ background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .artwork-card img {{ width: 100%; height: 200px; object-fit: cover; }}
        .artwork-info {{ padding: 12px; }}
        .artwork-info h4 {{ margin: 0 0 8px 0; font-size: 14px; }}
        .artist {{ color: #666; font-size: 12px; margin: 0; }}
        .activation {{ color: #0066cc; font-weight: bold; font-size: 12px; }}
        .ratings {{ font-size: 10px; color: #888; margin-top: 8px; }}
        .rating {{ display: inline-block; margin-right: 8px; }}
        a {{ color: #0066cc; }}
    </style>
</head>
<body>
    <div class="container">
        <p><a href="index_layer{layer}.html">&larr; Back to Layer {layer} overview</a></p>
        <h1>SAE Feature {feat_idx} (Layer {layer})</h1>

        <div class="stats">
            <h3>Correlations with Aesthetic Ratings</h3>
            <div class="corr-grid">
                <div>
                    <h4>Numeric Ratings</h4>
                    <table>
                        <tr><th>Rating</th><th>Correlation</th></tr>
                        {''.join(corr_rows)}
                    </table>
                </div>
                <div>{subject_section}</div>
                <div>{mood_section}</div>
            </div>
            <p><strong>Max Activation:</strong> {feat_data.get('max_activation', 0):.4f}</p>
            <p><strong>Mean Activation:</strong> {feat_data.get('mean_activation', 0):.6f}</p>
            <p><strong>Non-zero in:</strong> {feat_data.get('num_nonzero', 0)} artworks</p>
        </div>

        <h2>Top Activating Artworks</h2>
        <p>These artworks have the highest activation for this feature:</p>
        <div class="artwork-grid">
            {''.join(artwork_cards)}
        </div>
    </div>
</body>
</html>
"""

    output_path = output_dir / f"feature_{feat_idx}_layer{layer}.html"
    with open(output_path, "w") as f:
        f.write(html)

    return f"feature_{feat_idx}_layer{layer}.html"


def generate_index_page(
    results: dict,
    feature_pages: dict[int, str],
    output_dir: Path,
    layer: int = 8,
) -> None:
    """Generate main index page for a specific layer."""
    top_features_data = results.get("top_features_by_label", results.get("top_features_by_rating", {}))

    # Build rating sections
    rating_sections = []
    for category, ratings in RATING_CATEGORIES.items():
        rating_tables = []
        for rating in ratings:
            rating_results = top_features_data.get(rating, [])
            rows = []
            for r in rating_results[:10]:
                feat_idx = r["feature_idx"]
                link = feature_pages.get(feat_idx, "#")
                corr = r["correlation"]
                color = "green" if corr > 0 else "red"
                q_val = r.get('q_value', r.get('p_value', 0))
                rows.append(f"""
                    <tr>
                        <td><a href="{link}">Feature {feat_idx}</a></td>
                        <td style="color:{color}">{corr:.4f}</td>
                        <td>{q_val:.2e}</td>
                    </tr>
                """)

            rating_tables.append(f"""
                <div class="rating-table">
                    <h4>{rating}</h4>
                    <table>
                        <tr><th>Feature</th><th>Correlation</th><th>q-value</th></tr>
                        {''.join(rows) if rows else '<tr><td colspan="3"><em>No FDR-significant correlations</em></td></tr>'}
                    </table>
                </div>
            """)

        rating_sections.append(f"""
            <div class="category-section">
                <h3>{category}</h3>
                <div class="rating-grid">{''.join(rating_tables)}</div>
            </div>
        """)

    # Build differentiating features section
    diff_features = results.get("differentiating_features", {})
    diff_sections = []
    # Show key differentiating types
    for diff_type in ["alexander_specific", "askell_specific", "critical_specific"]:
        if diff_type in diff_features:
            features = diff_features[diff_type]
            links = [f'<a href="{feature_pages.get(f, "#")}">F{f}</a>' for f in features[:10]]
            diff_sections.append(f"<p><strong>{diff_type}:</strong> {', '.join(links)}</p>")

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SAE Feature Correlation Analysis - Layer {layer}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1, h2, h3 {{ color: #333; }}
        .intro {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .category-section {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .rating-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .rating-table h4 {{ margin: 0 0 10px 0; color: #555; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
        th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f9f9f9; }}
        a {{ color: #0066cc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .images {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
        .images img {{ max-width: 45%; border-radius: 8px; min-width: 300px; }}
        .diff-section {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .layer-nav {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .layer-nav a {{ margin-right: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SAE Feature Correlation Analysis - Layer {layer}</h1>

        <div class="layer-nav">
            <strong>Layers:</strong>
            <a href="index_layer7.html">Layer 7 (Early)</a>
            <a href="index_layer8.html">Layer 8 (Mid)</a>
            <a href="index_layer11.html">Layer 11 (Late)</a>
        </div>

        <div class="intro">
            <h2>Research Question</h2>
            <p><strong>"What visual features do vision models use to predict aesthetic judgments?"</strong></p>
            <p>This analysis correlates 49,152 sparse autoencoder (SAE) features from CLIP ViT-B-32 Layer {layer}
               with aesthetic ratings and categorical labels.</p>
            <p><strong>Layer {layer} interpretation:</strong>
            {"Early features - textures, patterns, low-level visual elements" if layer == 7 else
             "Mid-level features - compositional elements, object parts" if layer == 8 else
             "Late features - semantic concepts, high-level meanings"}</p>
            <p><em>All correlations are FDR-corrected (Benjamini-Hochberg, q &lt; 0.05).</em></p>
        </div>

        <div class="images">
            <img src="correlation_heatmap_layer{layer}.png" alt="Correlation Heatmap">
            <img src="feature_importance_layer{layer}.png" alt="Feature Importance">
        </div>

        {''.join(rating_sections)}

        <div class="diff-section">
            <h2>Differentiating Features</h2>
            <p>Features that distinguish between rating types:</p>
            {''.join(diff_sections) if diff_sections else '<p><em>Analysis pending</em></p>'}
        </div>
    </div>
</body>
</html>
"""

    output_path = output_dir / f"index_layer{layer}.html"
    with open(output_path, "w") as f:
        f.write(html)
    logger.info(f"Saved index page to {output_path}")


def main(layer: int = 8):
    """Main visualization pipeline for a single layer."""
    logger.info("=" * 60)
    logger.info(f"SAE Visualization Generator (Layer {layer})")
    logger.info("=" * 60)

    # Load analysis results
    results_path = ANALYSIS_DIR / f"correlation_results_layer{layer}.json"
    if not results_path.exists():
        # Try old format for backwards compatibility
        results_path = ANALYSIS_DIR / "correlation_results.json"
        if not results_path.exists():
            logger.error(f"Analysis results not found for layer {layer}")
            logger.error(f"Run: python scripts/analyze_sae_correlations.py --layer {layer}")
            return

    with open(results_path) as f:
        results = json.load(f)
    logger.info(f"Loaded analysis results from {results_path}")

    # Load features
    features_dir = get_features_dir(layer)
    if not features_dir.exists():
        # Try old format for backwards compatibility
        features_dir = BASE_FEATURES_DIR

    logger.info(f"Loading SAE features from {features_dir}...")
    features = {}
    for batch_file in sorted(features_dir.glob("batch_*.json")):
        batch_features = load_features_batch(batch_file, to_dense=True)
        features.update(batch_features)
    logger.info(f"Loaded features for {len(features)} artworks")

    # Load artwork metadata
    logger.info("Loading artwork metadata...")
    metadata = load_artwork_metadata()
    logger.info(f"Loaded metadata for {len(metadata)} artworks")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate charts
    logger.info("Generating charts...")
    generate_correlation_heatmap(results, OUTPUT_DIR / f"correlation_heatmap_layer{layer}.png", layer=layer)
    generate_feature_importance_chart(results, OUTPUT_DIR / f"feature_importance_layer{layer}.png", layer=layer)

    # Generate feature pages
    logger.info("Generating feature pages...")
    feature_pages = {}
    feature_analyses = results.get("feature_analyses", {})
    for feat_idx_str, feat_data in feature_analyses.items():
        feat_idx = int(feat_idx_str)
        page_name = generate_feature_page(feat_idx, feat_data, features, metadata, OUTPUT_DIR, layer=layer)
        feature_pages[feat_idx] = page_name
    logger.info(f"Generated {len(feature_pages)} feature pages")

    # Generate index page
    logger.info("Generating index page...")
    generate_index_page(results, feature_pages, OUTPUT_DIR, layer=layer)

    logger.info("\n" + "=" * 60)
    logger.info(f"Visualization Complete! (Layer {layer})")
    logger.info(f"Open {OUTPUT_DIR / f'index_layer{layer}.html'} to view results")
    logger.info("=" * 60)


def main_multi_layer(layers: list[int] = None):
    """Generate visualizations for multiple layers."""
    if layers is None:
        layers = list(SAE_CONFIGS.keys())

    logger.info("=" * 60)
    logger.info(f"Multi-Layer Visualization: Layers {layers}")
    logger.info("=" * 60)

    for layer in layers:
        main(layer=layer)

    logger.info("\n" + "=" * 60)
    logger.info("Multi-Layer Visualization Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SAE visualization")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        choices=list(SAE_CONFIGS.keys()),
        help="CLIP transformer layer to visualize (default: all layers)",
    )
    args = parser.parse_args()

    if args.layer is not None:
        main(layer=args.layer)
    else:
        main_multi_layer()
