#!/usr/bin/env python3
"""
mirror_self Feature Analysis

Deep dive into SAE features that correlate with "mirror_self" (reflection of self).
Identifies which visual features increase or decrease the sense of self-reflection.

Usage:
    python scripts/analyze_mirror_self_features.py [--layer N] [--top-n 50]
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import get_db_context
from database.models import Artwork
from services.sae_features import SAE_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
BASE_FEATURES_DIR = Path(__file__).parent.parent / "data" / "sae_features"
ANALYSIS_DIR = Path(__file__).parent.parent / "output" / "sae_analysis"


@dataclass
class MirrorSelfFeature:
    """A feature that correlates with mirror_self."""
    feature_idx: int
    correlation: float
    direction: str  # "positive" or "negative"
    monosemanticity_score: float | None
    is_monosemantic: bool
    is_labeled: bool
    gemini_label: str | None
    gemini_description: str | None
    other_correlations: dict  # Other ratings this feature correlates with
    top_artworks: list[str]  # Artwork IDs with highest activation


def load_correlation_results(layer: int) -> dict:
    """Load correlation results for a layer."""
    path = ANALYSIS_DIR / f"correlation_results_layer{layer}.json"
    if not path.exists():
        raise FileNotFoundError(f"Correlation results not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_monosemanticity_scores(layer: int) -> dict:
    """Load monosemanticity scores for a layer."""
    path = ANALYSIS_DIR / f"monosemanticity_scores_layer{layer}.json"
    if not path.exists():
        logger.warning(f"Monosemanticity scores not found: {path}")
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("scores", {})


def load_gemini_labels(layer: int) -> dict:
    """Load Gemini labels for a layer."""
    path = ANALYSIS_DIR / f"feature_labels_gemini_layer{layer}.json"
    if not path.exists():
        logger.warning(f"Gemini labels not found: {path}")
        return {}
    with open(path) as f:
        data = json.load(f)

    # Convert to dict keyed by feature_idx
    labels = {}
    for item in data.get("labels", []):
        labels[str(item["feature_idx"])] = item
    return labels


def get_mirror_self_features(
    correlation_results: dict,
    mono_scores: dict,
    gemini_labels: dict,
    top_n: int = 50,
) -> list[MirrorSelfFeature]:
    """Extract top features correlated with mirror_self."""

    # Get feature analyses
    feature_analyses = correlation_results.get("feature_analyses", {})

    # Find all features with mirror_self correlation
    mirror_self_features = []

    for feat_idx_str, analysis in feature_analyses.items():
        correlations = analysis.get("correlations", {})
        if "mirror_self" not in correlations:
            continue

        corr = correlations["mirror_self"]
        if abs(corr) < 0.1:  # Skip weak correlations
            continue

        # Get monosemanticity
        mono_data = mono_scores.get(feat_idx_str, {})
        mono_score = mono_data.get("monosemanticity_score")
        is_mono = mono_score is not None and mono_score >= 0.50  # Calibrated threshold

        # Get Gemini label
        gemini = gemini_labels.get(feat_idx_str, {})

        # Get other strong correlations
        other_corrs = {}
        for rating, val in correlations.items():
            if rating != "mirror_self" and abs(val) >= 0.2:
                other_corrs[rating] = val

        # Get top artworks
        top_arts = mono_data.get("top_artworks", analysis.get("top_artworks", []))

        feature = MirrorSelfFeature(
            feature_idx=int(feat_idx_str),
            correlation=corr,
            direction="positive" if corr > 0 else "negative",
            monosemanticity_score=mono_score,
            is_monosemantic=is_mono,
            is_labeled=bool(gemini),
            gemini_label=gemini.get("short_label"),
            gemini_description=gemini.get("description"),
            other_correlations=other_corrs,
            top_artworks=top_arts[:5] if top_arts else [],
        )
        mirror_self_features.append(feature)

    # Sort by absolute correlation (strongest first)
    mirror_self_features.sort(key=lambda x: abs(x.correlation), reverse=True)

    return mirror_self_features[:top_n]


def load_artwork_metadata(artwork_ids: list[str]) -> dict[str, dict]:
    """Load metadata for specific artworks."""
    if not artwork_ids:
        return {}

    metadata = {}
    with get_db_context() as db:
        artworks = db.query(Artwork).filter(Artwork.id.in_(artwork_ids)).all()
        for art in artworks:
            metadata[art.id] = {
                "title": art.title or "Untitled",
                "artist": art.artist_name or "Unknown",
                "year": art.year,
                "image_url": art.image_url,
                "thumbnail_url": art.thumbnail_url or art.image_url,
            }
    return metadata


def generate_feature_html(
    feature: MirrorSelfFeature,
    artwork_metadata: dict,
    layer: int,
    output_dir: Path,
) -> Path:
    """Generate HTML page for a single feature."""

    direction_color = "#2ecc71" if feature.direction == "positive" else "#e74c3c"
    direction_symbol = "+" if feature.direction == "positive" else ""

    # Build artwork grid
    artwork_html = ""
    for art_id in feature.top_artworks:
        meta = artwork_metadata.get(art_id, {})
        img_url = meta.get("thumbnail_url") or meta.get("image_url", "")
        title = meta.get("title", "Unknown")
        artist = meta.get("artist", "Unknown")

        if img_url:
            artwork_html += f"""
            <div class="artwork-card">
                <img src="{img_url}" alt="{title}" loading="lazy">
                <div class="artwork-info">
                    <strong>{title}</strong><br>
                    <span>{artist}</span>
                </div>
            </div>
            """

    # Build other correlations table
    other_corr_html = ""
    if feature.other_correlations:
        other_corr_html = "<h3>Other Strong Correlations</h3><table><tr><th>Rating</th><th>Correlation</th></tr>"
        for rating, val in sorted(feature.other_correlations.items(), key=lambda x: -abs(x[1])):
            color = "#2ecc71" if val > 0 else "#e74c3c"
            other_corr_html += f'<tr><td>{rating}</td><td style="color:{color}">{val:+.3f}</td></tr>'
        other_corr_html += "</table>"

    # Label section
    label_html = ""
    if feature.is_labeled:
        label_html = f"""
        <div class="label-section">
            <h3>Gemini Label: {feature.gemini_label}</h3>
            <p>{feature.gemini_description}</p>
        </div>
        """
    else:
        label_html = '<div class="unlabeled-notice">This feature has not been labeled yet.</div>'

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Feature {feature.feature_idx} - mirror_self Analysis</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .correlation {{ font-size: 48px; font-weight: bold; color: {direction_color}; }}
        .direction {{ font-size: 24px; color: {direction_color}; margin-top: 10px; }}
        .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; }}
        .stat-label {{ color: #666; font-size: 14px; }}
        .artwork-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .artwork-card {{ border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
        .artwork-card img {{ width: 100%; height: 200px; object-fit: cover; }}
        .artwork-info {{ padding: 10px; font-size: 12px; }}
        .label-section {{ background: #e8f5e9; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .unlabeled-notice {{ background: #fff3e0; padding: 15px; border-radius: 8px; margin: 20px 0; color: #e65100; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; }}
        .mono-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; }}
        .mono-yes {{ background: #c8e6c9; color: #2e7d32; }}
        .mono-no {{ background: #ffcdd2; color: #c62828; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Feature {feature.feature_idx}</h1>
        <div class="correlation">{direction_symbol}{feature.correlation:.3f}</div>
        <div class="direction">{feature.direction.upper()} correlation with mirror_self</div>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{f'{feature.monosemanticity_score:.3f}' if feature.monosemanticity_score else 'N/A'}</div>
            <div class="stat-label">Monosemanticity Score</div>
            <span class="mono-badge {'mono-yes' if feature.is_monosemantic else 'mono-no'}">
                {'Monosemantic' if feature.is_monosemantic else 'Polysemantic'}
            </span>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(feature.other_correlations)}</div>
            <div class="stat-label">Other Strong Correlations</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">Layer {layer}</div>
            <div class="stat-label">SAE Layer</div>
        </div>
    </div>

    {label_html}

    <h2>Top Activating Artworks</h2>
    <p>These artworks most strongly activate this feature:</p>
    <div class="artwork-grid">
        {artwork_html if artwork_html else '<p>No artwork data available</p>'}
    </div>

    {other_corr_html}

    <p style="margin-top: 40px; color: #666; font-size: 12px;">
        <a href="mirror_self_analysis_index.html">Back to mirror_self Analysis Index</a>
    </p>
</body>
</html>"""

    output_path = output_dir / f"feature_{feature.feature_idx}_mirror_self.html"
    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def generate_index_html(
    features: list[MirrorSelfFeature],
    layer: int,
    output_dir: Path,
) -> Path:
    """Generate index page for mirror_self analysis."""

    # Split into positive and negative
    positive = [f for f in features if f.direction == "positive"]
    negative = [f for f in features if f.direction == "negative"]

    # Stats
    total = len(features)
    labeled = sum(1 for f in features if f.is_labeled)
    monosemantic = sum(1 for f in features if f.is_monosemantic)

    def feature_row(f: MirrorSelfFeature) -> str:
        color = "#2ecc71" if f.direction == "positive" else "#e74c3c"
        mono_badge = '<span class="badge mono">Mono</span>' if f.is_monosemantic else ''
        label_badge = '<span class="badge labeled">Labeled</span>' if f.is_labeled else '<span class="badge unlabeled">Unlabeled</span>'
        label_text = f.gemini_label or "—"
        return f"""
        <tr>
            <td><a href="feature_{f.feature_idx}_mirror_self.html">{f.feature_idx}</a></td>
            <td style="color:{color};font-weight:bold">{f.correlation:+.3f}</td>
            <td>{f'{f.monosemanticity_score:.3f}' if f.monosemanticity_score else '—'}</td>
            <td>{mono_badge} {label_badge}</td>
            <td>{label_text}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>mirror_self Feature Analysis - Layer {layer}</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 10px; margin-bottom: 30px; }}
        h1 {{ margin: 0; }}
        .subtitle {{ opacity: 0.9; margin-top: 10px; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 30px 0; }}
        .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 36px; font-weight: bold; color: #333; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .section {{ margin: 40px 0; }}
        .section-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 15px; }}
        .section-header h2 {{ margin: 0; }}
        .indicator {{ width: 20px; height: 20px; border-radius: 50%; }}
        .indicator.positive {{ background: #2ecc71; }}
        .indicator.negative {{ background: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f5f5f5; font-weight: 600; }}
        tr:hover {{ background: #f9f9f9; }}
        a {{ color: #667eea; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; margin-right: 4px; }}
        .badge.mono {{ background: #c8e6c9; color: #2e7d32; }}
        .badge.labeled {{ background: #e3f2fd; color: #1565c0; }}
        .badge.unlabeled {{ background: #fff3e0; color: #e65100; }}
        .hypothesis {{ background: #f3e5f5; padding: 20px; border-radius: 8px; margin: 30px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>mirror_self Feature Analysis</h1>
        <p class="subtitle">Which visual features make art feel like "a mirror for the self"?</p>
        <p class="subtitle">Layer {layer} SAE Features</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{total}</div>
            <div class="stat-label">Features Analyzed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#2ecc71">{len(positive)}</div>
            <div class="stat-label">Increase mirror_self</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#e74c3c">{len(negative)}</div>
            <div class="stat-label">Decrease mirror_self</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{labeled}/{total}</div>
            <div class="stat-label">Features Labeled</div>
        </div>
    </div>

    <div class="hypothesis">
        <h3>Research Question</h3>
        <p><strong>What visual features make viewers see themselves reflected in an artwork?</strong></p>
        <p>Features with positive correlation INCREASE the sense of self-reflection.<br>
        Features with negative correlation DECREASE it (e.g., external, objective content).</p>
    </div>

    <div class="section">
        <div class="section-header">
            <div class="indicator positive"></div>
            <h2>Features that INCREASE mirror_self ({len(positive)} features)</h2>
        </div>
        <table>
            <tr><th>Feature</th><th>Correlation</th><th>Mono Score</th><th>Status</th><th>Label</th></tr>
            {''.join(feature_row(f) for f in positive)}
        </table>
    </div>

    <div class="section">
        <div class="section-header">
            <div class="indicator negative"></div>
            <h2>Features that DECREASE mirror_self ({len(negative)} features)</h2>
        </div>
        <table>
            <tr><th>Feature</th><th>Correlation</th><th>Mono Score</th><th>Status</th><th>Label</th></tr>
            {''.join(feature_row(f) for f in negative)}
        </table>
    </div>

    <p style="margin-top: 40px; color: #666; font-size: 12px;">
        Generated from Layer {layer} SAE correlation analysis
    </p>
</body>
</html>"""

    output_path = output_dir / "mirror_self_analysis_index.html"
    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Analyze mirror_self correlating features")
    parser.add_argument("--layer", type=int, default=11, choices=list(SAE_CONFIGS.keys()),
                        help="SAE layer to analyze (default: 11)")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Number of top features to analyze (default: 50)")
    args = parser.parse_args()

    layer = args.layer
    top_n = args.top_n

    print("=" * 60)
    print(f"mirror_self Feature Analysis (Layer {layer})")
    print("=" * 60)

    # Load data
    print("\nLoading correlation results...")
    corr_results = load_correlation_results(layer)

    print("Loading monosemanticity scores...")
    mono_scores = load_monosemanticity_scores(layer)

    print("Loading Gemini labels...")
    gemini_labels = load_gemini_labels(layer)

    # Extract mirror_self features
    print(f"\nExtracting top {top_n} mirror_self features...")
    features = get_mirror_self_features(corr_results, mono_scores, gemini_labels, top_n)

    print(f"Found {len(features)} features with |correlation| >= 0.1")

    # Stats
    positive = [f for f in features if f.direction == "positive"]
    negative = [f for f in features if f.direction == "negative"]
    labeled = [f for f in features if f.is_labeled]
    unlabeled = [f for f in features if not f.is_labeled]
    monosemantic = [f for f in features if f.is_monosemantic]

    print(f"\n  Positive correlation (increase mirror_self): {len(positive)}")
    print(f"  Negative correlation (decrease mirror_self): {len(negative)}")
    print(f"  Already labeled: {len(labeled)}")
    print(f"  Unlabeled (need Gemini): {len(unlabeled)}")
    print(f"  Monosemantic (score >= 0.50): {len(monosemantic)}")

    # Collect all artwork IDs for metadata
    all_artwork_ids = set()
    for f in features:
        all_artwork_ids.update(f.top_artworks)

    print(f"\nLoading metadata for {len(all_artwork_ids)} artworks...")
    artwork_metadata = load_artwork_metadata(list(all_artwork_ids))

    # Generate HTML pages
    print("\nGenerating HTML visualizations...")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    for feature in features:
        generate_feature_html(feature, artwork_metadata, layer, ANALYSIS_DIR)

    # Generate index
    index_path = generate_index_html(features, layer, ANALYSIS_DIR)
    print(f"  Index: {index_path}")

    # Save JSON data
    output_data = {
        "layer": layer,
        "total_features": len(features),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "labeled_count": len(labeled),
        "unlabeled_count": len(unlabeled),
        "monosemantic_count": len(monosemantic),
        "features": [asdict(f) for f in features],
        "unlabeled_feature_ids": [f.feature_idx for f in unlabeled],
    }

    json_path = ANALYSIS_DIR / f"mirror_self_features_layer{layer}.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  JSON: {json_path}")

    # Print top unlabeled features (for Gemini labeling)
    print("\n" + "=" * 60)
    print("TOP UNLABELED FEATURES (for Gemini labeling)")
    print("=" * 60)

    for f in unlabeled[:20]:
        sign = "+" if f.direction == "positive" else ""
        mono = "MONO" if f.is_monosemantic else "poly"
        print(f"  Feature {f.feature_idx}: {sign}{f.correlation:.3f} [{mono}]")

    print(f"\nTotal unlabeled to label: {len(unlabeled)}")
    print("\nDone!")


if __name__ == "__main__":
    main()
