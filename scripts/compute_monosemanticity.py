#!/usr/bin/env python3
"""
Monosemanticity Scoring for SAE Features (Optimized with NumPy/SciPy)

Measures which SAE features are truly "monosemantic" (single-concept) vs
"polysemantic" (mixed concepts). Only ~60-70% of vision SAE features are
monosemantic, so this helps prioritize interpretation efforts.

Method (based on SAE-for-VLM, NeurIPS 2025):
1. For each feature, get all artworks and their activation values
2. Get DINOv2 embeddings for those artworks (independent of CLIP to avoid circularity)
3. Compute activation-weighted pairwise cosine similarity
4. MS score = weighted mean similarity (high = clear single concept)

Formula: MS^k = (1/Z) Σ Σ (a^k_n * a^k_m) * sim(n, m)
where a^k_n is normalized activation, sim is cosine similarity of DINOv2 embeddings

IMPORTANT: Uses DINOv2 embeddings (not CLIP) because the SAE was trained on CLIP.
Using CLIP embeddings would be circular - we'd just be measuring whether the SAE
selects coherent regions in CLIP space, which it does by construction.

Research basis: https://arxiv.org/abs/2504.02821
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import get_db_context
from database.models import Artwork
from services.sae_features import SAE_CONFIGS, SAE_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
BASE_FEATURES_DIR = Path(__file__).parent.parent / "data" / "sae_features"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "sae_analysis"
FINE_ART_IDS_FILE = Path(__file__).parent.parent / "data" / "fine_art_artwork_ids.json"

# Configuration
TOP_K_ARTWORKS = 20  # Number of top-activating artworks to consider
MIN_ACTIVATIONS = 10  # Minimum artworks where feature activates for scoring
STREAMING_THRESHOLD_GB = 5  # Use streaming mode for layers with data > this size

# Calibrated thresholds based on random art pair similarity distribution
# Random pairs: mean=0.195, std=0.154 (computed on 50k pairs from 16k artworks)
BASELINE_MEAN = 0.195
BASELINE_STD = 0.154
POLYSEMANTIC_THRESHOLD = BASELINE_MEAN + 1 * BASELINE_STD  # ~0.35 (+1σ)
MONOSEMANTIC_THRESHOLD = BASELINE_MEAN + 2 * BASELINE_STD  # ~0.50 (+2σ)


def get_features_dir(layer: int) -> Path:
    """Get layer-specific features directory."""
    return BASE_FEATURES_DIR / f"layer_{layer}"


def load_fine_art_ids() -> set[str]:
    """Load the fine art artwork IDs filter."""
    if not FINE_ART_IDS_FILE.exists():
        raise FileNotFoundError(f"Fine art IDs file not found: {FINE_ART_IDS_FILE}")
    with open(FINE_ART_IDS_FILE) as f:
        data = json.load(f)
    return set(data["artwork_ids"])


def load_artwork_embeddings(
    exclude_sources: list[str] | None = None,
    fine_art_only: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Load DINOv2 embeddings from database.

    IMPORTANT: We use DINOv2 embeddings (1024-dim) instead of CLIP because:
    1. The SAE was trained on CLIP - using CLIP would be circular
    2. DINOv2 provides independent visual similarity measure
    3. This matches the methodology in SAE-for-VLM (NeurIPS 2025)

    Args:
        exclude_sources: List of sources to exclude (e.g., ['nypl'])
        fine_art_only: If True, filter to fine art IDs only (excludes NYPL + carriage drawings)

    Returns:
        embeddings_matrix: Shape (num_artworks, 1024)
        artwork_ids: List of artwork IDs
    """
    logger.info("Loading DINOv2 embeddings from database...")

    # Load fine art filter if requested
    fine_art_ids = None
    if fine_art_only:
        fine_art_ids = load_fine_art_ids()
        logger.info(f"Fine art filter: {len(fine_art_ids)} artwork IDs")

    with get_db_context() as db:
        query = db.query(Artwork).filter(Artwork.embedding_dinov2.isnot(None))

        if exclude_sources:
            query = query.filter(~Artwork.source.in_(exclude_sources))

        if fine_art_ids:
            query = query.filter(Artwork.id.in_(list(fine_art_ids)))

        artworks = query.all()

        artwork_ids = []
        embeddings_list = []

        for art in artworks:
            emb = np.frombuffer(art.embedding_dinov2, dtype=np.float32).copy()
            # Normalize for cosine similarity
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings_list.append(emb)
            artwork_ids.append(art.id)

    # Stack into matrix (num_artworks x embedding_dim)
    embeddings_matrix = np.vstack(embeddings_list)
    logger.info(f"Loaded {len(artwork_ids)} DINOv2 embeddings, shape: {embeddings_matrix.shape}")
    return embeddings_matrix, artwork_ids


def load_sae_features_as_sparse(
    artwork_ids: list[str],
    layer: int = 8,
) -> tuple[sparse.csr_matrix, list[str]]:
    """
    Load SAE features into a sparse matrix (memory-efficient streaming).

    Args:
        artwork_ids: List of artwork IDs to include (filter to these only)
        layer: SAE layer to load features from

    Returns:
        sparse_matrix: Shape (num_artworks, SAE_DIM) in CSR format
        valid_artwork_ids: List of artwork IDs that have SAE features
    """
    features_dir = get_features_dir(layer)
    logger.info(f"Loading SAE features from {features_dir}...")

    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    # Create set for O(1) lookup
    target_ids = set(artwork_ids)

    # Stream through batch files, only keeping artworks we need
    rows = []
    cols = []
    data = []
    valid_ids = []
    id_to_row = {}

    batch_files = sorted(features_dir.glob("batch_*.json"))
    for batch_file in tqdm(batch_files, desc="Loading SAE features"):
        with open(batch_file) as f:
            batch_data = json.load(f)

        for artwork_id, features in batch_data.items():
            if artwork_id not in target_ids:
                continue

            # Assign row index
            if artwork_id not in id_to_row:
                row_idx = len(valid_ids)
                id_to_row[artwork_id] = row_idx
                valid_ids.append(artwork_id)
            else:
                row_idx = id_to_row[artwork_id]

            # Add non-zero entries
            for feat_idx_str, activation in features.items():
                rows.append(row_idx)
                cols.append(int(feat_idx_str))
                data.append(activation)

        # Clear batch_data to free memory
        del batch_data

    logger.info(f"Found {len(valid_ids)} artworks with both embeddings and SAE features")

    # Create sparse matrix
    sparse_matrix = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(valid_ids), SAE_DIM),
        dtype=np.float32,
    ).tocsr()

    # Free COO arrays
    del rows, cols, data

    logger.info(f"Sparse matrix shape: {sparse_matrix.shape}, nnz: {sparse_matrix.nnz:,}")
    return sparse_matrix, valid_ids


def get_layer_data_size_gb(layer: int) -> float:
    """Estimate data size for a layer in GB."""
    features_dir = get_features_dir(layer)
    if not features_dir.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in features_dir.glob("batch_*.json"))
    return total_bytes / (1024 ** 3)


def build_feature_to_artworks_streaming(
    artwork_ids: list[str],
    embeddings_matrix: np.ndarray,
    layer: int,
    top_k: int = TOP_K_ARTWORKS,
    min_activations: int = MIN_ACTIVATIONS,
) -> dict[int, dict]:
    """
    Stream through batch files and compute monosemanticity scores incrementally.

    Memory-efficient approach for layers with large activation data:
    1. Build feature -> [(artwork_idx, activation)] mapping incrementally
    2. Keep only top-k per feature using a heap
    3. Compute scores as we go

    Args:
        artwork_ids: List of artwork IDs (maps to rows in embeddings_matrix)
        embeddings_matrix: Pre-loaded DINOv2 embeddings (num_artworks, 1024)
        layer: SAE layer to process
        top_k: Number of top-activating artworks per feature
        min_activations: Minimum activations required for scoring

    Returns:
        Mapping of feature_idx -> {score, num_activations, top_artworks, ...}
    """
    import heapq
    from collections import defaultdict

    features_dir = get_features_dir(layer)
    logger.info(f"Streaming SAE features from {features_dir}...")

    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    # Create mapping from artwork_id -> index in embeddings_matrix
    id_to_idx = {aid: idx for idx, aid in enumerate(artwork_ids)}

    # Feature -> list of (activation, artwork_idx) - using min-heap for top-k
    # Also track total activation count per feature
    feature_heaps: dict[int, list] = defaultdict(list)
    feature_counts: dict[int, int] = defaultdict(int)

    batch_files = sorted(features_dir.glob("batch_*.json"))
    artworks_processed = 0

    for batch_file in tqdm(batch_files, desc="Streaming SAE features"):
        with open(batch_file) as f:
            batch_data = json.load(f)

        for artwork_id, features in batch_data.items():
            if artwork_id not in id_to_idx:
                continue

            artwork_idx = id_to_idx[artwork_id]
            artworks_processed += 1

            for feat_idx_str, activation in features.items():
                feat_idx = int(feat_idx_str)
                feature_counts[feat_idx] += 1

                # Maintain top-k using min-heap
                heap = feature_heaps[feat_idx]
                if len(heap) < top_k:
                    heapq.heappush(heap, (activation, artwork_idx))
                elif activation > heap[0][0]:
                    heapq.heapreplace(heap, (activation, artwork_idx))

        # Free batch data
        del batch_data

    logger.info(f"Processed {artworks_processed} artworks, found {len(feature_heaps)} active features")

    # Now compute monosemanticity scores for each feature
    scores = {}
    skipped_low = 0

    for feat_idx in tqdm(sorted(feature_heaps.keys()), desc="Computing scores"):
        count = feature_counts[feat_idx]
        if count < min_activations:
            skipped_low += 1
            continue

        # Get top-k as sorted list (highest first)
        heap = feature_heaps[feat_idx]
        top_items = sorted(heap, reverse=True)  # Sort by activation descending

        top_activations = np.array([act for act, _ in top_items], dtype=np.float32)
        top_artwork_indices = [idx for _, idx in top_items]

        # Get embeddings
        top_embeddings = embeddings_matrix[top_artwork_indices]

        if len(top_embeddings) < min_activations // 2:
            continue

        # Compute score
        mono_score = compute_weighted_pairwise_similarity(top_embeddings, top_activations)

        # Get artwork IDs for top artworks
        top_artwork_ids = [artwork_ids[idx] for idx in top_artwork_indices[:5]]

        scores[feat_idx] = {
            "monosemanticity_score": float(mono_score),
            "num_activations": int(count),
            "num_embeddings_used": int(len(top_embeddings)),
            "top_artworks": top_artwork_ids,
            "top_activation_value": float(top_activations[0]) if len(top_activations) > 0 else 0,
        }

    logger.info(f"Scored {len(scores)} features (skipped {skipped_low} with < {min_activations} activations)")
    return scores


def compute_weighted_pairwise_similarity(
    embeddings: np.ndarray,
    activations: np.ndarray,
) -> float:
    """
    Compute activation-weighted pairwise cosine similarity.

    Implements the MS score from SAE-for-VLM (NeurIPS 2025):
    MS^k = (1/Z) Σ_n Σ_m (ã^k_n * ã^k_m) * sim(n, m)

    where ã^k_n is min-max normalized activation and sim is cosine similarity.

    Args:
        embeddings: Shape (k, embedding_dim), normalized
        activations: Shape (k,), raw activation values

    Returns:
        Activation-weighted mean pairwise cosine similarity
    """
    if embeddings.shape[0] < 2:
        return 0.0

    # Min-max normalize activations to [0, 1]
    a_min, a_max = activations.min(), activations.max()
    if a_max - a_min > 1e-8:
        norm_activations = (activations - a_min) / (a_max - a_min)
    else:
        norm_activations = np.ones_like(activations)

    # Compute all pairwise similarities at once: (k, k)
    similarity_matrix = embeddings @ embeddings.T

    # Compute weight matrix: w_nm = ã_n * ã_m
    weight_matrix = np.outer(norm_activations, norm_activations)

    # Zero out diagonal (self-similarity)
    np.fill_diagonal(weight_matrix, 0)
    np.fill_diagonal(similarity_matrix, 0)

    # Compute weighted mean (only upper triangle to avoid double counting)
    k = embeddings.shape[0]
    upper_tri = np.triu_indices(k, k=1)

    weights = weight_matrix[upper_tri]
    sims = similarity_matrix[upper_tri]

    total_weight = weights.sum()
    if total_weight < 1e-8:
        return float(np.mean(sims))  # Fallback to unweighted

    weighted_sim = (weights * sims).sum() / total_weight
    return float(weighted_sim)


def compute_monosemanticity_scores(
    sae_matrix: sparse.csr_matrix,
    embeddings_matrix: np.ndarray,
    artwork_ids: list[str],
    top_k: int = TOP_K_ARTWORKS,
    min_activations: int = MIN_ACTIVATIONS,
) -> dict[int, dict]:
    """
    Compute monosemanticity score for each SAE feature.

    Uses activation-weighted pairwise DINOv2 similarity (SAE-for-VLM method).

    Args:
        sae_matrix: Sparse matrix (num_artworks, SAE_DIM)
        embeddings_matrix: Dense matrix (num_artworks, embedding_dim) - DINOv2
        artwork_ids: List of artwork IDs corresponding to matrix rows
        top_k: Number of top activating artworks to consider
        min_activations: Minimum artworks required for scoring

    Returns:
        Mapping of feature_idx -> {score, num_activations, top_artworks}
    """
    scores = {}
    skipped_low_activations = 0
    skipped_no_embeddings = 0

    # Convert to CSC for efficient column access
    sae_csc = sae_matrix.tocsc()

    for feat_idx in tqdm(range(SAE_DIM), desc="Computing monosemanticity"):
        # Get column for this feature
        col = sae_csc.getcol(feat_idx)
        nnz = col.nnz

        if nnz < min_activations:
            skipped_low_activations += 1
            continue

        # Get non-zero entries
        col_coo = col.tocoo()
        row_indices = col_coo.row
        activations = col_coo.data

        # Get top-K by activation value
        if len(activations) > top_k:
            top_k_idx = np.argpartition(activations, -top_k)[-top_k:]
            top_k_idx = top_k_idx[np.argsort(-activations[top_k_idx])]
        else:
            top_k_idx = np.argsort(-activations)

        top_rows = row_indices[top_k_idx]
        top_activations = activations[top_k_idx]

        # Get embeddings for top artworks
        top_embeddings = embeddings_matrix[top_rows]

        if top_embeddings.shape[0] < min_activations // 2:
            skipped_no_embeddings += 1
            continue

        # Compute activation-weighted monosemanticity score
        mono_score = compute_weighted_pairwise_similarity(top_embeddings, top_activations)

        # Get artwork IDs for top artworks
        top_artwork_ids = [artwork_ids[r] for r in top_rows[:5]]

        # Store result
        scores[feat_idx] = {
            "monosemanticity_score": float(mono_score),
            "num_activations": int(nnz),
            "num_embeddings_used": int(top_embeddings.shape[0]),
            "top_artworks": top_artwork_ids,
            "top_activation_value": float(top_activations[0]) if len(top_activations) > 0 else 0,
        }

    logger.info(f"Scored {len(scores)} features")
    logger.info(f"Skipped {skipped_low_activations} features (< {min_activations} activations)")
    logger.info(f"Skipped {skipped_no_embeddings} features (insufficient embeddings)")

    return scores


def analyze_score_distribution(scores: dict[int, dict]) -> dict:
    """Analyze the distribution of monosemanticity scores."""
    all_scores = [data["monosemanticity_score"] for data in scores.values()]

    if not all_scores:
        return {}

    arr = np.array(all_scores)

    distribution = {
        "total_features_scored": len(all_scores),
        "mean_score": float(np.mean(arr)),
        "median_score": float(np.median(arr)),
        "std_score": float(np.std(arr)),
        "min_score": float(np.min(arr)),
        "max_score": float(np.max(arr)),
        "percentiles": {
            "10th": float(np.percentile(arr, 10)),
            "25th": float(np.percentile(arr, 25)),
            "50th": float(np.percentile(arr, 50)),
            "75th": float(np.percentile(arr, 75)),
            "90th": float(np.percentile(arr, 90)),
        },
        "calibration": {
            "baseline_mean": BASELINE_MEAN,
            "baseline_std": BASELINE_STD,
            "polysemantic_threshold": POLYSEMANTIC_THRESHOLD,
            "monosemantic_threshold": MONOSEMANTIC_THRESHOLD,
        },
        "category_counts": {
            f"polysemantic (<{POLYSEMANTIC_THRESHOLD:.2f}, +1σ)": int(
                np.sum(arr < POLYSEMANTIC_THRESHOLD)
            ),
            f"moderate ({POLYSEMANTIC_THRESHOLD:.2f}-{MONOSEMANTIC_THRESHOLD:.2f}, +1-2σ)": int(
                np.sum((arr >= POLYSEMANTIC_THRESHOLD) & (arr < MONOSEMANTIC_THRESHOLD))
            ),
            f"monosemantic (>={MONOSEMANTIC_THRESHOLD:.2f}, +2σ)": int(
                np.sum(arr >= MONOSEMANTIC_THRESHOLD)
            ),
        },
    }

    return distribution


def get_top_monosemantic_features(
    scores: dict[int, dict], n: int = 100, min_score: float | None = None
) -> list[tuple[int, dict]]:
    """Get top N most monosemantic features."""
    if min_score is None:
        min_score = MONOSEMANTIC_THRESHOLD
    filtered = [
        (feat_idx, data)
        for feat_idx, data in scores.items()
        if data["monosemanticity_score"] >= min_score
    ]
    filtered.sort(key=lambda x: x[1]["monosemanticity_score"], reverse=True)
    return filtered[:n]


def generate_report(scores: dict[int, dict], distribution: dict, output_path: Path):
    """Generate markdown report of monosemanticity analysis."""
    lines = [
        "# SAE Feature Monosemanticity Analysis\n",
        "## Overview\n",
        f"Analyzed {distribution.get('total_features_scored', 0)} features with sufficient activations.\n",
        "",
        "## Score Distribution\n",
        f"- **Mean**: {distribution.get('mean_score', 0):.3f}",
        f"- **Median**: {distribution.get('median_score', 0):.3f}",
        f"- **Std Dev**: {distribution.get('std_score', 0):.3f}",
        f"- **Range**: {distribution.get('min_score', 0):.3f} - {distribution.get('max_score', 0):.3f}",
        "",
        "### Percentiles\n",
        "| Percentile | Score |",
        "|------------|-------|",
    ]

    for p, val in distribution.get("percentiles", {}).items():
        lines.append(f"| {p} | {val:.3f} |")

    lines.extend([
        "",
        "### Category Breakdown\n",
        "| Category | Count | Percentage |",
        "|----------|-------|------------|",
    ])

    total = distribution.get("total_features_scored", 1)
    for category, count in distribution.get("category_counts", {}).items():
        pct = 100 * count / total
        lines.append(f"| {category} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Calibration Method\n",
        f"Thresholds calibrated against random art pair similarity (n=50,000 pairs):",
        f"- Random pair baseline: mean={BASELINE_MEAN:.3f}, std={BASELINE_STD:.3f}",
        f"- **Polysemantic** (<{POLYSEMANTIC_THRESHOLD:.2f}): Within +1σ of random (indistinguishable)",
        f"- **Moderate** ({POLYSEMANTIC_THRESHOLD:.2f}-{MONOSEMANTIC_THRESHOLD:.2f}): +1σ to +2σ above random",
        f"- **Monosemantic** (≥{MONOSEMANTIC_THRESHOLD:.2f}): +2σ above random (clearly above chance)",
        "",
        "## Interpretation\n",
        f"- **Polysemantic features** (score < {POLYSEMANTIC_THRESHOLD:.2f}): Top-activating images are no more ",
        "  similar than random art pairs. Feature fires on unrelated concepts.",
        f"- **Moderate features** ({POLYSEMANTIC_THRESHOLD:.2f}-{MONOSEMANTIC_THRESHOLD:.2f}): Some consistency ",
        "  but not clearly single-concept. May be worth examining.",
        f"- **Monosemantic features** (score >= {MONOSEMANTIC_THRESHOLD:.2f}): Top-activating images are ",
        "  significantly more similar than random. Prioritize for interpretation.",
        "",
        "## Top Monosemantic Features\n",
        f"Features with monosemanticity score >= {MONOSEMANTIC_THRESHOLD:.2f}:\n",
        "| Feature | Score | # Activations | Top Artworks |",
        "|---------|-------|---------------|--------------|",
    ])

    top_features = get_top_monosemantic_features(scores, n=50)
    for feat_idx, data in top_features[:50]:
        top_arts = ", ".join(data.get("top_artworks", [])[:3])
        lines.append(
            f"| {feat_idx} | {data['monosemanticity_score']:.3f} | "
            f"{data['num_activations']} | {top_arts} |"
        )

    lines.extend([
        "",
        "## Recommendations\n",
        f"1. Focus interpretation efforts on features with score >= {MONOSEMANTIC_THRESHOLD:.2f}",
        "2. Use monosemantic features for correlation analysis with human labels",
        "3. Features in 'moderate' category may still be interpretable - examine manually",
        "",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Report saved to {output_path}")


def main(
    layer: int = 8,
    exclude_sources: list[str] | None = None,
    force_streaming: bool = False,
    fine_art_only: bool = False,
):
    """Run monosemanticity analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Monosemanticity Analysis (Layer {layer})")
    logger.info("Using DINOv2 embeddings (independent of CLIP SAE)")
    if fine_art_only:
        logger.info("FINE ART ONLY mode (excludes NYPL + carriage drawings)")
    if exclude_sources:
        logger.info(f"Excluding sources: {exclude_sources}")
    logger.info("=" * 60)

    # Load DINOv2 embeddings from database
    logger.info("Loading DINOv2 embeddings...")
    embeddings_matrix, artwork_ids = load_artwork_embeddings(
        exclude_sources=exclude_sources,
        fine_art_only=fine_art_only,
    )

    # Check data size to decide approach
    data_size_gb = get_layer_data_size_gb(layer)
    use_streaming = force_streaming or data_size_gb > STREAMING_THRESHOLD_GB
    logger.info(f"Layer {layer} data size: {data_size_gb:.2f} GB")

    if use_streaming:
        logger.info("Using STREAMING mode (memory-efficient for large layers)")
        # Stream through features, computing scores incrementally
        scores = build_feature_to_artworks_streaming(
            artwork_ids,
            embeddings_matrix,
            layer=layer,
            top_k=TOP_K_ARTWORKS,
            min_activations=MIN_ACTIVATIONS,
        )
    else:
        logger.info("Using MATRIX mode (faster for smaller layers)")
        # Load SAE features as sparse matrix
        sae_matrix, valid_artwork_ids = load_sae_features_as_sparse(artwork_ids, layer=layer)

        # Filter embeddings to match valid artwork IDs
        id_to_idx = {aid: idx for idx, aid in enumerate(artwork_ids)}
        valid_indices = [id_to_idx[aid] for aid in valid_artwork_ids]
        embeddings_matrix = embeddings_matrix[valid_indices]
        logger.info(f"Filtered embeddings to {embeddings_matrix.shape[0]} artworks")

        # Compute scores
        logger.info("Computing monosemanticity scores...")
        scores = compute_monosemanticity_scores(
            sae_matrix,
            embeddings_matrix,
            valid_artwork_ids,
            top_k=TOP_K_ARTWORKS,
        )

    # Analyze distribution
    distribution = analyze_score_distribution(scores)

    # Determine output suffix
    suffix = "_fine_art" if fine_art_only else ""

    # Save results
    output_file = OUTPUT_DIR / f"monosemanticity_scores_layer{layer}{suffix}.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "metadata": {
                    "layer": layer,
                    "embedding_type": "DINOv2",
                    "top_k_artworks": TOP_K_ARTWORKS,
                    "min_activations": MIN_ACTIVATIONS,
                    "sae_dim": SAE_DIM,
                    "num_artworks": embeddings_matrix.shape[0],
                    "exclude_sources": exclude_sources,
                    "fine_art_only": fine_art_only,
                },
                "distribution": distribution,
                "scores": {str(k): v for k, v in scores.items()},
            },
            f,
            indent=2,
        )
    logger.info(f"Scores saved to {output_file}")

    # Generate report
    report_path = OUTPUT_DIR / f"monosemanticity_report_layer{layer}{suffix}.md"
    generate_report(scores, distribution, report_path)

    # Print summary
    print("\n" + "=" * 60)
    print(f"MONOSEMANTICITY ANALYSIS SUMMARY (Layer {layer})")
    print("=" * 60)
    print(f"Embedding type: DINOv2 (1024-dim)")
    print(f"Artworks analyzed: {embeddings_matrix.shape[0]}")
    print(f"Total features scored: {distribution.get('total_features_scored', 0)}")
    print(f"Mean score: {distribution.get('mean_score', 0):.3f}")
    print(f"Median score: {distribution.get('median_score', 0):.3f}")
    print()
    print("Category breakdown:")
    for category, count in distribution.get("category_counts", {}).items():
        total = distribution.get("total_features_scored", 1)
        pct = 100 * count / total
        print(f"  {category}: {count} ({pct:.1f}%)")
    print()
    print(f"Results saved to {OUTPUT_DIR}")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute monosemanticity scores for SAE features")
    parser.add_argument(
        "--layer",
        type=int,
        default=8,
        choices=list(SAE_CONFIGS.keys()),
        help="SAE layer to analyze (default: 8)",
    )
    parser.add_argument(
        "--exclude-sources",
        type=str,
        nargs="+",
        default=None,
        help="Sources to exclude (e.g., nypl)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Force streaming mode (auto-enabled for layers > 5GB)",
    )
    parser.add_argument(
        "--fine-art-only",
        action="store_true",
        help="Filter to fine art only (excludes NYPL + Met carriage drawings)",
    )
    args = parser.parse_args()

    main(
        layer=args.layer,
        exclude_sources=args.exclude_sources,
        force_streaming=args.streaming,
        fine_art_only=args.fine_art_only,
    )
