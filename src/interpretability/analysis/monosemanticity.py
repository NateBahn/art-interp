"""
Monosemanticity Scoring for SAE Features

Measures which SAE features are truly "monosemantic" (single-concept) vs
"polysemantic" (mixed concepts). Only ~60-70% of vision SAE features are
monosemantic, so this helps prioritize interpretation efforts.

Method (based on SAE-for-VLM, NeurIPS 2025):
1. For each feature, get all images and their activation values
2. Get embeddings for those images (from an INDEPENDENT model to avoid circularity)
3. Compute activation-weighted pairwise cosine similarity
4. MS score = weighted mean similarity (high = clear single concept)

Formula: MS^k = (1/Z) Σ Σ (ã^k_n * ã^k_m) * sim(n, m)
where ã^k_n is normalized activation, sim is cosine similarity

IMPORTANT: Uses embeddings from a DIFFERENT model than the SAE base model.
For CLIP-based SAEs, use DINOv2 embeddings. Using CLIP would be circular.

Reference: https://arxiv.org/abs/2504.02821

Example:
    >>> from interpretability.analysis import MonosemanticityScorer
    >>> from interpretability.storage import SampleImageProvider, SampleEmbeddingProvider
    >>>
    >>> scorer = MonosemanticityScorer(
    ...     embedding_provider=SampleEmbeddingProvider(),
    ... )
    >>> # Score a single feature given its activations
    >>> activations = {"img1": 2.5, "img2": 1.8, "img3": 1.2}
    >>> score = scorer.score_feature(activations)
    >>> print(f"Monosemanticity: {score:.3f}")
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from interpretability.storage.protocols import EmbeddingProvider

logger = logging.getLogger(__name__)

# Calibrated thresholds based on random art pair similarity distribution
# Random pairs: mean=0.195, std=0.154 (computed on 50k pairs from 16k artworks)
BASELINE_MEAN = 0.195
BASELINE_STD = 0.154
POLYSEMANTIC_THRESHOLD = BASELINE_MEAN + 1 * BASELINE_STD  # ~0.35 (+1σ)
MONOSEMANTIC_THRESHOLD = BASELINE_MEAN + 2 * BASELINE_STD  # ~0.50 (+2σ)


@dataclass
class MonosemanticityResult:
    """Result of monosemanticity scoring for a single feature."""

    feature_idx: int
    score: float
    num_activations: int
    num_embeddings_used: int
    top_items: list[str]  # IDs of top-activating items
    top_activation_value: float
    category: str = field(init=False)

    def __post_init__(self) -> None:
        """Compute category based on calibrated thresholds."""
        if self.score >= MONOSEMANTIC_THRESHOLD:
            self.category = "monosemantic"
        elif self.score >= POLYSEMANTIC_THRESHOLD:
            self.category = "moderate"
        else:
            self.category = "polysemantic"


@dataclass
class MonosemanticityReport:
    """Summary report of monosemanticity analysis across features."""

    total_features_scored: int
    mean_score: float
    median_score: float
    std_score: float
    min_score: float
    max_score: float
    percentiles: dict[str, float]
    category_counts: dict[str, int]
    top_monosemantic: list[MonosemanticityResult]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_features_scored": self.total_features_scored,
            "mean_score": self.mean_score,
            "median_score": self.median_score,
            "std_score": self.std_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "percentiles": self.percentiles,
            "calibration": {
                "baseline_mean": BASELINE_MEAN,
                "baseline_std": BASELINE_STD,
                "polysemantic_threshold": POLYSEMANTIC_THRESHOLD,
                "monosemantic_threshold": MONOSEMANTIC_THRESHOLD,
            },
            "category_counts": self.category_counts,
            "top_monosemantic": [
                {
                    "feature_idx": r.feature_idx,
                    "score": r.score,
                    "num_activations": r.num_activations,
                    "top_items": r.top_items[:5],
                }
                for r in self.top_monosemantic[:20]
            ],
        }


class MonosemanticityScorer:
    """
    Compute monosemanticity scores for SAE features.

    Monosemanticity measures whether a feature activates on a coherent
    set of inputs (single concept) or diverse unrelated inputs (polysemantic).

    The scoring uses an embedding model INDEPENDENT of the SAE's base model
    to avoid circular reasoning. For CLIP-based SAEs, DINOv2 is recommended.

    Attributes:
        embedding_provider: Source of embeddings for similarity computation
        top_k: Number of top-activating items to consider per feature
        min_activations: Minimum activations required to score a feature

    Example:
        >>> scorer = MonosemanticityScorer(embedding_provider)
        >>> activations = {"img1": 2.5, "img2": 1.8, "img3": 1.2, ...}
        >>> score = scorer.score_feature(activations)
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        top_k: int = 20,
        min_activations: int = 10,
    ) -> None:
        """
        Initialize the scorer.

        Args:
            embedding_provider: Provides embeddings for items (e.g., DINOv2)
            top_k: Number of top-activating items to consider
            min_activations: Minimum items required for scoring
        """
        self.embedding_provider = embedding_provider
        self.top_k = top_k
        self.min_activations = min_activations

    def score_feature(
        self,
        activations: dict[str, float],
        feature_idx: int = 0,
    ) -> MonosemanticityResult | None:
        """
        Compute monosemanticity score for a single feature.

        Args:
            activations: Mapping of item_id -> activation value
            feature_idx: Index of the feature (for tracking)

        Returns:
            MonosemanticityResult or None if insufficient data
        """
        if len(activations) < self.min_activations:
            return None

        # Sort by activation value (descending)
        sorted_items = sorted(activations.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[: self.top_k]

        # Get embeddings for top items
        item_ids = [item_id for item_id, _ in top_items]
        embeddings_dict = self.embedding_provider.get_embeddings_batch(item_ids)

        # Filter to items with embeddings
        valid_items = [(item_id, act) for item_id, act in top_items if item_id in embeddings_dict]

        if len(valid_items) < self.min_activations // 2:
            return None

        # Build arrays
        embeddings = np.array([embeddings_dict[item_id] for item_id, _ in valid_items])
        activation_values = np.array([act for _, act in valid_items], dtype=np.float32)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)

        # Compute monosemanticity score
        score = compute_weighted_pairwise_similarity(embeddings, activation_values)

        return MonosemanticityResult(
            feature_idx=feature_idx,
            score=score,
            num_activations=len(activations),
            num_embeddings_used=len(valid_items),
            top_items=[item_id for item_id, _ in valid_items[:5]],
            top_activation_value=activation_values[0] if len(activation_values) > 0 else 0,
        )

    def score_features(
        self,
        feature_activations: Iterator[tuple[int, dict[str, float]]],
    ) -> list[MonosemanticityResult]:
        """
        Score multiple features.

        Args:
            feature_activations: Iterator of (feature_idx, {item_id: activation})

        Returns:
            List of MonosemanticityResult for features that could be scored
        """
        results = []
        for feature_idx, activations in feature_activations:
            result = self.score_feature(activations, feature_idx)
            if result is not None:
                results.append(result)
        return results

    def generate_report(
        self,
        results: list[MonosemanticityResult],
    ) -> MonosemanticityReport:
        """
        Generate summary report from scoring results.

        Args:
            results: List of MonosemanticityResult from score_features

        Returns:
            MonosemanticityReport with statistics and top features
        """
        if not results:
            return MonosemanticityReport(
                total_features_scored=0,
                mean_score=0.0,
                median_score=0.0,
                std_score=0.0,
                min_score=0.0,
                max_score=0.0,
                percentiles={},
                category_counts={},
                top_monosemantic=[],
            )

        scores = np.array([r.score for r in results])

        # Category counts
        category_counts = {
            "polysemantic": sum(1 for r in results if r.category == "polysemantic"),
            "moderate": sum(1 for r in results if r.category == "moderate"),
            "monosemantic": sum(1 for r in results if r.category == "monosemantic"),
        }

        # Top monosemantic features
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        top_mono = [r for r in sorted_results if r.category == "monosemantic"][:50]

        return MonosemanticityReport(
            total_features_scored=len(results),
            mean_score=float(np.mean(scores)),
            median_score=float(np.median(scores)),
            std_score=float(np.std(scores)),
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
            percentiles={
                "10th": float(np.percentile(scores, 10)),
                "25th": float(np.percentile(scores, 25)),
                "50th": float(np.percentile(scores, 50)),
                "75th": float(np.percentile(scores, 75)),
                "90th": float(np.percentile(scores, 90)),
            },
            category_counts=category_counts,
            top_monosemantic=top_mono,
        )


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
        embeddings: Shape (k, embedding_dim), should be L2-normalized
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

    # Compute all pairwise similarities: (k, k)
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


def classify_monosemanticity(score: float) -> str:
    """
    Classify a monosemanticity score into a category.

    Categories are based on calibrated thresholds from random pairs:
    - polysemantic: score < 0.35 (within +1σ of random)
    - moderate: 0.35 <= score < 0.50 (between +1σ and +2σ)
    - monosemantic: score >= 0.50 (+2σ above random)

    Args:
        score: Monosemanticity score (weighted pairwise similarity)

    Returns:
        Category string: "polysemantic", "moderate", or "monosemantic"
    """
    if score >= MONOSEMANTIC_THRESHOLD:
        return "monosemantic"
    elif score >= POLYSEMANTIC_THRESHOLD:
        return "moderate"
    else:
        return "polysemantic"
