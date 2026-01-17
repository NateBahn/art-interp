"""
Feature-Rating Correlation Analysis

Computes correlations between SAE features and human-provided ratings.
Identifies which features predict specific aesthetic dimensions.

Methods:
- Pearson correlation for numeric ratings (1-10 scales)
- Point-biserial correlation for binary/categorical ratings
- Benjamini-Hochberg FDR correction for multiple testing

Example:
    >>> from interpretability.analysis import CorrelationAnalyzer
    >>>
    >>> analyzer = CorrelationAnalyzer(rating_dimensions=["mirror_self", "drawn_to"])
    >>> # features: {item_id: feature_vector}
    >>> # ratings: {item_id: {"mirror_self": 8.0, "drawn_to": 7.5}}
    >>> results = analyzer.compute_correlations(features, ratings)
    >>> print(results.top_features_for("mirror_self"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class FeatureCorrelation:
    """Correlation between a single feature and a rating dimension."""

    feature_idx: int
    rating_dimension: str
    correlation: float
    p_value: float
    p_value_corrected: float  # After FDR correction
    n_samples: int
    is_significant: bool  # After FDR correction

    def __repr__(self) -> str:
        sig = "*" if self.is_significant else ""
        return (
            f"FeatureCorrelation({self.feature_idx}, "
            f"{self.rating_dimension}: r={self.correlation:.3f}{sig})"
        )


@dataclass
class CorrelationResults:
    """Results of correlation analysis across all features and ratings."""

    correlations: list[FeatureCorrelation]
    rating_dimensions: list[str]
    n_features: int
    n_items: int
    alpha: float  # Significance threshold used

    def top_features_for(
        self,
        rating_dimension: str,
        n: int = 20,
        significant_only: bool = True,
    ) -> list[FeatureCorrelation]:
        """
        Get top features correlated with a rating dimension.

        Args:
            rating_dimension: Which rating to query
            n: Number of top features to return
            significant_only: If True, only return significant correlations

        Returns:
            List of FeatureCorrelation, sorted by absolute correlation
        """
        filtered = [
            c
            for c in self.correlations
            if c.rating_dimension == rating_dimension
            and (not significant_only or c.is_significant)
        ]
        return sorted(filtered, key=lambda c: abs(c.correlation), reverse=True)[:n]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        by_dimension = {}
        for dim in self.rating_dimensions:
            top = self.top_features_for(dim, n=100, significant_only=False)
            by_dimension[dim] = [
                {
                    "feature_idx": c.feature_idx,
                    "correlation": c.correlation,
                    "p_value": c.p_value,
                    "p_value_corrected": c.p_value_corrected,
                    "is_significant": c.is_significant,
                }
                for c in top
            ]

        return {
            "n_features": self.n_features,
            "n_items": self.n_items,
            "alpha": self.alpha,
            "rating_dimensions": self.rating_dimensions,
            "correlations_by_dimension": by_dimension,
        }


class CorrelationAnalyzer:
    """
    Analyze correlations between SAE features and ratings.

    Computes Pearson correlations between each feature's activation
    pattern and each rating dimension across the dataset.

    Attributes:
        rating_dimensions: List of rating dimensions to analyze
        alpha: Significance threshold (default 0.05)
        min_variance: Minimum variance required to compute correlation

    Example:
        >>> analyzer = CorrelationAnalyzer(["mirror_self", "drawn_to"])
        >>> features = {"img1": np.array([...]), "img2": np.array([...])}
        >>> ratings = {"img1": {"mirror_self": 8.0}, "img2": {"mirror_self": 6.0}}
        >>> results = analyzer.compute_correlations(features, ratings)
    """

    def __init__(
        self,
        rating_dimensions: list[str],
        alpha: float = 0.05,
        min_variance: float = 1e-6,
    ) -> None:
        """
        Initialize the analyzer.

        Args:
            rating_dimensions: Rating dimensions to correlate with
            alpha: Significance threshold for FDR correction
            min_variance: Minimum variance in feature/rating to compute correlation
        """
        self.rating_dimensions = rating_dimensions
        self.alpha = alpha
        self.min_variance = min_variance

    def compute_correlations(
        self,
        features: dict[str, np.ndarray],
        ratings: dict[str, dict[str, float]],
    ) -> CorrelationResults:
        """
        Compute correlations between features and ratings.

        Args:
            features: Mapping of item_id -> feature vector
            ratings: Mapping of item_id -> {dimension: value}

        Returns:
            CorrelationResults with all correlations
        """
        # Find items with both features and ratings
        common_ids = sorted(set(features.keys()) & set(ratings.keys()))

        if len(common_ids) < 10:
            logger.warning(f"Only {len(common_ids)} items have both features and ratings")
            return CorrelationResults(
                correlations=[],
                rating_dimensions=self.rating_dimensions,
                n_features=0,
                n_items=len(common_ids),
                alpha=self.alpha,
            )

        logger.info(f"Computing correlations for {len(common_ids)} items")

        # Build feature matrix
        feature_matrix = np.array([features[item_id] for item_id in common_ids])
        n_items, n_features = feature_matrix.shape

        # Build rating matrix
        rating_matrix = np.zeros((n_items, len(self.rating_dimensions)))
        for i, item_id in enumerate(common_ids):
            for j, dim in enumerate(self.rating_dimensions):
                rating_matrix[i, j] = ratings[item_id].get(dim, np.nan)

        # Compute correlations
        all_correlations = []

        for feat_idx in range(n_features):
            feature_values = feature_matrix[:, feat_idx]

            # Skip features with no variance
            if np.var(feature_values) < self.min_variance:
                continue

            for dim_idx, dim in enumerate(self.rating_dimensions):
                rating_values = rating_matrix[:, dim_idx]

                # Skip if ratings have no variance
                valid_mask = ~np.isnan(rating_values)
                if valid_mask.sum() < 5 or np.var(rating_values[valid_mask]) < self.min_variance:
                    continue

                # Compute Pearson correlation
                r, p = stats.pearsonr(
                    feature_values[valid_mask], rating_values[valid_mask]
                )

                all_correlations.append(
                    FeatureCorrelation(
                        feature_idx=feat_idx,
                        rating_dimension=dim,
                        correlation=float(r),
                        p_value=float(p),
                        p_value_corrected=float(p),  # Will be updated
                        n_samples=int(valid_mask.sum()),
                        is_significant=False,  # Will be updated
                    )
                )

        # Apply FDR correction
        all_correlations = self._apply_fdr_correction(all_correlations)

        logger.info(
            f"Computed {len(all_correlations)} correlations, "
            f"{sum(1 for c in all_correlations if c.is_significant)} significant"
        )

        return CorrelationResults(
            correlations=all_correlations,
            rating_dimensions=self.rating_dimensions,
            n_features=n_features,
            n_items=n_items,
            alpha=self.alpha,
        )

    def _apply_fdr_correction(
        self,
        correlations: list[FeatureCorrelation],
    ) -> list[FeatureCorrelation]:
        """Apply Benjamini-Hochberg FDR correction."""
        if not correlations:
            return correlations

        # Extract p-values
        p_values = np.array([c.p_value for c in correlations])
        n = len(p_values)

        # Sort by p-value
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        # BH procedure
        threshold = (np.arange(1, n + 1) / n) * self.alpha
        significant_mask = sorted_p <= threshold

        # Find the largest k where p_k <= (k/n)*alpha
        if significant_mask.any():
            max_k = np.max(np.where(significant_mask)[0]) + 1
            significant_mask[: max_k] = True
            significant_mask[max_k:] = False
        else:
            significant_mask[:] = False

        # Compute adjusted p-values
        adjusted_p = np.zeros(n)
        adjusted_p[sorted_indices[-1]] = sorted_p[-1]
        for i in range(n - 2, -1, -1):
            idx = sorted_indices[i]
            adjusted_p[idx] = min(
                adjusted_p[sorted_indices[i + 1]], sorted_p[i] * n / (i + 1)
            )
        adjusted_p = np.minimum(adjusted_p, 1.0)

        # Update correlations
        for i, corr in enumerate(correlations):
            corr.p_value_corrected = float(adjusted_p[i])
            # Check if in significant set
            rank = np.searchsorted(sorted_indices, i)
            corr.is_significant = significant_mask[rank] if rank < n else False

        return correlations


def compute_point_biserial(
    feature_values: np.ndarray,
    categories: np.ndarray,
) -> tuple[float, float]:
    """
    Compute point-biserial correlation for binary/categorical data.

    Used when ratings are categorical rather than numeric.

    Args:
        feature_values: Numeric feature activations
        categories: Binary or categorical labels (converted to 0/1)

    Returns:
        Tuple of (correlation, p_value)
    """
    unique = np.unique(categories[~np.isnan(categories)])
    if len(unique) != 2:
        raise ValueError("Point-biserial requires exactly 2 categories")

    # Convert to 0/1
    binary = (categories == unique[1]).astype(float)
    valid = ~np.isnan(binary)

    return stats.pointbiserialr(binary[valid], feature_values[valid])
