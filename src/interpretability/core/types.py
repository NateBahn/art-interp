"""
Core data types for SAE interpretability.

This module defines the primary data structures used throughout the package.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SAEFeatureResult:
    """
    Result of SAE feature extraction from a single image.

    The SAE produces sparse feature activations where most values are zero.
    This result captures both the full feature vector and summary statistics.

    Attributes:
        features: Sparse feature activations (shape: SAE_DIM, typically 49152)
        num_active: Number of non-zero features (sparsity indicator)
        top_indices: Indices of the top-k activating features
        top_values: Activation values for the top-k features

    Example:
        >>> result = extractor.extract_from_image(image)
        >>> print(f"Active: {result.num_active} / {len(result.features)}")
        Active: 127 / 49152
        >>> print(f"Top feature: {result.top_indices[0]} = {result.top_values[0]:.2f}")
        Top feature: 23847 = 4.23
    """

    features: np.ndarray
    num_active: int
    top_indices: list[int]
    top_values: list[float]

    def to_sparse_dict(self) -> dict[int, float]:
        """Convert to sparse dictionary format for storage."""
        nonzero = np.nonzero(self.features)[0]
        return {int(idx): float(self.features[idx]) for idx in nonzero}

    @classmethod
    def from_sparse_dict(cls, sparse: dict[int, float], dim: int = 49152) -> "SAEFeatureResult":
        """Reconstruct from sparse dictionary format."""
        features = np.zeros(dim, dtype=np.float32)
        for idx_str, value in sparse.items():
            features[int(idx_str)] = value

        nonzero = np.nonzero(features)[0]
        num_active = len(nonzero)

        # Get top-k
        top_k = min(20, num_active)
        if top_k > 0:
            top_idx = np.argsort(features)[-top_k:][::-1]
            top_indices = top_idx.tolist()
            top_values = features[top_idx].tolist()
        else:
            top_indices = []
            top_values = []

        return cls(
            features=features,
            num_active=num_active,
            top_indices=top_indices,
            top_values=top_values,
        )


@dataclass
class SpatialFeatureResult:
    """
    Result of spatial SAE feature extraction from a single image.

    Spatial SAEs extract features from each of the 49 patches (7x7 grid),
    allowing localization of features within the image.

    Attributes:
        patch_features: Feature activations per patch (shape: 49 x SAE_DIM)
        aggregated_features: Max-pooled features across patches (shape: SAE_DIM)
        patch_top_indices: Top feature indices for each patch
        patch_top_values: Top feature values for each patch
        num_active_per_patch: Active feature count per patch

    Example:
        >>> result = extractor.extract_spatial(image)
        >>> # Find which patches activate feature 1234
        >>> activation_map = result.patch_features[:, 1234].reshape(7, 7)
    """

    patch_features: np.ndarray  # (49, SAE_DIM)
    aggregated_features: np.ndarray  # (SAE_DIM,) - max across patches
    patch_top_indices: list[list[int]]  # Top-k per patch
    patch_top_values: list[list[float]]
    num_active_per_patch: list[int]

    def get_feature_heatmap(self, feature_idx: int) -> np.ndarray:
        """
        Get 7x7 activation heatmap for a specific feature.

        Args:
            feature_idx: Index of the feature (0 to SAE_DIM-1)

        Returns:
            7x7 numpy array of activation values
        """
        return self.patch_features[:, feature_idx].reshape(7, 7)


@dataclass
class ImageItem:
    """
    An image with optional metadata for interpretability analysis.

    This is the standard input format for the analysis pipeline.
    Implementations of ImageProvider return instances of this class.

    Attributes:
        id: Unique identifier for the image
        image_url: URL to fetch the image (optional if image is provided)
        title: Human-readable title/name
        ratings: Dict of rating dimension -> score (e.g., {"mirror_self": 7.5})
        embedding: Pre-computed embedding for monosemanticity scoring
        metadata: Additional metadata (artist, date, etc.)

    Example:
        >>> item = ImageItem(
        ...     id="painting_001",
        ...     image_url="https://example.com/art.jpg",
        ...     title="Starry Night",
        ...     ratings={"mirror_self": 9.0, "drawn_to": 8.5},
        ... )
    """

    id: str
    image_url: str | None = None
    title: str | None = None
    ratings: dict[str, float] | None = None
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_ratings(self) -> bool:
        """Check if this item has rating data."""
        return self.ratings is not None and len(self.ratings) > 0

    def get_rating(self, dimension: str) -> float | None:
        """Get rating for a specific dimension."""
        if self.ratings is None:
            return None
        return self.ratings.get(dimension)
