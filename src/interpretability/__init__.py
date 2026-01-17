"""
Interpretability tools for vision models using Sparse Autoencoders (SAEs).

This package provides tools for understanding what vision models like CLIP learn,
using Sparse Autoencoders to decompose activations into interpretable features.

Key capabilities:
- Extract SAE features from images (CLS token and spatial patches)
- Score feature monosemanticity (single-concept vs mixed)
- Correlate features with human ratings
- Label features using Vision-Language Models
- Visualize spatial feature activations as heatmaps

Example:
    >>> from interpretability.core import SAEFeatureExtractor
    >>> extractor = SAEFeatureExtractor(layer=8)
    >>> result = extractor.extract_from_image(image)
    >>> print(f"Top feature: {result.top_indices[0]} with activation {result.top_values[0]:.2f}")
"""

__version__ = "0.1.0"

from interpretability.core.sae_extractor import SAEFeatureExtractor
from interpretability.core.types import ImageItem, SAEFeatureResult

__all__ = [
    "SAEFeatureExtractor",
    "SAEFeatureResult",
    "ImageItem",
    "__version__",
]
