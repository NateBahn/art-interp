"""
VLM-based feature labeling for SAE interpretability.

This module provides tools for labeling SAE features using vision-language
models (VLMs) like Google's Gemini. It sends top-activating images to the
VLM and asks it to identify the visual pattern that explains the feature.

Example:
    ```python
    from interpretability.labeling import GeminiLabeler, FeatureInfo

    labeler = GeminiLabeler()  # Uses GEMINI_API_KEY env var

    result = labeler.label_feature(
        feature=FeatureInfo(
            feature_idx=1234,
            top_artwork_ids=["art1", "art2", "art3"],
        ),
        get_image_url=my_image_lookup_function,
    )
    print(result.short_label)  # e.g., "Warm sunset colors"
    ```
"""

from interpretability.labeling.gemini_labeler import (
    FeatureInfo,
    FeatureLabelResult,
    GeminiLabeler,
    fetch_image,
    label_feature_with_gemini,
    parse_gemini_response,
)
from interpretability.labeling.prompts import (
    DEFAULT_PROMPTS,
    build_cls_prompt,
    build_spatial_prompt,
)

__all__ = [
    # Main classes
    "GeminiLabeler",
    "FeatureInfo",
    "FeatureLabelResult",
    # Convenience functions
    "label_feature_with_gemini",
    "fetch_image",
    "parse_gemini_response",
    # Prompt builders
    "build_cls_prompt",
    "build_spatial_prompt",
    "DEFAULT_PROMPTS",
]
