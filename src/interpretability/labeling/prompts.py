"""
Prompt templates for VLM feature labeling.

These prompts are designed to work with vision-language models like Gemini
to identify visual patterns in top-activating images for SAE features.
"""

from typing import Protocol


class PromptBuilder(Protocol):
    """Protocol for prompt builders."""

    def build(self, feature_idx: int, num_images: int, **kwargs) -> str:
        """Build a prompt for labeling a feature."""
        ...


def build_cls_prompt(
    feature_idx: int,
    num_images: int,
    strongest_rating: str | None = None,
    strongest_correlation: float | None = None,
) -> str:
    """
    Build a prompt for labeling CLS (global) SAE features.

    Args:
        feature_idx: The feature index being labeled
        num_images: Number of images being shown
        strongest_rating: The rating dimension most correlated with this feature
        strongest_correlation: The correlation coefficient

    Returns:
        Formatted prompt string for the VLM
    """
    if strongest_rating and strongest_correlation:
        direction = "positively" if strongest_correlation > 0 else "negatively"
        change = "increase" if strongest_correlation > 0 else "decrease"

        return f"""You are an art historian and computer vision researcher.

These {num_images} artworks all strongly activate the same neural network feature (Feature #{feature_idx}).
This feature correlates {direction} with people's "{strongest_rating}" ratings of artworks.
Correlation: r = {strongest_correlation:.3f}

Task: Identify what visual or conceptual pattern these artworks share that could explain why they {change} this rating.

Focus on:
- Visual elements (color, composition, brushwork, lighting)
- Subject matter and themes
- Artistic style and technique
- Emotional or psychological qualities

Return your analysis as JSON (and only JSON, no markdown code blocks):
{{
  "short_label": "2-5 word feature name",
  "description": "1-2 sentence explanation of what visual pattern this feature detects",
  "visual_elements": ["element1", "element2", "element3"],
  "explains_rating": "How this pattern might influence the {strongest_rating} rating",
  "non_obvious_insight": "What's surprising or non-obvious about this pattern?",
  "confidence": 0.0-1.0 (how confident are you that these images share a coherent pattern? 1.0 = very clear pattern, 0.5 = somewhat ambiguous, 0.0 = no clear pattern)
}}"""
    else:
        # Exploration prompt for features without strong correlations
        return f"""You are an art historian and computer vision researcher.

These {num_images} artworks all strongly activate the same neural network feature (Feature #{feature_idx}).
The model learned this pattern from training data.

Task: Identify what visual or conceptual pattern these artworks share.
Be specific about what makes these artworks similar.

Return your analysis as JSON (and only JSON, no markdown code blocks):
{{
  "short_label": "2-5 word feature name",
  "description": "1-2 sentence explanation of what visual pattern this feature detects",
  "visual_elements": ["element1", "element2", "element3"],
  "non_obvious_insight": "What's surprising about this pattern?",
  "confidence": 0.0-1.0 (how confident are you that these images share a coherent pattern? 1.0 = very clear pattern, 0.5 = somewhat ambiguous, 0.0 = no clear pattern)
}}"""


def build_spatial_prompt(
    feature_idx: int,
    num_images: int,
    rating: str,
    correlation: float,
) -> str:
    """
    Build a prompt for labeling spatial SAE features with heatmaps.

    For spatial features, we show both the original image and a heatmap overlay
    showing where the feature activates. This gives the VLM spatial context.

    Args:
        feature_idx: The feature index being labeled
        num_images: Number of artworks (each shown twice: original + heatmap)
        rating: The rating dimension being analyzed
        correlation: The correlation coefficient

    Returns:
        Formatted prompt string for the VLM
    """
    direction = "positively" if correlation > 0 else "negatively"
    change = "higher" if correlation > 0 else "lower"

    return f"""You are an art historian and computer vision researcher analyzing neural network features.

I'm showing you {num_images} artworks that strongly activate Spatial Feature #{feature_idx}.
Each artwork is shown TWICE:
- First: The original artwork
- Second: A HEATMAP overlay showing WHERE in the image this feature activates (green = high activation)

This feature correlates {direction} with people's "{rating}" ratings (r = {correlation:.3f}).
When this feature activates strongly, artworks tend to receive {change} "{rating}" scores.

IMPORTANT: Pay attention to the HEATMAPS. They show the specific regions where the feature fires.
Look for patterns in:
- What parts of images activate (faces, backgrounds, edges, textures, etc.)
- Common visual elements in those regions
- Why those regions might influence the "{rating}" perception

Return your analysis as JSON (only JSON, no markdown):
{{
  "short_label": "2-5 word feature name",
  "spatial_focus": "Where in images this feature typically activates (e.g., 'faces and figures', 'sky regions', 'textural details')",
  "visual_pattern": "What visual elements or textures appear in the activated regions",
  "description": "1-2 sentence explanation of what this feature detects",
  "explains_rating": "How detecting this pattern in these locations might influence {rating} ratings",
  "confidence": "high/medium/low"
}}"""


# Default prompt builders for different feature types
DEFAULT_PROMPTS = {
    "cls": build_cls_prompt,
    "spatial": build_spatial_prompt,
}
