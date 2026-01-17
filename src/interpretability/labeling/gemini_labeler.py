"""
VLM-based feature labeling using Google's Gemini API.

This module provides functionality to label SAE features by showing
top-activating images to a vision-language model and asking it to
identify the visual pattern that explains the feature's activations.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Callable

import httpx
from PIL import Image

from interpretability.labeling.prompts import build_cls_prompt, build_spatial_prompt


@dataclass
class FeatureLabelResult:
    """Result from labeling a single feature."""

    feature_idx: int
    short_label: str | None = None
    description: str | None = None
    visual_elements: list[str] | None = None
    explains_rating: str | None = None
    non_obvious_insight: str | None = None
    confidence: float | None = None
    spatial_focus: str | None = None  # For spatial features
    visual_pattern: str | None = None  # For spatial features

    # Metadata
    tier: int | None = None
    strongest_rating: str | None = None
    strongest_correlation: float | None = None
    monosemanticity_score: float | None = None
    num_images_used: int = 0
    model_used: str | None = None
    raw_response: dict | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            k: v
            for k, v in {
                "feature_idx": self.feature_idx,
                "short_label": self.short_label,
                "description": self.description,
                "visual_elements": self.visual_elements,
                "explains_rating": self.explains_rating,
                "non_obvious_insight": self.non_obvious_insight,
                "confidence": self.confidence,
                "spatial_focus": self.spatial_focus,
                "visual_pattern": self.visual_pattern,
                "tier": self.tier,
                "strongest_rating": self.strongest_rating,
                "strongest_correlation": self.strongest_correlation,
                "monosemanticity_score": self.monosemanticity_score,
                "num_images_used": self.num_images_used,
                "model_used": self.model_used,
            }.items()
            if v is not None
        }


@dataclass
class FeatureInfo:
    """Information about a feature to be labeled."""

    feature_idx: int
    top_artwork_ids: list[str]
    tier: int | None = None
    strongest_rating: str | None = None
    strongest_correlation: float | None = None
    monosemanticity_score: float | None = None


def parse_gemini_response(response_text: str) -> dict | None:
    """
    Parse JSON from Gemini response, handling markdown code blocks.

    Gemini sometimes wraps JSON in markdown code blocks even when asked not to.
    This function handles both clean JSON and markdown-wrapped JSON.

    Args:
        response_text: Raw text response from Gemini

    Returns:
        Parsed JSON dict or None if parsing fails
    """
    text = response_text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from the text using regex
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None


def fetch_image(url: str, timeout: float = 30.0) -> Image.Image | None:
    """
    Fetch image from URL and return as PIL Image.

    Args:
        url: URL of the image to fetch
        timeout: Request timeout in seconds

    Returns:
        PIL Image in RGB mode, or None if fetch fails
    """
    try:
        response = httpx.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Interpretability/1.0 (Research)"},
            follow_redirects=True,
        )
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        return None


class GeminiLabeler:
    """
    Label SAE features using Google's Gemini VLM.

    This class provides a high-level interface for labeling SAE features
    by sending top-activating images to Gemini and parsing the response.

    Example:
        ```python
        labeler = GeminiLabeler(api_key=os.environ["GEMINI_API_KEY"])

        def get_image_url(artwork_id: str) -> str | None:
            # Your logic to get image URL from artwork ID
            return image_urls.get(artwork_id)

        result = labeler.label_feature(
            feature=FeatureInfo(
                feature_idx=1234,
                top_artwork_ids=["art1", "art2", "art3"],
                strongest_rating="mirror_self",
                strongest_correlation=0.45,
            ),
            get_image_url=get_image_url,
        )
        print(result.short_label)
        ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.0-flash-exp",
    ):
        """
        Initialize the Gemini labeler.

        Args:
            api_key: Gemini API key. If not provided, reads from GEMINI_API_KEY env var.
            model_name: Gemini model to use for labeling

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Pass api_key or set GEMINI_API_KEY environment variable."
            )

        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazily initialize the Gemini model."""
        if self._model is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model_name)
            except ImportError:
                raise ImportError(
                    "google-generativeai package required for Gemini labeling. "
                    "Install with: pip install google-generativeai"
                )
        return self._model

    def label_feature(
        self,
        feature: FeatureInfo,
        get_image_url: Callable[[str], str | None],
        max_images: int = 5,
        prompt_builder: Callable[..., str] | None = None,
    ) -> FeatureLabelResult | None:
        """
        Label a single feature using Gemini.

        Args:
            feature: Feature information including artwork IDs
            get_image_url: Function that returns image URL for an artwork ID
            max_images: Maximum number of images to send to Gemini
            prompt_builder: Custom prompt builder function (default: build_cls_prompt)

        Returns:
            FeatureLabelResult with the label and metadata, or None if labeling fails
        """
        artwork_ids = feature.top_artwork_ids[:max_images]

        if not artwork_ids:
            return None

        # Fetch images
        images = []
        for artwork_id in artwork_ids:
            url = get_image_url(artwork_id)
            if url:
                img = fetch_image(url)
                if img:
                    images.append(img)

        if len(images) < 2:
            return None

        # Build prompt
        if prompt_builder is None:
            prompt_builder = build_cls_prompt

        prompt = prompt_builder(
            feature_idx=feature.feature_idx,
            num_images=len(images),
            strongest_rating=feature.strongest_rating,
            strongest_correlation=feature.strongest_correlation,
        )

        # Send to Gemini
        try:
            model = self._get_model()
            response = model.generate_content([prompt] + images)
            result_dict = parse_gemini_response(response.text)

            if not result_dict:
                return None

            # Parse confidence - handle both float and string formats
            confidence = result_dict.get("confidence")
            if isinstance(confidence, str):
                confidence_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
                confidence = confidence_map.get(confidence.lower(), 0.5)

            return FeatureLabelResult(
                feature_idx=feature.feature_idx,
                short_label=result_dict.get("short_label"),
                description=result_dict.get("description"),
                visual_elements=result_dict.get("visual_elements"),
                explains_rating=result_dict.get("explains_rating"),
                non_obvious_insight=result_dict.get("non_obvious_insight"),
                confidence=confidence,
                spatial_focus=result_dict.get("spatial_focus"),
                visual_pattern=result_dict.get("visual_pattern"),
                tier=feature.tier,
                strongest_rating=feature.strongest_rating,
                strongest_correlation=feature.strongest_correlation,
                monosemanticity_score=feature.monosemanticity_score,
                num_images_used=len(images),
                model_used=self.model_name,
                raw_response=result_dict,
            )

        except Exception:
            return None

    def label_features(
        self,
        features: list[FeatureInfo],
        get_image_url: Callable[[str], str | None],
        max_images: int = 5,
        delay: float = 1.0,
        on_progress: Callable[[int, int, FeatureLabelResult | None], None] | None = None,
    ) -> list[FeatureLabelResult]:
        """
        Label multiple features with rate limiting.

        Args:
            features: List of features to label
            get_image_url: Function that returns image URL for an artwork ID
            max_images: Maximum images per feature
            delay: Delay between API calls in seconds
            on_progress: Optional callback for progress updates (current, total, result)

        Returns:
            List of successful FeatureLabelResult objects
        """
        results = []

        for i, feature in enumerate(features):
            result = self.label_feature(
                feature=feature,
                get_image_url=get_image_url,
                max_images=max_images,
            )

            if result:
                results.append(result)

            if on_progress:
                on_progress(i + 1, len(features), result)

            # Rate limiting between calls
            if i < len(features) - 1:
                time.sleep(delay)

        return results


# Convenience function for one-off labeling
def label_feature_with_gemini(
    feature_idx: int,
    image_urls: list[str],
    strongest_rating: str | None = None,
    strongest_correlation: float | None = None,
    api_key: str | None = None,
) -> FeatureLabelResult | None:
    """
    Convenience function to label a single feature.

    Args:
        feature_idx: Feature index
        image_urls: List of image URLs for top-activating images
        strongest_rating: Rating dimension most correlated with this feature
        strongest_correlation: Correlation coefficient
        api_key: Gemini API key (or use GEMINI_API_KEY env var)

    Returns:
        FeatureLabelResult or None if labeling fails

    Example:
        ```python
        result = label_feature_with_gemini(
            feature_idx=1234,
            image_urls=["http://example.com/art1.jpg", "http://example.com/art2.jpg"],
            strongest_rating="wholeness",
            strongest_correlation=0.35,
        )
        if result:
            print(f"Feature {result.feature_idx}: {result.short_label}")
        ```
    """
    labeler = GeminiLabeler(api_key=api_key)

    # Create a simple URL lookup from the list
    url_map = {f"image_{i}": url for i, url in enumerate(image_urls)}

    feature = FeatureInfo(
        feature_idx=feature_idx,
        top_artwork_ids=list(url_map.keys()),
        strongest_rating=strongest_rating,
        strongest_correlation=strongest_correlation,
    )

    return labeler.label_feature(
        feature=feature,
        get_image_url=lambda aid: url_map.get(aid),
    )
