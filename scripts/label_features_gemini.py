#!/usr/bin/env python3
"""
Label SAE features using Gemini 2.0 Flash VLM.

Sends top-activating artwork images to Gemini and asks it to identify
the visual pattern that explains why these artworks activate the feature.
"""

import json
import argparse
import time
import re
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image
import google.generativeai as genai


# Database connection for getting image URLs
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from database.connection import get_db_context
from database.models import Artwork


# Gemini API configuration
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")


def get_artwork_image_url(db, artwork_id: str) -> str | None:
    """Get image URL for an artwork from database."""
    artwork = db.query(Artwork).filter(Artwork.id == artwork_id).first()
    if artwork and artwork.image_url:
        return artwork.image_url
    return None


def fetch_image(url: str) -> Image.Image | None:
    """Fetch image from URL and return as PIL Image."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return None


def build_prompt(feature: dict, num_images: int = 5) -> str:
    """Build the Gemini prompt for a feature."""
    rating = feature.get("strongest_rating")
    correlation = feature.get("strongest_correlation")

    if rating and correlation:
        direction = "positively" if correlation > 0 else "negatively"
        change = "increase" if correlation > 0 else "decrease"

        return f"""You are an art historian and computer vision researcher.

These {num_images} artworks all strongly activate the same neural network feature (Feature #{feature['feature_idx']}).
This feature correlates {direction} with people's "{rating}" ratings of artworks.
Correlation: r = {correlation:.3f}

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
  "explains_rating": "How this pattern might influence the {rating} rating",
  "non_obvious_insight": "What's surprising or non-obvious about this pattern?",
  "confidence": 0.0-1.0 (how confident are you that these images share a coherent pattern? 1.0 = very clear pattern, 0.5 = somewhat ambiguous, 0.0 = no clear pattern)
}}"""
    else:
        # Exploration prompt for features without strong correlations
        return f"""You are an art historian and computer vision researcher.

These {num_images} artworks all strongly activate the same neural network feature (Feature #{feature['feature_idx']}).
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


def parse_gemini_response(response_text: str) -> dict | None:
    """Parse JSON from Gemini response, handling markdown code blocks."""
    text = response_text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        # Find the end of the opening code fence
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        # Remove closing code fence
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from the text
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        print(f"  Failed to parse JSON: {text[:200]}...")
        return None


def label_feature(
    model: genai.GenerativeModel,
    feature: dict,
    db,
    max_images: int = 5,
) -> dict | None:
    """Label a single feature using Gemini."""
    feature_idx = feature["feature_idx"]
    artwork_ids = feature.get("top_5_artwork_ids", [])[:max_images]

    if not artwork_ids:
        print(f"  No artwork IDs for feature {feature_idx}")
        return None

    # Fetch images
    images = []
    for artwork_id in artwork_ids:
        url = get_artwork_image_url(db, artwork_id)
        if url:
            img = fetch_image(url)
            if img:
                images.append(img)

    if len(images) < 2:
        print(f"  Only {len(images)} images available for feature {feature_idx}")
        return None

    # Build prompt and send to Gemini
    prompt = build_prompt(feature, num_images=len(images))

    try:
        response = model.generate_content([prompt] + images)
        result = parse_gemini_response(response.text)

        if result:
            # Add metadata
            result["feature_idx"] = feature_idx
            result["tier"] = feature.get("tier")
            result["strongest_rating"] = feature.get("strongest_rating")
            result["strongest_correlation"] = feature.get("strongest_correlation")
            result["monosemanticity_score"] = feature.get("monosemanticity_score")
            result["num_images_used"] = len(images)

        return result

    except Exception as e:
        print(f"  Gemini API error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Label SAE features with Gemini VLM")
    parser.add_argument(
        "--layer",
        type=int,
        default=11,
        choices=[7, 8, 11],
        help="SAE layer to label features from (default: 11)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input features file (default: features_to_label_layer{layer}.json)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output labels file (default: feature_labels_gemini_layer{layer}.json)"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Maximum number of features to label (for testing)"
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Only label features from this tier"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds"
    )
    parser.add_argument(
        "--images",
        type=int,
        default=5,
        choices=[3, 5, 7, 10],
        help="Number of images to send to Gemini per feature (default: 5)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file"
    )
    args = parser.parse_args()

    layer = args.layer

    # Resolve paths with layer-specific defaults
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "output" / "sae_analysis"

    if args.input:
        input_path = script_dir / args.input
    else:
        input_path = output_dir / f"features_to_label_layer{layer}.json"

    if args.output:
        output_path = script_dir / args.output
    else:
        output_path = output_dir / f"feature_labels_gemini_layer{layer}.json"

    print("=" * 60)
    print(f"Labeling SAE Features with Gemini (Layer {layer})")
    print("=" * 60)

    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")

    # Load features
    print(f"\nLoading features from {input_path}")
    with open(input_path) as f:
        data = json.load(f)

    features = data["features"]
    print(f"  Loaded {len(features)} features")

    # Filter by tier if specified
    if args.tier:
        features = [f for f in features if f.get("tier") == args.tier]
        print(f"  Filtered to {len(features)} tier {args.tier} features")

    # Limit if specified
    if args.max_features:
        features = features[:args.max_features]
        print(f"  Limited to {len(features)} features")

    # Load existing results if resuming
    existing_labels = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing_data = json.load(f)
        for label in existing_data.get("labels", []):
            existing_labels[label["feature_idx"]] = label
        print(f"  Loaded {len(existing_labels)} existing labels")

    # Label features
    labels = list(existing_labels.values())
    labeled_ids = set(existing_labels.keys())

    with get_db_context() as db:
        for i, feature in enumerate(features):
            feature_idx = feature["feature_idx"]

            if feature_idx in labeled_ids:
                print(f"[{i+1}/{len(features)}] Feature {feature_idx}: already labeled")
                continue

            print(f"[{i+1}/{len(features)}] Labeling feature {feature_idx} "
                  f"(tier {feature.get('tier')}, mono={feature.get('monosemanticity_score', 0):.3f})")

            result = label_feature(model, feature, db, max_images=args.images)

            if result:
                labels.append(result)
                labeled_ids.add(feature_idx)
                print(f"  -> {result.get('short_label', 'No label')}")

                # Save incrementally
                output = {
                    "metadata": {
                        "layer": layer,
                        "total_labels": len(labels),
                        "model": "gemini-2.0-flash-exp",
                    },
                    "labels": labels,
                }
                with open(output_path, "w") as f:
                    json.dump(output, f, indent=2)
            else:
                print(f"  -> Failed to label")

            # Rate limiting
            if i < len(features) - 1:
                time.sleep(args.delay)

    print(f"\nLabeled {len(labels)} features")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
