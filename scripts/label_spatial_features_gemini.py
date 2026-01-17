#!/usr/bin/env python3
"""
Label Spatial SAE features using Gemini 2.0 Flash VLM.

Enhanced version that includes heatmaps showing WHERE features activate,
giving Gemini spatial context for better interpretations.
"""

import json
import argparse
import time
import re
from pathlib import Path
from io import BytesIO

import numpy as np
import requests
from PIL import Image
import google.generativeai as genai
import torch

# Database connection for getting image URLs
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from database.connection import get_db_context
from database.models import Artwork

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "sae_analysis"
SPATIAL_FEATURES_DIR = SCRIPT_DIR / "data" / "spatial_sae_features"
CORRELATIONS_FILE = OUTPUT_DIR / "spatial_correlations.json"

# Gemini API configuration
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Spatial SAE configuration
SPATIAL_SAE_REPO = "Prisma-Multimodal/imagenet-sweep-vanilla-x64-Spatial_max_8-hook_resid_post-965.125-99"


class SpatialHeatmapGenerator:
    """Generate heatmaps for spatial SAE features."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._load_clip_model()
        self._load_sae_weights()

    def _load_clip_model(self):
        """Load CLIP model with activation hook."""
        import open_clip
        print("Loading CLIP ViT-B-32...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.layer8_activations = None
        self._register_hook()

    def _register_hook(self):
        """Register forward hook on layer 8."""
        def hook_fn(module, input, output):
            self.layer8_activations = output.detach()
        self.model.visual.transformer.resblocks[8].register_forward_hook(hook_fn)

    def _load_sae_weights(self):
        """Load Spatial SAE weights."""
        from huggingface_hub import hf_hub_download
        print(f"Loading Spatial SAE...")
        weights_path = hf_hub_download(SPATIAL_SAE_REPO, "weights.pt")
        weights = torch.load(weights_path, map_location=self.device, weights_only=False)
        state_dict = weights["state_dict"]
        self.W_enc = state_dict["W_enc"].to(self.device)
        self.b_enc = state_dict["b_enc"].to(self.device)
        print(f"SAE loaded: 768 -> {self.W_enc.shape[1]} features")

    def generate_heatmap(self, image: Image.Image, feature_idx: int) -> np.ndarray:
        """Generate 7x7 heatmap for a specific feature on an image."""
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _ = self.model.encode_image(img_tensor)

        # Get spatial patches (exclude CLS token)
        spatial_acts = self.layer8_activations[0, 1:, :]  # (49, 768)

        # Pass through SAE encoder
        spatial_features = torch.relu(spatial_acts @ self.W_enc + self.b_enc)  # (49, 49152)

        # Extract target feature and reshape to 7x7
        heatmap = spatial_features[:, feature_idx].reshape(7, 7).cpu().numpy()

        return heatmap

    def create_overlay_image(self, image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
        """Create an image with heatmap overlay."""
        # Resize image to standard size
        img = image.copy()
        img = img.resize((224, 224), Image.Resampling.LANCZOS)

        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap_norm = heatmap / heatmap.max()
        else:
            heatmap_norm = heatmap

        # Upscale heatmap to image size
        heatmap_img = Image.fromarray((heatmap_norm * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize((224, 224), Image.Resampling.NEAREST)

        # Apply colormap (green for high activation)
        heatmap_colored = Image.new("RGBA", (224, 224))
        heatmap_data = np.array(heatmap_img)

        for y in range(224):
            for x in range(224):
                val = heatmap_data[y, x] / 255.0
                if val > 0.1:  # Only show significant activations
                    # Green with intensity based on activation
                    r = int(50 * (1 - val))
                    g = int(255 * val)
                    b = int(50 * (1 - val))
                    a = int(180 * val)
                    heatmap_colored.putpixel((x, y), (r, g, b, a))

        # Composite
        img_rgba = img.convert("RGBA")
        result = Image.alpha_composite(img_rgba, heatmap_colored)
        return result.convert("RGB")


def fetch_image(url: str) -> Image.Image | None:
    """Fetch image from URL and return as PIL Image."""
    try:
        response = requests.get(url, timeout=30, headers={
            "User-Agent": "ArtRecommender/1.0 (Research)"
        })
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return None


def load_spatial_features() -> dict:
    """Load all spatial features from batch files."""
    all_features = {}
    for batch_file in sorted(SPATIAL_FEATURES_DIR.glob("spatial_batch_*.json")):
        with open(batch_file) as f:
            batch = json.load(f)
            all_features.update(batch)
    return all_features


def get_top_activating_artworks(spatial_features: dict, feature_idx: int, top_k: int = 5) -> list:
    """Get artwork IDs with highest activation for a feature."""
    activations = []
    for artwork_id, data in spatial_features.items():
        sparse = data.get("sparse_features", {})
        activation = sparse.get(str(feature_idx), 0)
        if activation > 0:
            activations.append((artwork_id, activation))

    activations.sort(key=lambda x: -x[1])
    return activations[:top_k]


def build_prompt(feature_idx: int, rating: str, correlation: float) -> str:
    """Build the Gemini prompt for a spatial feature."""
    direction = "positively" if correlation > 0 else "negatively"
    change = "higher" if correlation > 0 else "lower"

    return f"""You are an art historian and computer vision researcher analyzing neural network features.

I'm showing you 5 artworks that strongly activate Spatial Feature #{feature_idx}.
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


def parse_gemini_response(response_text: str) -> dict | None:
    """Parse JSON from Gemini response."""
    text = response_text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
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
    heatmap_gen: SpatialHeatmapGenerator,
    feature_idx: int,
    rating: str,
    correlation: float,
    spatial_features: dict,
    artwork_data: dict,
    max_images: int = 5,
) -> dict | None:
    """Label a single spatial feature using Gemini with heatmaps."""

    # Get top activating artworks
    top_artworks = get_top_activating_artworks(spatial_features, feature_idx, max_images)

    if len(top_artworks) < 2:
        print(f"  Not enough activating artworks for feature {feature_idx}")
        return None

    # Prepare images: original + heatmap overlay for each artwork
    images = []
    artwork_info = []

    for artwork_id, activation in top_artworks:
        artwork = artwork_data.get(artwork_id)
        if not artwork:
            continue

        url = artwork.get("image_url")
        if not url:
            continue

        img = fetch_image(url)
        if not img:
            continue

        # Generate heatmap overlay
        heatmap = heatmap_gen.generate_heatmap(img, feature_idx)
        overlay = heatmap_gen.create_overlay_image(img, heatmap)

        # Add both original and overlay
        images.append(img.resize((224, 224), Image.Resampling.LANCZOS))
        images.append(overlay)

        artwork_info.append({
            "id": artwork_id,
            "title": artwork.get("title"),
            "artist": artwork.get("artist"),
            "activation": activation,
        })

    if len(images) < 4:  # Need at least 2 artworks (4 images)
        print(f"  Only {len(images)//2} valid artworks for feature {feature_idx}")
        return None

    # Build prompt and send to Gemini
    prompt = build_prompt(feature_idx, rating, correlation)

    try:
        response = model.generate_content([prompt] + images)
        result = parse_gemini_response(response.text)

        if result:
            result["feature_idx"] = feature_idx
            result["rating"] = rating
            result["correlation"] = correlation
            result["num_artworks"] = len(artwork_info)
            result["top_artworks"] = artwork_info

        return result

    except Exception as e:
        print(f"  Gemini API error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Label spatial SAE features with Gemini VLM + heatmaps")
    parser.add_argument(
        "--rating",
        type=str,
        default="mirror_self",
        help="Rating dimension to analyze"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top features to label"
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Only label positive correlations"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (default: spatial_labels_{rating}.json)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between API calls in seconds"
    )
    args = parser.parse_args()

    # Set output path
    if args.output is None:
        args.output = OUTPUT_DIR / f"spatial_labels_{args.rating}.json"

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load correlations
    print(f"\nLoading spatial correlations...")
    with open(CORRELATIONS_FILE) as f:
        corr_data = json.load(f)

    # Get features for this rating
    rating_features = corr_data["top_features_by_rating"].get(args.rating, [])
    if not rating_features:
        print(f"No features found for rating: {args.rating}")
        return

    # Filter and limit
    if args.positive_only:
        rating_features = [f for f in rating_features if f["correlation"] > 0]
    features_to_label = rating_features[:args.top_k]

    print(f"Will label {len(features_to_label)} features for '{args.rating}'")

    # Load spatial features
    print("\nLoading spatial features...")
    spatial_features = load_spatial_features()
    print(f"  Loaded {len(spatial_features)} artworks")

    # Load artwork metadata
    print("\nLoading artwork metadata...")
    artwork_data = {}
    with get_db_context() as db:
        for a in db.query(Artwork).all():
            artist_name = None
            if a.artist:
                artist_name = a.artist.name if hasattr(a.artist, 'name') else str(a.artist)
            artwork_data[a.id] = {
                "title": a.title,
                "artist": artist_name,
                "year": a.year,
                "image_url": a.image_url,
            }
    print(f"  Loaded {len(artwork_data)} artworks")

    # Initialize heatmap generator
    print("\nInitializing heatmap generator...")
    heatmap_gen = SpatialHeatmapGenerator(device=device)

    # Configure Gemini
    print("\nConfiguring Gemini...")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")

    # Label features
    print("\n" + "=" * 60)
    print(f"LABELING SPATIAL FEATURES FOR '{args.rating.upper()}'")
    print("=" * 60)

    labels = []
    for i, feat in enumerate(features_to_label):
        feature_idx = feat["feature_idx"]
        correlation = feat["correlation"]

        print(f"\n[{i+1}/{len(features_to_label)}] Feature #{feature_idx} (r = {correlation:+.4f})")

        result = label_feature(
            model=model,
            heatmap_gen=heatmap_gen,
            feature_idx=feature_idx,
            rating=args.rating,
            correlation=correlation,
            spatial_features=spatial_features,
            artwork_data=artwork_data,
        )

        if result:
            labels.append(result)
            print(f"  -> {result.get('short_label', 'No label')}")
            print(f"     Spatial focus: {result.get('spatial_focus', 'N/A')}")
        else:
            print(f"  -> Failed to label")

        # Rate limiting
        if i < len(features_to_label) - 1:
            time.sleep(args.delay)

    # Save results
    output = {
        "metadata": {
            "rating": args.rating,
            "model": "gemini-2.0-flash-exp",
            "num_labels": len(labels),
            "sae_type": "spatial",
        },
        "labels": labels,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Labeled {len(labels)} features")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
