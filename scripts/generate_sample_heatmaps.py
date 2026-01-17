#!/usr/bin/env python3
"""
Generate per-feature heatmap data for sample paintings.

For each painting, stores the full 7x7 activation grid for each of the
top correlated features, allowing the UI to show WHERE each feature fires.
"""

import json
from pathlib import Path
from io import BytesIO

import numpy as np
import torch
import requests
from PIL import Image
from huggingface_hub import hf_hub_download

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "heatmaps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load correlation data to get top features per question
ANALYSIS_DIR = Path(__file__).parent.parent / "output" / "sae_analysis"

# Spatial SAE repo for CLIP-B-32 layer 8
SPATIAL_SAE_REPO = "Prisma-Multimodal/imagenet-sweep-vanilla-x64-Spatial_max_8-hook_resid_post-965.125-99"

# Sample paintings to process
SAMPLE_PAINTINGS = [
    {
        "id": "met_435922",
        "title": "Salisbury Cathedral from the Bishop's Grounds",
        "artist": "John Constable",
        "year": 1825,
        "image_url": "https://images.metmuseum.org/CRDImages/ep/original/DP164837.jpg",
    },
    {
        "id": "met_435904",
        "title": "Still Life with a Skull and a Writing Quill",
        "artist": "Pieter Claesz",
        "year": 1628,
        "image_url": "https://images.metmuseum.org/CRDImages/ep/original/DP145929.jpg",
    },
    {
        "id": "met_435897",
        "title": "Head of Christ (Ecce Homo)",
        "artist": "Guido Reni",
        "year": 1640,
        "image_url": "https://images.metmuseum.org/CRDImages/ep/original/DT229079.jpg",
    },
]


def load_top_correlated_features() -> dict[str, list[int]]:
    """Load the top correlated features for each rating question."""
    correlations_path = ANALYSIS_DIR / "all_correlations.json"
    if not correlations_path.exists():
        print(f"Warning: {correlations_path} not found, using default features")
        return {}

    with open(correlations_path) as f:
        data = json.load(f)

    result = {}
    top_features_by_rating = data.get("top_features_by_rating", {})
    for question, features in top_features_by_rating.items():
        # Get top 20 feature indices for each question
        result[question] = [f["feature_idx"] for f in features[:20]]

    return result


class SpatialSAE:
    """Simple SAE class for spatial feature extraction."""

    def __init__(self, W_enc, b_enc, W_dec, b_dec):
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.W_dec = W_dec
        self.b_dec = b_dec
        self.num_features = W_enc.shape[1]

    def encode(self, x):
        """Encode activations to sparse features."""
        features = torch.relu(x @ self.W_enc + self.b_enc)
        return features

    @classmethod
    def from_pretrained(cls, repo_id: str, device: str = "cpu"):
        """Load SAE from HuggingFace."""
        weights_path = hf_hub_download(repo_id, "weights.pt")
        weights = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = weights["state_dict"]

        return cls(
            W_enc=state_dict["W_enc"].to(device),
            b_enc=state_dict["b_enc"].to(device),
            W_dec=state_dict["W_dec"].to(device),
            b_dec=state_dict["b_dec"].to(device),
        )


class CLIPWithHook:
    """CLIP model with hook to capture layer 8 activations."""

    def __init__(self, device: str = "cpu"):
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.layer8_activations = None
        self._register_hook()

    def _register_hook(self):
        """Register forward hook on layer 8."""
        def hook_fn(module, input, output):
            self.layer8_activations = output.detach()

        self.model.visual.transformer.resblocks[8].register_forward_hook(hook_fn)

    def get_spatial_activations(self, image: Image.Image) -> torch.Tensor:
        """Get layer 8 spatial patch activations for an image."""
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _ = self.model.encode_image(img_tensor)

        # Exclude CLS token at position 0
        spatial_acts = self.layer8_activations[0, 1:, :]
        return spatial_acts


def fetch_image(url: str) -> Image.Image | None:
    """Fetch image from URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return None


def generate_feature_heatmaps(
    painting: dict,
    clip_model: CLIPWithHook,
    sae: SpatialSAE,
    features_by_question: dict[str, list[int]],
) -> dict:
    """Generate per-feature heatmaps for a single painting."""
    painting_id = painting["id"]
    print(f"\nProcessing: {painting['title']}")

    # Fetch image
    image = fetch_image(painting["image_url"])
    if not image:
        return None

    # Get original image size
    orig_width, orig_height = image.size
    print(f"  Original size: {orig_width}x{orig_height}")

    # Get spatial activations
    spatial_acts = clip_model.get_spatial_activations(image)  # [49, 768]

    # Run through SAE to get all feature activations
    sae_features = sae.encode(spatial_acts)  # [49, 49152]
    print(f"  SAE features shape: {sae_features.shape}")

    # Collect all unique feature indices we need
    all_feature_indices = set()
    for features in features_by_question.values():
        all_feature_indices.update(features)

    # Also get top 20 most active features overall for this painting
    total_per_feature = sae_features.sum(dim=0)
    top_k = torch.topk(total_per_feature, 20)
    top_active_features = [
        {"feature_idx": int(idx), "total_activation": float(val)}
        for idx, val in zip(top_k.indices, top_k.values)
    ]
    all_feature_indices.update([f["feature_idx"] for f in top_active_features])

    print(f"  Extracting heatmaps for {len(all_feature_indices)} unique features")

    # Generate 7x7 heatmap for each feature
    feature_heatmaps = {}
    for feat_idx in all_feature_indices:
        # Get this feature's activation across all 49 patches
        feat_activations = sae_features[:, feat_idx].cpu().numpy()  # [49]
        # Reshape to 7x7 grid
        heatmap_7x7 = feat_activations.reshape(7, 7).tolist()
        feature_heatmaps[str(feat_idx)] = {
            "heatmap_7x7": heatmap_7x7,
            "max_activation": float(feat_activations.max()),
            "mean_activation": float(feat_activations.mean()),
            "total_activation": float(feat_activations.sum()),
        }

    # Save the resized image for consistency in the UI
    output_size = (512, 512)
    image_resized = image.resize(output_size, Image.Resampling.LANCZOS)
    image_path = OUTPUT_DIR / f"{painting_id}_image.jpg"
    image_resized.save(image_path, quality=90)
    print(f"  Saved: {image_path}")

    # Build the complete data structure
    heatmap_data = {
        "painting_id": painting_id,
        "title": painting["title"],
        "artist": painting.get("artist"),
        "year": painting.get("year"),
        "image_url": painting["image_url"],
        "feature_heatmaps": feature_heatmaps,
        "top_active_features": top_active_features,
    }

    return heatmap_data


def main():
    print("=" * 60)
    print("Generating Per-Feature Heatmaps for Sample Paintings")
    print("=" * 60)

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\nUsing device: {device}")

    # Load top correlated features per question
    print("\nLoading correlation data...")
    features_by_question = load_top_correlated_features()
    if features_by_question:
        for q, feats in features_by_question.items():
            print(f"  {q}: {len(feats)} features")
    else:
        print("  No correlation data found, will use top active features only")

    # Load models
    print("\nLoading CLIP model...")
    clip_model = CLIPWithHook(device=device)

    print("Loading Spatial SAE...")
    sae = SpatialSAE.from_pretrained(SPATIAL_SAE_REPO, device=device)
    print(f"  SAE loaded: 768 -> {sae.num_features} features")

    # Generate heatmaps
    all_data = []
    for painting in SAMPLE_PAINTINGS:
        data = generate_feature_heatmaps(painting, clip_model, sae, features_by_question)
        if data:
            all_data.append(data)

    # Save index
    index_path = OUTPUT_DIR / "index.json"
    with open(index_path, "w") as f:
        json.dump({
            "paintings": all_data,
            "total": len(all_data),
            "features_by_question": {q: feats for q, feats in features_by_question.items()},
        }, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Generated per-feature heatmaps for {len(all_data)} paintings")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Index file: {index_path}")

    # Summary
    if all_data:
        sample = all_data[0]
        print(f"\nExample: {sample['title']}")
        print(f"  Features with heatmaps: {len(sample['feature_heatmaps'])}")
        print(f"  Top active feature: #{sample['top_active_features'][0]['feature_idx']} "
              f"(activation: {sample['top_active_features'][0]['total_activation']:.1f})")


if __name__ == "__main__":
    main()
