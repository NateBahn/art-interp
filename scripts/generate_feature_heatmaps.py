#!/usr/bin/env python3
"""
Generate spatial activation heatmaps for SAE features.

Uses Prisma's spatial SAE for CLIP-B-32 to show WHERE in images
specific features activate most strongly.

Supports layers 7, 8, 9, and 11.
"""

import json
import argparse
import types
from pathlib import Path
from io import BytesIO

import numpy as np
import torch
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import zoom
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import open_clip

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from database.connection import get_db_context
from database.models import Artwork


# Create mock modules for loading Prisma SAE weights
# The weights.pt files were pickled with references to vit_prisma.sae classes
def _setup_mock_modules():
    """Create mock vit_prisma.sae modules to enable loading pickled weights."""
    def create_mock_module(name, attrs=None):
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        return mod

    class MockSAEConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockSparseAutoencoder:
        def __init__(self, cfg=None):
            self.cfg = cfg

    # Only create mocks if not already present
    if 'vit_prisma.sae' not in sys.modules:
        sys.modules['vit_prisma'] = create_mock_module('vit_prisma')
        sys.modules['vit_prisma.sae'] = create_mock_module('vit_prisma.sae')
        sys.modules['vit_prisma.sae.config'] = create_mock_module('vit_prisma.sae.config', {
            'VisionModelSAERunnerConfig': MockSAEConfig
        })
        sys.modules['vit_prisma.sae.sae'] = create_mock_module('vit_prisma.sae.sae', {
            'SparseAutoencoder': MockSparseAutoencoder
        })

_setup_mock_modules()


# Paths
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "sae_analysis"
HEATMAPS_DIR = OUTPUT_DIR / "heatmaps"
GEMINI_LABELS_FILE = OUTPUT_DIR / "feature_labels_gemini.json"
FEATURES_TO_LABEL_FILE = OUTPUT_DIR / "features_to_label.json"

# Spatial SAE repos for different layers
SPATIAL_SAE_REPOS = {
    7: "Prisma-Multimodal/imagenet-sweep-vanilla-x64-Spatial_max_7-hook_resid_post-984.1376953125-99",
    8: "Prisma-Multimodal/imagenet-sweep-vanilla-x64-Spatial_max_8-hook_resid_post-965.125-99",
    9: "Prisma-Multimodal/imagenet-sweep-vanilla-x64-Spatial_max_9-hook_resid_post-854.891540527344-99",
    11: "Prisma-Multimodal/imagenet-sweep-vanilla-x64-Spatial_max_11-hook_resid_post-829.0498046875-99",
}


class SpatialSAE:
    """Simple SAE class for spatial feature extraction."""

    def __init__(self, W_enc, b_enc, W_dec, b_dec):
        self.W_enc = W_enc  # [768, 49152]
        self.b_enc = b_enc  # [49152]
        self.W_dec = W_dec  # [49152, 768]
        self.b_dec = b_dec  # [768]

    def encode(self, x):
        """Encode activations to sparse features."""
        # x: [batch, 49, 768] or [49, 768]
        # Returns: [batch, 49, 49152] or [49, 49152]
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
    """CLIP model with hook to capture layer activations."""

    def __init__(self, layer: int = 8, device: str = "cpu"):
        self.device = device
        self.layer = layer
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model = self.model.to(device)
        self.model.eval()

        # Storage for hooked activations
        self.layer_activations = None

        # Register hook on specified layer
        self._register_hook()

    def _register_hook(self):
        """Register forward hook on specified layer."""
        def hook_fn(module, input, output):
            # output shape: [batch, seq_len, hidden_dim] = [B, 50, 768]
            # seq_len = 1 CLS token + 49 spatial patches (7x7)
            self.layer_activations = output.detach()

        # Hook on the output of target layer
        self.model.visual.transformer.resblocks[self.layer].register_forward_hook(hook_fn)

    def get_spatial_activations(self, image: Image.Image) -> torch.Tensor:
        """Get spatial patch activations for an image at configured layer."""
        # Preprocess
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Forward pass (triggers hook)
        with torch.no_grad():
            _ = self.model.encode_image(img_tensor)

        # Get spatial patches (exclude CLS token at position 0)
        # Shape: [1, 50, 768] -> [49, 768]
        spatial_acts = self.layer_activations[0, 1:, :]

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


def get_artwork_image_url(db, artwork_id: str) -> str | None:
    """Get image URL for an artwork."""
    artwork = db.query(Artwork).filter(Artwork.id == artwork_id).first()
    if artwork and artwork.image_url:
        return artwork.image_url
    return None


def create_heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    output_path: Path,
    alpha: float = 0.5,
    colormap: str = "hot",
):
    """Create and save heatmap overlay on image."""
    # Resize image to 224x224 (CLIP input size)
    img_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)

    # Upscale 7x7 heatmap to 224x224
    # Use zoom for smooth interpolation
    scale_factor = 224 / 7
    heatmap_upscaled = zoom(heatmap, scale_factor, order=3)  # cubic interpolation

    # Normalize heatmap to [0, 1]
    if heatmap_upscaled.max() > heatmap_upscaled.min():
        heatmap_normalized = (heatmap_upscaled - heatmap_upscaled.min()) / (
            heatmap_upscaled.max() - heatmap_upscaled.min()
        )
    else:
        heatmap_normalized = np.zeros_like(heatmap_upscaled)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Show image
    ax.imshow(img_array)

    # Overlay heatmap
    ax.imshow(heatmap_normalized, alpha=alpha, cmap=colormap)

    ax.axis("off")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close()


def create_side_by_side(
    image: Image.Image,
    heatmap: np.ndarray,
    output_path: Path,
    feature_idx: int,
    colormap: str = "hot",
):
    """Create side-by-side comparison: original + heatmap overlay."""
    # Resize image
    img_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)

    # Upscale heatmap
    scale_factor = 224 / 7
    heatmap_upscaled = zoom(heatmap, scale_factor, order=3)

    # Normalize
    if heatmap_upscaled.max() > heatmap_upscaled.min():
        heatmap_normalized = (heatmap_upscaled - heatmap_upscaled.min()) / (
            heatmap_upscaled.max() - heatmap_upscaled.min()
        )
    else:
        heatmap_normalized = np.zeros_like(heatmap_upscaled)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    ax1.imshow(img_array)
    ax1.set_title("Original", fontsize=12)
    ax1.axis("off")

    # Heatmap overlay
    ax2.imshow(img_array)
    im = ax2.imshow(heatmap_normalized, alpha=0.5, cmap=colormap)
    ax2.set_title(f"Feature {feature_idx} Activation", fontsize=12)
    ax2.axis("off")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Activation", fontsize=10)

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close()


def generate_aggregate_heatmap(
    artwork_id: str,
    image: Image.Image,
    clip_model: CLIPWithHook,
    sae: SpatialSAE,
    output_dir: Path,
) -> dict | None:
    """Generate aggregate spatial activation heatmap for an artwork.

    Shows total spatial SAE activation across all features - highlights
    which regions of the image have the strongest neural responses.
    """
    # Get spatial activations from CLIP layer 8
    spatial_acts = clip_model.get_spatial_activations(image)  # [49, 768]

    # Run through SAE to get features
    sae_features = sae.encode(spatial_acts)  # [49, 49152]

    # Create aggregate heatmap: sum of all feature activations per patch
    aggregate_heatmap = sae_features.sum(dim=1).cpu().numpy().reshape(7, 7)

    # Also find top active features for this image
    total_per_feature = sae_features.sum(dim=0)  # [49152]
    top_k = torch.topk(total_per_feature, 5)
    top_features = [
        {"feature_idx": int(idx), "total_activation": float(val)}
        for idx, val in zip(top_k.indices, top_k.values)
    ]

    # Generate outputs
    aggregate_path = output_dir / f"{artwork_id}_aggregate.png"

    # Create enhanced visualization with aggregate heatmap
    create_aggregate_visualization(image, aggregate_heatmap, aggregate_path, artwork_id, top_features)

    return {
        "artwork_id": artwork_id,
        "aggregate_path": str(aggregate_path.relative_to(OUTPUT_DIR)),
        "top_spatial_features": top_features,
        "max_aggregate_activation": float(aggregate_heatmap.max()),
        "mean_aggregate_activation": float(aggregate_heatmap.mean()),
    }


def create_aggregate_visualization(
    image: Image.Image,
    heatmap: np.ndarray,
    output_path: Path,
    artwork_id: str,
    top_features: list[dict],
    colormap: str = "viridis",
):
    """Create visualization showing aggregate spatial activations."""
    # Resize image
    img_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)

    # Upscale heatmap
    scale_factor = 224 / 7
    heatmap_upscaled = zoom(heatmap, scale_factor, order=3)

    # Normalize
    if heatmap_upscaled.max() > heatmap_upscaled.min():
        heatmap_normalized = (heatmap_upscaled - heatmap_upscaled.min()) / (
            heatmap_upscaled.max() - heatmap_upscaled.min()
        )
    else:
        heatmap_normalized = np.zeros_like(heatmap_upscaled)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original image
    ax1.imshow(img_array)
    ax1.set_title("Original", fontsize=12)
    ax1.axis("off")

    # Heatmap overlay
    ax2.imshow(img_array)
    im = ax2.imshow(heatmap_normalized, alpha=0.5, cmap=colormap)
    ax2.set_title(f"Spatial SAE Activation Map", fontsize=12)
    ax2.axis("off")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Activation Intensity", fontsize=10)

    # Add top features text
    feature_text = "Top Spatial Features:\n"
    for i, f in enumerate(top_features[:3]):
        feature_text += f"  #{i+1}: F{f['feature_idx']} ({f['total_activation']:.1f})\n"
    fig.text(0.02, 0.02, feature_text, fontsize=8, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close()


def generate_heatmaps_for_feature(
    feature_idx: int,
    artwork_ids: list[str],
    clip_model: CLIPWithHook,
    sae: SpatialSAE,
    db,
    output_dir: Path,
    max_artworks: int = 5,
) -> list[dict]:
    """Generate heatmaps for artworks associated with a CLS feature.

    Note: Since spatial SAE features differ from CLS SAE features, we generate
    aggregate spatial activation maps that show which regions are most active
    in the spatial SAE for these artworks.
    """
    results = []

    for artwork_id in artwork_ids[:max_artworks]:
        url = get_artwork_image_url(db, artwork_id)
        if not url:
            continue

        image = fetch_image(url)
        if not image:
            continue

        # Generate aggregate heatmap for this artwork
        result = generate_aggregate_heatmap(
            artwork_id=artwork_id,
            image=image,
            clip_model=clip_model,
            sae=sae,
            output_dir=output_dir,
        )

        if result:
            result["cls_feature_idx"] = feature_idx
            results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate spatial heatmaps for SAE features")
    parser.add_argument(
        "--layer", type=int, default=11,
        choices=list(SPATIAL_SAE_REPOS.keys()),
        help="CLIP layer for spatial SAE (default: 11)"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input features file (default: features_to_label_layer{layer}.json)"
    )
    parser.add_argument(
        "--feature", type=int, default=None,
        help="Generate heatmaps for a specific feature index"
    )
    parser.add_argument(
        "--max-features", type=int, default=None,
        help="Maximum number of features to process"
    )
    parser.add_argument(
        "--max-artworks", type=int, default=5,
        help="Maximum artworks per feature"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use (cpu/cuda/mps)"
    )
    parser.add_argument(
        "--mirror-self", action="store_true",
        help="Use mirror_self features file"
    )
    args = parser.parse_args()

    layer = args.layer

    print("=" * 60)
    print(f"Generating Spatial SAE Feature Heatmaps (Layer {layer})")
    print("=" * 60)

    # Determine input file
    if args.input:
        input_file = OUTPUT_DIR / args.input
    elif args.mirror_self:
        input_file = OUTPUT_DIR / "mirror_self_features_layer11.json"
    else:
        input_file = OUTPUT_DIR / f"features_to_label_layer{layer}.json"

    # Determine output directory (always layer-specific)
    if args.mirror_self:
        heatmaps_dir = OUTPUT_DIR / f"heatmaps_mirror_self_layer{layer}"
    else:
        heatmaps_dir = OUTPUT_DIR / f"heatmaps_layer{layer}"

    # Load feature data
    print(f"\nLoading features from {input_file}...")
    with open(input_file) as f:
        features_data = json.load(f)

    features = features_data.get("features", [])
    print(f"  Found {len(features)} features with artwork IDs")

    # Filter to specific feature if requested
    if args.feature is not None:
        features = [f for f in features if f["feature_idx"] == args.feature]
        if not features:
            print(f"  Feature {args.feature} not found!")
            return
        print(f"  Filtered to feature {args.feature}")

    # Limit features if requested
    if args.max_features:
        features = features[:args.max_features]
        print(f"  Limited to {len(features)} features")

    # Load models
    print(f"\nLoading CLIP model (layer={layer}, device={args.device})...")
    clip_model = CLIPWithHook(layer=layer, device=args.device)

    print(f"\nLoading Spatial SAE for layer {layer}...")
    sae_repo = SPATIAL_SAE_REPOS[layer]
    sae = SpatialSAE.from_pretrained(sae_repo, device=args.device)
    print(f"  SAE loaded: 768 -> 49152 features")

    # Generate heatmaps
    print(f"\nGenerating heatmaps for {len(features)} features...")
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    with get_db_context() as db:
        for feature in tqdm(features):
            feature_idx = feature["feature_idx"]
            # Handle both formats: "top_5_artwork_ids" and "top_artworks"
            artwork_ids = feature.get("top_5_artwork_ids") or feature.get("top_artworks", [])

            if not artwork_ids:
                continue

            results = generate_heatmaps_for_feature(
                feature_idx=feature_idx,
                artwork_ids=artwork_ids,
                clip_model=clip_model,
                sae=sae,
                db=db,
                output_dir=heatmaps_dir,
                max_artworks=args.max_artworks,
            )

            if results:
                all_results[feature_idx] = results

    # Save index (always layer-specific)
    index_name = f"heatmap_index_mirror_self_layer{layer}.json" if args.mirror_self else f"heatmap_index_layer{layer}.json"
    index_path = OUTPUT_DIR / index_name
    with open(index_path, "w") as f:
        json.dump({
            "layer": layer,
            "input_file": str(input_file.name),
            "total_features": len(all_results),
            "total_heatmaps": sum(len(v) for v in all_results.values()),
            "features": all_results,
        }, f, indent=2)

    print(f"\nGenerated heatmaps for {len(all_results)} features")
    print(f"Total heatmaps: {sum(len(v) for v in all_results.values())}")
    print(f"Saved index to {index_path}")


if __name__ == "__main__":
    main()
