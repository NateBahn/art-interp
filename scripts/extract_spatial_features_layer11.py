#!/usr/bin/env python3
"""
Extract spatial SAE features from Layer 11 for all artworks.

Stores sparse features (all non-zero activations) for each of the 49 spatial patches.
This enables discovering novel patterns without information loss.

Output format per artwork:
{
    "artwork_id": str,
    "patches": [
        {"idx": [feature_indices], "val": [activation_values]},  # patch 0 (top-left)
        ...  # 49 patches total (7x7 grid)
    ],
    "stats": {
        "total_active": int,  # total non-zero features across all patches
        "mean_active_per_patch": float,
        "max_activation": float,
    }
}

Storage estimate: ~50-100 KB per artwork (sparse), ~1-2 GB for 16k artworks
"""

import json
import argparse
import types
import sys
from pathlib import Path
from io import BytesIO
from datetime import datetime

import numpy as np
import torch
import httpx
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import open_clip

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.connection import get_db_context
from database.models import Artwork


# Create mock modules for loading Prisma SAE weights
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


# Configuration
LAYER = 11
SPATIAL_SAE_REPO = "Prisma-Multimodal/imagenet-sweep-vanilla-x64-Spatial_max_11-hook_resid_post-829.0498046875-99"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sae_features_spatial_layer11"
BATCH_SIZE = 100  # Save progress every N artworks
NUM_PATCHES = 49  # 7x7 grid
SAE_DIM = 49152


class SpatialSAE:
    """Spatial SAE for extracting patch-level features."""

    def __init__(self, W_enc, b_enc, device="cpu"):
        self.W_enc = W_enc.to(device)  # [768, 49152]
        self.b_enc = b_enc.to(device)  # [49152]
        self.device = device

    def encode(self, x):
        """Encode spatial activations to sparse features.

        Args:
            x: [49, 768] spatial patch activations

        Returns:
            [49, 49152] sparse feature activations
        """
        return torch.relu(x @ self.W_enc + self.b_enc)

    @classmethod
    def from_pretrained(cls, repo_id: str, device: str = "cpu"):
        """Load SAE from HuggingFace."""
        print(f"  Downloading SAE from {repo_id}...")
        weights_path = hf_hub_download(repo_id, "weights.pt")
        weights = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = weights["state_dict"]
        return cls(
            W_enc=state_dict["W_enc"],
            b_enc=state_dict["b_enc"],
            device=device,
        )


class CLIPWithHook:
    """CLIP model with hook to capture layer activations."""

    def __init__(self, layer: int = 11, device: str = "cpu"):
        self.device = device
        self.layer = layer
        print(f"  Loading CLIP ViT-B-32...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.layer_activations = None
        self._register_hook()

    def _register_hook(self):
        def hook_fn(module, input, output):
            self.layer_activations = output.detach()
        self.model.visual.transformer.resblocks[self.layer].register_forward_hook(hook_fn)

    def get_spatial_activations(self, image: Image.Image) -> torch.Tensor:
        """Get spatial patch activations for an image.

        Returns:
            [49, 768] tensor of spatial patch activations
        """
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _ = self.model.encode_image(img_tensor)
        # Exclude CLS token (position 0), keep 49 spatial patches
        return self.layer_activations[0, 1:, :]


def fetch_image(url: str, timeout: float = 30.0) -> Image.Image | None:
    """Fetch image from URL with retry logic."""
    headers = {"User-Agent": "ArtRecommender/1.0 (Research Project)"}
    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
    except Exception as e:
        return None


def extract_sparse_features(
    sae_features: torch.Tensor,
    top_k: int = 500,
) -> list[dict]:
    """Convert dense SAE features to sparse format (top-k per patch).

    Args:
        sae_features: [49, 49152] dense feature tensor
        top_k: number of top features to keep per patch (default 500)

    Returns:
        List of 49 dicts, each with 'idx' (feature indices) and 'val' (activations)
    """
    patches = []
    for patch_idx in range(NUM_PATCHES):
        patch_features = sae_features[patch_idx]

        # Get top-k features by activation value
        k = min(top_k, (patch_features > 0).sum().item())  # Don't exceed non-zero count
        if k > 0:
            topk = torch.topk(patch_features, k)
            indices = topk.indices.cpu().tolist()
            values = [round(v, 4) for v in topk.values.cpu().tolist()]
        else:
            indices = []
            values = []

        patches.append({
            "idx": indices,
            "val": values,
        })

    return patches


def process_artwork(
    artwork_id: str,
    image_url: str,
    clip_model: CLIPWithHook,
    sae: SpatialSAE,
    top_k: int = 500,
) -> dict | None:
    """Process a single artwork and return sparse spatial features."""
    # Fetch image
    image = fetch_image(image_url)
    if image is None:
        return None

    # Get CLIP spatial activations
    spatial_acts = clip_model.get_spatial_activations(image)  # [49, 768]

    # Run through SAE
    sae_features = sae.encode(spatial_acts)  # [49, 49152]

    # Convert to sparse format (top-k per patch)
    patches = extract_sparse_features(sae_features, top_k=top_k)

    # Compute stats
    total_stored = sum(len(p["idx"]) for p in patches)
    total_nonzero = int((sae_features > 0).sum().cpu())
    max_activation = float(sae_features.max().cpu())

    return {
        "artwork_id": artwork_id,
        "patches": patches,
        "stats": {
            "features_stored": total_stored,
            "features_nonzero": total_nonzero,
            "retention_pct": round(100 * total_stored / total_nonzero, 1) if total_nonzero > 0 else 0,
            "max_activation": round(max_activation, 2),
        }
    }


def save_batch(features: list[dict], batch_num: int, output_dir: Path):
    """Save a batch of features to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_file = output_dir / f"batch_{batch_num:04d}.json"
    with open(batch_file, "w") as f:
        json.dump(features, f)
    return batch_file


def load_progress(output_dir: Path) -> set[str]:
    """Load already-processed artwork IDs from existing batches."""
    processed = set()
    if not output_dir.exists():
        return processed

    for batch_file in output_dir.glob("batch_*.json"):
        with open(batch_file) as f:
            batch = json.load(f)
            for item in batch:
                processed.add(item["artwork_id"])

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Extract spatial SAE features (Layer 11) for all artworks"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of artworks to process (for testing)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use (cpu/cuda/mps)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Save progress every N artworks (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing progress"
    )
    parser.add_argument(
        "--top-k", type=int, default=500,
        help="Top-k features to store per patch (default: 500)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"Extracting Spatial SAE Features (Layer {LAYER})")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Top-k per patch: {args.top_k}")

    # Load models
    print("\nLoading models...")
    clip_model = CLIPWithHook(layer=LAYER, device=args.device)
    sae = SpatialSAE.from_pretrained(SPATIAL_SAE_REPO, device=args.device)
    print("  Models loaded successfully")

    # Load progress if resuming
    processed_ids = set()
    if args.resume:
        processed_ids = load_progress(OUTPUT_DIR)
        print(f"\nResuming: {len(processed_ids)} artworks already processed")

    # Get artworks from database
    print("\nLoading artworks from database...")
    with get_db_context() as db:
        query = db.query(Artwork.id, Artwork.image_url).filter(
            Artwork.image_url.isnot(None)
        )
        artworks = query.all()

    print(f"  Found {len(artworks)} artworks with images")

    # Filter out already processed
    if processed_ids:
        artworks = [(id, url) for id, url in artworks if id not in processed_ids]
        print(f"  {len(artworks)} remaining to process")

    # Apply limit if specified
    if args.limit:
        artworks = artworks[:args.limit]
        print(f"  Limited to {len(artworks)} artworks")

    # Process artworks
    print(f"\nProcessing {len(artworks)} artworks...")

    current_batch = []
    batch_num = len(list(OUTPUT_DIR.glob("batch_*.json"))) if OUTPUT_DIR.exists() else 0
    success_count = 0
    fail_count = 0

    start_time = datetime.now()

    for artwork_id, image_url in tqdm(artworks, desc="Extracting features"):
        result = process_artwork(artwork_id, image_url, clip_model, sae, top_k=args.top_k)

        if result:
            current_batch.append(result)
            success_count += 1
        else:
            fail_count += 1

        # Save batch when full
        if len(current_batch) >= args.batch_size:
            save_batch(current_batch, batch_num, OUTPUT_DIR)
            batch_num += 1
            current_batch = []

    # Save remaining
    if current_batch:
        save_batch(current_batch, batch_num, OUTPUT_DIR)

    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n{'=' * 70}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Time: {elapsed}")
    print(f"Rate: {success_count / elapsed.total_seconds():.1f} artworks/sec")
    print(f"Output: {OUTPUT_DIR}")

    # Estimate storage
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("batch_*.json"))
    print(f"Storage used: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
