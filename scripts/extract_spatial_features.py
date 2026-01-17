#!/usr/bin/env python3
"""
Extract Spatial SAE features for all artworks.

Uses the Prisma Spatial SAE (trained on patches, not CLS) to extract features
that can be used for both correlation analysis AND heatmaps consistently.

For each artwork, stores:
- max_pooled: Maximum activation per feature across 49 patches (for correlations)
- The script saves in batches to handle large datasets
"""

import json
import time
from pathlib import Path
from io import BytesIO
from dataclasses import dataclass

import numpy as np
import torch
import httpx
from PIL import Image
from huggingface_hub import hf_hub_download
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import get_db_context
from database.models import Artwork

# Paths
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "spatial_sae_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Spatial SAE configuration
SPATIAL_SAE_REPO = "Prisma-Multimodal/imagenet-sweep-vanilla-x64-Spatial_max_8-hook_resid_post-965.125-99"
SAE_DIM = 49152

# Batch settings
BATCH_SIZE = 100
MAX_RETRIES = 3


class SpatialSAEExtractor:
    """Extract spatial SAE features from images."""

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
        print(f"Loading Spatial SAE from {SPATIAL_SAE_REPO}...")
        weights_path = hf_hub_download(SPATIAL_SAE_REPO, "weights.pt")
        weights = torch.load(weights_path, map_location=self.device, weights_only=False)
        state_dict = weights["state_dict"]

        self.W_enc = state_dict["W_enc"].to(self.device)
        self.b_enc = state_dict["b_enc"].to(self.device)
        print(f"SAE loaded: 768 -> {self.W_enc.shape[1]} features")

    def extract_features(self, image: Image.Image) -> dict | None:
        """
        Extract spatial SAE features from an image.

        Returns dict with:
        - max_pooled: (49152,) max across patches
        - top_indices: indices of top 50 features by max activation
        - top_values: values of top 50 features
        """
        try:
            img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                _ = self.model.encode_image(img_tensor)

            # Get spatial patches (exclude CLS token)
            spatial_acts = self.layer8_activations[0, 1:, :]  # (49, 768)

            # Pass through SAE encoder
            spatial_features = torch.relu(spatial_acts @ self.W_enc + self.b_enc)  # (49, 49152)

            # Max-pool across patches for correlation analysis
            max_pooled = spatial_features.max(dim=0).values  # (49152,)

            # Get top features
            top_k = torch.topk(max_pooled, min(50, (max_pooled > 0).sum().item()))

            return {
                "max_pooled": max_pooled.cpu().numpy().astype(np.float32),
                "top_indices": top_k.indices.cpu().tolist(),
                "top_values": top_k.values.cpu().tolist(),
                "num_active": int((max_pooled > 0).sum().item()),
            }
        except Exception as e:
            print(f"  Error extracting features: {e}")
            return None


def fetch_image(url: str) -> Image.Image | None:
    """Fetch image from URL with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(follow_redirects=True, timeout=30.0) as client:
                response = client.get(url, headers={
                    "User-Agent": "ArtRecommender/1.0 (Research)"
                })
                if response.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(1)
    return None


def save_batch(features_dict: dict, batch_num: int):
    """Save a batch of features to JSON (sparse format)."""
    output_path = OUTPUT_DIR / f"spatial_batch_{batch_num:04d}.json"

    # Convert to sparse format for storage efficiency
    sparse_dict = {}
    for artwork_id, data in features_dict.items():
        max_pooled = data["max_pooled"]
        nonzero_idx = np.nonzero(max_pooled)[0]
        sparse_dict[artwork_id] = {
            "sparse_features": {int(idx): float(max_pooled[idx]) for idx in nonzero_idx},
            "top_indices": data["top_indices"],
            "top_values": data["top_values"],
            "num_active": data["num_active"],
        }

    with open(output_path, "w") as f:
        json.dump(sparse_dict, f)

    print(f"  Saved batch {batch_num} to {output_path}")


def main():
    print("=" * 60)
    print("Extracting Spatial SAE Features for All Artworks")
    print("=" * 60)

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\nUsing device: {device}")

    # Load extractor
    extractor = SpatialSAEExtractor(device=device)

    # Get artworks with images and labels
    print("\nLoading artworks from database...")
    with get_db_context() as db:
        artworks = db.query(Artwork).filter(
            Artwork.image_url.isnot(None),
            Artwork.labels.isnot(None)
        ).all()
        artwork_list = [(a.id, a.image_url, a.title) for a in artworks]

    print(f"Found {len(artwork_list)} artworks with images and labels")

    # Check for already processed
    existing_batches = list(OUTPUT_DIR.glob("spatial_batch_*.json"))
    already_processed = set()
    for batch_file in existing_batches:
        with open(batch_file) as f:
            batch_data = json.load(f)
            already_processed.update(batch_data.keys())

    remaining = [(aid, url, title) for aid, url, title in artwork_list if aid not in already_processed]
    print(f"Already processed: {len(already_processed)}, Remaining: {len(remaining)}")

    if not remaining:
        print("All artworks already processed!")
        return

    # Process in batches
    batch_num = len(existing_batches)
    current_batch = {}
    failed = []

    for artwork_id, image_url, title in tqdm(remaining, desc="Extracting features"):
        # Fetch image
        image = fetch_image(image_url)
        if image is None:
            failed.append(artwork_id)
            continue

        # Extract features
        features = extractor.extract_features(image)
        if features is None:
            failed.append(artwork_id)
            continue

        current_batch[artwork_id] = features

        # Save batch when full
        if len(current_batch) >= BATCH_SIZE:
            save_batch(current_batch, batch_num)
            batch_num += 1
            current_batch = {}

    # Save remaining
    if current_batch:
        save_batch(current_batch, batch_num)

    print(f"\n{'=' * 60}")
    print(f"Extraction complete!")
    print(f"  Processed: {len(remaining) - len(failed)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Output directory: {OUTPUT_DIR}")

    if failed:
        with open(OUTPUT_DIR / "failed_extractions.json", "w") as f:
            json.dump(failed, f)
        print(f"  Failed IDs saved to failed_extractions.json")


if __name__ == "__main__":
    main()
