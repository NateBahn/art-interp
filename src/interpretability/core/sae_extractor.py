"""
SAE Feature Extractor

Extracts sparse autoencoder features from images using Prisma-Multimodal SAEs
trained on CLIP ViT-B-32. Supports both CLS token (global) and spatial (7x7 patch)
feature extraction.

The SAE transforms CLIP's 768-dimensional internal activations into a sparse
49,152-dimensional feature space where each feature ideally represents a
single interpretable concept.

Example:
    >>> from interpretability.core import SAEFeatureExtractor
    >>> from PIL import Image
    >>>
    >>> # Load extractor (downloads SAE weights on first use)
    >>> extractor = SAEFeatureExtractor(layer=8)
    >>>
    >>> # Extract features from an image
    >>> image = Image.open("artwork.jpg")
    >>> result = extractor.extract_from_image(image)
    >>>
    >>> # Examine top features
    >>> for idx, val in zip(result.top_indices[:5], result.top_values[:5]):
    ...     print(f"Feature {idx}: {val:.2f}")

Reference:
    - Prisma-Multimodal SAEs: https://huggingface.co/Prisma-Multimodal
    - SAE for VLMs paper: https://arxiv.org/abs/2504.02821
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import httpx
import numpy as np
import open_clip
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from interpretability.core.configs import (
    SAE_CONFIG_FILENAME,
    SAE_CONFIGS,
    SAE_DIM,
    SPATIAL_SAE_CONFIGS,
    SAEConfig,
)
from interpretability.core.types import SAEFeatureResult, SpatialFeatureResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# HTTP retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF = [1.0, 2.0, 5.0]

# Thread pool for async image downloads
_executor = ThreadPoolExecutor(max_workers=4)


class SAEFeatureExtractor:
    """
    Extract sparse features from images using Prisma SAEs on CLIP ViT-B-32.

    The SAE operates on internal 768-dim activations at a configurable layer,
    producing 49,152 sparse feature activations. Each feature ideally represents
    a single interpretable visual concept.

    Supported layers:
        - Layer 7: Early features (textures, patterns, edges)
        - Layer 8: Mid-level features (shapes, composition)
        - Layer 9: (spatial only) Transitional features
        - Layer 11: Late features (semantic/conceptual)

    Two extraction modes:
        - CLS (default): Global image features from the [CLS] token
        - Spatial: Per-patch features from the 7x7 spatial grid

    Attributes:
        layer: CLIP transformer layer being analyzed
        feature_type: "cls" or "spatial"
        device: Computation device (cuda, mps, or cpu)
        config: SAE configuration (repo, dimensions, etc.)

    Example:
        >>> # CLS token extraction (global features)
        >>> extractor = SAEFeatureExtractor(layer=8, feature_type="cls")
        >>> result = extractor.extract_from_image(image)
        >>> print(f"Top feature: {result.top_indices[0]}")
        >>>
        >>> # Spatial extraction (localized features)
        >>> spatial_extractor = SAEFeatureExtractor(layer=8, feature_type="spatial")
        >>> spatial_result = spatial_extractor.extract_spatial(image)
        >>> heatmap = spatial_result.get_feature_heatmap(1234)  # 7x7 array
    """

    def __init__(
        self,
        layer: int = 8,
        feature_type: Literal["cls", "spatial"] = "cls",
        device: str | None = None,
    ) -> None:
        """
        Initialize the feature extractor.

        Downloads CLIP model and SAE weights on first use.
        Subsequent calls use cached weights from HuggingFace Hub.

        Args:
            layer: CLIP transformer layer to extract from.
                   CLS supports: 7, 8, 11
                   Spatial supports: 7, 8, 9, 11
            feature_type: "cls" for global features, "spatial" for 7x7 patch features
            device: Computation device. Auto-detected if None (CUDA > MPS > CPU)

        Raises:
            ValueError: If layer is not supported for the given feature_type
        """
        self.feature_type = feature_type
        self.layer = layer

        # Select appropriate config
        configs = SPATIAL_SAE_CONFIGS if feature_type == "spatial" else SAE_CONFIGS
        if layer not in configs:
            available = list(configs.keys())
            raise ValueError(
                f"Layer {layer} not supported for {feature_type}. Available: {available}"
            )
        self.config: SAEConfig = configs[layer]

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(
            f"Initializing SAEFeatureExtractor (layer={layer}, type={feature_type}) "
            f"on device: {self.device}"
        )

        # Load models
        self._load_clip_model()
        self._load_sae_weights()

        # Register activation hook
        self._activations: dict[str, torch.Tensor] = {}
        self._hook = self._register_hook()

        logger.info(f"SAEFeatureExtractor ready. SAE dim: {self.config.dim}")

    def _load_clip_model(self) -> None:
        """Load CLIP ViT-B-32 model with DataComp weights."""
        logger.info("Loading CLIP ViT-B-32 (datacomp_xl_s13b_b90k)...")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="datacomp_xl_s13b_b90k"
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        logger.info("CLIP model loaded")

    def _load_sae_weights(self) -> None:
        """Download and load SAE weights from HuggingFace Hub."""
        repo_id = self.config.repo_id
        weights_file = self.config.weights_file
        logger.info(f"Downloading SAE weights from {repo_id}...")

        weights_path = hf_hub_download(repo_id, filename=weights_file)

        # Try to load config (optional, not all repos have it)
        try:
            config_path = hf_hub_download(repo_id, filename=SAE_CONFIG_FILENAME)
            with open(config_path) as f:
                self.sae_config = json.load(f)
        except Exception:
            self.sae_config = {}

        # Load weights
        sae_weights = torch.load(weights_path, map_location=self.device, weights_only=True)

        self.W_enc = sae_weights["W_enc"].to(self.device)  # (768, 49152)
        self.b_enc = sae_weights["b_enc"].to(self.device)  # (49152,)
        self.W_dec = sae_weights["W_dec"].to(self.device)  # (49152, 768)
        self.b_dec = sae_weights["b_dec"].to(self.device)  # (768,)

        logger.info(f"SAE weights loaded. Encoder shape: {self.W_enc.shape}")

    def _register_hook(self) -> torch.utils.hooks.RemovableHandle:
        """Register forward hook to capture layer activations."""

        def hook_fn(
            module: torch.nn.Module,
            input: tuple,  # noqa: A002
            output: torch.Tensor,
        ) -> None:
            self._activations[f"layer_{self.layer}"] = output.detach()

        layer_module = self.clip_model.visual.transformer.resblocks[self.layer]
        return layer_module.register_forward_hook(hook_fn)

    def extract_from_image(
        self,
        image: Image.Image,
        top_k: int = 20,
    ) -> SAEFeatureResult | None:
        """
        Extract CLS token features from a PIL Image.

        Args:
            image: PIL Image to process (any size, will be resized)
            top_k: Number of top-activating features to track

        Returns:
            SAEFeatureResult with sparse features and statistics,
            or None if extraction fails
        """
        try:
            # Preprocess image for CLIP
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # Forward pass (hook captures activations)
            with torch.no_grad():
                _ = self.clip_model.encode_image(image_tensor)

            # Get layer activations: (1, num_tokens, 768)
            # num_tokens = 1 (CLS) + 49 (patches) = 50
            layer_act = self._activations[f"layer_{self.layer}"]

            # Extract CLS token (first token)
            cls_activation = layer_act[0, 0, :]  # (768,)

            # Pass through SAE encoder: ReLU(x @ W_enc + b_enc)
            sparse_features = torch.relu(cls_activation @ self.W_enc + self.b_enc)

            # Convert to numpy
            features_np = sparse_features.cpu().numpy().astype(np.float32)

            # Compute statistics
            num_active = int((features_np > 0).sum())
            k = min(top_k, num_active)
            if k > 0:
                top_k_result = torch.topk(sparse_features, k)
                top_indices = top_k_result.indices.cpu().tolist()
                top_values = top_k_result.values.cpu().tolist()
            else:
                top_indices = []
                top_values = []

            return SAEFeatureResult(
                features=features_np,
                num_active=num_active,
                top_indices=top_indices,
                top_values=top_values,
            )

        except Exception as e:
            logger.warning(f"Failed to extract features: {e}")
            return None

    def extract_spatial(
        self,
        image: Image.Image,
        top_k_per_patch: int = 10,
    ) -> SpatialFeatureResult | None:
        """
        Extract spatial (7x7 patch) features from a PIL Image.

        Each of the 49 patches gets its own feature vector, allowing
        localization of features within the image.

        Args:
            image: PIL Image to process
            top_k_per_patch: Top features to track per patch

        Returns:
            SpatialFeatureResult with per-patch and aggregated features,
            or None if extraction fails

        Raises:
            ValueError: If extractor was not initialized with feature_type="spatial"
        """
        if self.feature_type != "spatial":
            raise ValueError(
                "extract_spatial() requires feature_type='spatial'. "
                f"This extractor was initialized with feature_type='{self.feature_type}'"
            )

        try:
            # Preprocess image
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # Forward pass
            with torch.no_grad():
                _ = self.clip_model.encode_image(image_tensor)

            # Get activations: (1, 50, 768) -> patches are tokens 1-49
            layer_act = self._activations[f"layer_{self.layer}"]
            patch_activations = layer_act[0, 1:, :]  # (49, 768)

            # Apply SAE to each patch
            # Batch matrix multiply: (49, 768) @ (768, 49152) + (49152,)
            sparse_patches = torch.relu(patch_activations @ self.W_enc + self.b_enc)  # (49, 49152)

            # Convert to numpy
            patch_features = sparse_patches.cpu().numpy().astype(np.float32)

            # Aggregate via max-pooling
            aggregated = np.max(patch_features, axis=0)

            # Per-patch statistics
            patch_top_indices = []
            patch_top_values = []
            num_active_per_patch = []

            for patch_idx in range(49):
                patch_feats = sparse_patches[patch_idx]
                num_active = int((patch_feats > 0).sum())
                num_active_per_patch.append(num_active)

                k = min(top_k_per_patch, num_active)
                if k > 0:
                    top_k_result = torch.topk(patch_feats, k)
                    patch_top_indices.append(top_k_result.indices.cpu().tolist())
                    patch_top_values.append(top_k_result.values.cpu().tolist())
                else:
                    patch_top_indices.append([])
                    patch_top_values.append([])

            return SpatialFeatureResult(
                patch_features=patch_features,
                aggregated_features=aggregated,
                patch_top_indices=patch_top_indices,
                patch_top_values=patch_top_values,
                num_active_per_patch=num_active_per_patch,
            )

        except Exception as e:
            logger.warning(f"Failed to extract spatial features: {e}")
            return None

    async def extract_from_url(
        self,
        image_url: str,
        top_k: int = 20,
    ) -> SAEFeatureResult | None:
        """
        Download image from URL and extract features.

        Includes retry logic with exponential backoff for robustness.

        Args:
            image_url: URL of the image to download
            top_k: Number of top features to track

        Returns:
            SAEFeatureResult or None if download/extraction fails
        """
        try:
            image_data = await self._download_image_with_retry(image_url)
            if image_data is None:
                return None

            image = Image.open(BytesIO(image_data)).convert("RGB")
            return self.extract_from_image(image, top_k=top_k)

        except Exception as e:
            logger.warning(f"Failed to extract from URL {image_url}: {e}")
            return None

    async def _download_image_with_retry(self, image_url: str) -> bytes | None:
        """Download image with retry logic (runs in thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, lambda: self._sync_download_image(image_url)
        )

    def _sync_download_image(self, image_url: str) -> bytes | None:
        """Synchronous image download with retry logic."""
        headers = {"User-Agent": "Interpretability-Toolkit/1.0 (Research)"}

        for attempt in range(MAX_RETRIES):
            try:
                with httpx.Client(headers=headers, follow_redirects=True) as client:
                    response = client.get(image_url, timeout=30.0)

                    if response.status_code == 429:
                        wait_time = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                        logger.warning(f"Rate limited, retry {attempt + 1} in {wait_time}s")
                        time.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    return response.content

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_BACKOFF[attempt]
                    logger.warning(f"Download failed: {e}, retry {attempt + 1} in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Download failed after {MAX_RETRIES} attempts: {e}")

        return None

    def cleanup(self) -> None:
        """Remove hooks and free resources."""
        if hasattr(self, "_hook") and self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.cleanup()

    def __repr__(self) -> str:
        return (
            f"SAEFeatureExtractor(layer={self.layer}, "
            f"feature_type='{self.feature_type}', device='{self.device}')"
        )


# =============================================================================
# Utility functions for batch processing and storage
# =============================================================================


def save_features_batch(
    features_dict: dict[str, np.ndarray],
    output_path: Path,
    sparse: bool = True,
) -> None:
    """
    Save a batch of features to JSON file.

    Args:
        features_dict: Mapping of item_id -> feature array
        output_path: Path to save JSON file
        sparse: If True, only save non-zero values (recommended)

    Example:
        >>> features = {"img1": result1.features, "img2": result2.features}
        >>> save_features_batch(features, Path("features.json"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sparse:
        # Sparse format: {id: {idx: value, ...}}
        sparse_dict = {}
        for item_id, features in features_dict.items():
            nonzero_idx = np.nonzero(features)[0]
            sparse_dict[item_id] = {
                int(idx): float(features[idx]) for idx in nonzero_idx
            }
        data = sparse_dict
    else:
        # Dense format: {id: [values...]}
        data = {k: v.tolist() for k, v in features_dict.items()}

    with open(output_path, "w") as f:
        json.dump(data, f)


def load_features_batch(
    input_path: Path,
    to_dense: bool = True,
) -> dict[str, np.ndarray | dict]:
    """
    Load a batch of features from JSON file.

    Args:
        input_path: Path to JSON file
        to_dense: If True, convert sparse dict to dense numpy array

    Returns:
        Mapping of item_id -> feature array (or sparse dict if to_dense=False)

    Example:
        >>> features = load_features_batch(Path("features.json"))
        >>> for item_id, feats in features.items():
        ...     print(f"{item_id}: {(feats > 0).sum()} active features")
    """
    with open(input_path) as f:
        data = json.load(f)

    if to_dense:
        result = {}
        for item_id, sparse_features in data.items():
            if isinstance(sparse_features, dict):
                # Sparse format -> dense
                dense = np.zeros(SAE_DIM, dtype=np.float32)
                for idx_str, value in sparse_features.items():
                    dense[int(idx_str)] = value
                result[item_id] = dense
            else:
                # Already dense (list)
                result[item_id] = np.array(sparse_features, dtype=np.float32)
        return result
    else:
        return data
