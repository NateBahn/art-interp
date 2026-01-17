"""
SAE Model Configurations

Defines configuration for Prisma-Multimodal Sparse Autoencoders trained on CLIP ViT-B-32.
Models are hosted on HuggingFace and downloaded automatically on first use.

Two types of SAEs are available:
- CLS SAEs: Operate on the CLS token (global image representation)
- Spatial SAEs: Operate on 49 spatial patches (7x7 grid, localized features)

Reference: https://huggingface.co/Prisma-Multimodal
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SAEConfig:
    """Configuration for a single SAE model."""

    repo_id: str  # HuggingFace repository ID
    weights_file: str  # Filename of weights in repo
    dim: int  # SAE hidden dimension (typically 49152)
    num_patches: int | None = None  # None for CLS, 49 for spatial (7x7)

    @property
    def is_spatial(self) -> bool:
        """Whether this is a spatial SAE (vs CLS token)."""
        return self.num_patches is not None


# CLS Token SAEs - global image features
# These extract features from the [CLS] token at each layer
SAE_CONFIGS: dict[int, SAEConfig] = {
    7: SAEConfig(
        repo_id="Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-7-hook_resid_post-l1-0.0001",
        weights_file="n_images_2600058.pt",
        dim=49152,
    ),
    8: SAEConfig(
        repo_id="Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-8-hook_resid_post-l1-1e-05",
        weights_file="n_images_2600058.pt",
        dim=49152,
    ),
    11: SAEConfig(
        repo_id="Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-8e-05",
        weights_file="n_images_2600058.pt",
        dim=49152,
    ),
}

# Spatial SAEs - localized features across 7x7 grid
# These extract features from each of the 49 spatial patches (excluding CLS)
SPATIAL_SAE_CONFIGS: dict[int, SAEConfig] = {
    7: SAEConfig(
        repo_id="Prisma-Multimodal/imagenet-sweep-vanilla-x64-Spatial_max_7-hook_resid_post-984.1376953125-99",
        weights_file="weights.pt",
        dim=49152,
        num_patches=49,
    ),
    8: SAEConfig(
        repo_id="Prisma-Multimodal/imagenet-sweep-vanilla-x64-Spatial_max_8-hook_resid_post-965.125-99",
        weights_file="weights.pt",
        dim=49152,
        num_patches=49,
    ),
    9: SAEConfig(
        repo_id="Prisma-Multimodal/imagenet-sweep-vanilla-x64-Spatial_max_9-hook_resid_post-854.891540527344-99",
        weights_file="weights.pt",
        dim=49152,
        num_patches=49,
    ),
    11: SAEConfig(
        repo_id="Prisma-Multimodal/imagenet-sweep-vanilla-x64-Spatial_max_11-hook_resid_post-829.0498046875-99",
        weights_file="weights.pt",
        dim=49152,
        num_patches=49,
    ),
}

# Common constants
SAE_DIM = 49152  # All SAEs produce this many features
CLIP_HIDDEN_DIM = 768  # CLIP ViT-B-32 internal dimension
SAE_CONFIG_FILENAME = "config.json"  # Config file in HF repo

# Available layers for each SAE type
CLS_LAYERS = list(SAE_CONFIGS.keys())  # [7, 8, 11]
SPATIAL_LAYERS = list(SPATIAL_SAE_CONFIGS.keys())  # [7, 8, 9, 11]
