"""
File-based Storage Implementations

Concrete implementations of storage protocols using JSON files.
Suitable for research workflows and moderate-scale datasets.

For production use with large datasets, consider implementing
database-backed storage using the same protocols.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from interpretability.core.configs import SAE_DIM
from interpretability.core.types import ImageItem

logger = logging.getLogger(__name__)


class JSONFeatureStore:
    """
    File-based feature storage using JSON format.

    Features are stored in a sparse format (only non-zero values) to
    minimize disk usage. Directory structure:
        base_dir/
            layer_7/
                cls/
                    batch_0.json
                    batch_1.json
                spatial/
                    batch_0.json
            layer_8/
                ...

    Each batch file contains multiple items:
        {"item_id": {"feature_idx": value, ...}, ...}

    Attributes:
        base_dir: Root directory for feature storage
        batch_size: Number of items per batch file (for writes)

    Example:
        >>> store = JSONFeatureStore(Path("./features"))
        >>> store.save_features("img001", features, layer=8)
        >>> loaded = store.load_features("img001", layer=8)
    """

    def __init__(
        self,
        base_dir: Path,
        batch_size: int = 100,
    ) -> None:
        """
        Initialize the feature store.

        Args:
            base_dir: Root directory for storage
            batch_size: Items per batch file when saving
        """
        self.base_dir = Path(base_dir)
        self.batch_size = batch_size
        self._cache: dict[tuple[int, str], dict[str, dict[int, float]]] = {}

    def _get_layer_dir(self, layer: int, feature_type: str) -> Path:
        """Get directory for a specific layer and type."""
        return self.base_dir / f"layer_{layer}" / feature_type

    def save_features(
        self,
        item_id: str,
        features: np.ndarray,
        layer: int,
        feature_type: str = "cls",
    ) -> None:
        """
        Save features for a single item.

        Features are converted to sparse format and saved to JSON.

        Args:
            item_id: Unique identifier
            features: Dense feature array (SAE_DIM,)
            layer: CLIP layer number
            feature_type: "cls" or "spatial"
        """
        # Convert to sparse
        nonzero_idx = np.nonzero(features)[0]
        sparse = {int(idx): float(features[idx]) for idx in nonzero_idx}

        # Save to individual file (simple approach)
        layer_dir = self._get_layer_dir(layer, feature_type)
        layer_dir.mkdir(parents=True, exist_ok=True)

        output_path = layer_dir / f"{item_id}.json"
        with open(output_path, "w") as f:
            json.dump(sparse, f)

    def save_features_batch(
        self,
        features_dict: dict[str, np.ndarray],
        layer: int,
        feature_type: str = "cls",
        batch_name: str | None = None,
    ) -> None:
        """
        Save multiple features to a single batch file.

        More efficient for bulk operations than individual saves.

        Args:
            features_dict: Mapping of item_id -> feature array
            layer: CLIP layer number
            feature_type: "cls" or "spatial"
            batch_name: Optional name for batch file (auto-generated if None)
        """
        layer_dir = self._get_layer_dir(layer, feature_type)
        layer_dir.mkdir(parents=True, exist_ok=True)

        # Convert all to sparse
        sparse_dict = {}
        for item_id, features in features_dict.items():
            nonzero_idx = np.nonzero(features)[0]
            sparse_dict[item_id] = {
                int(idx): float(features[idx]) for idx in nonzero_idx
            }

        # Determine batch filename
        if batch_name is None:
            existing = list(layer_dir.glob("batch_*.json"))
            batch_num = len(existing)
            batch_name = f"batch_{batch_num}"

        output_path = layer_dir / f"{batch_name}.json"
        with open(output_path, "w") as f:
            json.dump(sparse_dict, f)

        logger.info(f"Saved {len(features_dict)} features to {output_path}")

    def load_features(
        self,
        item_id: str,
        layer: int,
        feature_type: str = "cls",
    ) -> np.ndarray | None:
        """
        Load features for a single item.

        Searches both individual files and batch files.

        Args:
            item_id: Unique identifier
            layer: CLIP layer number
            feature_type: "cls" or "spatial"

        Returns:
            Dense feature array, or None if not found
        """
        layer_dir = self._get_layer_dir(layer, feature_type)

        # Check individual file first
        individual_path = layer_dir / f"{item_id}.json"
        if individual_path.exists():
            with open(individual_path) as f:
                sparse = json.load(f)
            return self._sparse_to_dense(sparse)

        # Search batch files
        cache_key = (layer, feature_type)
        if cache_key not in self._cache:
            self._load_batches_to_cache(layer, feature_type)

        cached_batches = self._cache.get(cache_key, {})
        if item_id in cached_batches:
            return self._sparse_to_dense(cached_batches[item_id])

        return None

    def _load_batches_to_cache(self, layer: int, feature_type: str) -> None:
        """Load all batch files into cache."""
        layer_dir = self._get_layer_dir(layer, feature_type)
        cache_key = (layer, feature_type)
        self._cache[cache_key] = {}

        if not layer_dir.exists():
            return

        for batch_path in layer_dir.glob("batch_*.json"):
            with open(batch_path) as f:
                batch_data = json.load(f)
            self._cache[cache_key].update(batch_data)

    def _sparse_to_dense(self, sparse: dict[str, float]) -> np.ndarray:
        """Convert sparse dict to dense numpy array."""
        dense = np.zeros(SAE_DIM, dtype=np.float32)
        for idx_str, value in sparse.items():
            dense[int(idx_str)] = value
        return dense

    def iter_features(
        self,
        layer: int,
        feature_type: str = "cls",
    ) -> Iterator[tuple[str, np.ndarray]]:
        """
        Iterate over all stored features for a layer.

        Yields:
            Tuples of (item_id, dense_features)
        """
        layer_dir = self._get_layer_dir(layer, feature_type)

        if not layer_dir.exists():
            return

        # Individual files
        for json_path in layer_dir.glob("*.json"):
            if json_path.name.startswith("batch_"):
                continue
            item_id = json_path.stem
            with open(json_path) as f:
                sparse = json.load(f)
            yield item_id, self._sparse_to_dense(sparse)

        # Batch files
        for batch_path in layer_dir.glob("batch_*.json"):
            with open(batch_path) as f:
                batch_data = json.load(f)
            for item_id, sparse in batch_data.items():
                yield item_id, self._sparse_to_dense(sparse)

    def has_features(
        self,
        item_id: str,
        layer: int,
        feature_type: str = "cls",
    ) -> bool:
        """Check if features exist for an item."""
        layer_dir = self._get_layer_dir(layer, feature_type)

        # Check individual file
        if (layer_dir / f"{item_id}.json").exists():
            return True

        # Check cache/batches
        cache_key = (layer, feature_type)
        if cache_key not in self._cache:
            self._load_batches_to_cache(layer, feature_type)

        return item_id in self._cache.get(cache_key, {})

    def get_stored_count(self, layer: int, feature_type: str = "cls") -> int:
        """Count total stored features for a layer."""
        count = 0
        layer_dir = self._get_layer_dir(layer, feature_type)

        if not layer_dir.exists():
            return 0

        # Count individual files
        count += len(list(layer_dir.glob("*.json"))) - len(
            list(layer_dir.glob("batch_*.json"))
        )

        # Count items in batches
        for batch_path in layer_dir.glob("batch_*.json"):
            with open(batch_path) as f:
                batch_data = json.load(f)
            count += len(batch_data)

        return count


class JSONImageProvider:
    """
    File-based image provider using JSON metadata.

    Reads image metadata from a JSON file with structure:
        [
            {
                "id": "img001",
                "image_url": "https://...",
                "title": "Starry Night",
                "ratings": {"mirror_self": 9.0, ...}
            },
            ...
        ]

    Attributes:
        images: Dict of item_id -> ImageItem

    Example:
        >>> provider = JSONImageProvider(Path("images.json"))
        >>> item = provider.get_image("img001")
        >>> print(item.title)
    """

    def __init__(self, json_path: Path) -> None:
        """
        Load images from JSON file.

        Args:
            json_path: Path to JSON file with image metadata
        """
        self.json_path = Path(json_path)
        self.images: dict[str, ImageItem] = {}
        self._load_images()

    def _load_images(self) -> None:
        """Load and parse the JSON file."""
        with open(self.json_path) as f:
            data = json.load(f)

        for item_data in data:
            item = ImageItem(
                id=item_data["id"],
                image_url=item_data.get("image_url"),
                title=item_data.get("title"),
                ratings=item_data.get("ratings"),
                metadata=item_data.get("metadata", {}),
            )
            self.images[item.id] = item

        logger.info(f"Loaded {len(self.images)} images from {self.json_path}")

    def get_image(self, item_id: str) -> ImageItem | None:
        """Get image by ID."""
        return self.images.get(item_id)

    def iter_images(
        self,
        filter_labeled: bool = True,
        limit: int | None = None,
    ) -> Iterator[ImageItem]:
        """Iterate over images."""
        count = 0
        for item in self.images.values():
            if filter_labeled and not item.has_ratings():
                continue
            yield item
            count += 1
            if limit is not None and count >= limit:
                break

    def get_count(self, filter_labeled: bool = True) -> int:
        """Count available images."""
        if not filter_labeled:
            return len(self.images)
        return sum(1 for item in self.images.values() if item.has_ratings())


class JSONEmbeddingProvider:
    """
    File-based embedding provider using JSON storage.

    Embeddings are stored as:
        {"item_id": [embedding_values...], ...}

    Attributes:
        embeddings: Dict of item_id -> embedding array

    Example:
        >>> provider = JSONEmbeddingProvider(Path("dinov2_embeddings.json"))
        >>> embedding = provider.get_embedding("img001")
    """

    def __init__(self, json_path: Path) -> None:
        """
        Load embeddings from JSON file.

        Args:
            json_path: Path to JSON file with embeddings
        """
        self.json_path = Path(json_path)
        self.embeddings: dict[str, np.ndarray] = {}
        self._load_embeddings()

    def _load_embeddings(self) -> None:
        """Load and convert embeddings."""
        with open(self.json_path) as f:
            data = json.load(f)

        for item_id, embedding_list in data.items():
            self.embeddings[item_id] = np.array(embedding_list, dtype=np.float32)

        logger.info(f"Loaded {len(self.embeddings)} embeddings from {self.json_path}")

    def get_embedding(self, item_id: str) -> np.ndarray | None:
        """Get embedding for an item."""
        return self.embeddings.get(item_id)

    def get_embeddings_batch(self, item_ids: list[str]) -> dict[str, np.ndarray]:
        """Get embeddings for multiple items."""
        return {
            item_id: self.embeddings[item_id]
            for item_id in item_ids
            if item_id in self.embeddings
        }
