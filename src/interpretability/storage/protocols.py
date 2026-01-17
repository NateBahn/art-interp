"""
Storage Protocols (Abstract Interfaces)

Defines abstract interfaces for data access, allowing the interpretability
package to work with different data sources (databases, files, APIs, etc.)
without coupling to any specific implementation.

Users implement these protocols to connect their data to the analysis pipeline.
Built-in implementations are provided for common cases (JSON files, sample data).

Example:
    >>> from interpretability.storage import ImageProvider, ImageItem
    >>>
    >>> # Implement for your data source
    >>> class MyDatabaseProvider(ImageProvider):
    ...     def get_image(self, item_id: str) -> ImageItem | None:
    ...         row = self.db.query(id=item_id)
    ...         return ImageItem(
    ...             id=row.id,
    ...             image_url=row.url,
    ...             ratings=row.ratings,
    ...         )
    ...
    >>> # Use with analysis tools
    >>> provider = MyDatabaseProvider(db_connection)
    >>> analyzer = CorrelationAnalyzer(provider)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from interpretability.core.types import ImageItem

if TYPE_CHECKING:
    pass

# Re-export ImageItem for convenience
__all__ = [
    "ImageProvider",
    "FeatureStore",
    "EmbeddingProvider",
    "ImageItem",
]


@runtime_checkable
class ImageProvider(Protocol):
    """
    Protocol for providing images and metadata to the analysis pipeline.

    Implementations connect different data sources (databases, file systems,
    APIs) to the interpretability tools. The package includes sample and
    file-based implementations; users implement this for their own data.

    Required methods:
        - get_image: Fetch a single image by ID
        - iter_images: Iterate over images (optionally filtered)
        - get_count: Count available images

    Example implementation for a SQLAlchemy database:

        class DatabaseImageProvider:
            def __init__(self, session_factory):
                self.session_factory = session_factory

            def get_image(self, item_id: str) -> ImageItem | None:
                with self.session_factory() as session:
                    artwork = session.query(Artwork).get(item_id)
                    if not artwork:
                        return None
                    return ImageItem(
                        id=artwork.id,
                        image_url=artwork.image_url,
                        title=artwork.title,
                        ratings=json.loads(artwork.labels) if artwork.labels else None,
                    )

            def iter_images(self, filter_labeled: bool = True, limit: int | None = None):
                with self.session_factory() as session:
                    query = session.query(Artwork)
                    if filter_labeled:
                        query = query.filter(Artwork.labels.isnot(None))
                    if limit:
                        query = query.limit(limit)
                    for artwork in query:
                        yield self.get_image(artwork.id)

            def get_count(self, filter_labeled: bool = True) -> int:
                with self.session_factory() as session:
                    query = session.query(Artwork)
                    if filter_labeled:
                        query = query.filter(Artwork.labels.isnot(None))
                    return query.count()
    """

    def get_image(self, item_id: str) -> ImageItem | None:
        """
        Get a single image by its unique identifier.

        Args:
            item_id: Unique identifier for the image

        Returns:
            ImageItem with metadata, or None if not found
        """
        ...

    def iter_images(
        self,
        filter_labeled: bool = True,
        limit: int | None = None,
    ) -> Iterator[ImageItem]:
        """
        Iterate over available images.

        Args:
            filter_labeled: If True, only yield images with rating data
            limit: Maximum number of images to yield (None for all)

        Yields:
            ImageItem instances
        """
        ...

    def get_count(self, filter_labeled: bool = True) -> int:
        """
        Count available images.

        Args:
            filter_labeled: If True, only count images with rating data

        Returns:
            Number of available images
        """
        ...


@runtime_checkable
class FeatureStore(Protocol):
    """
    Protocol for storing and loading SAE features.

    Implementations handle persistence of extracted features, allowing
    caching of expensive computations. Features can be stored in various
    formats (JSON, NPZ, databases).

    Required methods:
        - save_features: Store features for an item
        - load_features: Retrieve features for an item
        - iter_features: Iterate over all stored features
        - has_features: Check if features exist for an item

    Example implementation for JSON files:

        class JSONFeatureStore:
            def __init__(self, base_dir: Path):
                self.base_dir = base_dir

            def save_features(self, item_id: str, features: np.ndarray, layer: int):
                path = self.base_dir / f"layer_{layer}" / f"{item_id}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                sparse = {int(i): float(features[i]) for i in np.nonzero(features)[0]}
                with open(path, "w") as f:
                    json.dump(sparse, f)

            def load_features(self, item_id: str, layer: int) -> np.ndarray | None:
                path = self.base_dir / f"layer_{layer}" / f"{item_id}.json"
                if not path.exists():
                    return None
                with open(path) as f:
                    sparse = json.load(f)
                dense = np.zeros(49152, dtype=np.float32)
                for idx, val in sparse.items():
                    dense[int(idx)] = val
                return dense
    """

    def save_features(
        self,
        item_id: str,
        features: np.ndarray,
        layer: int,
        feature_type: str = "cls",
    ) -> None:
        """
        Save features for an item.

        Args:
            item_id: Unique identifier for the item
            features: Feature array (typically shape SAE_DIM)
            layer: CLIP layer these features are from
            feature_type: "cls" or "spatial"
        """
        ...

    def load_features(
        self,
        item_id: str,
        layer: int,
        feature_type: str = "cls",
    ) -> np.ndarray | None:
        """
        Load features for an item.

        Args:
            item_id: Unique identifier for the item
            layer: CLIP layer to load features from
            feature_type: "cls" or "spatial"

        Returns:
            Feature array, or None if not found
        """
        ...

    def iter_features(
        self,
        layer: int,
        feature_type: str = "cls",
    ) -> Iterator[tuple[str, np.ndarray]]:
        """
        Iterate over all stored features for a layer.

        Args:
            layer: CLIP layer to iterate
            feature_type: "cls" or "spatial"

        Yields:
            Tuples of (item_id, features)
        """
        ...

    def has_features(
        self,
        item_id: str,
        layer: int,
        feature_type: str = "cls",
    ) -> bool:
        """
        Check if features exist for an item.

        Args:
            item_id: Unique identifier for the item
            layer: CLIP layer to check
            feature_type: "cls" or "spatial"

        Returns:
            True if features are stored, False otherwise
        """
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol for providing embeddings for monosemanticity scoring.

    Monosemanticity scoring requires embeddings from a DIFFERENT model than
    the one used by the SAE (to avoid circular reasoning). DINOv2 embeddings
    are recommended since the SAEs are trained on CLIP.

    Required methods:
        - get_embedding: Get embedding for a single item
        - get_embeddings_batch: Get embeddings for multiple items

    Example implementation:

        class DINOv2EmbeddingProvider:
            def __init__(self, embeddings_file: Path):
                # Pre-computed embeddings stored as {id: embedding_list}
                with open(embeddings_file) as f:
                    self.embeddings = json.load(f)

            def get_embedding(self, item_id: str) -> np.ndarray | None:
                if item_id not in self.embeddings:
                    return None
                return np.array(self.embeddings[item_id], dtype=np.float32)

            def get_embeddings_batch(self, item_ids: list[str]) -> dict[str, np.ndarray]:
                return {
                    item_id: self.get_embedding(item_id)
                    for item_id in item_ids
                    if item_id in self.embeddings
                }
    """

    def get_embedding(self, item_id: str) -> np.ndarray | None:
        """
        Get embedding for a single item.

        Args:
            item_id: Unique identifier for the item

        Returns:
            Embedding array (e.g., 1024-dim for DINOv2), or None if not found
        """
        ...

    def get_embeddings_batch(self, item_ids: list[str]) -> dict[str, np.ndarray]:
        """
        Get embeddings for multiple items.

        More efficient than calling get_embedding repeatedly if the
        implementation can batch database/file access.

        Args:
            item_ids: List of item identifiers

        Returns:
            Dict mapping item_id -> embedding (only includes found items)
        """
        ...
