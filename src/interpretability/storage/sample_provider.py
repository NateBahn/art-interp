"""
Sample Data Provider

Provides built-in sample data for testing and demonstration.
Uses public domain artworks from Wikimedia Commons.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np

from interpretability.core.types import ImageItem

logger = logging.getLogger(__name__)

# Sample images: public domain artworks from Wikimedia Commons
SAMPLE_IMAGES = [
    {
        "id": "sample_001",
        "title": "Starry Night",
        "artist": "Vincent van Gogh",
        "year": 1889,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "ratings": {
            "mirror_self": 8.5,
            "wholeness": 9.0,
            "inner_light": 8.0,
            "deepest_honest": 7.5,
            "drawn_to": 9.0,
            "choose_to_look": 8.5,
            "technical_skill": 8.5,
            "emotional_impact": 9.0,
        },
    },
    {
        "id": "sample_002",
        "title": "The Great Wave off Kanagawa",
        "artist": "Katsushika Hokusai",
        "year": 1831,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/1280px-Tsunami_by_hokusai_19th_century.jpg",
        "ratings": {
            "mirror_self": 7.0,
            "wholeness": 9.5,
            "inner_light": 7.5,
            "deepest_honest": 6.5,
            "drawn_to": 9.0,
            "choose_to_look": 8.0,
            "technical_skill": 9.0,
            "emotional_impact": 8.5,
        },
    },
    {
        "id": "sample_003",
        "title": "Girl with a Pearl Earring",
        "artist": "Johannes Vermeer",
        "year": 1665,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/1665_Girl_with_a_Pearl_Earring.jpg/800px-1665_Girl_with_a_Pearl_Earring.jpg",
        "ratings": {
            "mirror_self": 8.0,
            "wholeness": 8.5,
            "inner_light": 9.0,
            "deepest_honest": 7.0,
            "drawn_to": 8.5,
            "choose_to_look": 9.0,
            "technical_skill": 9.5,
            "emotional_impact": 8.0,
        },
    },
    {
        "id": "sample_004",
        "title": "The Birth of Venus",
        "artist": "Sandro Botticelli",
        "year": 1485,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg/1280px-Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg",
        "ratings": {
            "mirror_self": 6.5,
            "wholeness": 9.0,
            "inner_light": 8.5,
            "deepest_honest": 5.5,
            "drawn_to": 8.0,
            "choose_to_look": 7.5,
            "technical_skill": 9.0,
            "emotional_impact": 7.5,
        },
    },
    {
        "id": "sample_005",
        "title": "The Scream",
        "artist": "Edvard Munch",
        "year": 1893,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg/800px-Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg",
        "ratings": {
            "mirror_self": 9.0,
            "wholeness": 7.5,
            "inner_light": 6.0,
            "deepest_honest": 9.5,
            "drawn_to": 8.0,
            "choose_to_look": 7.0,
            "technical_skill": 8.0,
            "emotional_impact": 9.5,
        },
    },
    {
        "id": "sample_006",
        "title": "Water Lilies",
        "artist": "Claude Monet",
        "year": 1906,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg/1280px-Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg",
        "ratings": {
            "mirror_self": 7.5,
            "wholeness": 9.5,
            "inner_light": 9.0,
            "deepest_honest": 6.0,
            "drawn_to": 8.5,
            "choose_to_look": 9.0,
            "technical_skill": 9.0,
            "emotional_impact": 8.0,
        },
    },
    {
        "id": "sample_007",
        "title": "American Gothic",
        "artist": "Grant Wood",
        "year": 1930,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Grant_Wood_-_American_Gothic_-_Google_Art_Project.jpg/800px-Grant_Wood_-_American_Gothic_-_Google_Art_Project.jpg",
        "ratings": {
            "mirror_self": 6.0,
            "wholeness": 8.5,
            "inner_light": 5.5,
            "deepest_honest": 7.0,
            "drawn_to": 7.0,
            "choose_to_look": 6.5,
            "technical_skill": 8.5,
            "emotional_impact": 6.5,
        },
    },
    {
        "id": "sample_008",
        "title": "The Persistence of Memory",
        "artist": "Salvador Dalí",
        "year": 1931,
        "image_url": "https://upload.wikimedia.org/wikipedia/en/d/dd/The_Persistence_of_Memory.jpg",
        "ratings": {
            "mirror_self": 8.0,
            "wholeness": 7.0,
            "inner_light": 6.5,
            "deepest_honest": 8.5,
            "drawn_to": 8.5,
            "choose_to_look": 8.0,
            "technical_skill": 9.0,
            "emotional_impact": 8.0,
        },
    },
    {
        "id": "sample_009",
        "title": "Nighthawks",
        "artist": "Edward Hopper",
        "year": 1942,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Nighthawks_by_Edward_Hopper_1942.jpg/1280px-Nighthawks_by_Edward_Hopper_1942.jpg",
        "ratings": {
            "mirror_self": 8.5,
            "wholeness": 8.0,
            "inner_light": 6.0,
            "deepest_honest": 9.0,
            "drawn_to": 8.0,
            "choose_to_look": 8.5,
            "technical_skill": 9.0,
            "emotional_impact": 8.5,
        },
    },
    {
        "id": "sample_010",
        "title": "The Kiss",
        "artist": "Gustav Klimt",
        "year": 1908,
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/The_Kiss_-_Gustav_Klimt_-_Google_Cultural_Institute.jpg/800px-The_Kiss_-_Gustav_Klimt_-_Google_Cultural_Institute.jpg",
        "ratings": {
            "mirror_self": 7.5,
            "wholeness": 9.0,
            "inner_light": 9.5,
            "deepest_honest": 7.0,
            "drawn_to": 9.5,
            "choose_to_look": 9.0,
            "technical_skill": 9.5,
            "emotional_impact": 9.0,
        },
    },
]

# Rating dimensions available in sample data
SAMPLE_RATING_DIMENSIONS = [
    "mirror_self",
    "wholeness",
    "inner_light",
    "deepest_honest",
    "drawn_to",
    "choose_to_look",
    "technical_skill",
    "emotional_impact",
]


class SampleImageProvider:
    """
    Provider for built-in sample data.

    Includes 10 famous public domain artworks with aesthetic ratings.
    Useful for testing, demos, and getting started without your own data.

    Example:
        >>> from interpretability.storage import SampleImageProvider
        >>> provider = SampleImageProvider()
        >>> for item in provider.iter_images(limit=3):
        ...     print(f"{item.title} by {item.metadata['artist']}")
        Starry Night by Vincent van Gogh
        The Great Wave off Kanagawa by Katsushika Hokusai
        Girl with a Pearl Earring by Johannes Vermeer
    """

    def __init__(self) -> None:
        """Initialize with built-in sample data."""
        self.images: dict[str, ImageItem] = {}

        for data in SAMPLE_IMAGES:
            item = ImageItem(
                id=data["id"],
                image_url=data["image_url"],
                title=data["title"],
                ratings=data["ratings"],
                metadata={
                    "artist": data["artist"],
                    "year": data["year"],
                },
            )
            self.images[item.id] = item

        logger.info(f"Initialized SampleImageProvider with {len(self.images)} images")

    def get_image(self, item_id: str) -> ImageItem | None:
        """Get image by ID."""
        return self.images.get(item_id)

    def iter_images(
        self,
        filter_labeled: bool = True,
        limit: int | None = None,
    ) -> Iterator[ImageItem]:
        """Iterate over sample images."""
        count = 0
        for item in self.images.values():
            if filter_labeled and not item.has_ratings():
                continue
            yield item
            count += 1
            if limit is not None and count >= limit:
                break

    def get_count(self, filter_labeled: bool = True) -> int:
        """Count sample images."""
        if not filter_labeled:
            return len(self.images)
        return sum(1 for item in self.images.values() if item.has_ratings())

    @staticmethod
    def get_rating_dimensions() -> list[str]:
        """Get the list of rating dimensions in sample data."""
        return SAMPLE_RATING_DIMENSIONS.copy()


class SampleEmbeddingProvider:
    """
    Mock embedding provider for sample data.

    Generates random embeddings for demonstration purposes.
    For real analysis, use actual embeddings (e.g., from DINOv2).

    Note: Random embeddings will NOT produce meaningful monosemanticity
    scores. This is only for testing the pipeline.
    """

    def __init__(self, dim: int = 1024, seed: int = 42) -> None:
        """
        Initialize with random embeddings.

        Args:
            dim: Embedding dimension (1024 for DINOv2 ViT-L)
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.embeddings: dict[str, np.ndarray] = {}

        # Generate embeddings for sample images
        for data in SAMPLE_IMAGES:
            # Random unit vector
            emb = self.rng.standard_normal(dim).astype(np.float32)
            emb /= np.linalg.norm(emb)
            self.embeddings[data["id"]] = emb

        logger.info(
            f"Initialized SampleEmbeddingProvider with {len(self.embeddings)} "
            f"random {dim}-dim embeddings"
        )

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
