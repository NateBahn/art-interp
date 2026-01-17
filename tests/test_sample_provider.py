"""Tests for sample data providers."""


from interpretability.storage.sample_provider import (
    SAMPLE_RATING_DIMENSIONS,
    SampleImageProvider,
)


class TestSampleImageProvider:
    """Tests for SampleImageProvider."""

    def test_get_image(self, sample_provider):
        """Test getting a specific image."""
        item = sample_provider.get_image("sample_001")

        assert item is not None
        assert item.id == "sample_001"
        assert item.title == "Starry Night"
        assert item.image_url is not None
        assert item.has_ratings()

    def test_get_image_not_found(self, sample_provider):
        """Test getting non-existent image."""
        item = sample_provider.get_image("nonexistent")
        assert item is None

    def test_iter_images(self, sample_provider):
        """Test iterating over images."""
        images = list(sample_provider.iter_images())

        assert len(images) == 10
        assert all(img.has_ratings() for img in images)

    def test_iter_images_with_limit(self, sample_provider):
        """Test iteration with limit."""
        images = list(sample_provider.iter_images(limit=3))
        assert len(images) == 3

    def test_get_count(self, sample_provider):
        """Test counting images."""
        assert sample_provider.get_count() == 10

    def test_rating_dimensions(self):
        """Test that sample data has all rating dimensions."""
        provider = SampleImageProvider()
        item = provider.get_image("sample_001")

        for dim in SAMPLE_RATING_DIMENSIONS:
            assert dim in item.ratings


class TestSampleEmbeddingProvider:
    """Tests for SampleEmbeddingProvider."""

    def test_get_embedding(self, sample_embedding_provider):
        """Test getting an embedding."""
        emb = sample_embedding_provider.get_embedding("sample_001")

        assert emb is not None
        assert emb.shape == (1024,)  # DINOv2 dimension
        # Should be unit normalized
        import numpy as np
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    def test_get_embedding_not_found(self, sample_embedding_provider):
        """Test getting non-existent embedding."""
        emb = sample_embedding_provider.get_embedding("nonexistent")
        assert emb is None

    def test_get_embeddings_batch(self, sample_embedding_provider):
        """Test batch embedding retrieval."""
        ids = ["sample_001", "sample_002", "nonexistent"]
        embeddings = sample_embedding_provider.get_embeddings_batch(ids)

        assert len(embeddings) == 2  # Only existing ones
        assert "sample_001" in embeddings
        assert "sample_002" in embeddings
        assert "nonexistent" not in embeddings
