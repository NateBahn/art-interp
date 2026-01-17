"""Tests for core data types."""

import numpy as np
import pytest

from interpretability.core.types import ImageItem, SAEFeatureResult


class TestSAEFeatureResult:
    """Tests for SAEFeatureResult."""

    def test_to_sparse_dict(self, sample_feature_result):
        """Test conversion to sparse dictionary."""
        sparse = sample_feature_result.to_sparse_dict()

        assert isinstance(sparse, dict)
        assert len(sparse) == 5  # 5 non-zero features
        assert sparse[100] == pytest.approx(2.5)
        assert sparse[500] == pytest.approx(1.8)

    def test_from_sparse_dict(self):
        """Test reconstruction from sparse dictionary."""
        sparse = {100: 2.5, 500: 1.8, 1000: 1.2}
        result = SAEFeatureResult.from_sparse_dict(sparse)

        assert result.num_active == 3
        assert result.features[100] == 2.5
        assert result.features[500] == 1.8
        assert result.features[0] == 0  # Should be zero
        assert len(result.top_indices) <= 20

    def test_roundtrip(self, sample_feature_result):
        """Test sparse dict roundtrip preserves data."""
        sparse = sample_feature_result.to_sparse_dict()
        reconstructed = SAEFeatureResult.from_sparse_dict(sparse)

        np.testing.assert_array_almost_equal(
            sample_feature_result.features,
            reconstructed.features,
        )


class TestImageItem:
    """Tests for ImageItem."""

    def test_has_ratings_true(self):
        """Test has_ratings with ratings present."""
        item = ImageItem(
            id="test",
            ratings={"mirror_self": 8.0, "drawn_to": 7.5},
        )
        assert item.has_ratings() is True

    def test_has_ratings_false(self):
        """Test has_ratings when ratings are None."""
        item = ImageItem(id="test")
        assert item.has_ratings() is False

    def test_has_ratings_empty(self):
        """Test has_ratings when ratings dict is empty."""
        item = ImageItem(id="test", ratings={})
        assert item.has_ratings() is False

    def test_get_rating(self):
        """Test getting specific rating value."""
        item = ImageItem(
            id="test",
            ratings={"mirror_self": 8.0, "drawn_to": 7.5},
        )
        assert item.get_rating("mirror_self") == 8.0
        assert item.get_rating("unknown") is None
