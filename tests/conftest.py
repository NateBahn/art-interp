"""Test fixtures for interpretability package."""

import numpy as np
import pytest

from interpretability.core.types import SAEFeatureResult
from interpretability.storage.sample_provider import SampleEmbeddingProvider, SampleImageProvider


@pytest.fixture
def sample_provider():
    """Provide sample image data."""
    return SampleImageProvider()


@pytest.fixture
def sample_embedding_provider():
    """Provide sample embeddings (random, for testing only)."""
    return SampleEmbeddingProvider()


@pytest.fixture
def sample_feature_result():
    """Create a sample SAEFeatureResult for testing."""
    features = np.zeros(49152, dtype=np.float32)
    # Set some non-zero values
    features[100] = 2.5
    features[500] = 1.8
    features[1000] = 1.2
    features[5000] = 0.9
    features[10000] = 0.5

    return SAEFeatureResult(
        features=features,
        num_active=5,
        top_indices=[100, 500, 1000, 5000, 10000],
        top_values=[2.5, 1.8, 1.2, 0.9, 0.5],
    )


@pytest.fixture
def sample_activations():
    """Sample feature activations for multiple items."""
    return {
        "sample_001": {"100": 2.5, "500": 1.8},
        "sample_002": {"100": 2.0, "1000": 1.5},
        "sample_003": {"100": 1.5, "500": 2.0, "1000": 1.0},
        "sample_004": {"500": 1.0, "2000": 0.8},
        "sample_005": {"100": 0.5, "500": 0.5, "1000": 0.5},
    }


@pytest.fixture
def sample_ratings():
    """Sample ratings for testing correlation analysis."""
    return {
        "sample_001": {"mirror_self": 8.5, "drawn_to": 9.0},
        "sample_002": {"mirror_self": 7.0, "drawn_to": 9.0},
        "sample_003": {"mirror_self": 8.0, "drawn_to": 8.5},
        "sample_004": {"mirror_self": 6.5, "drawn_to": 8.0},
        "sample_005": {"mirror_self": 9.0, "drawn_to": 8.0},
    }
