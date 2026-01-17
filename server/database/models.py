"""Database models for art-interp - SAE features only."""

from datetime import UTC, datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base


def utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(UTC)


Base = declarative_base()


class ArtworkMeta(Base):
    """Artwork metadata - maps to existing Supabase 'artworks' table."""

    __tablename__ = "artworks"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    artist_name = Column(String, nullable=True)
    year = Column(Integer, nullable=True)
    image_url = Column(String, nullable=False)
    thumbnail_url = Column(String, nullable=True)
    labels = Column(Text, nullable=True)  # JSON: AI-generated labels
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class SAEFeature(Base):
    """SAE (Sparse Autoencoder) feature metadata from interpretability analysis."""

    __tablename__ = "sae_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    layer = Column(Integer, nullable=False, index=True)  # Transformer layer (7, 8, 11)
    feature_idx = Column(Integer, nullable=False)  # Feature index within layer
    monosemanticity_score = Column(Float, nullable=True)  # Semantic coherence [0-1]
    tier = Column(Integer, nullable=False, index=True)  # Quality tier (1=elite, 2=good, 3=diverse)
    strongest_rating = Column(String(100), nullable=True)  # Best correlated rating type
    strongest_correlation = Column(Float, nullable=True)  # Correlation coefficient
    top_artwork_ids = Column(Text, nullable=True)  # JSON array of artwork IDs
    all_correlations = Column(Text, nullable=True)  # JSON object of all rating correlations
    created_at = Column(DateTime, default=utc_now)

    __table_args__ = (
        UniqueConstraint("layer", "feature_idx", name="uq_sae_layer_feature"),
        Index("idx_sae_monosemanticity", "monosemanticity_score"),
    )


class SAEFeatureLabel(Base):
    """Gemini-generated labels for SAE features."""

    __tablename__ = "sae_feature_labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    layer = Column(Integer, nullable=False, index=True)
    feature_idx = Column(Integer, nullable=False)
    short_label = Column(String(255), nullable=False)  # 2-5 word feature name
    description = Column(Text, nullable=False)  # 1-2 sentence explanation
    visual_elements = Column(Text, nullable=True)  # JSON array of visual properties
    explains_rating = Column(Text, nullable=True)  # How pattern influences rating
    non_obvious_insight = Column(Text, nullable=True)  # Surprising observations
    confidence = Column(Float, nullable=True)  # VLM confidence [0-1]
    model_used = Column(String(100), nullable=True)  # e.g., "gemini-2.0-flash-exp"
    num_images_used = Column(Integer, nullable=True)  # Images analyzed
    created_at = Column(DateTime, default=utc_now)

    __table_args__ = (
        UniqueConstraint("layer", "feature_idx", name="uq_sae_label_layer_feature"),
    )


class ArtworkTopFeatures(Base):
    """Top SAE feature activations per artwork - inverted index for fast lookup."""

    __tablename__ = "artwork_top_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    artwork_id = Column(String, nullable=False, index=True)
    layer = Column(Integer, nullable=False)  # Transformer layer (7, 8, 11)
    feature_idx = Column(Integer, nullable=False)  # Feature index within layer
    activation = Column(Float, nullable=False)  # Activation value
    rank = Column(Integer, nullable=False)  # Rank within this artwork (1 = highest)

    __table_args__ = (
        # Fast lookup: "get top features for artwork X in layer Y"
        Index("idx_artwork_top_features_lookup", "artwork_id", "layer", "rank"),
        # Fast lookup: "get artworks with high activation for feature X"
        Index("idx_artwork_top_features_feature", "layer", "feature_idx", "activation"),
        # Prevent duplicates
        UniqueConstraint("artwork_id", "layer", "feature_idx", name="uq_artwork_layer_feature"),
    )
