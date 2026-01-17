#!/usr/bin/env python3
"""
Import SAE feature labels from JSON files into Supabase PostgreSQL.

Usage:
    python scripts/import_feature_labels_to_db.py

This script:
1. Loads feature metadata from features_to_label_layer{7,8,11}.json
2. Loads labels from feature_labels_gemini_layer{7,8,11}.json
3. Creates tables if they don't exist
4. Inserts/updates records with upsert logic
5. Sets up RLS policies for public read access
"""

import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file before importing config
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from sqlalchemy import text

from config import DATABASE_TYPE, DATABASE_URL
from database.connection import engine, get_db_context
from database.models import Base, SAEFeature, SAEFeatureLabel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Log database configuration
logger.info(f"Database type: {DATABASE_TYPE}")
if DATABASE_TYPE == "postgresql":
    # Mask password in URL for logging
    masked_url = DATABASE_URL.split("@")[1] if "@" in DATABASE_URL else DATABASE_URL
    logger.info(f"Database host: {masked_url}")

# Paths to JSON files
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "sae_analysis"
LAYERS = [7, 8, 11]


def load_features_to_label(layer: int) -> list[dict]:
    """Load feature selection data for a layer."""
    path = OUTPUT_DIR / f"features_to_label_layer{layer}.json"
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return []

    with open(path) as f:
        data = json.load(f)
    return data.get("features", [])


def load_feature_labels(layer: int) -> tuple[dict, list[dict]]:
    """Load Gemini-generated labels for a layer."""
    path = OUTPUT_DIR / f"feature_labels_gemini_layer{layer}.json"
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return {}, []

    with open(path) as f:
        data = json.load(f)
    return data.get("metadata", {}), data.get("labels", [])


def create_tables() -> None:
    """Create tables if they don't exist."""
    logger.info("Creating tables...")
    Base.metadata.create_all(bind=engine, tables=[SAEFeature.__table__, SAEFeatureLabel.__table__])
    logger.info("Tables created")


def setup_rls_policies() -> None:
    """Set up Row Level Security for public read access."""
    logger.info("Setting up RLS policies...")

    with engine.connect() as conn:
        # Enable RLS on both tables
        conn.execute(text("ALTER TABLE sae_features ENABLE ROW LEVEL SECURITY"))
        conn.execute(text("ALTER TABLE sae_feature_labels ENABLE ROW LEVEL SECURITY"))

        # Drop existing policies if they exist (to allow re-running)
        conn.execute(text("DROP POLICY IF EXISTS \"Public read access\" ON sae_features"))
        conn.execute(text("DROP POLICY IF EXISTS \"Public read access\" ON sae_feature_labels"))

        # Create public read policies
        conn.execute(
            text('CREATE POLICY "Public read access" ON sae_features FOR SELECT USING (true)')
        )
        conn.execute(
            text('CREATE POLICY "Public read access" ON sae_feature_labels FOR SELECT USING (true)')
        )

        conn.commit()
        logger.info("RLS policies configured for public read access")


def import_features(layer: int, features: list[dict]) -> int:
    """Import feature metadata for a layer."""
    if not features:
        return 0

    with get_db_context() as db:
        count = 0
        for feature in features:
            # Check if exists
            existing = (
                db.query(SAEFeature)
                .filter(SAEFeature.layer == layer, SAEFeature.feature_idx == feature["feature_idx"])
                .first()
            )

            if existing:
                # Update existing
                existing.monosemanticity_score = feature.get("monosemanticity_score")
                existing.tier = feature["tier"]
                existing.strongest_rating = feature.get("strongest_rating")
                existing.strongest_correlation = feature.get("strongest_correlation")
                existing.top_artwork_ids = json.dumps(feature.get("top_5_artwork_ids", []))
                existing.all_correlations = json.dumps(feature.get("all_correlations", {}))
            else:
                # Insert new
                db.add(
                    SAEFeature(
                        layer=layer,
                        feature_idx=feature["feature_idx"],
                        monosemanticity_score=feature.get("monosemanticity_score"),
                        tier=feature["tier"],
                        strongest_rating=feature.get("strongest_rating"),
                        strongest_correlation=feature.get("strongest_correlation"),
                        top_artwork_ids=json.dumps(feature.get("top_5_artwork_ids", [])),
                        all_correlations=json.dumps(feature.get("all_correlations", {})),
                    )
                )
            count += 1

        db.commit()
        return count


def import_labels(layer: int, metadata: dict, labels: list[dict]) -> int:
    """Import Gemini-generated labels for a layer."""
    if not labels:
        return 0

    model_used = metadata.get("model", "gemini-2.0-flash-exp")

    with get_db_context() as db:
        count = 0
        for label in labels:
            # Check if exists
            existing = (
                db.query(SAEFeatureLabel)
                .filter(
                    SAEFeatureLabel.layer == layer,
                    SAEFeatureLabel.feature_idx == label["feature_idx"],
                )
                .first()
            )

            if existing:
                # Update existing
                existing.short_label = label["short_label"]
                existing.description = label["description"]
                existing.visual_elements = json.dumps(label.get("visual_elements", []))
                existing.explains_rating = label.get("explains_rating")
                existing.non_obvious_insight = label.get("non_obvious_insight")
                existing.confidence = label.get("confidence")
                existing.model_used = model_used
                existing.num_images_used = label.get("num_images_used")
            else:
                # Insert new
                db.add(
                    SAEFeatureLabel(
                        layer=layer,
                        feature_idx=label["feature_idx"],
                        short_label=label["short_label"],
                        description=label["description"],
                        visual_elements=json.dumps(label.get("visual_elements", [])),
                        explains_rating=label.get("explains_rating"),
                        non_obvious_insight=label.get("non_obvious_insight"),
                        confidence=label.get("confidence"),
                        model_used=model_used,
                        num_images_used=label.get("num_images_used"),
                    )
                )
            count += 1

        db.commit()
        return count


def main() -> None:
    """Main import function."""
    logger.info("Starting feature labels import to database...")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Create tables
    create_tables()

    total_features = 0
    total_labels = 0

    # Import each layer
    for layer in LAYERS:
        logger.info(f"\n=== Processing Layer {layer} ===")

        # Load and import features
        features = load_features_to_label(layer)
        if features:
            count = import_features(layer, features)
            logger.info(f"Layer {layer}: Imported {count} features")
            total_features += count

        # Load and import labels
        metadata, labels = load_feature_labels(layer)
        if labels:
            count = import_labels(layer, metadata, labels)
            logger.info(f"Layer {layer}: Imported {count} labels")
            total_labels += count

    # Set up RLS policies for public access
    try:
        setup_rls_policies()
    except Exception as e:
        logger.warning(f"Could not set up RLS policies (may already exist or not PostgreSQL): {e}")

    # Print summary
    logger.info("\n=== Import Summary ===")
    logger.info(f"Total features imported: {total_features}")
    logger.info(f"Total labels imported: {total_labels}")

    # Verify counts in database
    with get_db_context() as db:
        db_features = db.query(SAEFeature).count()
        db_labels = db.query(SAEFeatureLabel).count()
        logger.info(f"Features in database: {db_features}")
        logger.info(f"Labels in database: {db_labels}")

        # Breakdown by layer
        for layer in LAYERS:
            feature_count = db.query(SAEFeature).filter(SAEFeature.layer == layer).count()
            label_count = db.query(SAEFeatureLabel).filter(SAEFeatureLabel.layer == layer).count()
            logger.info(f"  Layer {layer}: {feature_count} features, {label_count} labels")

    logger.info("\nImport complete!")
    logger.info("You can now access these tables from any Supabase client:")
    logger.info("  const { data } = await supabase.from('sae_feature_labels').select('*')")


if __name__ == "__main__":
    main()
