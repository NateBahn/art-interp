#!/usr/bin/env python3
"""
Build inverted index: artwork_id -> top N activated features.

Usage:
    python scripts/build_artwork_top_features.py [--top-n 20] [--layer 11]

This script:
1. Loads SAE activation data from batch files
2. For each artwork, finds top N features by activation
3. Optionally filters to only labeled features (interpretable)
4. Stores results in artwork_top_features table
"""

import argparse
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

from config import DATABASE_TYPE
from database.connection import engine, get_db_context
from database.models import ArtworkTopFeatures, Base, SAEFeature

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data" / "sae_features"
LAYERS = [7, 8, 11]


def get_labeled_feature_indices(layer: int) -> set[int]:
    """Get the set of feature indices that have labels (interpretable features)."""
    with get_db_context() as db:
        features = db.query(SAEFeature.feature_idx).filter(SAEFeature.layer == layer).all()
        return {f[0] for f in features}


def load_sparse_batch(batch_path: Path) -> dict[str, dict[str, float]]:
    """Load a batch file as sparse dict (don't convert to dense)."""
    with open(batch_path) as f:
        return json.load(f)


def process_layer(
    layer: int,
    top_n: int = 20,
    labeled_only: bool = True,
    batch_size: int = 1000,
) -> int:
    """
    Process all artworks for a layer and store top features.

    Args:
        layer: Transformer layer (7, 8, 11)
        top_n: Number of top features to store per artwork
        labeled_only: If True, only consider features that have labels
        batch_size: Number of records to commit at once

    Returns:
        Number of records inserted
    """
    layer_dir = DATA_DIR / f"layer_{layer}"
    if not layer_dir.exists():
        logger.warning(f"Layer directory not found: {layer_dir}")
        return 0

    # Get labeled feature indices if filtering
    labeled_indices = None
    if labeled_only:
        labeled_indices = get_labeled_feature_indices(layer)
        logger.info(f"Layer {layer}: Found {len(labeled_indices)} labeled features")
        if not labeled_indices:
            logger.warning(f"No labeled features for layer {layer}, skipping")
            return 0

    # Find all batch files
    batch_files = sorted(layer_dir.glob("batch_*.json"))
    logger.info(f"Layer {layer}: Found {len(batch_files)} batch files")

    total_inserted = 0
    records_buffer = []

    for batch_idx, batch_path in enumerate(batch_files):
        logger.info(f"Processing {batch_path.name} ({batch_idx + 1}/{len(batch_files)})")

        # Load sparse data
        batch_data = load_sparse_batch(batch_path)

        for artwork_id, sparse_features in batch_data.items():
            # Filter to labeled features if requested
            if labeled_only:
                filtered = {
                    int(idx): val
                    for idx, val in sparse_features.items()
                    if int(idx) in labeled_indices
                }
            else:
                filtered = {int(idx): val for idx, val in sparse_features.items()}

            if not filtered:
                continue

            # Sort by activation value (descending) and take top N
            sorted_features = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n]

            # Create records
            for rank, (feature_idx, activation) in enumerate(sorted_features, start=1):
                records_buffer.append(
                    ArtworkTopFeatures(
                        artwork_id=artwork_id,
                        layer=layer,
                        feature_idx=feature_idx,
                        activation=activation,
                        rank=rank,
                    )
                )

        # Commit in batches
        if len(records_buffer) >= batch_size:
            with get_db_context() as db:
                db.bulk_save_objects(records_buffer)
            total_inserted += len(records_buffer)
            logger.info(f"  Inserted {total_inserted} records so far...")
            records_buffer = []

    # Insert remaining records
    if records_buffer:
        with get_db_context() as db:
            db.bulk_save_objects(records_buffer)
        total_inserted += len(records_buffer)

    return total_inserted


def clear_layer_data(layer: int) -> int:
    """Clear existing data for a layer (for re-runs)."""
    with get_db_context() as db:
        deleted = db.query(ArtworkTopFeatures).filter(ArtworkTopFeatures.layer == layer).delete()
        return deleted


def setup_public_access():
    """Grant public read access for Supabase."""
    if DATABASE_TYPE != "postgresql":
        logger.info("Skipping public access setup (not PostgreSQL)")
        return

    from sqlalchemy import text

    with engine.connect() as conn:
        logger.info("Setting up public access for artwork_top_features...")
        conn.execute(text("ALTER TABLE artwork_top_features DISABLE ROW LEVEL SECURITY"))
        conn.execute(text("GRANT SELECT ON artwork_top_features TO anon"))
        conn.commit()
        logger.info("Public access configured")


def main():
    parser = argparse.ArgumentParser(description="Build artwork -> top features inverted index")
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top features to store per artwork (default: 20)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        choices=[7, 8, 11],
        help="Process only this layer (default: all layers)",
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Include all features, not just labeled ones",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before inserting",
    )
    args = parser.parse_args()

    logger.info(f"Database type: {DATABASE_TYPE}")
    logger.info(f"Top N features per artwork: {args.top_n}")
    logger.info(f"Labeled features only: {not args.all_features}")

    # Create table if not exists
    logger.info("Creating table if not exists...")
    Base.metadata.create_all(bind=engine, tables=[ArtworkTopFeatures.__table__])

    layers_to_process = [args.layer] if args.layer else LAYERS

    for layer in layers_to_process:
        logger.info(f"\n=== Processing Layer {layer} ===")

        if args.clear:
            deleted = clear_layer_data(layer)
            logger.info(f"Cleared {deleted} existing records for layer {layer}")

        inserted = process_layer(
            layer=layer,
            top_n=args.top_n,
            labeled_only=not args.all_features,
        )
        logger.info(f"Layer {layer}: Inserted {inserted} records")

    # Set up public access
    setup_public_access()

    # Print summary
    logger.info("\n=== Summary ===")
    with get_db_context() as db:
        total = db.query(ArtworkTopFeatures).count()
        logger.info(f"Total records in artwork_top_features: {total}")

        for layer in LAYERS:
            count = db.query(ArtworkTopFeatures).filter(ArtworkTopFeatures.layer == layer).count()
            artworks = (
                db.query(ArtworkTopFeatures.artwork_id)
                .filter(ArtworkTopFeatures.layer == layer)
                .distinct()
                .count()
            )
            logger.info(f"  Layer {layer}: {count} records, {artworks} artworks")

    logger.info("\nDone! You can now query top features for any artwork:")
    logger.info("  supabase.from('artwork_top_features').select('*').eq('artwork_id', 'met_123')")


if __name__ == "__main__":
    main()
