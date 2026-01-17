#!/usr/bin/env python3
"""
Batch SAE Feature Extraction Script

Extracts sparse autoencoder features for all labeled artworks in the database.
Stores results as compressed JSON files for correlation analysis.

Usage:
    python scripts/extract_sae_features.py [--limit N] [--batch-size N] [--resume]
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sqlalchemy import text

from database.connection import get_db_context
from database.models import Artwork
from services.sae_features import SAE_CONFIGS, SAEFeatureService, save_features_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Output configuration
BASE_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sae_features"
BATCH_SIZE = 100  # Number of artworks per JSON file
RATE_LIMIT_DELAY = 0.3  # Seconds between requests


def get_output_dir(layer: int) -> Path:
    """Get layer-specific output directory."""
    return BASE_OUTPUT_DIR / f"layer_{layer}"


def get_labeled_artworks_with_images(limit: int | None = None) -> list[dict]:
    """
    Query artworks that have both labels and valid image URLs.

    Returns list of dicts with id, image_url, and labels.
    """
    with get_db_context() as db:
        query = db.query(Artwork).filter(
            Artwork.labels.isnot(None),
            Artwork.image_url.isnot(None),
        )

        if limit:
            query = query.limit(limit)

        artworks = query.all()

        results = []
        for artwork in artworks:
            try:
                labels = json.loads(artwork.labels) if artwork.labels else {}
            except json.JSONDecodeError:
                labels = {}

            results.append(
                {
                    "id": artwork.id,
                    "title": artwork.title,
                    "image_url": artwork.image_url,
                    "labels": labels,
                }
            )

        return results


def get_already_processed_ids(output_dir: Path) -> set[str]:
    """Get IDs of artworks already processed from existing batch files."""
    processed = set()

    if not output_dir.exists():
        return processed

    for batch_file in output_dir.glob("batch_*.json"):
        try:
            with open(batch_file) as f:
                data = json.load(f)
                processed.update(data.keys())
        except Exception as e:
            logger.warning(f"Error reading {batch_file}: {e}")

    return processed


async def extract_features_batch(
    service: SAEFeatureService,
    artworks: list[dict],
    batch_num: int,
    output_dir: Path,
) -> tuple[int, int]:
    """
    Extract features for a batch of artworks.

    Returns (success_count, failure_count).
    """
    features_dict = {}
    success_count = 0
    failure_count = 0

    for i, artwork in enumerate(artworks):
        artwork_id = artwork["id"]
        image_url = artwork["image_url"]

        try:
            result = await service.get_features_from_url(image_url)

            if result:
                features_dict[artwork_id] = result.features
                success_count += 1
                logger.debug(f"  [{i+1}/{len(artworks)}] {artwork_id}: {result.num_active} active features")
            else:
                failure_count += 1
                logger.warning(f"  [{i+1}/{len(artworks)}] {artwork_id}: Failed to extract features")

        except Exception as e:
            failure_count += 1
            logger.warning(f"  [{i+1}/{len(artworks)}] {artwork_id}: Error - {e}")

        # Rate limiting
        await asyncio.sleep(RATE_LIMIT_DELAY)

    # Save batch
    if features_dict:
        output_path = output_dir / f"batch_{batch_num:04d}.json"
        save_features_batch(features_dict, output_path, compress=True)
        logger.info(f"Saved batch {batch_num}: {len(features_dict)} artworks to {output_path}")

    return success_count, failure_count


async def main(
    layer: int = 8,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
    resume: bool = False,
):
    """Main extraction pipeline."""
    output_dir = get_output_dir(layer)

    logger.info("=" * 60)
    logger.info(f"SAE Feature Extraction Pipeline (Layer {layer})")
    logger.info("=" * 60)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get artworks to process
    logger.info("Querying labeled artworks...")
    all_artworks = get_labeled_artworks_with_images(limit=limit)
    logger.info(f"Found {len(all_artworks)} labeled artworks with images")

    # Filter already processed if resuming
    if resume:
        processed_ids = get_already_processed_ids(output_dir)
        logger.info(f"Resume mode: {len(processed_ids)} artworks already processed")
        all_artworks = [a for a in all_artworks if a["id"] not in processed_ids]
        logger.info(f"Remaining to process: {len(all_artworks)}")

    if not all_artworks:
        logger.info("No artworks to process!")
        return

    # Initialize service with specified layer
    logger.info(f"Initializing SAE Feature Service for layer {layer}...")
    service = SAEFeatureService(layer=layer)

    # Process in batches
    total_batches = (len(all_artworks) + batch_size - 1) // batch_size
    total_success = 0
    total_failure = 0
    start_time = time.time()

    # Find starting batch number
    existing_batches = list(output_dir.glob("batch_*.json"))
    start_batch_num = len(existing_batches) + 1 if resume else 1

    for batch_idx in range(total_batches):
        batch_num = start_batch_num + batch_idx
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(all_artworks))
        batch_artworks = all_artworks[batch_start:batch_end]

        logger.info(f"\nProcessing batch {batch_num}/{start_batch_num + total_batches - 1} ({len(batch_artworks)} artworks)")

        success, failure = await extract_features_batch(
            service, batch_artworks, batch_num, output_dir
        )

        total_success += success
        total_failure += failure

        # Progress estimate
        elapsed = time.time() - start_time
        processed = total_success + total_failure
        if processed > 0:
            rate = processed / elapsed
            remaining = len(all_artworks) - processed
            eta_seconds = remaining / rate if rate > 0 else 0
            logger.info(
                f"Progress: {processed}/{len(all_artworks)} "
                f"({total_success} ok, {total_failure} failed) "
                f"- ETA: {eta_seconds/60:.1f} min"
            )

    # Cleanup
    service.cleanup()

    # Save metadata
    sae_config = SAE_CONFIGS[layer]
    metadata = {
        "extraction_timestamp": datetime.utcnow().isoformat(),
        "total_artworks": len(all_artworks),
        "successful": total_success,
        "failed": total_failure,
        "sae_repo": sae_config["repo_id"],
        "sae_layer": layer,
        "sae_dim": sae_config["dim"],
        "clip_model": "ViT-B-32 (datacomp_xl_s13b_b90k)",
        "feature_type": "CLS token only",
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Final summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info(f"Extraction Complete! (Layer {layer})")
    logger.info(f"  Total processed: {total_success + total_failure}")
    logger.info(f"  Successful: {total_success}")
    logger.info(f"  Failed: {total_failure}")
    logger.info(f"  Time: {elapsed/60:.1f} minutes")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SAE features for labeled artworks")
    parser.add_argument(
        "--layer",
        type=int,
        default=8,
        choices=list(SAE_CONFIGS.keys()),
        help="CLIP transformer layer to extract from (default: 8)",
    )
    parser.add_argument("--limit", type=int, help="Limit number of artworks to process")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Artworks per batch file")
    parser.add_argument("--resume", action="store_true", help="Resume from existing progress")
    args = parser.parse_args()

    asyncio.run(main(layer=args.layer, limit=args.limit, batch_size=args.batch_size, resume=args.resume))
