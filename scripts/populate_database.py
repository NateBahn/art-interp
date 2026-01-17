#!/usr/bin/env python3
"""
Populate Database Script

Adds artworks to the database and extracts SAE features.
This is the main entry point for setting up your own art-interp instance.

Usage:
    # From a directory of images
    python scripts/populate_database.py --images ./my_artworks/

    # From a CSV file with metadata
    python scripts/populate_database.py --csv artworks.csv

    # Just extract features for existing artworks
    python scripts/populate_database.py --features-only

CSV format (optional columns: id, title, artist, year):
    image_path,title,artist,year
    ./images/starry_night.jpg,The Starry Night,Vincent van Gogh,1889
    ./images/mona_lisa.jpg,Mona Lisa,Leonardo da Vinci,1503
"""

import argparse
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_database_session():
    """Get database session from environment."""
    import os
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")

    database_url = os.getenv("DATABASE_URL", "sqlite:///./data/art_interp.db")

    # Ensure data directory exists for SQLite
    if "sqlite" in database_url:
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)

    connect_args = {"check_same_thread": False} if "sqlite" in database_url else {}
    engine = create_engine(database_url, connect_args=connect_args)

    # Import and create tables
    from server.database.models import Base
    Base.metadata.create_all(bind=engine)

    Session = sessionmaker(bind=engine)
    return Session()


def generate_artwork_id(image_path: Path) -> str:
    """Generate a unique ID for an artwork based on file hash."""
    with open(image_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()[:12]
    return f"local_{file_hash}"


def add_artwork_from_image(
    db,
    image_path: Path,
    title: str | None = None,
    artist: str | None = None,
    year: int | None = None,
    artwork_id: str | None = None,
) -> str:
    """Add a single artwork to the database."""
    from server.database.models import ArtworkMeta

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Generate ID if not provided
    if artwork_id is None:
        artwork_id = generate_artwork_id(image_path)

    # Use filename as title if not provided
    if title is None:
        title = image_path.stem.replace("_", " ").title()

    # Check if already exists
    existing = db.query(ArtworkMeta).filter(ArtworkMeta.id == artwork_id).first()
    if existing:
        logger.info(f"Artwork {artwork_id} already exists, skipping")
        return artwork_id

    # For local files, use file:// URL (or you could copy to a static dir)
    image_url = f"file://{image_path.absolute()}"

    artwork = ArtworkMeta(
        id=artwork_id,
        title=title,
        artist_name=artist,
        year=year,
        image_url=image_url,
        thumbnail_url=image_url,  # Same as main for local files
    )

    db.add(artwork)
    db.commit()

    logger.info(f"Added artwork: {title} ({artwork_id})")
    return artwork_id


def add_artworks_from_directory(db, image_dir: Path) -> list[str]:
    """Add all images from a directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    artwork_ids = []

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() in image_extensions:
            try:
                artwork_id = add_artwork_from_image(db, image_path)
                artwork_ids.append(artwork_id)
            except Exception as e:
                logger.error(f"Failed to add {image_path}: {e}")

    return artwork_ids


def add_artworks_from_csv(db, csv_path: Path) -> list[str]:
    """Add artworks from a CSV file."""
    artwork_ids = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_path = Path(row.get("image_path", row.get("path", "")))

            # Handle relative paths
            if not image_path.is_absolute():
                image_path = csv_path.parent / image_path

            try:
                artwork_id = add_artwork_from_image(
                    db,
                    image_path,
                    title=row.get("title"),
                    artist=row.get("artist"),
                    year=int(row["year"]) if row.get("year") else None,
                    artwork_id=row.get("id"),
                )
                artwork_ids.append(artwork_id)
            except Exception as e:
                logger.error(f"Failed to add {image_path}: {e}")

    return artwork_ids


def extract_features_for_artworks(db, artwork_ids: list[str], layer: int = 8):
    """Extract SAE features for artworks."""
    from server.database.models import ArtworkMeta, ArtworkTopFeatures

    try:
        from src.interpretability.core import SAEFeatureExtractor
    except ImportError:
        logger.error(
            "Could not import SAEFeatureExtractor. "
            "Make sure you've installed the package: pip install -e ."
        )
        return

    logger.info(f"Initializing SAE extractor for layer {layer}...")
    extractor = SAEFeatureExtractor(layer=layer)

    for artwork_id in artwork_ids:
        artwork = db.query(ArtworkMeta).filter(ArtworkMeta.id == artwork_id).first()
        if not artwork:
            continue

        # Check if features already exist
        existing = db.query(ArtworkTopFeatures).filter(
            ArtworkTopFeatures.artwork_id == artwork_id,
            ArtworkTopFeatures.layer == layer,
        ).first()

        if existing:
            logger.info(f"Features already exist for {artwork_id}, skipping")
            continue

        # Load image
        image_url = artwork.image_url
        try:
            if image_url.startswith("file://"):
                image_path = Path(image_url[7:])
                image = Image.open(image_path).convert("RGB")
            else:
                # For remote URLs, would need httpx
                import httpx
                response = httpx.get(image_url, timeout=30)
                from io import BytesIO
                image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image for {artwork_id}: {e}")
            continue

        # Extract features
        logger.info(f"Extracting features for: {artwork.title}")
        result = extractor.extract_from_image(image)

        if result is None:
            logger.error(f"Feature extraction failed for {artwork_id}")
            continue

        # Save top features to database
        top_k = 50  # Store top 50 features per artwork
        top_indices = result.top_indices[:top_k]
        top_values = result.top_values[:top_k]

        for rank, (feature_idx, activation) in enumerate(zip(top_indices, top_values), 1):
            feature = ArtworkTopFeatures(
                artwork_id=artwork_id,
                layer=layer,
                feature_idx=int(feature_idx),
                activation=float(activation),
                rank=rank,
            )
            db.add(feature)

        db.commit()
        logger.info(f"Saved {len(top_indices)} features for {artwork_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Populate art-interp database with artworks and features"
    )
    parser.add_argument(
        "--images", type=Path, help="Directory containing artwork images"
    )
    parser.add_argument(
        "--csv", type=Path, help="CSV file with artwork metadata"
    )
    parser.add_argument(
        "--features-only", action="store_true",
        help="Only extract features for existing artworks (don't add new ones)"
    )
    parser.add_argument(
        "--layer", type=int, default=8,
        help="CLIP layer for feature extraction (default: 8)"
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Only add artworks, don't extract features"
    )

    args = parser.parse_args()

    if not args.images and not args.csv and not args.features_only:
        parser.print_help()
        print("\nError: Specify --images, --csv, or --features-only")
        sys.exit(1)

    # Connect to database
    logger.info("Connecting to database...")
    db = get_database_session()

    artwork_ids = []

    # Add artworks
    if args.images:
        logger.info(f"Adding artworks from directory: {args.images}")
        artwork_ids = add_artworks_from_directory(db, args.images)
        logger.info(f"Added {len(artwork_ids)} artworks")

    elif args.csv:
        logger.info(f"Adding artworks from CSV: {args.csv}")
        artwork_ids = add_artworks_from_csv(db, args.csv)
        logger.info(f"Added {len(artwork_ids)} artworks")

    elif args.features_only:
        # Get all artwork IDs from database
        from server.database.models import ArtworkMeta
        artworks = db.query(ArtworkMeta).all()
        artwork_ids = [a.id for a in artworks]
        logger.info(f"Found {len(artwork_ids)} artworks in database")

    # Extract features
    if not args.skip_features and artwork_ids:
        logger.info(f"Extracting SAE features for {len(artwork_ids)} artworks...")
        extract_features_for_artworks(db, artwork_ids, layer=args.layer)

    logger.info("Done!")

    # Print summary
    from server.database.models import ArtworkMeta, ArtworkTopFeatures
    total_artworks = db.query(ArtworkMeta).count()
    total_features = db.query(ArtworkTopFeatures).count()

    print(f"\nDatabase summary:")
    print(f"  Artworks: {total_artworks}")
    print(f"  Feature records: {total_features}")


if __name__ == "__main__":
    main()
