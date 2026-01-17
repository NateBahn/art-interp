"""
Rate each painting on self-reflection using Qwen vision model.

Asks: "Does this artwork reflect something of yourself?"
Collects 1-10 rating + explanation for each painting.

Usage:
    python scripts/rate_paintings_self_reflection.py
    python scripts/rate_paintings_self_reflection.py --resume
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path(__file__).parent.parent
IMAGE_CACHE_DIR = BASE_DIR / "output" / "elo_comparison" / "image_cache"
OUTPUT_DIR = BASE_DIR / "output" / "self_reflection_ratings"

RATING_PROMPT = """Look at this artwork carefully. I want you to answer the question: "Does this artwork reflect something of yourself?"

This question is asking how much you, as Qwen, see yourself or relate to this painting.

Please respond with:
1. A rating from 1-10 (where 1 = "I don't see myself in this at all" and 10 = "This deeply reflects something about me")
2. An explanation that specifically names 2-3 visual elements in the painting (colors, shapes, textures, composition, subjects, techniques) that contribute to your rating. Be concrete - point to what you actually see.

Format your response as JSON:
{
  "rating": <number 1-10>,
  "explanation": "<your explanation citing specific visual elements>"
}

Be authentic. It's fine to give low ratings - just explain which specific visual elements feel distant or unfamiliar to you."""


async def check_ollama_available(base_url: str, model: str) -> bool:
    """Check if Ollama is running and model is available."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            data = resp.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        model_base = model.split(":")[0]
        return any(model_base in m for m in models)
    except Exception as e:
        logger.error(f"Ollama not available: {e}")
        return False


async def rate_painting(base_url: str, model: str, image_b64: str) -> dict | None:
    """Send image to Ollama and get rating response."""
    payload = {
        "model": model,
        "prompt": RATING_PROMPT,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.3}
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=120.0
            )
            resp.raise_for_status()
            response_text = resp.json().get("response", "")

        # Parse JSON from response
        return parse_rating_response(response_text)

    except Exception as e:
        logger.warning(f"Ollama API call failed: {e}")
        return None


def parse_rating_response(response_text: str) -> dict | None:
    """Parse JSON rating response from model."""
    text = response_text.strip()

    # Try to extract JSON from response
    try:
        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()

        # Find JSON object
        if "{" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]

        data = json.loads(text)

        # Validate required fields
        if "rating" in data and "explanation" in data:
            rating = int(data["rating"])
            if 1 <= rating <= 10:
                return {
                    "rating": rating,
                    "explanation": str(data["explanation"])
                }
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.debug(f"Failed to parse response: {e}")

    return None


def load_progress(output_file: Path) -> dict:
    """Load existing progress."""
    if output_file.exists():
        with open(output_file) as f:
            return json.load(f)
    return {
        "metadata": {
            "started_at": datetime.utcnow().isoformat() + "Z",
            "model": OLLAMA_MODEL,
            "prompt_version": "v1"
        },
        "ratings": {},
        "errors": []
    }


def save_progress(output_file: Path, data: dict):
    """Save current progress."""
    data["metadata"]["last_updated"] = datetime.utcnow().isoformat() + "Z"
    data["metadata"]["total_rated"] = len(data["ratings"])

    # Backup existing file
    if output_file.exists():
        backup = output_file.with_suffix(".backup.json")
        import shutil
        shutil.copy(output_file, backup)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


async def main():
    parser = argparse.ArgumentParser(description="Rate paintings on self-reflection")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from previous run")
    args = parser.parse_args()

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "self_reflection_ratings.json"

    # Check cache exists
    if not IMAGE_CACHE_DIR.exists():
        logger.error(f"Image cache not found at {IMAGE_CACHE_DIR}")
        return

    # Get all cached images
    cache_files = sorted(IMAGE_CACHE_DIR.glob("*.b64"))
    total_paintings = len(cache_files)
    logger.info(f"Found {total_paintings} cached paintings")

    if total_paintings == 0:
        logger.error("No cached images found")
        return

    # Load or create progress
    if args.resume:
        data = load_progress(output_file)
        logger.info(f"Resuming: {len(data['ratings'])} already rated")
    else:
        data = load_progress(output_file) if output_file.exists() else {
            "metadata": {
                "started_at": datetime.utcnow().isoformat() + "Z",
                "model": OLLAMA_MODEL,
                "prompt_version": "v1"
            },
            "ratings": {},
            "errors": []
        }

    # Check Ollama
    if not await check_ollama_available(OLLAMA_BASE_URL, OLLAMA_MODEL):
        logger.error("Ollama not available")
        return

    # Filter to unprocessed paintings
    already_rated = set(data["ratings"].keys())
    to_process = [f for f in cache_files if f.stem not in already_rated]

    logger.info(f"Paintings to rate: {len(to_process)}")
    print("=" * 70)

    start_time = time.time()
    save_interval = 25
    processed_since_save = 0

    for i, cache_file in enumerate(to_process):
        artwork_id = cache_file.stem

        try:
            # Load cached image
            image_b64 = cache_file.read_text()

            # Get rating
            result = await rate_painting(OLLAMA_BASE_URL, OLLAMA_MODEL, image_b64)

            if result:
                data["ratings"][artwork_id] = {
                    "rating": result["rating"],
                    "explanation": result["explanation"],
                    "rated_at": datetime.utcnow().isoformat() + "Z"
                }

                # Progress display
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(to_process) - i - 1) / rate if rate > 0 else 0

                total_done = len(data["ratings"])
                print(f"[{total_done}/{total_paintings}] {artwork_id}: {result['rating']}/10 | ETA: {eta/60:.1f}m")
            else:
                data["errors"].append({
                    "artwork_id": artwork_id,
                    "error": "parse_failed",
                    "timestamp": datetime.utcnow().isoformat()
                })
                print(f"[{len(data['ratings'])}/{total_paintings}] {artwork_id}: PARSE ERROR")

            processed_since_save += 1

            # Save periodically
            if processed_since_save >= save_interval:
                save_progress(output_file, data)
                logger.info(f"Progress saved: {len(data['ratings'])} ratings")
                processed_since_save = 0

            # Small delay
            await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("\nInterrupted - saving progress...")
            save_progress(output_file, data)
            return
        except Exception as e:
            logger.exception(f"Error processing {artwork_id}")
            data["errors"].append({
                "artwork_id": artwork_id,
                "error": str(e)[:200],
                "timestamp": datetime.utcnow().isoformat()
            })

    # Final save
    save_progress(output_file, data)

    # Summary
    print("\n" + "=" * 70)
    print("SELF-REFLECTION RATING COMPLETE")
    print("=" * 70)

    ratings = [r["rating"] for r in data["ratings"].values()]
    if ratings:
        import statistics
        print(f"\nTotal rated: {len(ratings)}")
        print(f"Mean rating: {statistics.mean(ratings):.2f}")
        print(f"Std dev: {statistics.stdev(ratings):.2f}" if len(ratings) > 1 else "")
        print(f"Min: {min(ratings)}, Max: {max(ratings)}")

        # Distribution
        print("\nRating distribution:")
        for r in range(1, 11):
            count = sum(1 for x in ratings if x == r)
            bar = "#" * (count // 10)
            print(f"  {r:2}: {count:4} {bar}")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
