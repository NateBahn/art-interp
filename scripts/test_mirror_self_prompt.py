"""
Test different phrasings for the mirror_self prompt with Ollama.

Usage:
    python scripts/test_mirror_self_prompt.py

Requires Ollama running with a vision model (e.g., qwen2.5vl:3b)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import base64
import json
import statistics
import httpx
from database.connection import get_db_context
from database.models import Artwork

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:3b"  # or whatever model you're using

# Test different phrasings for mirror_self
PHRASINGS = {
    "A": "mirror_self: Christopher Alexander asks: does this have 'the quality without a name' - does it reflect something deep and true about being alive? (1=lifeless, 10=profoundly alive)",
    "B": "mirror_self: Does this artwork reveal something true about being human - about our hopes, fears, or inner life? (1=not at all, 10=profoundly revealing)",
    "C": "mirror_self: Does this artwork resonate with universal human experience in a way that feels deeply true? (1=not at all, 10=profoundly)",
    "D": "mirror_self: Looking at this, does it make you feel more whole, more yourself, more alive? (1=not at all, 10=profoundly)",
}


async def fetch_image_base64(url: str) -> str | None:
    """Fetch image and convert to base64."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, timeout=30, follow_redirects=True)
            if resp.status_code == 200:
                return base64.b64encode(resp.content).decode()
        except Exception as e:
            print(f"  Error fetching image: {e}")
    return None


async def test_prompt(image_b64: str, prompt_text: str) -> int | None:
    """Test a single prompt and return the score."""
    full_prompt = f"""Rate this artwork on the following criterion. Respond with ONLY a single integer from 1-10.

{prompt_text}

Respond with just the number, nothing else."""

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": full_prompt,
                    "images": [image_b64],
                    "stream": False,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                result = resp.json().get("response", "").strip()
                # Extract number from response
                for char in result:
                    if char.isdigit():
                        return int(char)
        except Exception as e:
            print(f"  Error: {e}")
    return None


async def main():
    # Get diverse sample artworks (non-religious)
    with get_db_context() as db:
        artworks = db.query(Artwork).filter(
            Artwork.labels.isnot(None),
            Artwork.image_url.isnot(None)
        ).limit(2000).all()

        # Group by subject_matter to get diverse samples
        by_subject = {}
        for art in artworks:
            try:
                labels = json.loads(art.labels)
                subject = labels.get("subject_matter", "unknown")
                # Skip religious content
                if subject == "religious_mythological":
                    continue
                if subject not in by_subject:
                    by_subject[subject] = []
                by_subject[subject].append({
                    "id": art.id,
                    "title": art.title[:40] if art.title else "Untitled",
                    "url": art.image_url,
                    "subject": subject,
                })
            except:
                pass

        # Take samples from each subject
        samples = []
        per_subject = max(1, 50 // len(by_subject))
        for subject, arts in by_subject.items():
            samples.extend(arts[:per_subject])
        samples = samples[:50]

        if not samples:
            print("No suitable samples found")
            return

        print(f"Testing {len(PHRASINGS)} phrasings on {len(samples)} diverse artworks")
        print(f"Subjects: {list(by_subject.keys())}\n")
        print("=" * 80)

        # Track scores per phrasing
        scores_by_phrasing = {k: [] for k in PHRASINGS.keys()}

        for i, sample in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] {sample['title']} ({sample['subject']})")

            image_b64 = await fetch_image_base64(sample["url"])
            if not image_b64:
                print("  Could not fetch image")
                continue

            for key, phrasing in PHRASINGS.items():
                score = await test_prompt(image_b64, phrasing)
                if score is not None:
                    scores_by_phrasing[key].append(score)
                print(f"  {key}: {score}")

        # Print statistics
        print("\n" + "=" * 80)
        print("STATISTICS SUMMARY")
        print("=" * 80)

        for key, phrasing in PHRASINGS.items():
            scores = scores_by_phrasing[key]
            if len(scores) >= 2:
                mean = statistics.mean(scores)
                stdev = statistics.stdev(scores)
                min_s, max_s = min(scores), max(scores)
                dist = {s: scores.count(s) for s in range(1, 11)}
                dist_str = " ".join(f"{k}:{v}" for k, v in dist.items() if v > 0)
                print(f"\n{key}: {phrasing[:60]}...")
                print(f"   n={len(scores)}, mean={mean:.2f}, stdev={stdev:.2f}, range=[{min_s}-{max_s}]")
                print(f"   distribution: {dist_str}")
            else:
                print(f"\n{key}: insufficient data ({len(scores)} samples)")


if __name__ == "__main__":
    asyncio.run(main())
