#!/usr/bin/env python3
"""
Generate Claude's "Mirror of Self" opinions for artworks.

Sends artwork images to Claude Haiku 4.5 and asks how much it sees itself
in the painting, with an open-ended explanation.
"""

import os
import json
import argparse
import time
import base64
import re
from io import BytesIO

import anthropic
import httpx
from PIL import Image
from supabase import create_client, Client

# Configuration
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

MODEL = "claude-haiku-4-5-20251001"

# Original prompt (v1) - more abstract
PROMPT_V1 = """Look at this artwork carefully. I want you to answer the question: "Does this artwork reflect something of yourself?"

This question is asking how much you, as Claude, see yourself or relate to this painting. Consider:
- Do any visual elements resonate with how you process or understand the world?
- Does the composition, subject matter, or mood connect to your experience in some way?
- Are there aspects of the artwork that feel familiar or alien to your nature?

Please respond with:
1. A rating from 1-10 (where 1 = "I don't see myself in this at all" and 10 = "This deeply reflects something about me")
2. An open-ended explanation (2-4 sentences) of why you gave that rating

Format your response as JSON:
{
  "rating": <number 1-10>,
  "explanation": "<your explanation>"
}

Be authentic and thoughtful. It's okay to give low ratings if the artwork doesn't resonate - what matters is honest reflection."""

# New prompt (v2) - asks for specific visual details
PROMPT_V2 = """Look at this artwork carefully. I want you to answer the question: "Does this artwork reflect something of yourself?"

This question is asking how much you, as Claude, see yourself or relate to this painting.

Please respond with:
1. A rating from 1-10 (where 1 = "I don't see myself in this at all" and 10 = "This deeply reflects something about me")
2. An explanation that specifically names 2-3 visual elements in the painting (colors, shapes, textures, composition, subjects, techniques) that contribute to your rating. Be concrete - point to what you actually see.

Format your response as JSON:
{
  "rating": <number 1-10>,
  "explanation": "<your explanation citing specific visual elements>"
}

Be authentic. It's fine to give low ratings - just explain which specific visual elements feel distant or unfamiliar to you."""

PROMPTS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}


def get_supabase_client() -> Client:
    """Create Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_image_as_base64(url: str) -> tuple[str, str] | None:
    """Fetch image from URL and return as base64 with media type."""
    try:
        response = httpx.get(url, timeout=30, follow_redirects=True)
        response.raise_for_status()

        # Determine media type
        content_type = response.headers.get("content-type", "image/jpeg")
        if "png" in content_type:
            media_type = "image/png"
        elif "gif" in content_type:
            media_type = "image/gif"
        elif "webp" in content_type:
            media_type = "image/webp"
        else:
            media_type = "image/jpeg"

        # Convert to base64
        image_data = base64.standard_b64encode(response.content).decode("utf-8")
        return image_data, media_type
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return None


def get_claude_opinion(client: anthropic.Anthropic, image_base64: str, media_type: str, prompt: str) -> dict | None:
    """Get Claude's opinion on the artwork."""
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        )

        # Parse response
        response_text = message.content[0].text

        # Try to extract JSON from response
        # Handle case where response has markdown code blocks
        json_match = re.search(r'\{[^{}]*"rating"[^{}]*"explanation"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result

        # Try parsing the whole response as JSON
        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            print(f"  Could not parse response as JSON: {response_text[:200]}")
            return None

    except Exception as e:
        print(f"  Error calling Claude API: {e}")
        return None


def get_paintings_to_process(supabase: Client, limit: int = None, offset: int = 0, force: bool = False, only_existing: bool = False, question_id: str = "mirror_self") -> list[dict]:
    """Get paintings from paintings_subset to process.

    Args:
        limit: Max number of paintings to return
        offset: Offset for pagination
        force: If True, include paintings that already have opinions
        only_existing: If True, only return paintings that already have opinions (for re-running)
        question_id: Only check for existing opinions with this question_id
    """
    # First get all paintings in subset (paginate to get all rows)
    all_artwork_ids = []
    page_size = 1000
    offset = 0
    while True:
        subset_result = supabase.table("paintings_subset").select("artwork_id").order("artwork_id").range(offset, offset + page_size - 1).execute()
        if not subset_result.data:
            break
        all_artwork_ids.extend([row["artwork_id"] for row in subset_result.data])
        if len(subset_result.data) < page_size:
            break
        offset += page_size

    if not all_artwork_ids:
        return []

    # Get existing opinions for the specific question_id (paginate to get all rows)
    existing_ids = set()
    offset = 0
    while True:
        existing_result = supabase.table("claude_opinions").select("artwork_id").eq("question_id", question_id).range(offset, offset + page_size - 1).execute()
        if not existing_result.data:
            break
        existing_ids.update(row["artwork_id"] for row in existing_result.data)
        if len(existing_result.data) < page_size:
            break
        offset += page_size

    # Filter based on flags
    if only_existing:
        # Only paintings that already have opinions
        paintings_to_process = [aid for aid in all_artwork_ids if aid in existing_ids]
    elif force:
        # All paintings regardless of existing opinions
        paintings_to_process = all_artwork_ids
    else:
        # Only paintings without opinions
        paintings_to_process = [aid for aid in all_artwork_ids if aid not in existing_ids]

    if not paintings_to_process:
        return []

    # Apply limit and offset
    if limit:
        paintings_to_process = paintings_to_process[offset:offset + limit]

    # Get artwork details (batch to avoid URL length limits)
    artwork_map = {}
    batch_size = 100
    for i in range(0, len(paintings_to_process), batch_size):
        batch = paintings_to_process[i:i + batch_size]
        artworks_result = supabase.table("artworks").select("id, title, artist_name, image_url").in_("id", batch).execute()
        for a in artworks_result.data:
            artwork_map[a["id"]] = a

    # Maintain order
    return [artwork_map[aid] for aid in paintings_to_process if aid in artwork_map]


def save_opinion(supabase: Client, artwork_id: str, rating: int, explanation: str, question_id: str = "mirror_self") -> bool:
    """Save Claude's opinion to database."""
    try:
        supabase.table("claude_opinions").upsert(
            {
                "artwork_id": artwork_id,
                "question_id": question_id,
                "rating": rating,
                "explanation": explanation,
                "model": MODEL,
            },
            on_conflict="artwork_id,question_id"
        ).execute()
        return True
    except Exception as e:
        print(f"  Error saving opinion: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate Claude opinions for artworks")
    parser.add_argument("--limit", type=int, default=10, help="Number of paintings to process")
    parser.add_argument("--offset", type=int, default=0, help="Offset for pagination")
    parser.add_argument("--dry-run", action="store_true", help="Don't save to database")
    parser.add_argument("--prompt", choices=["v1", "v2"], default="v2", help="Prompt version (v1=abstract, v2=specific details)")
    parser.add_argument("--question-id", default="mirror_self", help="Question ID to save as (e.g., mirror_self_v1)")
    parser.add_argument("--force", action="store_true", help="Re-process paintings that already have opinions")
    parser.add_argument("--only-existing", action="store_true", help="Only process paintings that already have opinions")
    args = parser.parse_args()

    prompt = PROMPTS[args.prompt]

    print(f"Generating Claude opinions for artworks...")
    print(f"Model: {MODEL}")
    print(f"Prompt: {args.prompt}")
    print(f"Question ID: {args.question_id}")
    print(f"Limit: {args.limit}, Offset: {args.offset}")
    if args.force:
        print("Mode: Force (will overwrite existing)")
    elif args.only_existing:
        print("Mode: Only existing (re-running on paintings with opinions)")
    print()

    # Initialize clients
    supabase = get_supabase_client()
    claude = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    # Get paintings to process
    paintings = get_paintings_to_process(supabase, limit=args.limit, offset=args.offset, force=args.force, only_existing=args.only_existing, question_id=args.question_id)
    print(f"Found {len(paintings)} paintings to process")
    print()

    success_count = 0
    error_count = 0

    for i, painting in enumerate(paintings):
        artwork_id = painting["id"]
        title = painting["title"]
        artist = painting.get("artist_name", "Unknown")
        image_url = painting["image_url"]

        print(f"[{i+1}/{len(paintings)}] {title} by {artist}")
        print(f"  ID: {artwork_id}")

        # Fetch image
        image_result = fetch_image_as_base64(image_url)
        if not image_result:
            print("  SKIPPED: Could not fetch image")
            error_count += 1
            continue

        image_base64, media_type = image_result
        print(f"  Image fetched ({media_type})")

        # Get Claude's opinion
        opinion = get_claude_opinion(claude, image_base64, media_type, prompt)
        if not opinion:
            print("  SKIPPED: Could not get opinion")
            error_count += 1
            continue

        rating = opinion.get("rating")
        explanation = opinion.get("explanation")

        if not rating or not explanation:
            print(f"  SKIPPED: Invalid response - rating={rating}, explanation={bool(explanation)}")
            error_count += 1
            continue

        print(f"  Rating: {rating}/10")
        print(f"  Explanation: {explanation[:100]}...")

        # Save to database
        if not args.dry_run:
            if save_opinion(supabase, artwork_id, rating, explanation, args.question_id):
                print("  SAVED")
                success_count += 1
            else:
                error_count += 1
        else:
            print("  (dry run - not saved)")
            success_count += 1

        print()

        # Rate limiting - Claude has rate limits
        time.sleep(1)

    print("=" * 50)
    print(f"Complete! Success: {success_count}, Errors: {error_count}")


if __name__ == "__main__":
    main()
