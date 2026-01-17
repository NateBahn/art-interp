#!/usr/bin/env python3
"""
Compare different prompts for Claude's "Mirror of Self" opinions.

Runs a new prompt on paintings that already have opinions and compares the results.
"""

import os
import json
import base64
import re
import random

import anthropic
import httpx
from supabase import create_client, Client

# Configuration
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

MODEL = "claude-3-5-haiku-20241022"

# Original prompt
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

# New prompt - asks for specific visual details
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


def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_image_as_base64(url: str) -> tuple[str, str] | None:
    try:
        response = httpx.get(url, timeout=30, follow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "image/jpeg")
        if "png" in content_type:
            media_type = "image/png"
        elif "gif" in content_type:
            media_type = "image/gif"
        elif "webp" in content_type:
            media_type = "image/webp"
        else:
            media_type = "image/jpeg"
        image_data = base64.standard_b64encode(response.content).decode("utf-8")
        return image_data, media_type
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return None


def get_claude_opinion(client: anthropic.Anthropic, image_base64: str, media_type: str, prompt: str) -> dict | None:
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
        response_text = message.content[0].text
        json_match = re.search(r'\{[^{}]*"rating"[^{}]*"explanation"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            print(f"  Could not parse: {response_text[:100]}")
            return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    print("Comparing prompts for Claude's Mirror of Self opinions")
    print("=" * 60)
    print()

    supabase = get_supabase_client()
    claude = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    # Get paintings that already have opinions
    result = supabase.table("claude_opinions").select("artwork_id, rating, explanation").eq("question_id", "mirror_self").execute()
    existing_opinions = {row["artwork_id"]: row for row in result.data}

    # Get artwork details for these
    artwork_ids = list(existing_opinions.keys())
    artworks_result = supabase.table("artworks").select("id, title, artist_name, image_url").in_("id", artwork_ids).execute()
    artworks = {a["id"]: a for a in artworks_result.data}

    # Pick 10 random ones
    sample_ids = random.sample(artwork_ids, min(10, len(artwork_ids)))

    for i, artwork_id in enumerate(sample_ids):
        artwork = artworks[artwork_id]
        old_opinion = existing_opinions[artwork_id]

        print(f"[{i+1}/10] {artwork['title']}")
        print(f"        by {artwork.get('artist_name', 'Unknown')}")
        print("-" * 60)

        # Fetch image
        image_result = fetch_image_as_base64(artwork["image_url"])
        if not image_result:
            print("  SKIPPED: Could not fetch image")
            print()
            continue

        image_base64, media_type = image_result

        # Get new opinion with v2 prompt
        new_opinion = get_claude_opinion(claude, image_base64, media_type, PROMPT_V2)
        if not new_opinion:
            print("  SKIPPED: Could not get new opinion")
            print()
            continue

        # Compare
        print(f"  OLD (v1): {old_opinion['rating']}/10")
        print(f"    {old_opinion['explanation'][:200]}...")
        print()
        print(f"  NEW (v2): {new_opinion['rating']}/10")
        print(f"    {new_opinion['explanation'][:200]}...")
        print()

        rating_diff = new_opinion['rating'] - old_opinion['rating']
        if rating_diff == 0:
            print(f"  DIFF: Same rating")
        elif rating_diff > 0:
            print(f"  DIFF: New is +{rating_diff} higher")
        else:
            print(f"  DIFF: New is {rating_diff} lower")

        print()
        print()

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
