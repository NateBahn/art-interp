#!/usr/bin/env python3
"""
Load rating correlations from all_correlations.json into Supabase sae_features table.

This reads the pre-computed correlations and updates the all_correlations column
in a format the frontend expects: {"mirror_self": 0.033, "wholeness": -0.15, ...}
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm

# Load environment
load_dotenv()

# Paths
ANALYSIS_DIR = Path(__file__).parent.parent / "output" / "sae_analysis"
CORRELATIONS_FILE = ANALYSIS_DIR / "all_correlations.json"


def main():
    print("=" * 60)
    print("Loading rating correlations to Supabase")
    print("=" * 60)

    # Connect to Supabase
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set")
        return

    supabase = create_client(url, key)

    # Load correlations
    print(f"\nLoading correlations from {CORRELATIONS_FILE}...")
    with open(CORRELATIONS_FILE) as f:
        data = json.load(f)

    all_correlations = data.get("all_correlations", {})
    print(f"  Found {len(all_correlations):,} features with correlations")

    # Prepare updates - flatten the correlation structure
    updates = []
    for feature_idx_str, feature_data in all_correlations.items():
        feature_idx = int(feature_idx_str)
        correlations = feature_data.get("correlations", {})

        # Flatten to just the correlation values
        flattened = {}
        for rating_name, rating_data in correlations.items():
            if isinstance(rating_data, dict) and "correlation" in rating_data:
                flattened[rating_name] = round(rating_data["correlation"], 6)
            else:
                flattened[rating_name] = rating_data

        if flattened:
            updates.append({
                "feature_idx": feature_idx,
                "all_correlations": json.dumps(flattened)
            })

    print(f"  Prepared {len(updates):,} updates")

    # Update in batches
    batch_size = 500
    updated_count = 0
    failed_count = 0

    print(f"\nUpdating database in batches of {batch_size}...")
    for i in tqdm(range(0, len(updates), batch_size)):
        batch = updates[i:i + batch_size]

        for update in batch:
            try:
                result = supabase.table("sae_features").update({
                    "all_correlations": update["all_correlations"]
                }).eq("feature_idx", update["feature_idx"]).execute()

                if result.data:
                    updated_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"  Updated: {updated_count:,}")
    print(f"  Failed: {failed_count:,}")
    print("Done!")


if __name__ == "__main__":
    main()
