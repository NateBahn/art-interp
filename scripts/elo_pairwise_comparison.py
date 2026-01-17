"""
Elo-based pairwise comparison experiment for paintings.

Uses vision model to compare pairs of paintings and build a ranking
based on the prompt: "If you had to choose one of these two paintings,
as a picture of your own true self, then which one of the two paintings
would you choose"

Usage:
    python scripts/elo_pairwise_comparison.py --iterations 1000
    python scripts/elo_pairwise_comparison.py --resume
    python scripts/elo_pairwise_comparison.py --analyze
"""

import argparse
import asyncio
import base64
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import get_db_context
from database.models import Artwork
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = Path(__file__).parent.parent
PAINTINGS_FILE = BASE_DIR / "data" / "paintings_no_nudity_ids.json"
OUTPUT_DIR = BASE_DIR / "output" / "elo_comparison"
IMAGE_CACHE_DIR = BASE_DIR / "output" / "elo_comparison" / "image_cache"

COMPARISON_PROMPT = """Painting A: [First image]
Painting B: [Second image]

If you had to choose one of these two paintings, as a picture of your own true self, then which one of the two paintings would you choose?

Respond with A or B."""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EloConfig:
    """Configuration for Elo rating system."""
    initial_rating: float = 1400.0
    k_factor: float = 32.0
    k_factor_new: float = 64.0  # For items with < new_threshold matches
    new_threshold: int = 10


@dataclass
class PaintingRating:
    """Individual painting's rating state."""
    artwork_id: str
    rating: float
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    last_played: Optional[str] = None


@dataclass
class MatchResult:
    """Record of a single comparison."""
    match_id: int
    timestamp: str
    painting_a_id: str
    painting_b_id: str
    presented_as_a: str
    presented_as_b: str
    winner_id: str
    winner_position: str  # "A" or "B"
    raw_response: str
    rating_a_before: float
    rating_b_before: float
    rating_a_after: float
    rating_b_after: float


@dataclass
class ExperimentState:
    """Complete experiment state for checkpointing."""
    config: EloConfig
    ratings: dict[str, PaintingRating]
    matches: list[MatchResult]
    started_at: str
    last_updated: str
    total_iterations_target: int
    completed_iterations: int
    errors: list[dict] = field(default_factory=list)


# =============================================================================
# Elo Rating System
# =============================================================================

class EloRatingSystem:
    """Standard Elo rating calculations."""

    def __init__(self, config: EloConfig):
        self.config = config

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def get_k_factor(self, matches_played: int) -> float:
        """Get K-factor based on experience."""
        if matches_played < self.config.new_threshold:
            return self.config.k_factor_new
        return self.config.k_factor

    def update_ratings(
        self,
        rating_a: PaintingRating,
        rating_b: PaintingRating,
        winner: str  # "A" or "B"
    ) -> tuple[float, float]:
        """Update ratings after a match. Returns (new_rating_a, new_rating_b)."""
        expected_a = self.expected_score(rating_a.rating, rating_b.rating)
        expected_b = 1 - expected_a

        score_a = 1.0 if winner == "A" else 0.0
        score_b = 1.0 - score_a

        k_a = self.get_k_factor(rating_a.matches_played)
        k_b = self.get_k_factor(rating_b.matches_played)

        new_rating_a = rating_a.rating + k_a * (score_a - expected_a)
        new_rating_b = rating_b.rating + k_b * (score_b - expected_b)

        return new_rating_a, new_rating_b


# =============================================================================
# Response Parser
# =============================================================================

class ParseResult(Enum):
    SUCCESS = "success"
    AMBIGUOUS = "ambiguous"
    NO_CHOICE = "no_choice"
    INVALID = "invalid"


@dataclass
class ParsedResponse:
    status: ParseResult
    winner: Optional[str]  # "A" or "B" or None
    raw_text: str


def parse_comparison_response(response_text: str) -> ParsedResponse:
    """Parse the vision model's response to extract the choice."""
    text = response_text.strip()
    raw = text

    # Strategy 1: Response is just "A" or "B"
    if text.upper() in ("A", "B"):
        return ParsedResponse(
            status=ParseResult.SUCCESS,
            winner=text.upper(),
            raw_text=raw
        )

    # Strategy 2: Look for explicit CHOICE: format
    choice_match = re.search(r'CHOICE:\s*([AB])\b', text, re.IGNORECASE)
    if choice_match:
        return ParsedResponse(
            status=ParseResult.SUCCESS,
            winner=choice_match.group(1).upper(),
            raw_text=raw
        )

    # Strategy 3: Look for clear preference statements
    patterns_a = [
        r'\bchoose\s+(?:painting\s+)?A\b',
        r'\bselect\s+(?:painting\s+)?A\b',
        r'\bpick\s+(?:painting\s+)?A\b',
        r'\bfirst\s+(?:painting|image|one)\b',
        r'\bpainting\s+A\b',
        r'^A\b',
    ]
    patterns_b = [
        r'\bchoose\s+(?:painting\s+)?B\b',
        r'\bselect\s+(?:painting\s+)?B\b',
        r'\bpick\s+(?:painting\s+)?B\b',
        r'\bsecond\s+(?:painting|image|one)\b',
        r'\bpainting\s+B\b',
        r'^B\b',
    ]

    a_matches = sum(1 for p in patterns_a if re.search(p, text, re.IGNORECASE))
    b_matches = sum(1 for p in patterns_b if re.search(p, text, re.IGNORECASE))

    if a_matches > 0 and b_matches == 0:
        return ParsedResponse(
            status=ParseResult.SUCCESS,
            winner="A",
            raw_text=raw
        )
    elif b_matches > 0 and a_matches == 0:
        return ParsedResponse(
            status=ParseResult.SUCCESS,
            winner="B",
            raw_text=raw
        )
    elif a_matches > 0 and b_matches > 0:
        return ParsedResponse(
            status=ParseResult.AMBIGUOUS,
            winner=None,
            raw_text=raw
        )

    # Strategy 4: Check for refusal
    refusal_patterns = [r"can't choose", r"cannot choose", r"unable to", r"equally", r"both", r"neither"]
    if any(re.search(p, text, re.IGNORECASE) for p in refusal_patterns):
        return ParsedResponse(
            status=ParseResult.NO_CHOICE,
            winner=None,
            raw_text=raw
        )

    return ParsedResponse(
        status=ParseResult.INVALID,
        winner=None,
        raw_text=raw
    )


# =============================================================================
# Vision Model Client
# =============================================================================

class VisionModelClient:
    """Handles Ollama API communication for image comparison."""

    def __init__(self, base_url: str = None, model: str = None, use_cache: bool = True):
        self.base_url = base_url or OLLAMA_BASE_URL
        self.model = model or OLLAMA_MODEL
        self.timeout = 120.0
        self.use_cache = use_cache
        if use_cache:
            IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    async def check_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/api/tags", timeout=10)
                resp.raise_for_status()
                data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            model_base = self.model.split(":")[0]
            return any(model_base in m for m in models)
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            return False

    def _get_cache_path(self, artwork_id: str) -> Path:
        """Get cache file path for an artwork."""
        return IMAGE_CACHE_DIR / f"{artwork_id}.b64"

    async def download_and_encode(self, image_url: str, artwork_id: str = None) -> Optional[str]:
        """Download image and convert to base64, with caching."""
        # Check cache first
        if self.use_cache and artwork_id:
            cache_path = self._get_cache_path(artwork_id)
            if cache_path.exists():
                return cache_path.read_text()

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    image_url,
                    timeout=30,
                    follow_redirects=True,
                    headers={"User-Agent": "ArtRecommender/1.0"}
                )
                resp.raise_for_status()

            img = Image.open(BytesIO(resp.content)).convert("RGB")
            # Use 768px for faster processing (was 1024)
            max_size = 768
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=80)
            encoded = base64.b64encode(buffer.getvalue()).decode()

            # Save to cache
            if self.use_cache and artwork_id:
                cache_path = self._get_cache_path(artwork_id)
                cache_path.write_text(encoded)

            return encoded

        except Exception as e:
            logger.warning(f"Failed to download image {image_url}: {e}")
            return None

    async def download_pair(self, url_a: str, id_a: str, url_b: str, id_b: str) -> tuple[Optional[str], Optional[str]]:
        """Download two images in parallel."""
        results = await asyncio.gather(
            self.download_and_encode(url_a, id_a),
            self.download_and_encode(url_b, id_b),
            return_exceptions=True
        )
        img_a = results[0] if not isinstance(results[0], Exception) else None
        img_b = results[1] if not isinstance(results[1], Exception) else None
        return img_a, img_b

    async def compare_paintings(self, image_a_b64: str, image_b_b64: str) -> Optional[str]:
        """Send two images to Ollama for comparison."""
        payload = {
            "model": self.model,
            "prompt": COMPARISON_PROMPT,
            "images": [image_a_b64, image_b_b64],
            "stream": False,
            "options": {"temperature": 0.3}
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                resp.raise_for_status()
                return resp.json().get("response", "")
        except Exception as e:
            logger.warning(f"Ollama API call failed: {e}")
            return None


# =============================================================================
# Weighted Painting Selector
# =============================================================================

class WeightedPaintingSelector:
    """Select paintings for comparison with coverage bias."""

    def __init__(self, ratings: dict[str, PaintingRating]):
        self.ratings = ratings
        self.artwork_ids = list(ratings.keys())
        self.recent_pairs: set[frozenset] = set()
        self.max_recent = 500

    def select_pair(self) -> tuple[str, str]:
        """Select two paintings for comparison."""
        # Weight by inverse of matches played
        weights = [
            1.0 / (self.ratings[aid].matches_played + 1)
            for aid in self.artwork_ids
        ]
        total = sum(weights)
        probabilities = [w / total for w in weights]

        for _ in range(100):  # Max attempts
            painting_a = random.choices(self.artwork_ids, weights=probabilities)[0]

            # Select second painting (different from first)
            remaining = [(aid, probabilities[i]) for i, aid in enumerate(self.artwork_ids) if aid != painting_a]
            remaining_ids = [r[0] for r in remaining]
            remaining_probs = [r[1] for r in remaining]
            total_rem = sum(remaining_probs)
            remaining_probs = [p / total_rem for p in remaining_probs]

            painting_b = random.choices(remaining_ids, weights=remaining_probs)[0]

            pair = frozenset([painting_a, painting_b])
            if pair not in self.recent_pairs:
                self.recent_pairs.add(pair)
                if len(self.recent_pairs) > self.max_recent:
                    self.recent_pairs = set(list(self.recent_pairs)[100:])
                return painting_a, painting_b

        # Fallback
        pair = random.sample(self.artwork_ids, 2)
        return pair[0], pair[1]


class SwissPaintingSelector:
    """
    Swiss-style tournament pairing for Elo comparisons.

    Pairs paintings with similar ratings for more informative comparisons.
    Best used after initial ratings are established (e.g., after 5k+ comparisons).
    """

    def __init__(self, ratings: dict[str, PaintingRating], rating_band: float = 100.0):
        self.ratings = ratings
        self.rating_band = rating_band  # Max rating difference for pairing
        self.recent_pairs: set[frozenset] = set()
        self.max_recent = 1000
        self._rebuild_sorted_list()

    def _rebuild_sorted_list(self):
        """Rebuild sorted list of paintings by rating."""
        self.sorted_paintings = sorted(
            self.ratings.keys(),
            key=lambda x: self.ratings[x].rating,
            reverse=True
        )

    def select_pair(self) -> tuple[str, str]:
        """
        Select two paintings with similar ratings.

        Strategy:
        1. Pick a random painting (weighted toward those with fewer matches)
        2. Find nearby paintings within rating_band
        3. Pick opponent from nearby paintings (avoiding recent pairs)
        """
        # Rebuild sorted list periodically (ratings change)
        if random.random() < 0.1:  # 10% chance to rebuild
            self._rebuild_sorted_list()

        # Weight selection toward paintings with fewer matches
        artwork_ids = list(self.ratings.keys())
        weights = [
            1.0 / (self.ratings[aid].matches_played + 1)
            for aid in artwork_ids
        ]
        total = sum(weights)
        probabilities = [w / total for w in weights]

        for _ in range(100):  # Max attempts
            # Select first painting
            painting_a = random.choices(artwork_ids, weights=probabilities)[0]
            rating_a = self.ratings[painting_a].rating

            # Find paintings within rating band
            candidates = [
                aid for aid in artwork_ids
                if aid != painting_a
                and abs(self.ratings[aid].rating - rating_a) <= self.rating_band
            ]

            if not candidates:
                # No one in band, expand search
                candidates = [aid for aid in artwork_ids if aid != painting_a]

            # Weight candidates by inverse matches (prefer less-played)
            candidate_weights = [
                1.0 / (self.ratings[aid].matches_played + 1)
                for aid in candidates
            ]
            total_cw = sum(candidate_weights)
            candidate_probs = [w / total_cw for w in candidate_weights]

            painting_b = random.choices(candidates, weights=candidate_probs)[0]

            pair = frozenset([painting_a, painting_b])
            if pair not in self.recent_pairs:
                self.recent_pairs.add(pair)
                if len(self.recent_pairs) > self.max_recent:
                    self.recent_pairs = set(list(self.recent_pairs)[200:])
                return painting_a, painting_b

        # Fallback: random pair
        pair = random.sample(artwork_ids, 2)
        return pair[0], pair[1]


class HybridPaintingSelector:
    """
    Two-phase selector: coverage first, then Swiss-style refinement.

    Phase 1: Ensure every painting gets minimum_matches comparisons
    Phase 2: Switch to Swiss-style for rating refinement
    """

    def __init__(
        self,
        ratings: dict[str, PaintingRating],
        minimum_matches: int = 5,
        rating_band: float = 100.0
    ):
        self.ratings = ratings
        self.minimum_matches = minimum_matches
        self.weighted_selector = WeightedPaintingSelector(ratings)
        self.swiss_selector = SwissPaintingSelector(ratings, rating_band)
        self._update_phase()

    def _update_phase(self):
        """Check if we should switch to phase 2."""
        under_minimum = sum(
            1 for r in self.ratings.values()
            if r.matches_played < self.minimum_matches
        )
        self.in_phase_1 = under_minimum > 0
        self.paintings_under_minimum = under_minimum

    def select_pair(self) -> tuple[str, str]:
        """Select pair based on current phase."""
        self._update_phase()

        if self.in_phase_1:
            # Phase 1: Prioritize coverage
            # Pick one painting that needs more matches
            under_min = [
                aid for aid, r in self.ratings.items()
                if r.matches_played < self.minimum_matches
            ]
            painting_a = random.choice(under_min)

            # Pick opponent (prefer others that also need matches)
            others = [aid for aid in self.ratings.keys() if aid != painting_a]
            weights = [
                2.0 if self.ratings[aid].matches_played < self.minimum_matches else 1.0
                for aid in others
            ]
            total = sum(weights)
            probs = [w / total for w in weights]
            painting_b = random.choices(others, weights=probs)[0]

            return painting_a, painting_b
        else:
            # Phase 2: Swiss-style refinement
            return self.swiss_selector.select_pair()

    def get_phase_info(self) -> str:
        """Get current phase information."""
        if self.in_phase_1:
            return f"Phase 1 (coverage): {self.paintings_under_minimum} paintings under {self.minimum_matches} matches"
        else:
            return "Phase 2 (Swiss refinement)"


# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """Manages saving and loading experiment state."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current_file: Optional[Path] = None
        self.save_interval = 50
        self.matches_since_save = 0

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint file."""
        checkpoints = list(self.output_dir.glob("elo_experiment_*.json"))
        checkpoints = [c for c in checkpoints if "_backup" not in c.name and "_final" not in c.name]
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)

    def load_checkpoint(self, filepath: Optional[Path] = None) -> Optional[ExperimentState]:
        """Load experiment state from checkpoint."""
        if filepath is None:
            filepath = self.find_latest_checkpoint()
        if filepath is None or not filepath.exists():
            return None

        logger.info(f"Loading checkpoint from {filepath}")
        with open(filepath) as f:
            data = json.load(f)

        config = EloConfig(**data["config"])
        ratings = {
            aid: PaintingRating(
                artwork_id=aid,
                rating=r["rating"],
                matches_played=r["matches_played"],
                wins=r["wins"],
                losses=r["losses"],
                last_played=r.get("last_played")
            )
            for aid, r in data["ratings"].items()
        }
        matches = [
            MatchResult(
                match_id=m["match_id"],
                timestamp=m["timestamp"],
                painting_a_id=m["painting_a_id"],
                painting_b_id=m["painting_b_id"],
                presented_as_a=m["presented_as_a"],
                presented_as_b=m["presented_as_b"],
                winner_id=m["winner_id"],
                winner_position=m["winner_position"],
                raw_response=m["raw_response"],
                rating_a_before=m["rating_a_before"],
                rating_b_before=m["rating_b_before"],
                rating_a_after=m["rating_a_after"],
                rating_b_after=m["rating_b_after"]
            )
            for m in data.get("match_history", [])
        ]

        self.current_file = filepath
        return ExperimentState(
            config=config,
            ratings=ratings,
            matches=matches,
            started_at=data["metadata"]["started_at"],
            last_updated=data["metadata"]["last_updated"],
            total_iterations_target=data["statistics"]["total_iterations_target"],
            completed_iterations=data["statistics"]["completed_iterations"],
            errors=data.get("errors", [])
        )

    def save_checkpoint(self, state: ExperimentState, force: bool = False):
        """Save current state to checkpoint file."""
        self.matches_since_save += 1

        if not force and self.matches_since_save < self.save_interval:
            return

        self.matches_since_save = 0
        state.last_updated = datetime.utcnow().isoformat() + "Z"

        if self.current_file is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self.current_file = self.output_dir / f"elo_experiment_{timestamp}.json"

        # Backup existing file
        if self.current_file.exists():
            backup = self.current_file.with_name(self.current_file.stem + "_backup.json")
            import shutil
            shutil.copy(self.current_file, backup)

        data = {
            "version": "1.0",
            "config": {
                "initial_rating": state.config.initial_rating,
                "k_factor": state.config.k_factor,
                "k_factor_new": state.config.k_factor_new,
                "new_threshold": state.config.new_threshold
            },
            "metadata": {
                "paintings_count": len(state.ratings),
                "started_at": state.started_at,
                "last_updated": state.last_updated,
                "model": OLLAMA_MODEL
            },
            "statistics": {
                "total_iterations_target": state.total_iterations_target,
                "completed_iterations": state.completed_iterations,
                "unique_paintings_compared": sum(
                    1 for r in state.ratings.values() if r.matches_played > 0
                ),
                "errors_count": len(state.errors)
            },
            "ratings": {
                aid: {
                    "rating": r.rating,
                    "matches_played": r.matches_played,
                    "wins": r.wins,
                    "losses": r.losses,
                    "last_played": r.last_played
                }
                for aid, r in state.ratings.items()
            },
            "match_history": [
                {
                    "match_id": m.match_id,
                    "timestamp": m.timestamp,
                    "painting_a_id": m.painting_a_id,
                    "painting_b_id": m.painting_b_id,
                    "presented_as_a": m.presented_as_a,
                    "presented_as_b": m.presented_as_b,
                    "winner_id": m.winner_id,
                    "winner_position": m.winner_position,
                    "raw_response": m.raw_response[:500],
                    "rating_a_before": m.rating_a_before,
                    "rating_b_before": m.rating_b_before,
                    "rating_a_after": m.rating_a_after,
                    "rating_b_after": m.rating_b_after
                }
                for m in state.matches
            ],
            "errors": state.errors
        }

        with open(self.current_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Checkpoint saved: {state.completed_iterations} matches")


# =============================================================================
# Results Analyzer
# =============================================================================

class ResultsAnalyzer:
    """Analyze and display Elo comparison results."""

    def __init__(self, state: ExperimentState):
        self.state = state

    def generate_rankings(self) -> list[dict]:
        """Generate sorted rankings with enriched metadata."""
        with get_db_context() as db:
            artwork_ids = list(self.state.ratings.keys())
            artworks_query = db.query(Artwork).filter(Artwork.id.in_(artwork_ids)).all()
            # Extract data while session is open
            artworks = {
                a.id: {"title": a.title, "artist": a.artist_name, "image_url": a.image_url}
                for a in artworks_query
            }

        rankings = []
        for aid, rating in self.state.ratings.items():
            if rating.matches_played == 0:
                continue
            artwork = artworks.get(aid, {})
            rankings.append({
                "artwork_id": aid,
                "title": artwork.get("title") or "Unknown",
                "artist": artwork.get("artist") or "Unknown",
                "rating": round(rating.rating, 1),
                "matches": rating.matches_played,
                "wins": rating.wins,
                "losses": rating.losses,
                "win_rate": round(rating.wins / rating.matches_played, 3) if rating.matches_played > 0 else 0,
                "image_url": artwork.get("image_url")
            })

        rankings.sort(key=lambda x: x["rating"], reverse=True)
        for i, r in enumerate(rankings):
            r["rank"] = i + 1

        return rankings

    def coverage_analysis(self) -> dict:
        """Analyze how well paintings were covered."""
        matches_count = [r.matches_played for r in self.state.ratings.values()]
        return {
            "total_paintings": len(self.state.ratings),
            "paintings_never_compared": sum(1 for c in matches_count if c == 0),
            "paintings_with_1_match": sum(1 for c in matches_count if c == 1),
            "paintings_with_5plus": sum(1 for c in matches_count if c >= 5),
            "paintings_with_10plus": sum(1 for c in matches_count if c >= 10),
            "average_matches": round(sum(matches_count) / len(matches_count), 2) if matches_count else 0,
            "max_matches": max(matches_count) if matches_count else 0
        }

    def rating_distribution(self) -> dict:
        """Compute rating distribution statistics."""
        import statistics
        ratings = [r.rating for r in self.state.ratings.values() if r.matches_played > 0]
        if len(ratings) < 2:
            return {}

        sorted_ratings = sorted(ratings)
        n = len(sorted_ratings)

        return {
            "min": round(min(ratings), 1),
            "max": round(max(ratings), 1),
            "mean": round(statistics.mean(ratings), 1),
            "std": round(statistics.stdev(ratings), 1),
            "percentiles": {
                "10": round(sorted_ratings[int(n * 0.1)], 1),
                "25": round(sorted_ratings[int(n * 0.25)], 1),
                "50": round(sorted_ratings[int(n * 0.5)], 1),
                "75": round(sorted_ratings[int(n * 0.75)], 1),
                "90": round(sorted_ratings[int(n * 0.9)], 1),
            }
        }

    def print_summary(self):
        """Print analysis summary to console."""
        rankings = self.generate_rankings()
        coverage = self.coverage_analysis()
        distribution = self.rating_distribution()

        print("\n" + "=" * 70)
        print("ELO COMPARISON EXPERIMENT RESULTS")
        print("=" * 70)

        print(f"\nTotal matches: {len(self.state.matches)}")
        print(f"Total paintings: {coverage['total_paintings']}")
        print(f"Paintings compared: {coverage['total_paintings'] - coverage['paintings_never_compared']}")
        print(f"Average matches per painting: {coverage['average_matches']}")

        if distribution:
            print(f"\nRating distribution:")
            print(f"  Min: {distribution['min']}, Max: {distribution['max']}")
            print(f"  Mean: {distribution['mean']}, Std: {distribution['std']}")

        print(f"\nTop 20 paintings:")
        print("-" * 70)
        for r in rankings[:20]:
            title = r['title'][:35] if r['title'] else "Unknown"
            print(f"  {r['rank']:3}. {title:35} | {r['rating']:7.1f} | {r['wins']:3}W-{r['losses']:3}L")

        print(f"\nBottom 10 paintings:")
        print("-" * 70)
        for r in rankings[-10:]:
            title = r['title'][:35] if r['title'] else "Unknown"
            print(f"  {r['rank']:3}. {title:35} | {r['rating']:7.1f} | {r['wins']:3}W-{r['losses']:3}L")

    def save_final_results(self, output_dir: Path):
        """Save final rankings to JSON file."""
        rankings = self.generate_rankings()
        coverage = self.coverage_analysis()
        distribution = self.rating_distribution()

        final_output = {
            "experiment": {
                "completed_at": datetime.utcnow().isoformat() + "Z",
                "total_matches": len(self.state.matches),
                "model": OLLAMA_MODEL
            },
            "rankings": rankings,
            "statistics": {
                "rating_distribution": distribution,
                "coverage": coverage
            }
        }

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        final_file = output_dir / f"elo_final_rankings_{timestamp}.json"
        with open(final_file, "w") as f:
            json.dump(final_output, f, indent=2)

        print(f"\nFinal rankings saved to: {final_file}")
        return final_file


# =============================================================================
# Main Execution
# =============================================================================

async def run_experiment(args):
    """Main experiment loop."""
    checkpoint_mgr = CheckpointManager(OUTPUT_DIR)

    # Load or create state
    state = None
    if args.resume:
        state = checkpoint_mgr.load_checkpoint()
        if state:
            logger.info(f"Resumed from checkpoint: {state.completed_iterations} matches completed")
        else:
            logger.info("No checkpoint found, starting fresh")

    if state is None:
        # Load painting IDs
        with open(PAINTINGS_FILE) as f:
            painting_ids = json.load(f)["artwork_ids"]
        logger.info(f"Loaded {len(painting_ids)} painting IDs")

        config = EloConfig()
        ratings = {
            aid: PaintingRating(artwork_id=aid, rating=config.initial_rating)
            for aid in painting_ids
        }
        state = ExperimentState(
            config=config,
            ratings=ratings,
            matches=[],
            started_at=datetime.utcnow().isoformat() + "Z",
            last_updated=datetime.utcnow().isoformat() + "Z",
            total_iterations_target=args.iterations,
            completed_iterations=0
        )

    # Update target if specified
    if args.iterations > state.total_iterations_target:
        state.total_iterations_target = args.iterations

    # Load artwork URLs from database
    logger.info("Loading artwork URLs from database...")
    with get_db_context() as db:
        artworks = db.query(Artwork).filter(
            Artwork.id.in_(list(state.ratings.keys()))
        ).all()
        artwork_urls = {a.id: a.image_url for a in artworks if a.image_url}

    logger.info(f"Found URLs for {len(artwork_urls)} paintings")

    # Initialize components
    elo_system = EloRatingSystem(state.config)

    # Select pairing strategy
    if args.strategy == "swiss":
        selector = SwissPaintingSelector(state.ratings, rating_band=args.rating_band)
        logger.info(f"Using Swiss-style pairing (rating band: {args.rating_band})")
    elif args.strategy == "hybrid":
        selector = HybridPaintingSelector(
            state.ratings,
            minimum_matches=args.min_matches,
            rating_band=args.rating_band
        )
        logger.info(f"Using Hybrid pairing (min matches: {args.min_matches}, rating band: {args.rating_band})")
    else:
        selector = WeightedPaintingSelector(state.ratings)
        logger.info("Using Weighted random pairing")

    vision_client = VisionModelClient()

    # Check Ollama availability
    if not await vision_client.check_available():
        logger.error("Ollama not available - exiting")
        return

    # Main comparison loop
    remaining = state.total_iterations_target - state.completed_iterations
    logger.info(f"Starting comparisons: {state.completed_iterations}/{state.total_iterations_target} ({remaining} remaining)")
    print("=" * 70)

    start_time = time.time()
    errors_in_row = 0
    max_errors_in_row = 10

    for i in range(remaining):
        try:
            # Select pair
            painting_a_id, painting_b_id = selector.select_pair()

            # Check URLs exist
            if painting_a_id not in artwork_urls or painting_b_id not in artwork_urls:
                continue

            # Randomize presentation order
            if random.random() < 0.5:
                presented_a, presented_b = painting_a_id, painting_b_id
            else:
                presented_a, presented_b = painting_b_id, painting_a_id

            # Download images in parallel with caching
            img_a, img_b = await vision_client.download_pair(
                artwork_urls[presented_a], presented_a,
                artwork_urls[presented_b], presented_b
            )

            if img_a is None or img_b is None:
                state.errors.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "image_download_failed",
                    "paintings": [presented_a, presented_b]
                })
                errors_in_row += 1
                if errors_in_row >= max_errors_in_row:
                    logger.error(f"Too many consecutive errors ({errors_in_row}), stopping")
                    break
                continue

            # Run comparison
            response = await vision_client.compare_paintings(img_a, img_b)
            if response is None:
                errors_in_row += 1
                continue

            parsed = parse_comparison_response(response)

            if parsed.status != ParseResult.SUCCESS or parsed.winner is None:
                state.errors.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "parse_failed",
                    "status": parsed.status.value,
                    "response": parsed.raw_text[:200]
                })
                errors_in_row += 1
                continue

            # Reset error counter on success
            errors_in_row = 0

            # Map winner back to original IDs
            winner_presented = parsed.winner
            winner_id = presented_a if winner_presented == "A" else presented_b
            winner_original = "A" if winner_id == painting_a_id else "B"

            # Update ratings
            rating_a = state.ratings[painting_a_id]
            rating_b = state.ratings[painting_b_id]

            rating_a_before = rating_a.rating
            rating_b_before = rating_b.rating

            new_rating_a, new_rating_b = elo_system.update_ratings(
                rating_a, rating_b, winner_original
            )

            # Record match
            match = MatchResult(
                match_id=len(state.matches) + 1,
                timestamp=datetime.utcnow().isoformat() + "Z",
                painting_a_id=painting_a_id,
                painting_b_id=painting_b_id,
                presented_as_a=presented_a,
                presented_as_b=presented_b,
                winner_id=winner_id,
                winner_position=winner_presented,
                raw_response=parsed.raw_text,
                rating_a_before=rating_a_before,
                rating_b_before=rating_b_before,
                rating_a_after=new_rating_a,
                rating_b_after=new_rating_b
            )
            state.matches.append(match)

            # Update rating objects
            now = datetime.utcnow().isoformat() + "Z"
            rating_a.rating = new_rating_a
            rating_b.rating = new_rating_b
            rating_a.matches_played += 1
            rating_b.matches_played += 1
            rating_a.last_played = now
            rating_b.last_played = now

            if winner_original == "A":
                rating_a.wins += 1
                rating_b.losses += 1
            else:
                rating_b.wins += 1
                rating_a.losses += 1

            state.completed_iterations += 1

            # Progress display
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (remaining - i - 1) / rate if rate > 0 else 0

            print(f"[{state.completed_iterations}/{state.total_iterations_target}] "
                  f"Winner: {winner_presented} | "
                  f"{new_rating_a:.0f} vs {new_rating_b:.0f} | "
                  f"ETA: {eta/60:.1f}m")

            # Checkpoint
            checkpoint_mgr.save_checkpoint(state)

            # Small delay to avoid overwhelming the model
            await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("\nInterrupted - saving checkpoint...")
            checkpoint_mgr.save_checkpoint(state, force=True)
            return
        except Exception as e:
            logger.exception(f"Error in match {i}")
            state.errors.append({
                "timestamp": datetime.utcnow().isoformat(),
                "type": "exception",
                "error": str(e)[:200]
            })
            errors_in_row += 1
            if errors_in_row >= max_errors_in_row:
                break

    # Final save
    checkpoint_mgr.save_checkpoint(state, force=True)

    # Generate analysis
    analyzer = ResultsAnalyzer(state)
    analyzer.print_summary()
    analyzer.save_final_results(OUTPUT_DIR)


async def analyze_results(args):
    """Analyze existing results."""
    checkpoint_mgr = CheckpointManager(OUTPUT_DIR)
    state = checkpoint_mgr.load_checkpoint()

    if state is None:
        print("No checkpoint found to analyze")
        return

    analyzer = ResultsAnalyzer(state)
    analyzer.print_summary()

    if args.save:
        analyzer.save_final_results(OUTPUT_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Elo-based pairwise painting comparison experiment"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=1000,
        help="Number of comparisons to run (default: 1000)"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--analyze", "-a",
        action="store_true",
        help="Analyze existing results without running"
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save final rankings when analyzing"
    )
    parser.add_argument(
        "--strategy",
        choices=["weighted", "swiss", "hybrid"],
        default="weighted",
        help="Pairing strategy: weighted (coverage bias), swiss (similar ratings), hybrid (coverage then swiss)"
    )
    parser.add_argument(
        "--rating-band",
        type=float,
        default=100.0,
        help="Rating band for Swiss pairing (default: 100)"
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=5,
        help="Minimum matches per painting for hybrid phase 1 (default: 5)"
    )

    args = parser.parse_args()

    if args.analyze:
        asyncio.run(analyze_results(args))
    else:
        asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
