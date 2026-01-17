"""Server configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")

# Database - use PostgreSQL if DATABASE_URL is set, otherwise SQLite
_env_database_url = os.getenv("DATABASE_URL")
if _env_database_url and _env_database_url.startswith("postgresql"):
    DATABASE_URL = _env_database_url
else:
    DATABASE_URL = "sqlite:///./data/art_interp.db"

# Data directories
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
ANALYSIS_DIR = DATA_DIR / "sae_analysis"
HEATMAPS_DIR = DATA_DIR / "heatmaps"
CLS_FEATURES_DIR = DATA_DIR / "sae_features"

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Feature types
FEATURE_TYPES = ["cls", "spatial"]

# Rating questions with descriptions
QUESTIONS = [
    {"id": "drawn_to", "label": "Drawn To", "description": "How drawn is the AI to this image?"},
    {"id": "emotional_impact", "label": "Emotional Impact", "description": "How emotionally impactful does the AI find this?"},
    {"id": "technical_skill", "label": "Technical Skill", "description": "How does the AI rate the technical execution?"},
    {"id": "choose_to_look", "label": "Choose to Look", "description": "Would the AI choose to look at this longer?"},
    {"id": "wholeness", "label": "Wholeness", "description": "How complete/unified does this feel?"},
    {"id": "inner_light", "label": "Inner Light", "description": "Does this image have an inner luminosity?"},
    {"id": "mirror_self", "label": "Mirror of Self", "description": "Does this reflect something meaningful?"},
    {"id": "deepest_honest", "label": "Deepest Honest", "description": "The most honest aesthetic response"},
]
