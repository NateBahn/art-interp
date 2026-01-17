"""
art-interp API server.

Provides data for the "What Does AI See in Art?" interactive explorer.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.config import (
    ANALYSIS_DIR,
    CLS_FEATURES_DIR,
    CORS_ORIGINS,
    FEATURE_TYPES,
    HEATMAPS_DIR,
    HOST,
    PORT,
    QUESTIONS,
)
from server.database import ArtworkMeta, get_db, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# JSON data caches
_cls_correlations_data = None
_spatial_correlations_data = None
_feature_labels = None
_spatial_labels = None
_artwork_cls_features = None
_features_to_label = None
_heatmaps_index = None


def load_json_file(filename: str) -> dict | None:
    """Load a JSON file from the analysis directory."""
    filepath = ANALYSIS_DIR / filename
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


def get_correlations_data(feature_type: str = "spatial"):
    """Get correlations data for the specified feature type."""
    global _cls_correlations_data, _spatial_correlations_data

    if feature_type == "cls":
        if _cls_correlations_data is None:
            _cls_correlations_data = load_json_file("all_correlations.json") or {}
        return _cls_correlations_data
    else:
        if _spatial_correlations_data is None:
            _spatial_correlations_data = load_json_file("spatial_correlations.json")
            if _spatial_correlations_data is None:
                _spatial_correlations_data = load_json_file("all_correlations.json") or {}
        return _spatial_correlations_data


def get_feature_labels():
    """Get CLS feature labels from Gemini."""
    global _feature_labels
    if _feature_labels is None:
        _feature_labels = load_json_file("feature_labels_gemini.json") or {}
    return _feature_labels


def get_spatial_labels():
    """Get spatial feature labels from Gemini (merged from all spatial_labels_*.json files)."""
    global _spatial_labels
    if _spatial_labels is None:
        _spatial_labels = {"labels": []}
        for label_file in sorted(ANALYSIS_DIR.glob("spatial_labels_*.json")):
            try:
                with open(label_file) as f:
                    data = json.load(f)
                    if "labels" in data:
                        _spatial_labels["labels"].extend(data["labels"])
            except (json.JSONDecodeError, IOError):
                pass
    return _spatial_labels


def get_labels_for_feature_type(feature_type: str = "spatial"):
    """Get labels for the specified feature type."""
    if feature_type == "spatial":
        return get_spatial_labels()
    return get_feature_labels()


def get_artwork_cls_features():
    """Load CLS feature activations for all artworks from batch files."""
    global _artwork_cls_features
    if _artwork_cls_features is None:
        _artwork_cls_features = {}
        if CLS_FEATURES_DIR.exists():
            for batch_file in sorted(CLS_FEATURES_DIR.glob("batch_*.json")):
                try:
                    with open(batch_file) as f:
                        batch_data = json.load(f)
                        _artwork_cls_features.update(batch_data)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to load {batch_file}: {e}")
    return _artwork_cls_features


def get_features_to_label():
    """Get features selected for labeling (includes top artwork IDs)."""
    global _features_to_label
    if _features_to_label is None:
        _features_to_label = load_json_file("features_to_label.json") or {}
    return _features_to_label


def get_heatmaps_index():
    """Get heatmaps index."""
    global _heatmaps_index
    if _heatmaps_index is None:
        index_path = HEATMAPS_DIR / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                _heatmaps_index = json.load(f)
        else:
            _heatmaps_index = {"paintings": [], "total": 0}
    return _heatmaps_index


# Pydantic models
class PaintingBasic(BaseModel):
    id: str
    title: str
    artist: str | None
    year: int | None
    image_url: str
    thumbnail_url: str | None

    class Config:
        from_attributes = True


class PaintingDetail(PaintingBasic):
    labels: dict | None
    description: str | None


class QuestionInfo(BaseModel):
    id: str
    label: str
    description: str


class FeatureCorrelation(BaseModel):
    feature_idx: int
    label: str | None
    correlation: float
    p_value: float | None
    monosemanticity: float | None
    activation: float | None = None


class PaintingFeatures(BaseModel):
    painting_id: str
    question_id: str
    features: list[FeatureCorrelation]
    painting_specific: bool = False


class HeatmapFeature(BaseModel):
    feature_idx: int
    activation: float


class FeatureHeatmap(BaseModel):
    feature_idx: int
    heatmap_7x7: list[list[float]]
    max_activation: float
    mean_activation: float
    total_activation: float


class PaintingHeatmapData(BaseModel):
    painting_id: str
    title: str
    artist: str | None
    year: int | None
    image_url: str
    top_active_features: list[HeatmapFeature]
    available_features: list[int]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown."""
    logger.info("Starting art-interp server...")
    init_db()
    logger.info("Server ready!")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="art-interp API",
    description="API for SAE-based art interpretability explorer",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for heatmaps
if HEATMAPS_DIR.exists():
    app.mount("/static/heatmaps", StaticFiles(directory=str(HEATMAPS_DIR)), name="heatmaps")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "art-interp"}


@app.get("/api/interp/paintings", response_model=list[PaintingBasic])
def list_paintings(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    has_labels: bool = Query(True, description="Only return paintings with AI labels"),
    db: Session = Depends(get_db),
):
    """List paintings for the gallery view."""
    query = db.query(ArtworkMeta)

    if has_labels:
        query = query.filter(ArtworkMeta.labels.isnot(None))

    query = query.filter(ArtworkMeta.image_url.isnot(None))
    query = query.order_by(ArtworkMeta.id)

    paintings = query.offset(offset).limit(limit).all()

    return [
        PaintingBasic(
            id=p.id,
            title=p.title,
            artist=p.artist_name,
            year=p.year,
            image_url=p.image_url,
            thumbnail_url=p.thumbnail_url,
        )
        for p in paintings
    ]


@app.get("/api/interp/paintings/{painting_id}", response_model=PaintingDetail)
def get_painting(painting_id: str, db: Session = Depends(get_db)):
    """Get detailed info about a specific painting."""
    painting = db.query(ArtworkMeta).filter(ArtworkMeta.id == painting_id).first()
    if not painting:
        raise HTTPException(status_code=404, detail="Painting not found")

    labels = None
    if painting.labels:
        try:
            labels = json.loads(painting.labels)
        except json.JSONDecodeError:
            pass

    return PaintingDetail(
        id=painting.id,
        title=painting.title,
        artist=painting.artist_name,
        year=painting.year,
        image_url=painting.image_url,
        thumbnail_url=painting.thumbnail_url,
        labels=labels,
        description=painting.description,
    )


@app.get("/api/interp/questions", response_model=list[QuestionInfo])
def list_questions():
    """List available rating questions/dimensions."""
    return [QuestionInfo(**q) for q in QUESTIONS]


@app.get("/api/interp/paintings/{painting_id}/features", response_model=PaintingFeatures)
def get_painting_features(
    painting_id: str,
    question: str = Query(..., description="Question ID (e.g., 'drawn_to')"),
    limit: int = Query(10, ge=1, le=50),
    feature_type: str = Query("cls", description="SAE type: 'cls' (global) or 'spatial' (local)"),
    db: Session = Depends(get_db),
):
    """Get SAE features correlated with a specific rating question for a painting."""
    painting = db.query(ArtworkMeta).filter(ArtworkMeta.id == painting_id).first()
    if not painting:
        raise HTTPException(status_code=404, detail="Painting not found")

    valid_questions = {q["id"] for q in QUESTIONS}
    if question not in valid_questions:
        raise HTTPException(status_code=400, detail=f"Invalid question. Must be one of: {valid_questions}")

    if feature_type not in FEATURE_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid feature_type. Must be one of: {FEATURE_TYPES}")

    correlations_data = get_correlations_data("cls")
    feature_labels = get_labels_for_feature_type(feature_type)
    all_correlations = correlations_data.get("all_correlations", {})

    artwork_features = get_artwork_cls_features()
    painting_features = artwork_features.get(painting_id)
    painting_specific = painting_features is not None

    if painting_specific:
        feature_correlations = []

        for feat_idx_str, activation in painting_features.items():
            feat_idx = int(feat_idx_str)
            feat_corr_data = all_correlations.get(feat_idx_str, {})
            if not feat_corr_data:
                continue

            correlations = feat_corr_data.get("correlations", {})
            question_corr = correlations.get(question, {})
            correlation_value = question_corr.get("correlation", 0)
            p_value = question_corr.get("p_value")

            if abs(correlation_value) >= 0.05:
                feature_correlations.append({
                    "feature_idx": feat_idx,
                    "correlation": correlation_value,
                    "p_value": p_value,
                    "activation": activation,
                })

        feature_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        features = []
        for feat in feature_correlations[:limit]:
            feat_idx = feat["feature_idx"]

            label = None
            monosemanticity = None
            if "labels" in feature_labels:
                for labeled_feat in feature_labels["labels"]:
                    if labeled_feat.get("feature_idx") == feat_idx:
                        label = labeled_feat.get("short_label")
                        monosemanticity = labeled_feat.get("monosemanticity_score")
                        break

            features.append(FeatureCorrelation(
                feature_idx=feat_idx,
                label=label,
                correlation=feat["correlation"],
                p_value=feat["p_value"],
                monosemanticity=monosemanticity,
                activation=feat["activation"],
            ))
    else:
        top_features = correlations_data.get("top_features_by_rating", {}).get(question, [])

        features = []
        for feat in top_features[:limit]:
            feat_idx = feat["feature_idx"]

            label = None
            if "labels" in feature_labels:
                for labeled_feat in feature_labels["labels"]:
                    if labeled_feat.get("feature_idx") == feat_idx:
                        label = labeled_feat.get("short_label")
                        break

            features.append(FeatureCorrelation(
                feature_idx=feat_idx,
                label=label,
                correlation=feat["correlation"],
                p_value=feat.get("p_value"),
                monosemanticity=feat.get("monosemanticity"),
                activation=None,
            ))

    return PaintingFeatures(
        painting_id=painting_id,
        question_id=question,
        features=features,
        painting_specific=painting_specific,
    )


@app.get("/api/interp/features/{feature_idx}")
def get_feature_info(feature_idx: int, db: Session = Depends(get_db)):
    """Get detailed info about a specific SAE feature."""
    feature_labels = get_feature_labels()
    correlations_data = get_correlations_data("cls")
    features_to_label = get_features_to_label()

    label_info = None
    if "labels" in feature_labels:
        for feat in feature_labels["labels"]:
            if feat.get("feature_idx") == feature_idx:
                label_info = feat
                break

    feature_data = None
    for feat in features_to_label.get("features", []):
        if feat.get("feature_idx") == feature_idx:
            feature_data = feat
            break

    correlation_info = correlations_data.get("all_correlations", {}).get(str(feature_idx))

    if not label_info and not correlation_info and not feature_data:
        raise HTTPException(status_code=404, detail="Feature not found")

    top_artwork_ids = feature_data.get("top_5_artwork_ids", []) if feature_data else []

    if not top_artwork_ids:
        cls_features = get_artwork_cls_features()
        if cls_features:
            feature_key = str(feature_idx)
            artwork_activations = []
            for artwork_id, features in cls_features.items():
                if feature_key in features:
                    artwork_activations.append((artwork_id, features[feature_key]))
            artwork_activations.sort(key=lambda x: x[1], reverse=True)
            top_artwork_ids = [a[0] for a in artwork_activations[:5]]

    top_artworks = []
    if top_artwork_ids:
        artworks = db.query(ArtworkMeta).filter(ArtworkMeta.id.in_(top_artwork_ids)).all()
        artwork_map = {a.id: a for a in artworks}
        for art_id in top_artwork_ids:
            if art_id in artwork_map:
                a = artwork_map[art_id]
                top_artworks.append({
                    "id": a.id,
                    "title": a.title,
                    "artist": a.artist_name,
                    "image_url": a.image_url,
                    "thumbnail_url": a.thumbnail_url,
                })

    all_correlations = {}
    if correlation_info and "correlations" in correlation_info:
        for rating_id, corr_data in correlation_info["correlations"].items():
            all_correlations[rating_id] = {
                "correlation": corr_data["correlation"],
                "p_value": corr_data["p_value"],
            }

    return {
        "feature_idx": feature_idx,
        "label": label_info.get("short_label") if label_info else None,
        "description": label_info.get("description") if label_info else None,
        "visual_elements": label_info.get("visual_elements") if label_info else None,
        "explains_rating": label_info.get("explains_rating") if label_info else None,
        "tier": feature_data.get("tier") if feature_data else None,
        "monosemanticity": feature_data.get("monosemanticity_score") if feature_data else (
            label_info.get("monosemanticity_score") if label_info else None
        ),
        "strongest_rating": feature_data.get("strongest_rating") if feature_data else (
            label_info.get("strongest_rating") if label_info else None
        ),
        "strongest_correlation": feature_data.get("strongest_correlation") if feature_data else (
            label_info.get("strongest_correlation") if label_info else None
        ),
        "all_correlations": all_correlations,
        "top_artworks": top_artworks,
    }


@app.get("/api/interp/feature-types")
def get_feature_types():
    """Get available SAE feature types."""
    cls_data = get_correlations_data("cls")
    spatial_file = ANALYSIS_DIR / "spatial_correlations.json"
    spatial_available = spatial_file.exists()

    return {
        "types": [
            {
                "id": "spatial",
                "label": "Spatial (Local Patterns)",
                "description": "Features trained on image patches. Supports heatmaps showing WHERE features activate.",
                "available": True,
                "has_correlations": spatial_available,
                "supports_heatmaps": True,
            },
            {
                "id": "cls",
                "label": "CLS (Global)",
                "description": "Features trained on CLS token. Captures global image characteristics.",
                "available": bool(cls_data.get("top_features_by_rating")),
                "has_correlations": bool(cls_data.get("top_features_by_rating")),
                "supports_heatmaps": False,
            },
        ],
        "default": "spatial",
        "note": "Spatial is recommended as it provides consistent correlations and heatmaps.",
    }


@app.get("/api/interp/stats")
def get_stats(
    feature_type: str = Query("spatial", description="SAE type: 'cls' or 'spatial'"),
    db: Session = Depends(get_db),
):
    """Get statistics about the interpretability data."""
    correlations_data = get_correlations_data(feature_type)
    feature_labels = get_feature_labels()
    heatmaps_index = get_heatmaps_index()

    labeled_count = db.query(ArtworkMeta).filter(ArtworkMeta.labels.isnot(None)).count()

    return {
        "feature_type": feature_type,
        "paintings_with_labels": labeled_count,
        "features_analyzed": correlations_data.get("metadata", {}).get("num_features_analyzed", 0),
        "features_labeled": len(feature_labels.get("labels", [])),
        "rating_questions": len(QUESTIONS),
        "paintings_with_heatmaps": heatmaps_index.get("total", 0),
        "sae_type": correlations_data.get("metadata", {}).get("sae_type", "cls"),
    }


@app.get("/api/interp/paintings/{painting_id}/heatmap-data", response_model=PaintingHeatmapData)
def get_painting_heatmap_data(painting_id: str, db: Session = Depends(get_db)):
    """Get metadata about available heatmaps for a painting."""
    painting = db.query(ArtworkMeta).filter(ArtworkMeta.id == painting_id).first()
    if not painting:
        raise HTTPException(status_code=404, detail="Painting not found")

    heatmaps_index = get_heatmaps_index()
    heatmap_data = None
    for p in heatmaps_index.get("paintings", []):
        if p["painting_id"] == painting_id:
            heatmap_data = p
            break

    if not heatmap_data:
        raise HTTPException(
            status_code=404,
            detail="Heatmap not available for this painting."
        )

    feature_heatmaps = heatmap_data.get("feature_heatmaps", {})
    available_features = [int(k) for k in feature_heatmaps.keys()]

    return PaintingHeatmapData(
        painting_id=painting_id,
        title=heatmap_data["title"],
        artist=heatmap_data.get("artist"),
        year=heatmap_data.get("year"),
        image_url=f"/static/heatmaps/{painting_id}_image.jpg",
        top_active_features=[
            HeatmapFeature(feature_idx=f["feature_idx"], activation=f["total_activation"])
            for f in heatmap_data.get("top_active_features", [])
        ],
        available_features=sorted(available_features),
    )


@app.get("/api/interp/paintings/{painting_id}/heatmap/{feature_idx}", response_model=FeatureHeatmap)
def get_feature_heatmap(painting_id: str, feature_idx: int, db: Session = Depends(get_db)):
    """Get the heatmap for a specific feature in a painting."""
    painting = db.query(ArtworkMeta).filter(ArtworkMeta.id == painting_id).first()
    if not painting:
        raise HTTPException(status_code=404, detail="Painting not found")

    heatmaps_index = get_heatmaps_index()
    heatmap_data = None
    for p in heatmaps_index.get("paintings", []):
        if p["painting_id"] == painting_id:
            heatmap_data = p
            break

    if not heatmap_data:
        raise HTTPException(status_code=404, detail="Heatmap not available for this painting.")

    feature_heatmaps = heatmap_data.get("feature_heatmaps", {})
    feat_data = feature_heatmaps.get(str(feature_idx))

    if not feat_data:
        raise HTTPException(
            status_code=404,
            detail=f"Heatmap for feature {feature_idx} not available for this painting."
        )

    return FeatureHeatmap(
        feature_idx=feature_idx,
        heatmap_7x7=feat_data["heatmap_7x7"],
        max_activation=feat_data["max_activation"],
        mean_activation=feat_data["mean_activation"],
        total_activation=feat_data["total_activation"],
    )


@app.get("/api/interp/heatmaps", response_model=list[PaintingHeatmapData])
def list_paintings_with_heatmaps(db: Session = Depends(get_db)):
    """List all paintings that have pre-computed heatmaps."""
    heatmaps_index = get_heatmaps_index()
    results = []

    for heatmap_data in heatmaps_index.get("paintings", []):
        painting_id = heatmap_data["painting_id"]
        feature_heatmaps = heatmap_data.get("feature_heatmaps", {})
        available_features = [int(k) for k in feature_heatmaps.keys()]

        results.append(PaintingHeatmapData(
            painting_id=painting_id,
            title=heatmap_data["title"],
            artist=heatmap_data.get("artist"),
            year=heatmap_data.get("year"),
            image_url=f"/static/heatmaps/{painting_id}_image.jpg",
            top_active_features=[
                HeatmapFeature(feature_idx=f["feature_idx"], activation=f["total_activation"])
                for f in heatmap_data.get("top_active_features", [])
            ],
            available_features=sorted(available_features),
        ))

    return results


@app.get("/api/interp/features-by-question")
def get_features_by_question():
    """Get the top correlated features for each rating question."""
    heatmaps_index = get_heatmaps_index()
    return heatmaps_index.get("features_by_question", {})


def run():
    """Run the server."""
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    run()
