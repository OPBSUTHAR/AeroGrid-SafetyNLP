"""Vercel entrypoint for AeroGrid SafetyNLP (Aviation & Power-Grid Safety NLP).

Serves the FastAPI backend (aviation-safety-app/backend) under its /api routes
and the built React dashboard (aviation-safety-app/frontend/dist).

The dashboard build is produced by the Vercel buildCommand (npm ci +
npm run build), while the trained ML artifacts and NLTK corpora are shipped
straight from aviation-safety-app/backend/{artifacts,nltk_data}.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE / "aviation-safety-app" / "backend"
DIST_DIR = HERE / "aviation-safety-app" / "frontend" / "dist"

# 1) Make the backend package importable (main.py, ml_service.py, ...).
sys.path.insert(0, str(BACKEND_DIR))

# 2) Point the backend at the deployment copies of its ML artifacts. On Vercel
#    the pipeline output directory (safety_nlp_pipeline/data) is not shipped;
#    aviation-safety-app/backend/artifacts holds the trained model, vectorizer,
#    dataset, plots and metrics instead.
_ARTIFACTS = BACKEND_DIR / "artifacts"
if _ARTIFACTS.is_dir():
    _artifact_env = {
        "MODEL_PATH": "safety_model.pkl",
        "VECTORIZER_PATH": "tfidf_vectorizer.pkl",
        "DATASET_PATH": "real_safety_dataset.csv",
        "METRICS_JSON": "metrics.json",
        "TRAIN_CONFIG_PATH": "training_config.json",
        "CLASSIFICATION_REPORT_TXT": "classification_report.txt",
        "CLASSIFICATION_REPORT_CSV": "classification_report.csv",
    }
    for key, filename in _artifact_env.items():
        os.environ.setdefault(key, str(_ARTIFACTS / filename))
    os.environ.setdefault("PLOTS_DIR", str(_ARTIFACTS / "plots"))

# 3) NLTK corpora: prefer the bundled copy, otherwise use a writable /tmp
#    directory (serverless filesystems are read-only outside /tmp); the backend
#    downloads whatever is still missing there on first load.
_NLTK_BUNDLE = BACKEND_DIR / "nltk_data"
if _NLTK_BUNDLE.is_dir():
    os.environ.setdefault("NLTK_DATA", str(_NLTK_BUNDLE))
else:
    _NLTK_TMP = Path("/tmp/nltk_data")
    try:
        _NLTK_TMP.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    os.environ.setdefault("NLTK_DATA", str(_NLTK_TMP))

from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from main import app  # noqa: E402  (FastAPI instance exposing every /api route)

# 4) Warm up the ML artifacts at cold start so the first API request does not
#    race the background loader started by the lifespan handler.
try:
    from ml_service import ml_service

    if ml_service._model is None:
        ml_service.load_artifacts()
except Exception as exc:  # pragma: no cover - the app must still boot
    import logging

    logging.getLogger(__name__).warning("Artifact preload failed: %s", exc)

# 5) Frontend: the hashed bundles under /assets are static (Vercel promotes
#    StaticFiles mounts to the CDN); every other unknown path falls back to
#    index.html so client-side routes such as /performance keep working.
if (DIST_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="spa-assets")

_DIST_ROOT = DIST_DIR.resolve()


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    """Serve the SPA build; unknown client-side routes return index.html."""
    if full_path == "api" or full_path.startswith("api/"):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    if full_path:
        candidate = (_DIST_ROOT / full_path).resolve()
        if candidate.is_file() and str(candidate).startswith(str(_DIST_ROOT)):
            return FileResponse(candidate)
    index_file = DIST_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return JSONResponse(
        {"detail": "Frontend build not found - run 'npm run build' in aviation-safety-app/frontend"},
        status_code=404,
    )
