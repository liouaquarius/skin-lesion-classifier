"""FastAPI inference server.

Start (dev):
    uv run uvicorn backend.main:app --reload

Environment variables:
    CHECKPOINT_PATH  path to a .pt checkpoint
                     (default: results/checkpoints/vit_tiny-wce-seed42_best.pt)
    MODEL_NAME       build_model architecture name (default: vit_tiny_patch16_224)
    ALLOWED_ORIGINS  comma-separated CORS origins allowed to call the API
                     (default: the local Vite dev server)
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.schemas import ExplainResponse, PredictionResponse
from src.inference import Predictor

# CORS: in production the GitHub Pages frontend lives on a different origin than
# the HF Spaces backend, so the allowed origin(s) must be set via env var.
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    # Demo serves vit_tiny-wce: best macro-F1 with strong melanoma sensitivity.
    # Override both via env vars (e.g. in Docker) to serve a different model.
    ckpt = os.environ.get(
        "CHECKPOINT_PATH", "results/checkpoints/vit_tiny-wce-seed42_best.pt"
    )
    model_name = os.environ.get("MODEL_NAME", "vit_tiny_patch16_224")
    _predictor = Predictor(ckpt, model_name)
    yield
    _predictor = None


app = FastAPI(
    title="Skin Lesion Classifier",
    description="HAM10000 multi-class dermoscopy classifier (CNN vs ViT).",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, Exception):
        raise HTTPException(status_code=422, detail="Invalid image file.")
    result = _predictor.predict(image)
    return PredictionResponse(**result)


@app.post("/explain", response_model=ExplainResponse)
async def explain(file: UploadFile = File(...)) -> ExplainResponse:
    """Like /predict but also returns a Grad-CAM heatmap overlay as a data-URI."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, Exception):
        raise HTTPException(status_code=422, detail="Invalid image file.")
    result = _predictor.explain(image)
    return ExplainResponse(**result)
