"""FastAPI inference server.

Start (dev):
    uv run uvicorn backend.main:app --reload

Environment variables:
    CHECKPOINT_PATH  path to a .pt checkpoint  (default: checkpoints/model_best.pt)
    MODEL_NAME       timm model name            (default: resnet18)
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.schemas import ExplainResponse, PredictionResponse
from src.inference import Predictor

_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    ckpt = os.environ.get("CHECKPOINT_PATH", "results/checkpoints/best_model.pt")
    model_name = os.environ.get("MODEL_NAME", "resnet18")
    _predictor = Predictor(ckpt, model_name)
    yield
    _predictor = None


app = FastAPI(
    title="Skin Lesion Classifier",
    description="HAM10000 multi-class dermoscopy classifier (CNN vs ViT).",
    version="0.1.0",
    lifespan=lifespan,
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
