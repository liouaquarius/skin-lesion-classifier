"""FastAPI inference server.

Start (dev):
    uv run uvicorn backend.main:app --reload

Environment variables:
    CHECKPOINT_PATH    path to a .pt checkpoint
                       (default: results/checkpoints/vit_tiny-wce-seed42_best.pt)
    MODEL_NAME         build_model architecture name (default: vit_tiny_patch16_224)
    ALLOWED_ORIGINS    comma-separated CORS origins allowed to call the API
                       (default: the local Vite dev server)
    MAX_UPLOAD_MB      reject uploads larger than this many MB (default: 5)
    RATE_LIMIT_PER_MIN per-IP request cap per minute; 0 disables (default: 30)
"""

from __future__ import annotations

import io
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.schemas import ExplainResponse, PredictionResponse
from src.inference import Predictor

# CORS: in production the GitHub Pages frontend lives on a different origin than
# the HF Spaces backend, so the allowed origin(s) must be set via env var.
# NOTE: CORS only restrains browser JS from other origins — it is not auth and
# does not stop direct (curl/script) calls. Abuse protection is the limits below.
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

_MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "5")) * 1024 * 1024
_RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "30"))
_RATE_WINDOW_S = 60.0

# Per-IP sliding-window request log. In-memory is sufficient for a single
# scale-to-zero container; it is not shared across replicas.
_request_log: dict[str, deque[float]] = defaultdict(deque)

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


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:  # behind the HF Spaces / Pages proxy
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    """Per-IP sliding-window limiter; raises 429 when the cap is exceeded."""
    if _RATE_LIMIT_PER_MIN <= 0:
        return
    now = time.monotonic()
    log = _request_log[_client_ip(request)]
    while log and log[0] <= now - _RATE_WINDOW_S:
        log.popleft()
    if len(log) >= _RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Too many requests; please slow down.")
    log.append(now)


async def _read_image(file: UploadFile) -> Image.Image:
    """Size-limit and decode an upload into an RGB PIL image.

    The content-type header is client-supplied and spoofable, so validation
    relies on the size cap plus an actual decode (any non-image -> 422).
    """
    # Read at most MAX+1 bytes so an oversized upload cannot exhaust memory.
    data = await file.read(_MAX_UPLOAD_BYTES + 1)
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large (max {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=422, detail="Invalid image file.") from None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...), _: None = Depends(rate_limit)
) -> PredictionResponse:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    image = await _read_image(file)
    return PredictionResponse(**_predictor.predict(image))


@app.post("/explain", response_model=ExplainResponse)
async def explain(
    file: UploadFile = File(...), _: None = Depends(rate_limit)
) -> ExplainResponse:
    """Like /predict but also returns a Grad-CAM heatmap overlay as a data-URI."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    image = await _read_image(file)
    return ExplainResponse(**_predictor.explain(image))
