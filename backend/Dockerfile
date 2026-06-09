# syntax=docker/dockerfile:1
# CPU-only inference image — no CUDA required at runtime.
#
# Build (from repo root):
#   docker build -f backend/Dockerfile -t skin-lesion-classifier .
#
# Run (mount a trained checkpoint):
#   docker run -p 8000:8000 \
#     -e CHECKPOINT_PATH=/ckpt/best_model.pt \
#     -v /path/to/results/checkpoints:/ckpt:ro \
#     skin-lesion-classifier
#
# Or bake the model in: copy the best checkpoint to
# results/checkpoints/best_model.pt before building — it is the only
# .pt file the .dockerignore allows through.

FROM python:3.11-slim

WORKDIR /app

# Pillow needs libglib2.0-0 for JPEG/PNG decoding at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch (CPU-only) ────────────────────────────────────────────────────────
# Install before requirements.txt to avoid pulling the 2 GB CUDA wheel from PyPI.
RUN pip install --no-cache-dir \
    "torch==2.12.0+cpu" \
    "torchvision==0.27.0+cpu" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ── Remaining inference deps ──────────────────────────────────────────────────
# timm must follow torch (detects the installed backend at install time).
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir \
    "timm==1.0.27" \
    -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY src/ src/
COPY backend/ backend/

# Add project root to PYTHONPATH so `from src.inference import …` resolves.
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
