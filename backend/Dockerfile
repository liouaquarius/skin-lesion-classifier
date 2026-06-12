# syntax=docker/dockerfile:1
# CPU-only inference image — no CUDA required at runtime.
# Used for local Docker and Hugging Face Spaces (Docker SDK).
#
# Bake the deployment checkpoint first, then build from the repo root:
#   cp results/checkpoints/vit_tiny-wce-seed42_best.pt results/checkpoints/best_model.pt
#   docker build -f backend/Dockerfile -t skin-lesion-classifier .
#   docker run -p 8000:8000 skin-lesion-classifier
#
# best_model.pt is the only .pt the .dockerignore lets into the build context.
# On HF Spaces: set the Space README `app_port: 8000` and `ALLOWED_ORIGINS`
# (the GitHub Pages origin) as a Space variable.

FROM python:3.11-slim

WORKDIR /app

# Runtime libs: libglib2.0-0 (Pillow), libgl1 (OpenCV, pulled in by grad-cam).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
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

# ── Application source + baked model ──────────────────────────────────────────
COPY src/ src/
COPY backend/ backend/
COPY results/checkpoints/best_model.pt model/best_model.pt

# Add project root to PYTHONPATH; point the server at the baked vit_tiny-wce model.
ENV PYTHONPATH=/app \
    CHECKPOINT_PATH=/app/model/best_model.pt \
    MODEL_NAME=vit_tiny_patch16_224

EXPOSE 8000

# Shell form so a platform-provided $PORT (e.g. Cloud Run) can override;
# defaults to 8000, which matches the HF Space `app_port: 8000`.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
