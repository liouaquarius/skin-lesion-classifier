"""FastAPI endpoint tests — no real checkpoint or GPU needed."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from src.data.dataset import CLASSES

_MOCK_RESULT = {
    "predicted_class": "nv",
    "confidence": 0.95,
    "probabilities": {cls: 1.0 / len(CLASSES) for cls in CLASSES},
}


@pytest.fixture
def client():
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = _MOCK_RESULT
    with patch("backend.main.Predictor", return_value=mock_predictor):
        with TestClient(app) as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────


def test_health_returns_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["model_loaded"] is True


# ── /predict ──────────────────────────────────────────────────────────────────


def test_predict_valid_image_returns_schema(client: TestClient) -> None:
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    resp = client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert resp.status_code == 200

    body = resp.json()
    assert body["predicted_class"] in CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys()) == set(CLASSES)


def test_predict_invalid_file_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/predict",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 422
