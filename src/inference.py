"""Single-image inference utility used by the FastAPI backend."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.data import build_transforms
from src.data.dataset import CLASSES
from src.models import build_model

_NUM_CLASSES = len(CLASSES)


class Predictor:
    """Load a saved checkpoint and classify a single dermoscopy image."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_name: str,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = build_model(model_name, num_classes=_NUM_CLASSES, pretrained=False)
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self._transform = build_transforms(train=False)

    def predict(self, image: Image.Image) -> dict:
        """Return predicted class, confidence, and full probability distribution.

        Returns::

            {
                "predicted_class": "mel",
                "confidence": 0.923,
                "probabilities": {"akiec": 0.01, "bcc": 0.02, ...},
            }
        """
        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=-1).squeeze(0).cpu()
        idx = int(probs.argmax())
        return {
            "predicted_class": CLASSES[idx],
            "confidence": float(probs[idx]),
            "probabilities": {cls: float(probs[i]) for i, cls in enumerate(CLASSES)},
        }
