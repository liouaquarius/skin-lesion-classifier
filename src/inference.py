"""Single-image inference utility used by the FastAPI backend."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data import build_transforms
from src.data.dataset import CLASS_TO_IDX, CLASSES
from src.models import build_model

_NUM_CLASSES = len(CLASSES)

# vit_tiny_patch16_224: 224 / 16 = 14 patches per side
_VIT_GRID = 14


def _target_layers(model: torch.nn.Module, name: str) -> list:
    """Return the conv/attention layer used as Grad-CAM target."""
    if name == "resnet18":
        return [model.layer4[-1]]
    if name == "efficientnet_b0":
        return [model.conv_head]
    if name == "vit_tiny_patch16_224":
        return [model.blocks[-1].norm1]
    raise ValueError(f"No Grad-CAM target defined for '{name}'")


def _vit_reshape(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape ViT patch tokens (B, 1+N, D) → (B, D, H, W) for Grad-CAM."""
    b, _seq, d = tensor.shape
    patches = tensor[:, 1:, :].reshape(b, _VIT_GRID, _VIT_GRID, d)
    return patches.permute(0, 3, 1, 2)


class Predictor:
    """Load a saved checkpoint and classify a single dermoscopy image."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_name: str,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model_name = model_name
        self.model = build_model(model_name, num_classes=_NUM_CLASSES, pretrained=False)
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self._transform = build_transforms(train=False)

    def predict(self, image: Image.Image) -> dict:
        """Return predicted class, confidence, and full probability distribution."""
        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=-1).squeeze(0).cpu()
        idx = int(probs.argmax())
        return {
            "predicted_class": CLASSES[idx],
            "confidence": float(probs[idx]),
            "probabilities": {cls: float(probs[i]) for i, cls in enumerate(CLASSES)},
        }

    def explain(self, image: Image.Image) -> dict:
        """Return prediction + a base64-encoded Grad-CAM overlay (JPEG).

        The overlay is encoded as a data-URI so it can be embedded directly
        in an ``<img src="...">`` tag without a separate file endpoint.
        """
        result = self.predict(image)
        target_class = CLASS_TO_IDX[result["predicted_class"]]

        rgb = image.convert("RGB").resize((224, 224))
        tensor = self._transform(rgb).unsqueeze(0).to(self.device)

        reshape = _vit_reshape if "vit" in self.model_name else None
        targets = [ClassifierOutputTarget(target_class)]

        with GradCAM(
            model=self.model,
            target_layers=_target_layers(self.model, self.model_name),
            reshape_transform=reshape,
        ) as cam:
            grayscale = cam(input_tensor=tensor, targets=targets)[0]

        rgb_np = np.array(rgb, dtype=np.float32) / 255.0
        overlay = show_cam_on_image(rgb_np, grayscale, use_rgb=True)

        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()

        return {**result, "grad_cam_image": f"data:image/jpeg;base64,{b64}"}
