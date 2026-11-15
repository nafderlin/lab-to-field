"""Lightweight inference with the exported ONNX model.

Needs only numpy, pillow and onnxruntime (no PyTorch), so it runs on any CPU:
the Streamlit app, the command line and the Hugging Face Space all use this module.

    from lab_to_field.inference import LeafDiagnoser
    result = LeafDiagnoser().diagnose("leaf.jpg")
    print(result.answer, result.confidence)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

EXPORT_DIR = Path(__file__).resolve().parents[2] / "export"
PRECISIONS = ("fp16", "int8", "fp32")
IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
NOT_SURE = "Not sure"


def preprocess(img: Image.Image, size: int = IMAGE_SIZE) -> tuple[np.ndarray, Image.Image]:
    """Same as training-time evaluation: resize the shorter side to 255, center-crop 224, normalise.

    Returns the (1, 3, size, size) float32 tensor and the cropped RGB image (for display / Grad-CAM).
    """
    img = img.convert("RGB")
    w, h = img.size
    short = int(size * 1.14)  # same rounding as torchvision Resize(255) + CenterCrop(224)
    new_w, new_h = (short, int(short * h / w)) if w <= h else (int(short * w / h), short)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left, top = int(round((new_w - size) / 2.0)), int(round((new_h - size) / 2.0))
    img = img.crop((left, top, left + size, top + size))
    x = (np.asarray(img, dtype=np.float32) / 255 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32), img


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax of logits divided by the temperature (T > 1 softens, T < 1 sharpens)."""
    z = np.asarray(z, dtype=np.float64) / temperature
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class Diagnosis:
    """Result for one photo. `answer` is the disease name, or "Not sure" below the threshold."""

    probs: np.ndarray = field(repr=False)
    classes: list[str] = field(repr=False)
    threshold: float
    image224: Image.Image | None = field(default=None, repr=False)

    @property
    def best_guess(self) -> str:
        return self.classes[int(self.probs.argmax())]

    @property
    def confidence(self) -> float:
        return float(self.probs.max())

    @property
    def is_sure(self) -> bool:
        return self.confidence >= self.threshold

    @property
    def answer(self) -> str:
        return self.best_guess if self.is_sure else NOT_SURE

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        idx = np.argsort(-self.probs)[:k]
        return [(self.classes[i], float(self.probs[i])) for i in idx]

    def to_dict(self, k: int = 3) -> dict:
        return {"answer": self.answer, "best_guess": self.best_guess, "confidence": round(self.confidence, 4),
                "top": [{"label": c, "prob": round(p, 4)} for c, p in self.top(k)]}


class LeafDiagnoser:
    """Loads one exported model plus its calibration (temperature) and abstention threshold."""

    def __init__(self, precision: str = "fp16", export_dir: str | Path = EXPORT_DIR):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        import onnxruntime as ort  # imported here so `preprocess`/`softmax` work without it

        export_dir = Path(export_dir)
        meta = json.loads((export_dir / "labels.json").read_text())
        self.precision = precision
        self.classes: list[str] = meta["classes"]
        self.temperature: float = meta["temperature"]
        self.threshold: float = meta["abstain_threshold"]
        self.session = ort.InferenceSession(str(export_dir / f"plant_{precision}.onnx"),
                                            providers=["CPUExecutionProvider"])

    def logits(self, img: Image.Image) -> tuple[np.ndarray, Image.Image]:
        x, img224 = preprocess(img)
        return self.session.run(None, {"image": x})[0][0], img224

    def diagnose(self, img: Image.Image | str | Path, temperature: float | None = None) -> Diagnosis:
        """Calibrated diagnosis of one photo. Pass temperature=1.0 to see the uncalibrated probabilities."""
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        z, img224 = self.logits(img)
        probs = softmax(z, self.temperature if temperature is None else temperature)
        return Diagnosis(probs=probs, classes=self.classes, threshold=self.threshold, image224=img224)
