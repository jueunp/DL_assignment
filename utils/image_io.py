from __future__ import annotations

from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image


def load_image(image_source: str) -> Image.Image:
    if image_source.startswith(("http://", "https://")):
        response = requests.get(image_source, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    return Image.open(Path(image_source)).convert("RGB")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
