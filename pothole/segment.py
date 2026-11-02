"""YOLO-seg inference: one full-resolution binary mask per detected pothole."""

from pathlib import Path

import cv2
import numpy as np

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "pothole_seg.pt"


def load_model(weights=DEFAULT_WEIGHTS):
    from ultralytics import YOLO  # imported lazily: heavy, and only needed here

    if not Path(weights).is_file():
        raise FileNotFoundError(
            f"No trained weights at {weights}. Run scripts/train.py first."
        )
    return YOLO(str(weights))


def predict_masks(model, image, conf=0.25, device=None):
    """Return a list of (mask uint8 HxW in {0,255}, confidence)."""
    result = model.predict(image, conf=conf, device=device, retina_masks=True,
                           verbose=False)[0]
    if result.masks is None:
        return []
    height, width = image.shape[:2]
    masks = result.masks.data.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    out = []
    for mask, score in zip(masks, confs):
        binary = (mask > 0.5).astype(np.uint8) * 255
        if binary.shape != (height, width):
            binary = cv2.resize(binary, (width, height),
                                interpolation=cv2.INTER_NEAREST)
        out.append((binary, float(score)))
    return out
