"""Object detection inference wrapper around ultralytics YOLO.

Sync by design: YOLO inference is a CPU/GPU-bound torch call. Callers MUST
invoke `run_detect_sync` via `asyncio.to_thread(...)` so the FastAPI event
loop is not blocked.

Output format intentionally matches the existing
`backend.schemas.vision.DetectedObject`:

    {"label": str, "confidence": float, "bbox": [x, y, w, h]}

`bbox` is converted from YOLO's native [x1, y1, x2, y2] to top-left-plus-
size in absolute pixel units. The schema is permissive about units, so we
keep raw pixels rather than normalizing to [0, 1] — every existing
on-device caller already sends absolute pixel boxes too.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from backend.ml.registry import get_yolo


def run_detect_sync(image_bytes: bytes, conf: float = 0.25) -> List[Dict[str, Any]]:
    """Run YOLO detection on raw image bytes.

    Args:
        image_bytes: image payload (PNG/JPEG/etc.) as received from the client.
        conf: confidence threshold; detections below this are dropped.

    Returns:
        A list of `{label, confidence, bbox}` dicts. `bbox` is `[x, y, w, h]`
        in absolute pixel coordinates. Sorted by confidence desc.
    """
    model = get_yolo()

    # ultralytics accepts PIL.Image directly; going via PIL avoids depending
    # on cv2 here just for color-space gymnastics.
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(img, conf=conf, verbose=False)
    if not results:
        return []
    result = results[0]

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = getattr(result, "names", None) or getattr(model, "names", {})
    xyxy = boxes.xyxy.cpu().numpy().astype(float)
    confs = boxes.conf.cpu().numpy().astype(float)
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    detections: List[Dict[str, Any]] = []
    for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, cls_ids):
        label = names.get(int(cid), str(int(cid))) if isinstance(names, dict) else str(int(cid))
        detections.append(
            {
                "label": str(label),
                "confidence": round(float(c), 4),
                "bbox": [
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2 - x1), 2),
                    round(float(y2 - y1), 2),
                ],
            }
        )

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections
