"""Dual-model obstacle detection with priority scoring and voice guidance.

Sync by design. Callers MUST invoke ``run_obstacle_sync`` via
``asyncio.to_thread(...)``.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

from PIL import Image

from backend.ml.registry import get_obstacle_yolo, get_objects_yolo


_CANONICAL_LABELS = {
    "wall": "wall",
    "door": "door",
    "stairs": "stairs",
    "bench": "Bench",
    "person": "Person",
    "chair": "Chair",
    "table": "Table",
}
_CRITICAL_CLASSES = set(_CANONICAL_LABELS)


def _priority(det: Dict[str, Any], img_w: int, img_h: int) -> float:
    x, y, w, h = det["bbox"]
    cy = y + h / 2.0
    img_area = max(1, img_w * img_h)
    size_score = (w * h) / img_area
    vertical_score = cy / max(1, img_h)
    critical_bonus = 0.3 if det["is_critical"] else 0.0
    return size_score * 0.4 + vertical_score * 0.4 + critical_bonus + det["confidence"] * 0.2


def _voice_message(det: Dict[str, Any], img_w: int) -> str:
    x, y, w, h = det["bbox"]
    cx = x + w / 2.0
    if cx < img_w / 3.0:
        position = "on your left"
    elif cx > img_w * 2.0 / 3.0:
        position = "on your right"
    else:
        position = "ahead"
    area = w * h
    if area > 50_000:
        distance = "very close"
    elif area > 20_000:
        distance = "close"
    else:
        distance = "detected"
    return f"{det['label'].title()} {distance} {position}"


def _parse_model_results(
    model: Any,
    results: Any,
    img_w: int,
    img_h: int,
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    if not results:
        return detections

    result = results[0]
    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", None) or getattr(model, "names", {})
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy().astype(float)
    confs = boxes.conf.cpu().numpy().astype(float)
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), confidence, class_id in zip(xyxy, confs, cls_ids):
        if isinstance(names, dict):
            raw_label = names.get(int(class_id), str(int(class_id)))
        else:
            raw_label = names[int(class_id)] if int(class_id) < len(names) else str(int(class_id))
        label = _CANONICAL_LABELS.get(str(raw_label).strip().lower())
        if label is None:
            continue

        detection = {
            "label": label,
            "confidence": round(float(confidence), 4),
            "bbox": [
                round(float(x1), 2),
                round(float(y1), 2),
                round(float(x2 - x1), 2),
                round(float(y2 - y1), 2),
            ],
            "is_critical": label.lower() in _CRITICAL_CLASSES,
        }
        detection["priority_score"] = round(_priority(detection, img_w, img_h), 4)
        detections.append(detection)
    return detections


def run_obstacle_sync(
    image_bytes: bytes,
    conf: float = 0.25,
    top_n: int = 3,
) -> Dict[str, Any]:
    """Run both obstacle models and merge their seven agreed classes."""
    obstacle_model = get_obstacle_yolo()
    objects_model = get_objects_yolo()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = img.size

    obstacle_results = obstacle_model.predict(img, conf=conf, verbose=False)
    objects_results = objects_model.predict(img, conf=conf, verbose=False)

    detections = _parse_model_results(obstacle_model, obstacle_results, img_w, img_h)
    detections.extend(_parse_model_results(objects_model, objects_results, img_w, img_h))
    detections.sort(key=lambda d: d["priority_score"], reverse=True)
    highest: Optional[Dict[str, Any]] = detections[0] if detections else None

    if not detections:
        voice_guidance = ["Path is clear"]
    else:
        n = max(1, min(int(top_n), 10))
        voice_guidance = [_voice_message(d, img_w) for d in detections[:n]]

    return {
        "image_shape": {"width": img_w, "height": img_h},
        "detections": detections,
        "highest_priority": highest,
        "voice_guidance": voice_guidance,
    }
