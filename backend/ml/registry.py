"""Lazy loader for ML model singletons.

Each accessor is wrapped with `lru_cache(maxsize=1)` so the underlying
weights are loaded exactly once â€” on the first request that needs them â€”
and reused across subsequent calls. Application startup must NOT call
these functions; doing so would force every server process to load every
model at boot, defeating Phase 4's lazy-by-design rule.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from backend.core.config import settings


@lru_cache(maxsize=1)
def get_yolo():
    """Return a singleton YOLO model loaded from `ML_WEIGHTS_DIR/yolo.pt`."""
    # Imported lazily so importing this module does not require ultralytics
    # to be installed (matters for Phase 1 environments and unit tests that
    # mock detection without touching real weights).
    from ultralytics import YOLO

    weights = Path(settings.ML_WEIGHTS_DIR) / "yolo.pt"
    return YOLO(str(weights))


@lru_cache(maxsize=1)
def get_obstacle_yolo():
    """Return a singleton YOLO model trained for obstacle detection.

    Loaded from `ML_WEIGHTS_DIR/obstacle.pt`. Separate cache key from
    `get_yolo()` so the two models can coexist in memory and be replaced
    independently.
    """
    from ultralytics import YOLO

    weights = Path(settings.ML_WEIGHTS_DIR) / "obstacle.pt"
    return YOLO(str(weights))

@lru_cache(maxsize=1)
def get_objects_yolo():
    """Return the singleton indoor-object model for bench/person/chair/table."""
    from ultralytics import YOLO

    weights = Path(settings.ML_WEIGHTS_DIR) / "objects4.pt"
    return YOLO(str(weights))

