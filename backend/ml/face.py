"""Face recognition wrapper: detect a face, compute a 512-d ArcFace embedding.

Sync by design: ONNX inference and OpenCV decode block. Callers MUST invoke
`compute_embedding_sync` via `asyncio.to_thread(...)`.

Pipeline:
    bytes -> cv2.imdecode (BGR ndarray)
          -> Haar cascade (largest face crop)
          -> 112x112 resize, BGR->RGB, normalize to [-1, 1], NCHW
          -> ArcFace ONNX -> 512-d float32 vector
          -> L2-normalize
          -> list[float]

If no face is detected, returns None.

Detector choice: OpenCV's bundled Haar cascade. Trade-off vs. RetinaFace —
a bit less accurate on profile / heavily-occluded faces, but zero extra
deps and nothing to download. Embedding quality (ArcFace) is identical.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from backend.core.config import settings


_INPUT_SIZE = (112, 112)
_EMBEDDING_DIM = 512


@lru_cache(maxsize=1)
def _get_session():
    import onnxruntime as ort

    weights = Path(settings.ML_WEIGHTS_DIR) / "arcface.onnx"
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(weights), sess_options=opts, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def _get_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(str(cascade_path))


def _largest_face_crop(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    detector = _get_detector()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return image_bgr[y : y + h, x : x + w]


def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(face_bgr, _INPUT_SIZE, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - 127.5) / 128.0  # [-1, 1]
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(img, axis=0)  # NCHW


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-10)


def compute_embedding_sync(image_bytes: bytes) -> Optional[List[float]]:
    """Detect the largest face in `image_bytes` and return its 512-d L2-normalized embedding.

    Returns None if no face is detected. Caller decides how to surface that
    (HTTP 422 for an "add face" request, 200 with no match for "recognize").
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None

    face = _largest_face_crop(image_bgr)
    if face is None or face.size == 0:
        return None

    session = _get_session()
    blob = _preprocess(face)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    raw = session.run([output_name], {input_name: blob})[0]
    emb = _l2_normalize(raw.flatten().astype(np.float32))
    return emb.tolist()


def cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 if either vector is zero/None."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def compute_average_embedding_sync(image_bytes_list: List[bytes]) -> Optional[List[float]]:
    """Compute one embedding from multiple images of the same face.

    Used by face registration: the companion captures the subject from
    several angles (front, left, right), and we average the resulting
    embeddings into a single, more robust representative vector. The
    average is L2-normalized so cosine matching against it remains
    well-defined.

    Returns None if NO image yielded a face. Skips silently any image
    where no face was found, as long as at least one succeeds.
    """
    embeddings = []
    for image_bytes in image_bytes_list:
        emb = compute_embedding_sync(image_bytes)
        if emb is not None:
            embeddings.append(np.asarray(emb, dtype=np.float32))
    if not embeddings:
        return None
    avg = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = float(np.linalg.norm(avg))
    if norm > 0:
        avg = avg / norm
    return avg.astype(np.float32).tolist()
