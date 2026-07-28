"""OCR inference.

Two backends, picked automatically based on configuration:

  1. **OCR.space cloud API** (primary when ``OCR_SPACE_API_KEY`` is set).
     POSTs the JPEG to ``api.ocr.space``, parses the JSON response, and
     returns the recognized text.

  2. **Local Tesseract** (fallback). The original Phase-1 implementation —
     kept verbatim below so the project still works offline / without an
     API key, and so the existing tests keep mocking the same symbol.

Both backends are sync and CPU-bound; callers MUST invoke ``run_ocr_sync``
via ``asyncio.to_thread(...)`` so the FastAPI event loop is not blocked.
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import List, Optional

import pytesseract
import requests
from PIL import Image

from backend.core.config import settings


# ---------------------------------------------------------------------------
# OCR.space cloud backend (primary)
# ---------------------------------------------------------------------------

# Map our internal language codes (Tesseract-style 3-letter) to OCR.space's
# expected codes. They happen to align for the common cases. Adding this
# map up-front so future divergence has a single place to live.
_OCRSPACE_LANG_MAP = {
    "eng": "eng",
    "ara": "ara",
    "fre": "fre",
    "ger": "ger",
    "spa": "spa",
    "ita": "ita",
    "rus": "rus",
    "tur": "tur",
    "auto": "auto",
}


def _run_ocr_ocrspace_sync(image_bytes: bytes, lang: str = "eng") -> Optional[str]:
    """Call OCR.space and return the parsed text. Returns None on hard failure
    (network error, HTTP error, JSON error) so the caller can fall back to
    Tesseract. Returns "" on a successful call that produced no text."""
    api_key = settings.OCR_SPACE_API_KEY
    if not api_key:
        return None

    # OCR.space accepts ONE language per call. For "eng+ara" we run two
    # sequential calls and concatenate — that's how the project's existing
    # default behaves equivalently to multilingual Tesseract.
    raw_lang = (lang or "eng").lower().strip()
    langs = (
        [piece.strip() for piece in raw_lang.split("+") if piece.strip()]
        if "+" in raw_lang
        else [raw_lang]
    )

    parts: List[str] = []
    any_call_succeeded = False
    for one in langs:
        api_lang = _OCRSPACE_LANG_MAP.get(one, one)
        try:
            response = requests.post(
                settings.OCR_SPACE_URL,
                headers={"apikey": api_key},
                files={"file": ("img.jpg", image_bytes, "image/jpeg")},
                data={
                    "language": api_lang,
                    "OCREngine": str(settings.OCR_SPACE_ENGINE),
                    "isOverlayRequired": "false",
                    "scale": "true",
                    "isTable": "false",
                    "detectOrientation": "true",
                },
                timeout=settings.OCR_SPACE_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            # One language failed — try the next; only return None if all fail.
            continue

        if payload.get("IsErroredOnProcessing"):
            continue

        any_call_succeeded = True
        for parsed in payload.get("ParsedResults") or []:
            text = (parsed.get("ParsedText") or "").strip()
            if text:
                parts.append(text)

    if not any_call_succeeded:
        return None  # caller falls back to Tesseract
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tesseract backend (fallback / offline)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _configure_tesseract() -> None:
    if settings.OCR_TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = settings.OCR_TESSERACT_CMD


def _run_ocr_tesseract_sync(image_bytes: bytes, lang: str = "eng+ara") -> str:
    """Original Phase-1 Tesseract implementation. Kept as a fallback."""
    _configure_tesseract()
    img = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(img, lang=lang)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ocr_sync(image_bytes: bytes, lang: str = "eng") -> str:
    """Extract text from raw image bytes.

    Routing:
      1. If ``OCR_SPACE_API_KEY`` is set → call OCR.space.
         If the network call hard-fails, fall through to Tesseract.
      2. Otherwise → local Tesseract subprocess.

    Args:
        image_bytes: image payload (PNG/JPEG/etc.) as received from the client.
        lang: OCR language. Default ``"eng"`` keeps OCR.space to a single API
              request; pass ``"ara"`` for Arabic-only or ``"eng+ara"`` for
              both (which becomes two OCR.space requests).
    """
    if settings.OCR_SPACE_API_KEY:
        cloud_text = _run_ocr_ocrspace_sync(image_bytes, lang=lang)
        if cloud_text is not None:
            return cloud_text
        # Hard failure → keep going with Tesseract below.
    return _run_ocr_tesseract_sync(image_bytes, lang=lang)
