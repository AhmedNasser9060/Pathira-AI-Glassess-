"""Scene description: caption an image in English, optionally translate to Arabic.

Sync by design — BLIP-base inference takes ~2-5 s on CPU, MarianMT
translation ~50 ms. Callers MUST invoke `run_scene_sync` via
`asyncio.to_thread(...)`.

Pipeline (matches the project's original Colab notebook, scaled down to
fit a CPU dev box):
    bytes -> PIL.Image
          -> BLIP-base captioning   (Salesforce/blip-image-captioning-base)
          -> caption_en
          -> MarianMT EN->AR        (Helsinki-NLP/opus-mt-en-ar)
          -> caption_ar
          -> {"caption_en", "caption_ar" | None}

Models are downloaded from the Hugging Face hub on first call and cached
under `~/.cache/huggingface/`. Lazy-loaded via `lru_cache(1)` so server
boot stays under 2 s. Returning text only (no TTS) — the Expo app uses
on-device speech synthesis which is faster and works offline.
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import Any, Dict, Optional

from PIL import Image


# BLIP-base is the practical choice on a CPU laptop with limited RAM:
# ~1 GB on disk and ~1.5 GB RAM at runtime. BLIP-large would give
# slightly richer captions but it OOM'd alongside YOLO + ArcFace +
# MarianMT in this project. The detail loss is recovered with the
# tuned generation parameters below (longer beams, length penalty,
# repetition control, conditional prompt).
_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
_TRANSLATE_MODEL = "Helsinki-NLP/opus-mt-en-ar"

# Conditional prompt: BLIP appends its caption to this prefix, which
# nudges the model toward longer, more descriptive output. The prefix
# is stripped before returning so the user never sees it.
_CAPTION_PROMPT = "a detailed photograph of"


@lru_cache(maxsize=1)
def _get_caption_pipeline():
    # Lazy import: transformers is a heavy dependency; do not pay the cost
    # at module import time.
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained(_CAPTION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(_CAPTION_MODEL)
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def _get_translator_pipeline():
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(_TRANSLATE_MODEL)
    model = MarianMTModel.from_pretrained(_TRANSLATE_MODEL)
    model.eval()
    return tokenizer, model


def _caption_en(image_bytes: bytes) -> str:
    import torch

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processor, model = _get_caption_pipeline()
    # Conditional generation: feed the prompt as a leading caption. BLIP
    # continues from there, which produces longer, more descriptive output
    # than unconditional captioning alone.
    inputs = processor(img, _CAPTION_PROMPT, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,        # was 64 — allow longer captions
            num_beams=5,               # was 4 — wider search, better phrasing
            length_penalty=1.4,        # > 1 favours longer sequences
            repetition_penalty=1.15,   # discourage looping/echoing
            no_repeat_ngram_size=3,    # don't repeat 3-grams
            early_stopping=True,
        )
    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    # Strip the leading prompt so the caller sees the description only.
    if caption.lower().startswith(_CAPTION_PROMPT.lower()):
        caption = caption[len(_CAPTION_PROMPT):].lstrip(" ,.:;-—")
    return caption


def _translate_en_to_ar(text: str) -> Optional[str]:
    import torch

    if not text:
        return None
    tokenizer, model = _get_translator_pipeline()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        # Match the longer EN cap with a longer AR budget. MarianMT is
        # fast (~50 ms on CPU) so the bigger budget costs little.
        out = model.generate(**inputs, max_new_tokens=200, num_beams=4)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def run_scene_sync(image_bytes: bytes, translate: bool = True) -> Dict[str, Any]:
    """Caption an image and optionally translate the caption to Arabic.

    Args:
        image_bytes: image payload (PNG/JPEG/etc.) as received from the client.
        translate:   if True, also produce an Arabic caption. The Expo app
                     defaults to True so the screen reader can speak in
                     Arabic; pass False to skip the translator entirely.

    Returns:
        {"caption_en": str, "caption_ar": str | None}.
    """
    caption_en = _caption_en(image_bytes)
    caption_ar: Optional[str] = None
    if translate:
        try:
            caption_ar = _translate_en_to_ar(caption_en)
        except Exception:
            # Translation failures should not kill the whole call — the
            # English caption is still useful on its own.
            caption_ar = None
    return {"caption_en": caption_en, "caption_ar": caption_ar}
