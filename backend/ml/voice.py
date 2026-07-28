"""Voice command pipeline.

Two stages, both running locally — no external API keys required:

  1. **Speech-to-text**: faster-whisper, `tiny.en` model. On CPU, a
     2-3 s phrase transcribes in roughly 1 s. The model file is ~40 MB
     and is downloaded automatically on first call (cached under
     ~/.cache/huggingface).

  2. **Intent extraction**: a local Ollama instance running Mistral.
     We send the transcript inside a system prompt that constrains the
     output to a tiny JSON schema and parse the response. Mistral
     handles natural variants ("show me obstacles", "what do you see",
     "read this for me") that a regex matcher would miss.

Both stages are sync; callers MUST go through `asyncio.to_thread`.
"""

from __future__ import annotations

import io
import json
import re
from functools import lru_cache
from typing import Any, Dict, Optional

import requests

from backend.core.config import settings


# ---------------------------------------------------------------------------
# Speech-to-text via faster-whisper
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_whisper():
    # Lazy import: faster-whisper drags in CTranslate2 which we'd rather
    # not pay for at process boot.
    from faster_whisper import WhisperModel

    # int8 quantization keeps the memory footprint small (~150 MB RSS for
    # tiny.en) while staying within ~2 % of float16 accuracy.
    return WhisperModel(settings.WHISPER_MODEL, device="cpu", compute_type="int8")


def transcribe_sync(audio_bytes: bytes, language: str = "en") -> str:
    """Decode the user's recording into text.

    Accepts any container faster-whisper / FFmpeg can read (m4a/wav/mp3
    work out of the box). Returns the concatenated transcript.
    """
    model = _get_whisper()
    segments, _info = model.transcribe(
        io.BytesIO(audio_bytes),
        language=language,
        beam_size=1,           # tiny model + beam=1 is plenty for short commands
        condition_on_previous_text=False,
        vad_filter=True,       # drop silence so noise doesn't transcribe to gibberish
        vad_parameters={"min_silence_duration_ms": 250},
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


# ---------------------------------------------------------------------------
# Intent extraction via Ollama Mistral
# ---------------------------------------------------------------------------

_VALID_KINDS = {"obstacle", "detect", "ocr", "scene", "face"}
_VALID_ACTIONS = {"set_kind", "set_camera_source", "toggle_vision", "navigate", "greet", "noop"}
_VALID_SOURCES = {"phone", "glasses"}
_VALID_TARGETS = {"home", "vision", "settings", "tracking", "companions", "emergency", "support"}


# Cheap keyword pre-match — fires before we ever call Mistral. Catches
# the common-case commands ("open face recognition", "read this", "what
# do I see") instantly and works even if Ollama is slow / unreachable.
def _keyword_intent(transcript: str) -> Optional[Dict[str, Any]]:
    t = transcript.lower()

    # Wake word / greeting — "Pathira", "Hey Pathira"
    # Whisper tiny.en commonly mis-transcribes the name, so we match
    # as many phonetic variants as possible.
    if any(w in t for w in (
        "pathira", "batira", "patira", "pathera",
        "passirah", "passira", "passera", "bathira",
        "pathirah", "pathyra", "patirah", "patiera",
        "basira", "bashira", "pathera", "patera",
        "hey pathira", "hey patira", "hey passira",
    )):
        return {"action": "greet", "kind": None, "source": None, "target": None}

    # Navigation commands
    if any(w in t for w in ("open home", "go home", "go to home", "take me home")):
        return {"action": "navigate", "kind": None, "source": None, "target": "home"}
    if any(w in t for w in ("open vision", "start vision", "go to vision", "open camera", "start assist")):
        return {"action": "navigate", "kind": None, "source": None, "target": "vision"}
    if any(w in t for w in ("open setting", "go to setting", "open preferences")):
        return {"action": "navigate", "kind": None, "source": None, "target": "settings"}
    if any(w in t for w in ("open tracking", "go to tracking", "where am i", "open map", "show map")):
        return {"action": "navigate", "kind": None, "source": None, "target": "tracking"}
    if any(w in t for w in ("open companion", "go to companion", "my companion", "show companion")):
        return {"action": "navigate", "kind": None, "source": None, "target": "companions"}
    if any(w in t for w in ("emergency", "help me", "i need help", "danger", "sos")):
        return {"action": "navigate", "kind": None, "source": None, "target": "emergency"}
    if any(w in t for w in ("call support", "open support", "customer service", "call center")):
        return {"action": "navigate", "kind": None, "source": None, "target": "support"}

    # Stop / pause / resume — any synonym
    for kw in ("stop", "pause", "turn off", "shut down", "exit", "close", "start", "begin", "resume", "turn on"):
        if kw in t:
            return {"action": "toggle_vision", "kind": None, "source": None, "target": None}

    # Camera source switching
    if any(w in t for w in ("glasses", "spectacles", "wearable", "headset")):
        return {"action": "set_camera_source", "kind": None, "source": "glasses", "target": None}
    if any(w in t for w in ("phone camera", "use phone", "switch to phone", "phone cam", "mobile camera")):
        return {"action": "set_camera_source", "kind": None, "source": "phone", "target": None}

    # Mode keywords (most specific first)
    # Direct "switch to <mode>" commands
    if any(w in t for w in ("switch to face", "switch face", "face mode", "face recognition")):
        return {"action": "set_kind", "kind": "face", "source": None, "target": None}
    if any(w in t for w in ("switch to ocr", "switch ocr", "ocr mode", "reading mode")):
        return {"action": "set_kind", "kind": "ocr", "source": None, "target": None}
    if any(w in t for w in ("switch to obstacle", "switch obstacle", "obstacle mode", "navigation mode")):
        return {"action": "set_kind", "kind": "obstacle", "source": None, "target": None}
    if any(w in t for w in ("switch to scene", "switch scene", "scene mode", "describe mode")):
        return {"action": "set_kind", "kind": "scene", "source": None, "target": None}
    if any(w in t for w in ("switch to detect", "switch detect", "detect mode", "detection mode")):
        return {"action": "set_kind", "kind": "detect", "source": None, "target": None}
    # Natural language commands
    if any(w in t for w in ("face", "who is", "recognize person", "recognise person", "identify person")):
        return {"action": "set_kind", "kind": "face", "source": None, "target": None}
    if any(w in t for w in ("read", "ocr", "text", "letters", "writing")):
        return {"action": "set_kind", "kind": "ocr", "source": None, "target": None}
    if any(w in t for w in ("obstacle", "wall", "door", "navigate", "navigation", "in front of me", "in my way")):
        return {"action": "set_kind", "kind": "obstacle", "source": None, "target": None}
    if any(w in t for w in ("describe", "scene", "what do i see", "what is happening", "what's around", "describe the")):
        return {"action": "set_kind", "kind": "scene", "source": None, "target": None}
    if any(w in t for w in ("object", "things", "items", "what are these", "identify object")):
        return {"action": "set_kind", "kind": "detect", "source": None, "target": None}

    return None

_SYSTEM_PROMPT = """You are an intent classifier for a vision-assistance app for blind users. The app has 5 vision modes:

- "obstacle": detect walls, doors, people, furniture in the user's path
- "detect" : general object detection (chairs, laptops, cups, etc.)
- "ocr"    : read printed text out loud
- "scene"  : describe the whole scene in a sentence
- "face"   : recognize faces of people the user already registered

It also has two camera sources:
- "phone"  : the device's own camera
- "glasses": a Raspberry Pi camera the user wears

It has navigation screens:
- "home", "vision", "settings", "tracking", "companions", "emergency", "support"

Read the user's spoken command and return ONE JSON object with this exact shape:
{"action": "<action>", "kind": "<kind>", "source": "<source>", "target": "<target>"}

Rules:
- "action" MUST be one of: "set_kind", "set_camera_source", "toggle_vision", "navigate", "greet", "noop"
- For "set_kind", "kind" is one of the 5 modes. Set "source" and "target" to null.
- For "set_camera_source", "source" is "phone" or "glasses". Set rest to null.
- For "toggle_vision", set kind/source/target to null.
- For "navigate", "target" is one of the screens. Set kind/source to null.
- For "greet" (user says the app name "Pathira"), set everything else to null.
- For "noop", set all to null.
- Output ONLY the JSON. No prose.

Examples:
"Pathira"               -> {"action":"greet","kind":null,"source":null,"target":null}
"Open home"             -> {"action":"navigate","kind":null,"source":null,"target":"home"}
"Open vision"           -> {"action":"navigate","kind":null,"source":null,"target":"vision"}
"Open settings"         -> {"action":"navigate","kind":null,"source":null,"target":"settings"}
"Emergency"             -> {"action":"navigate","kind":null,"source":null,"target":"emergency"}
"Open face recognition" -> {"action":"set_kind","kind":"face","source":null,"target":null}
"Read this text"        -> {"action":"set_kind","kind":"ocr","source":null,"target":null}
"Show obstacles"        -> {"action":"set_kind","kind":"obstacle","source":null,"target":null}
"Use the glasses"       -> {"action":"set_camera_source","kind":null,"source":"glasses","target":null}
"Stop"                  -> {"action":"toggle_vision","kind":null,"source":null,"target":null}
"Hello"                 -> {"action":"noop","kind":null,"source":null,"target":null}
"""


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Mistral occasionally wraps the JSON in markdown fences or adds a
    leading sentence. Strip and retry."""
    text = text.strip()
    # Strip ```json fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block in the text.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def interpret_intent_sync(transcript: str) -> Dict[str, Any]:
    """Send the transcript through Ollama and return a normalised intent dict.

    Falls back to a `noop` action if Ollama is unreachable or returns
    something we can't parse — that way the phone shows a sensible
    "couldn't understand" message instead of a 500.
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return {"action": "noop", "kind": None, "source": None, "transcript": ""}

    # 1) Cheap keyword shortcut — catches the common commands without
    # ever hitting Ollama. Works while Mistral loads in the background.
    kw = _keyword_intent(transcript)
    if kw is not None:
        kw["transcript"] = transcript
        kw["matched_via"] = "keywords"
        return kw

    # 2) Otherwise fall back to Mistral for anything fuzzier.
    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        "stream": False,
        "format": "json",  # Ollama 0.1.30+ honours this and forces JSON output
        "options": {"temperature": 0.0, "num_predict": 64},
    }

    try:
        r = requests.post(
            f"{settings.OLLAMA_URL.rstrip('/')}/api/chat",
            json=payload,
            timeout=settings.OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        body = r.json()
        # Ollama /api/chat returns {"message": {"role":"assistant","content":"..."}, ...}
        content = (body.get("message") or {}).get("content") or ""
        parsed = _safe_json_loads(content) or {}
    except Exception as exc:
        return {
            "action": "noop",
            "kind": None,
            "source": None,
            "transcript": transcript,
            "error": f"ollama: {exc}",
        }

    action = parsed.get("action") if parsed.get("action") in _VALID_ACTIONS else "noop"
    kind = parsed.get("kind") if parsed.get("kind") in _VALID_KINDS else None
    source = parsed.get("source") if parsed.get("source") in _VALID_SOURCES else None
    target = parsed.get("target") if parsed.get("target") in _VALID_TARGETS else None

    # Sanity: enforce shape contracts.
    if action == "set_kind" and kind is None:
        action = "noop"
    if action == "set_camera_source" and source is None:
        action = "noop"
    if action == "navigate" and target is None:
        action = "noop"

    return {
        "action": action,
        "kind": kind,
        "source": source,
        "target": target,
        "transcript": transcript,
        "matched_via": "mistral",
    }
