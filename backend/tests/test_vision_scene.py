"""Tests for the scene-description endpoint.

`run_scene_sync` is mocked end-to-end; we never load BLIP or MarianMT.
"""

import pytest
from httpx import AsyncClient

from backend.api.endpoints import vision as vision_module


_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)


_FAKE = {
    "caption_en": "a person standing in a kitchen next to a refrigerator",
    "caption_ar": "شخص يقف في مطبخ بجوار ثلاجة",
}


async def _signup(client: AsyncClient, email: str) -> dict:
    r = await client.post(
        "/api/auth/signup",
        json={
            "name": "X",
            "email": email,
            "password": "Pa55word!",
            "user_type": "main",
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


@pytest.mark.asyncio
async def test_scene_returns_captions_and_persists(client, unique_email, monkeypatch):
    def _fake(b, translate=True):
        assert isinstance(b, (bytes, bytearray)) and len(b) > 0
        assert translate is True
        return _FAKE

    monkeypatch.setattr(vision_module, "run_scene_sync", _fake)

    body = await _signup(client, unique_email)
    h = {"Authorization": f"Bearer {body['access_token']}"}

    r = await client.post(
        "/api/vision/scene",
        headers=h,
        files={"image": ("scene.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["logged"] is True
    assert out["event_id"]
    assert out["caption_en"] == _FAKE["caption_en"]
    assert out["caption_ar"] == _FAKE["caption_ar"]

    hist = await client.get("/api/vision/history", headers=h)
    assert hist.status_code == 200
    rows = hist.json()
    scene_rows = [row for row in rows if row["summary"] == "scene"]
    assert len(scene_rows) == 1
    assert scene_rows[0]["detections"] == [
        {"caption_en": _FAKE["caption_en"], "caption_ar": _FAKE["caption_ar"]}
    ]


@pytest.mark.asyncio
async def test_scene_translate_false_skips_arabic(client, unique_email, monkeypatch):
    captured: list = []

    def _fake(b, translate=True):
        captured.append(translate)
        return {"caption_en": "a chair", "caption_ar": None}

    monkeypatch.setattr(vision_module, "run_scene_sync", _fake)

    body = await _signup(client, unique_email)
    h = {"Authorization": f"Bearer {body['access_token']}"}
    r = await client.post(
        "/api/vision/scene?translate=false",
        headers=h,
        files={"image": ("scene.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 200
    out = r.json()
    assert captured == [False]
    assert out["caption_en"] == "a chair"
    assert out["caption_ar"] is None


@pytest.mark.asyncio
async def test_scene_rejects_oversize(client, unique_email, monkeypatch):
    def _boom(*_a, **_kw):  # pragma: no cover
        raise AssertionError("run_scene_sync called for oversized payload")

    monkeypatch.setattr(vision_module, "run_scene_sync", _boom)

    body = await _signup(client, unique_email)
    h = {"Authorization": f"Bearer {body['access_token']}"}
    big = b"\x00" * (6 * 1024 * 1024)
    r = await client.post(
        "/api/vision/scene",
        headers=h,
        files={"image": ("big.bin", big, "application/octet-stream")},
    )
    assert r.status_code == 413, r.text


@pytest.mark.asyncio
async def test_scene_requires_auth(client):
    r = await client.post(
        "/api/vision/scene",
        files={"image": ("img.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_scene_pushes_to_ws(client, unique_email, monkeypatch):
    monkeypatch.setattr(vision_module, "run_scene_sync", lambda b, translate=True: _FAKE)

    calls: list = []

    async def _fake_send(user_id, payload):
        calls.append((user_id, payload))
        return 1

    monkeypatch.setattr(vision_module.vision_manager, "send_to_user", _fake_send)

    body = await _signup(client, unique_email)
    h = {"Authorization": f"Bearer {body['access_token']}"}
    r = await client.post(
        "/api/vision/scene",
        headers=h,
        files={"image": ("scene.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 200
    assert len(calls) == 1
    user_id, payload = calls[0]
    assert str(user_id) == body["user"]["id"]
    assert payload["type"] == "detection"
    assert payload["kind"] == "scene"
    assert payload["event_id"] == r.json()["event_id"]
    assert payload["caption_en"] == _FAKE["caption_en"]
    assert payload["caption_ar"] == _FAKE["caption_ar"]
    # backwards-compat: WS clients that read `text` get the EN caption.
    assert payload["text"] == _FAKE["caption_en"]
