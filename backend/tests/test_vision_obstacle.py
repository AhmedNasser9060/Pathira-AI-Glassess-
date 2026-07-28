"""Tests for the Phase-5 server-side obstacle detection endpoint.

`run_obstacle_sync` is mocked end-to-end; we never load the real weights.
"""

import pytest
from httpx import AsyncClient

from backend.api.endpoints import vision as vision_module


_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)


_FAKE_RESULT = {
    "image_shape": {"width": 640, "height": 480},
    "detections": [
        {
            "label": "person",
            "confidence": 0.91,
            "bbox": [10.0, 20.0, 100.0, 200.0],
            "is_critical": True,
            "priority_score": 0.74,
        },
        {
            "label": "chair",
            "confidence": 0.55,
            "bbox": [200.0, 50.0, 80.0, 90.0],
            "is_critical": True,
            "priority_score": 0.31,
        },
    ],
    "highest_priority": {
        "label": "person",
        "confidence": 0.91,
        "bbox": [10.0, 20.0, 100.0, 200.0],
        "is_critical": True,
        "priority_score": 0.74,
    },
    "voice_guidance": ["Person very close on your left", "Chair detected ahead"],
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
async def test_obstacle_returns_full_payload_and_persists(client, unique_email, monkeypatch):
    def _fake(b, conf=0.25, top_n=3):
        assert isinstance(b, (bytes, bytearray)) and len(b) > 0
        return _FAKE_RESULT

    monkeypatch.setattr(vision_module, "run_obstacle_sync", _fake)

    body = await _signup(client, unique_email)
    h = {"Authorization": f"Bearer {body['access_token']}"}

    r = await client.post(
        "/api/vision/obstacle",
        headers=h,
        files={"image": ("scene.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["logged"] is True
    assert out["event_id"]
    assert out["detections"][0]["label"] == "person"
    assert out["highest_priority"]["label"] == "person"
    assert out["voice_guidance"] == ["Person very close on your left", "Chair detected ahead"]
    assert out["image_shape"] == {"width": 640, "height": 480}

    # Persisted row visible in /history with summary='obstacle'.
    hist = await client.get("/api/vision/history", headers=h)
    assert hist.status_code == 200
    rows = hist.json()
    obstacle_rows = [row for row in rows if row["summary"] == "obstacle"]
    assert len(obstacle_rows) == 1
    assert obstacle_rows[0]["detections"] == _FAKE_RESULT["detections"]


@pytest.mark.asyncio
async def test_obstacle_rejects_oversize(client, unique_email, monkeypatch):
    def _boom(*_a, **_kw):  # pragma: no cover
        raise AssertionError("run_obstacle_sync called for oversized payload")

    monkeypatch.setattr(vision_module, "run_obstacle_sync", _boom)

    body = await _signup(client, unique_email)
    h = {"Authorization": f"Bearer {body['access_token']}"}

    big = b"\x00" * (6 * 1024 * 1024)
    r = await client.post(
        "/api/vision/obstacle",
        headers=h,
        files={"image": ("big.bin", big, "application/octet-stream")},
    )
    assert r.status_code == 413, r.text


@pytest.mark.asyncio
async def test_obstacle_requires_auth(client):
    r = await client.post(
        "/api/vision/obstacle",
        files={"image": ("img.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_obstacle_pushes_to_ws(client, unique_email, monkeypatch):
    """Phase 4 bridge regression: WS push includes voice_guidance."""
    monkeypatch.setattr(vision_module, "run_obstacle_sync", lambda b, conf=0.25, top_n=3: _FAKE_RESULT)

    calls: list = []

    async def _fake_send(user_id, payload):
        calls.append((user_id, payload))
        return 1

    monkeypatch.setattr(vision_module.vision_manager, "send_to_user", _fake_send)

    body = await _signup(client, unique_email)
    h = {"Authorization": f"Bearer {body['access_token']}"}
    r = await client.post(
        "/api/vision/obstacle",
        headers=h,
        files={"image": ("scene.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 200
    assert len(calls) == 1
    user_id, payload = calls[0]
    assert str(user_id) == body["user"]["id"]
    assert payload["type"] == "detection"
    assert payload["kind"] == "obstacle"
    assert payload["event_id"] == r.json()["event_id"]
    assert payload["objects"] == _FAKE_RESULT["detections"]
    assert payload["faces"] == []
    assert payload["voice_guidance"] == _FAKE_RESULT["voice_guidance"]


@pytest.mark.asyncio
async def test_obstacle_clear_path_returns_default_voice(client, unique_email, monkeypatch):
    """Empty detections -> single 'Path is clear' voice line."""
    empty_result = {
        "image_shape": {"width": 640, "height": 480},
        "detections": [],
        "highest_priority": None,
        "voice_guidance": ["Path is clear"],
    }
    monkeypatch.setattr(vision_module, "run_obstacle_sync", lambda b, conf=0.25, top_n=3: empty_result)

    body = await _signup(client, unique_email)
    h = {"Authorization": f"Bearer {body['access_token']}"}
    r = await client.post(
        "/api/vision/obstacle",
        headers=h,
        files={"image": ("scene.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 200
    out = r.json()
    assert out["detections"] == []
    assert out["highest_priority"] is None
    assert out["voice_guidance"] == ["Path is clear"]
