"""Tests for the dual-mode (JSON or multipart) /api/vision/detect endpoint.

The existing JSON-only tests live in test_vision.py and remain unchanged;
this file covers the new multipart/server-YOLO branch.
"""

import pytest
from httpx import AsyncClient

from backend.api.endpoints import vision as vision_module


# Smallest valid 1x1 PNG.
_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)


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
async def test_detect_multipart_runs_detector_and_logs(client, unique_email, monkeypatch):
    fake_objects = [
        {"label": "person", "confidence": 0.91, "bbox": [10.0, 20.0, 100.0, 200.0]},
        {"label": "chair", "confidence": 0.55, "bbox": [200.0, 50.0, 80.0, 90.0]},
    ]

    def _fake_detect(b, conf=0.25):
        assert isinstance(b, (bytes, bytearray)) and len(b) > 0
        return fake_objects

    monkeypatch.setattr(vision_module, "run_detect_sync", _fake_detect)

    body = await _signup(client, unique_email)
    h = {"Authorization": f"Bearer {body['access_token']}"}

    r = await client.post(
        "/api/vision/detect",
        headers=h,
        files={"image": ("scene.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["logged"] is True
    assert out["event_id"]

    hist = await client.get("/api/vision/history", headers=h)
    assert hist.status_code == 200
    rows = hist.json()
    detect_rows = [row for row in rows if row["summary"] == "detect"]
    assert len(detect_rows) == 1
    assert detect_rows[0]["detections"] == fake_objects


@pytest.mark.asyncio
async def test_detect_multipart_rejects_oversize(client, unique_email, monkeypatch):
    def _boom(*_a, **_kw):  # pragma: no cover - shouldn't be called
        raise AssertionError("run_detect_sync called for oversized payload")

    monkeypatch.setattr(vision_module, "run_detect_sync", _boom)

    body = await _signup(client, unique_email)
    h = {"Authorization": f"Bearer {body['access_token']}"}

    big = b"\x00" * (6 * 1024 * 1024)
    r = await client.post(
        "/api/vision/detect",
        headers=h,
        files={"image": ("big.bin", big, "application/octet-stream")},
    )
    assert r.status_code == 413, r.text


@pytest.mark.asyncio
async def test_detect_multipart_missing_image_field(client, unique_email, monkeypatch):
    def _boom(*_a, **_kw):  # pragma: no cover
        raise AssertionError("run_detect_sync called without an image")

    monkeypatch.setattr(vision_module, "run_detect_sync", _boom)

    body = await _signup(client, unique_email)
    h = {"Authorization": f"Bearer {body['access_token']}"}

    # multipart with the WRONG field name — endpoint must 400.
    r = await client.post(
        "/api/vision/detect",
        headers=h,
        files={"not_image": ("img.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 400, r.text


@pytest.mark.asyncio
async def test_detect_json_path_unchanged(client, unique_email, monkeypatch):
    """Regression guard: the JSON branch must NOT call run_detect_sync."""
    def _boom(*_a, **_kw):  # pragma: no cover
        raise AssertionError("run_detect_sync called for JSON request")

    monkeypatch.setattr(vision_module, "run_detect_sync", _boom)

    body = await _signup(client, unique_email)
    h = {"Authorization": f"Bearer {body['access_token']}"}

    r = await client.post(
        "/api/vision/detect",
        json={
            "objects": [
                {"label": "stop_sign", "confidence": 0.88, "bbox": [0.0, 0.0, 50.0, 50.0]}
            ],
            "text": "STOP",
        },
        headers=h,
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["logged"] is True
    assert out["event_id"]


@pytest.mark.asyncio
async def test_detect_requires_auth_for_multipart(client):
    r = await client.post(
        "/api/vision/detect",
        files={"image": ("img.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 401
