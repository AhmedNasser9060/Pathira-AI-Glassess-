"""Tests for the Phase-4 REST -> WebSocket bridge.

We mock `vision_manager.send_to_user` and assert it is called exactly once
per successful REST handler with the right user_id and payload shape.

Why mock at the manager level rather than open a real WS? TestClient's
WebSocket context is sync and would block the AsyncClient that submits
the REST request. Mocking the dispatcher proves the REST -> manager
contract; the WS endpoint's auth + loop behavior is tested separately
in test_websockets.py.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from backend.api.endpoints import faces as faces_module
from backend.api.endpoints import vision as vision_module


_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)


def _install_capture(monkeypatch, target_module) -> list:
    """Replace target_module.vision_manager.send_to_user with a capturing
    async stub. Returns a list that will receive (user_id, payload) tuples."""
    calls: list = []

    async def _fake_send(user_id, payload):
        calls.append((user_id, payload))
        return 1

    monkeypatch.setattr(target_module.vision_manager, "send_to_user", _fake_send)
    return calls


async def _signup(client: AsyncClient, email: str, utype: str = "main") -> dict:
    r = await client.post(
        "/api/auth/signup",
        json={
            "name": "X",
            "email": email,
            "password": "Pa55word!",
            "user_type": utype,
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# /api/vision/ocr
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ocr_pushes_to_ws(client, unique_email, monkeypatch):
    monkeypatch.setattr(vision_module, "run_ocr_sync", lambda b, lang="eng+ara": "hi")
    calls = _install_capture(monkeypatch, vision_module)

    body = await _signup(client, unique_email)
    h = {"Authorization": f"Bearer {body['access_token']}"}

    r = await client.post(
        "/api/vision/ocr",
        headers=h,
        files={"image": ("img.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 200
    assert len(calls) == 1
    user_id, payload = calls[0]
    assert str(user_id) == body["user"]["id"]
    assert payload["type"] == "detection"
    assert payload["kind"] == "ocr"
    assert payload["text"] == "hi"
    assert payload["objects"] == []
    assert payload["faces"] == []
    assert payload["event_id"] == r.json()["event_id"]


# ---------------------------------------------------------------------------
# /api/vision/detect (multipart YOLO)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_detect_multipart_pushes_to_ws(client, unique_email, monkeypatch):
    fake_objects = [
        {"label": "person", "confidence": 0.9, "bbox": [1.0, 2.0, 3.0, 4.0]},
    ]
    monkeypatch.setattr(vision_module, "run_detect_sync", lambda b, conf=0.25: fake_objects)
    calls = _install_capture(monkeypatch, vision_module)

    body = await _signup(client, unique_email)
    h = {"Authorization": f"Bearer {body['access_token']}"}

    r = await client.post(
        "/api/vision/detect",
        headers=h,
        files={"image": ("scene.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 200
    assert len(calls) == 1
    user_id, payload = calls[0]
    assert str(user_id) == body["user"]["id"]
    assert payload["kind"] == "detect"
    assert payload["objects"] == fake_objects
    assert payload["faces"] == []
    assert payload["event_id"] == r.json()["event_id"]


@pytest.mark.asyncio
async def test_detect_json_pushes_to_ws(client, unique_email, monkeypatch):
    """JSON branch (existing on-device pipeline) also bridges through the WS."""
    calls = _install_capture(monkeypatch, vision_module)

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
    assert r.status_code == 200
    assert len(calls) == 1
    user_id, payload = calls[0]
    assert str(user_id) == body["user"]["id"]
    assert payload["kind"] == "detect"
    assert payload["objects"][0]["label"] == "stop_sign"
    assert payload["text"] == "STOP"
    assert payload["faces"] == []


# ---------------------------------------------------------------------------
# /api/companion/faces/recognize — both paths bridge to the MAIN user
# ---------------------------------------------------------------------------

async def _setup_companion_link(client: AsyncClient, unique_email: str, perms: dict):
    cmp_email = unique_email
    cmp = await _signup(client, cmp_email, "companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, "main")
    r = await client.post(
        "/api/companions",
        json={"email": cmp_email, "permissions": perms},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert r.status_code in (200, 201)
    return cmp, main


@pytest.mark.asyncio
async def test_recognize_multipart_pushes_to_main_user_ws(client, unique_email, monkeypatch):
    """Multipart match -> push to TARGET user (the main user being monitored)."""
    fixed_emb = [1.0] + [0.0] * 511
    monkeypatch.setattr(faces_module, "compute_embedding_sync", lambda b: fixed_emb)
    calls = _install_capture(monkeypatch, faces_module)

    cmp, main = await _setup_companion_link(client, unique_email, {"camera": True, "alerts": True})
    h = {"Authorization": f"Bearer {cmp['access_token']}"}

    # Register Mom with the same embedding so we get cosine == 1.0.
    cr = await client.post(
        "/api/companion/faces",
        headers=h,
        json={
            "user_id": main["user"]["id"],
            "name": "Mom",
            "embedding": fixed_emb,
        },
    )
    assert cr.status_code == 201

    r = await client.post(
        "/api/companion/faces/recognize",
        headers=h,
        data={"user_id": main["user"]["id"]},
        files={"image": ("probe.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 200, r.text
    assert len(calls) == 1
    user_id, payload = calls[0]
    # IMPORTANT: bridge target is the main user, not the calling companion.
    assert str(user_id) == main["user"]["id"]
    assert payload["kind"] == "face"
    assert payload["objects"] == []
    assert len(payload["faces"]) == 1
    assert payload["faces"][0]["name"] == "Mom"
    assert payload["faces"][0]["similarity"] >= 0.99
    # alert_id propagates because perms include alerts:True.
    assert payload["faces"][0]["alert_id"] is not None


@pytest.mark.asyncio
async def test_recognize_multipart_no_match_pushes_empty_faces(client, unique_email, monkeypatch):
    """Multipart with no match -> still push (faces=[]) so the frontend can
    surface a no-match indicator."""
    monkeypatch.setattr(faces_module, "compute_embedding_sync", lambda b: [1.0] + [0.0] * 511)
    calls = _install_capture(monkeypatch, faces_module)

    cmp, main = await _setup_companion_link(client, unique_email, {"camera": True, "alerts": True})
    h = {"Authorization": f"Bearer {cmp['access_token']}"}
    # Register Mom with an ORTHOGONAL embedding: cosine == 0 -> no match.
    cr = await client.post(
        "/api/companion/faces",
        headers=h,
        json={
            "user_id": main["user"]["id"],
            "name": "Mom",
            "embedding": [0.0, 1.0] + [0.0] * 510,
        },
    )
    assert cr.status_code == 201

    r = await client.post(
        "/api/companion/faces/recognize",
        headers=h,
        data={"user_id": main["user"]["id"]},
        files={"image": ("probe.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 200
    assert len(calls) == 1
    _user_id, payload = calls[0]
    assert payload["kind"] == "face"
    assert payload["faces"] == []


@pytest.mark.asyncio
async def test_recognize_json_pushes_to_main_user_ws(client, unique_email, monkeypatch):
    """JSON branch: also bridges, also targets the main user."""
    calls = _install_capture(monkeypatch, faces_module)

    cmp, main = await _setup_companion_link(client, unique_email, {"camera": True, "alerts": True})
    h = {"Authorization": f"Bearer {cmp['access_token']}"}

    cr = await client.post(
        "/api/companion/faces",
        headers=h,
        json={"user_id": main["user"]["id"], "name": "Dad"},
    )
    assert cr.status_code == 201
    face_id = cr.json()["id"]

    r = await client.post(
        "/api/companion/faces/recognize",
        headers=h,
        json={
            "user_id": main["user"]["id"],
            "face_id": face_id,
            "confidence": 0.91,
        },
    )
    assert r.status_code == 200
    assert len(calls) == 1
    user_id, payload = calls[0]
    assert str(user_id) == main["user"]["id"]
    assert payload["kind"] == "face"
    assert payload["faces"][0]["face_id"] == face_id
    assert payload["faces"][0]["name"] == "Dad"
    assert payload["faces"][0]["similarity"] == 0.91
