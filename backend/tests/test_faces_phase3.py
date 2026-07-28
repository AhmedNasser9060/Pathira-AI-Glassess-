"""Tests for the Phase-3 server-side face recognition path.

The existing JSON-only tests live in `test_faces.py` and remain unchanged.
This file covers:
  - POST /api/companion/faces multipart -> embedding stored
  - POST /api/companion/faces multipart with no detected face -> 422
  - POST /api/companion/faces/recognize multipart -> match -> Alert created
  - POST /api/companion/faces/recognize multipart -> no match -> no Alert
  - POST /api/companion/faces/recognize multipart -> 5 MB cap -> 413
"""

import pytest
from httpx import AsyncClient

from backend.api.endpoints import faces as faces_module


_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)


async def _signup(client: AsyncClient, email: str, utype: str) -> dict:
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


async def _link(client: AsyncClient, main: dict, cmp_email: str, perms: dict) -> None:
    r = await client.post(
        "/api/companions",
        json={"email": cmp_email, "permissions": perms},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert r.status_code in (200, 201), r.text


# ---------------------------------------------------------------------------
# /faces multipart
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_face_multipart_stores_embedding(client, unique_email, monkeypatch):
    fake_emb = [0.1] * 512

    def _fake_embed(b):
        assert isinstance(b, (bytes, bytearray)) and len(b) > 0
        return fake_emb

    monkeypatch.setattr(faces_module, "compute_embedding_sync", _fake_embed)

    cmp_email = unique_email
    cmp = await _signup(client, cmp_email, "companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, "main")
    await _link(client, main, cmp_email, {"camera": True})

    h = {"Authorization": f"Bearer {cmp['access_token']}"}
    r = await client.post(
        "/api/companion/faces",
        headers=h,
        data={"user_id": main["user"]["id"], "name": "Mom", "relationship": "parent"},
        files={"image": ("mom.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["name"] == "Mom"
    assert out["relationship"] == "parent"
    assert out["embedding"] == fake_emb
    assert out["added_by"] == cmp["user"]["id"]


@pytest.mark.asyncio
async def test_add_face_multipart_no_face_detected_422(client, unique_email, monkeypatch):
    monkeypatch.setattr(faces_module, "compute_embedding_sync", lambda b: None)

    cmp_email = unique_email
    cmp = await _signup(client, cmp_email, "companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, "main")
    await _link(client, main, cmp_email, {"camera": True})

    h = {"Authorization": f"Bearer {cmp['access_token']}"}
    r = await client.post(
        "/api/companion/faces",
        headers=h,
        data={"user_id": main["user"]["id"], "name": "Nobody"},
        files={"image": ("blank.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 422, r.text


# ---------------------------------------------------------------------------
# /faces/recognize multipart
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recognize_multipart_match_creates_alert(client, unique_email, monkeypatch):
    """Probe embedding == registered embedding -> cosine == 1.0 -> match -> Alert."""
    fixed_emb = [1.0] + [0.0] * 511  # arbitrary unit vector
    monkeypatch.setattr(faces_module, "compute_embedding_sync", lambda b: fixed_emb)

    cmp_email = unique_email
    cmp = await _signup(client, cmp_email, "companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, "main")
    await _link(client, main, cmp_email, {"camera": True, "alerts": True})

    h = {"Authorization": f"Bearer {cmp['access_token']}"}
    # Register Mom with a JSON body so we can pin her embedding == fixed_emb.
    cr = await client.post(
        "/api/companion/faces",
        headers=h,
        json={
            "user_id": main["user"]["id"],
            "name": "Mom",
            "relationship": "parent",
            "embedding": fixed_emb,
        },
    )
    assert cr.status_code == 201, cr.text
    mom_id = cr.json()["id"]

    r = await client.post(
        "/api/companion/faces/recognize",
        headers=h,
        data={"user_id": main["user"]["id"]},
        files={"image": ("probe.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["logged"] is True
    assert body["alert_created"] is True
    assert body["alert_id"] is not None
    assert body["matched_face_id"] == mom_id
    assert body["matched_name"] == "Mom"
    assert body["similarity"] >= 0.99  # cosine of a vector with itself

    # Main user sees the alert
    alerts = await client.get(
        "/api/alerts",
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert any("Mom" in a["title"] for a in alerts.json())


@pytest.mark.asyncio
async def test_recognize_multipart_no_match_below_threshold(client, unique_email, monkeypatch):
    """Probe embedding orthogonal to registered embedding -> cosine == 0 -> no match."""
    monkeypatch.setattr(faces_module, "compute_embedding_sync", lambda b: [1.0] + [0.0] * 511)

    cmp_email = unique_email
    cmp = await _signup(client, cmp_email, "companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, "main")
    await _link(client, main, cmp_email, {"camera": True, "alerts": True})

    h = {"Authorization": f"Bearer {cmp['access_token']}"}
    # Register Mom with an ORTHOGONAL embedding so cosine == 0 < threshold.
    cr = await client.post(
        "/api/companion/faces",
        headers=h,
        json={
            "user_id": main["user"]["id"],
            "name": "Mom",
            "embedding": [0.0, 1.0] + [0.0] * 510,
        },
    )
    assert cr.status_code == 201, cr.text

    r = await client.post(
        "/api/companion/faces/recognize",
        headers=h,
        data={"user_id": main["user"]["id"]},
        files={"image": ("probe.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["logged"] is True
    assert body["alert_created"] is False
    assert body["alert_id"] is None
    assert body["matched_face_id"] is None
    assert body["matched_name"] is None
    assert body["similarity"] is not None and body["similarity"] < 0.6

    # No alert was created.
    alerts = await client.get(
        "/api/alerts",
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert alerts.json() == []


@pytest.mark.asyncio
async def test_recognize_multipart_rejects_oversize(client, unique_email, monkeypatch):
    def _boom(*_a, **_kw):  # pragma: no cover
        raise AssertionError("compute_embedding_sync called for oversized payload")

    monkeypatch.setattr(faces_module, "compute_embedding_sync", _boom)

    cmp_email = unique_email
    cmp = await _signup(client, cmp_email, "companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, "main")
    await _link(client, main, cmp_email, {"camera": True, "alerts": True})

    h = {"Authorization": f"Bearer {cmp['access_token']}"}
    big = b"\x00" * (6 * 1024 * 1024)
    r = await client.post(
        "/api/companion/faces/recognize",
        headers=h,
        data={"user_id": main["user"]["id"]},
        files={"image": ("big.bin", big, "application/octet-stream")},
    )
    assert r.status_code == 413, r.text


@pytest.mark.asyncio
async def test_recognize_json_path_unchanged(client, unique_email, monkeypatch):
    """Regression guard: JSON branch must NOT call compute_embedding_sync."""
    def _boom(*_a, **_kw):  # pragma: no cover
        raise AssertionError("compute_embedding_sync called for JSON request")

    monkeypatch.setattr(faces_module, "compute_embedding_sync", _boom)

    cmp_email = unique_email
    cmp = await _signup(client, cmp_email, "companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, "main")
    await _link(client, main, cmp_email, {"camera": True, "alerts": True})

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
            "confidence": 0.95,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["logged"] is True
    assert body["alert_created"] is True
    # Multipart-only fields stay None on the JSON path.
    assert body["matched_face_id"] is None
    assert body["matched_name"] is None
    assert body["similarity"] is None
