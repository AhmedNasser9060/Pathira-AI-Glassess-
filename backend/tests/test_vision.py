import pytest
from httpx import AsyncClient


async def _signup(
    client: AsyncClient, email: str, name: str = "X", utype: str | None = "main"
):
    body = {"name": name, "email": email, "password": "Pa55word!"}
    if utype is not None:
        body["user_type"] = utype
    r = await client.post("/api/auth/signup", json=body)
    assert r.status_code == 201, r.text
    return r.json()


@pytest.mark.asyncio
async def test_detect_logs_event(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    r = await client.post(
        "/api/vision/detect",
        json={
            "objects": [
                {"label": "person", "confidence": 0.92, "bbox": [0.1, 0.2, 0.3, 0.4]}
            ],
            "text": "stop sign",
        },
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["logged"] is True
    assert out["event_id"]


@pytest.mark.asyncio
async def test_detect_with_empty_objects(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    r = await client.post(
        "/api/vision/detect",
        json={"objects": []},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["logged"] is True
    assert out["event_id"]


@pytest.mark.asyncio
async def test_toggle_on_then_off_persists_two_rows(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    h = {"Authorization": f"Bearer {body['access_token']}"}

    r1 = await client.post("/api/vision/toggle", json={"is_active": True}, headers=h)
    assert r1.status_code == 200
    assert r1.json()["is_active"] is True

    r2 = await client.post("/api/vision/toggle", json={"is_active": False}, headers=h)
    assert r2.status_code == 200
    assert r2.json()["is_active"] is False

    hist = await client.get("/api/vision/history", headers=h)
    assert hist.status_code == 200
    rows = hist.json()
    toggle_rows = [r for r in rows if r["summary"] == "toggle"]
    assert len(toggle_rows) == 2


@pytest.mark.asyncio
async def test_history_returns_desc(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    h = {"Authorization": f"Bearer {body['access_token']}"}

    await client.post("/api/vision/detect", json={"objects": []}, headers=h)
    await client.post("/api/vision/detect", json={"objects": []}, headers=h)
    await client.post("/api/vision/detect", json={"objects": []}, headers=h)

    r = await client.get("/api/vision/history", headers=h)
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) >= 3
    # Most recent first.
    times = [row["created_at"] for row in rows]
    assert times == sorted(times, reverse=True)


@pytest.mark.asyncio
async def test_detect_requires_auth(client):
    r = await client.post("/api/vision/detect", json={"objects": []})
    assert r.status_code == 401
