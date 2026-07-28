import uuid
import pytest
from httpx import AsyncClient
from sqlalchemy.future import select

from backend.db.models.support_session import SupportSession


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
async def test_availability_stub(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    r = await client.get(
        "/api/support/availability",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["available"] is True
    assert out["wait_time_minutes"] == 2


@pytest.mark.asyncio
async def test_voice_call_returns_session(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    r = await client.post(
        "/api/support/call/voice",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["success"] is True
    assert out["session_id"]
    assert "stub" in out["message"].lower()


@pytest.mark.asyncio
async def test_video_call_returns_session(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    r = await client.post(
        "/api/support/call/video",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["success"] is True
    assert out["session_id"]
    assert "stub" in out["message"].lower()


@pytest.mark.asyncio
async def test_chat_returns_session(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    r = await client.post(
        "/api/support/chat",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["success"] is True
    assert out["session_id"]
    assert "stub" in out["message"].lower()


@pytest.mark.asyncio
async def test_voice_call_requires_auth(client):
    r = await client.post("/api/support/call/voice")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_voice_call_persists_row(client, db, unique_email):
    body = await _signup(client, unique_email, utype="main")
    r = await client.post(
        "/api/support/call/voice",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200
    sid = uuid.UUID(r.json()["session_id"])

    result = await db.execute(
        select(SupportSession).where(SupportSession.id == sid)
    )
    row = result.scalars().first()
    assert row is not None
    assert row.kind == "voice"
    assert row.status == "requested"
