import pytest
from httpx import AsyncClient


async def _signup(client: AsyncClient, email: str):
    r = await client.post(
        "/api/auth/signup",
        json={"name": "Me", "email": email, "password": "Pa55word!"},
    )
    assert r.status_code == 201, r.text
    return r.json()


@pytest.mark.asyncio
async def test_me_returns_current_user(client: AsyncClient, unique_email: str):
    body = await _signup(client, unique_email)
    r = await client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200, r.text
    user = r.json()
    assert user["email"] == unique_email
    assert user["name"] == "Me"
    assert "password_hash" not in user
    assert user["id"] == body["user"]["id"]


@pytest.mark.asyncio
async def test_me_requires_auth(client: AsyncClient):
    r = await client.get("/api/users/me")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_me_rejects_refresh_token(client: AsyncClient, unique_email: str):
    body = await _signup(client, unique_email)
    r = await client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {body['refresh_token']}"},  # wrong type
    )
    assert r.status_code == 401
