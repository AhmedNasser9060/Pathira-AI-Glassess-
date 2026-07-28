import pytest
from httpx import AsyncClient


async def _signup(client: AsyncClient, email: str):
    r = await client.post(
        "/api/auth/signup",
        json={"name": "T", "email": email, "password": "Pa55word!"},
    )
    assert r.status_code == 201
    return r.json()


@pytest.mark.asyncio
async def test_set_type_main(client: AsyncClient, unique_email: str):
    body = await _signup(client, unique_email)
    r = await client.post(
        "/api/users/set-type",
        json={"user_type": "main"},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["user_type"] == "main"


@pytest.mark.asyncio
async def test_set_type_companion(client: AsyncClient, unique_email: str):
    body = await _signup(client, unique_email)
    r = await client.post(
        "/api/users/set-type",
        json={"user_type": "companion"},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200
    assert r.json()["user_type"] == "companion"


@pytest.mark.asyncio
async def test_set_type_invalid_422(client: AsyncClient, unique_email: str):
    body = await _signup(client, unique_email)
    r = await client.post(
        "/api/users/set-type",
        json={"user_type": "admin"},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_set_type_requires_auth(client: AsyncClient):
    r = await client.post("/api/users/set-type", json={"user_type": "main"})
    assert r.status_code == 401
