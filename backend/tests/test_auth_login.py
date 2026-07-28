import pytest
from httpx import AsyncClient


async def _signup(client: AsyncClient, email: str, password: str = "Pa55word!"):
    r = await client.post(
        "/api/auth/signup",
        json={"name": "Login Test", "email": email, "password": password},
    )
    assert r.status_code == 201, r.text
    return r.json()


@pytest.mark.asyncio
async def test_login_returns_tokens(client: AsyncClient, unique_email: str):
    await _signup(client, unique_email)
    r = await client.post(
        "/api/auth/login",
        json={"email": unique_email, "password": "Pa55word!"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["token_type"] == "bearer"
    assert body["user"]["email"] == unique_email


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient, unique_email: str):
    await _signup(client, unique_email)
    r = await client.post(
        "/api/auth/login",
        json={"email": unique_email, "password": "WrongPass!"},
    )
    assert r.status_code == 400
    assert "incorrect" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_login_unknown_email(client: AsyncClient):
    r = await client.post(
        "/api/auth/login",
        json={"email": "ghost-pytest@example.test", "password": "Pa55word!"},
    )
    assert r.status_code == 400
