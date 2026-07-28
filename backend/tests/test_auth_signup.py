import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_signup_returns_tokens_and_user(client: AsyncClient, unique_email: str):
    payload = {
        "name": "Alice",
        "email": unique_email,
        "password": "Pa55word!",
        "user_type": "main",
    }
    r = await client.post("/api/auth/signup", json=payload)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["user"]["email"] == unique_email
    assert body["user"]["name"] == "Alice"
    assert body["user"]["user_type"] == "main"
    assert "password_hash" not in body["user"]
    assert "password" not in body["user"]


@pytest.mark.asyncio
async def test_signup_rejects_duplicate_email(client: AsyncClient, unique_email: str):
    payload = {"name": "A", "email": unique_email, "password": "Pa55word!"}
    r1 = await client.post("/api/auth/signup", json=payload)
    assert r1.status_code == 201
    r2 = await client.post("/api/auth/signup", json=payload)
    assert r2.status_code == 400
    assert "already exists" in r2.json()["detail"].lower()


@pytest.mark.asyncio
async def test_signup_rejects_short_password(client: AsyncClient, unique_email: str):
    payload = {"name": "A", "email": unique_email, "password": "short"}
    r = await client.post("/api/auth/signup", json=payload)
    assert r.status_code == 422
