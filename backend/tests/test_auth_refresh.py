import pytest
from httpx import AsyncClient


async def _signup(client: AsyncClient, email: str):
    r = await client.post(
        "/api/auth/signup",
        json={"name": "Refresh", "email": email, "password": "Pa55word!"},
    )
    assert r.status_code == 201, r.text
    return r.json()


@pytest.mark.asyncio
async def test_refresh_returns_new_tokens(client: AsyncClient, unique_email: str):
    body = await _signup(client, unique_email)
    r = await client.post(
        "/api/auth/refresh-token",
        json={"refresh_token": body["refresh_token"]},
    )
    assert r.status_code == 200, r.text
    new_body = r.json()
    assert new_body["access_token"]
    assert new_body["refresh_token"]
    assert new_body["refresh_token"] != body["refresh_token"]
    assert new_body["user"]["email"] == unique_email


@pytest.mark.asyncio
async def test_refresh_rejects_old_token_after_use(
    client: AsyncClient, unique_email: str
):
    body = await _signup(client, unique_email)
    r1 = await client.post(
        "/api/auth/refresh-token",
        json={"refresh_token": body["refresh_token"]},
    )
    assert r1.status_code == 200
    r2 = await client.post(
        "/api/auth/refresh-token",
        json={"refresh_token": body["refresh_token"]},
    )
    assert r2.status_code == 401


@pytest.mark.asyncio
async def test_refresh_rejects_garbage(client: AsyncClient):
    r = await client.post(
        "/api/auth/refresh-token", json={"refresh_token": "not-a-token"}
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_refresh_rejects_access_token(client: AsyncClient, unique_email: str):
    body = await _signup(client, unique_email)
    r = await client.post(
        "/api/auth/refresh-token",
        json={"refresh_token": body["access_token"]},  # wrong type
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_refresh_concurrent_only_one_succeeds(
    client: AsyncClient, unique_email: str
):
    """Two simultaneous refreshes with the same token: exactly one wins."""
    import asyncio
    r = await client.post(
        "/api/auth/signup",
        json={"name": "Concurrent", "email": unique_email, "password": "Pa55word!"},
    )
    assert r.status_code == 201
    refresh = r.json()["refresh_token"]

    async def hit():
        return await client.post(
            "/api/auth/refresh-token", json={"refresh_token": refresh}
        )

    r1, r2 = await asyncio.gather(hit(), hit())
    statuses = sorted([r1.status_code, r2.status_code])
    assert statuses == [200, 401], f"got {statuses} (expected one 200 + one 401)"
