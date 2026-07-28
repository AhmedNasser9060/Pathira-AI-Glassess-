import pytest
from httpx import AsyncClient


async def _signup(client: AsyncClient, email: str):
    r = await client.post(
        "/api/auth/signup",
        json={"name": "C", "email": email, "password": "Pa55word!"},
    )
    assert r.status_code == 201
    return r.json()


@pytest.mark.asyncio
async def test_change_password_succeeds(client: AsyncClient, unique_email: str):
    body = await _signup(client, unique_email)
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    r = await client.post(
        "/api/users/change-password",
        json={"current_password": "Pa55word!", "new_password": "Better1Pa55!"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    assert "message" in r.json()

    # Old password no longer works
    r_old = await client.post(
        "/api/auth/login",
        json={"email": unique_email, "password": "Pa55word!"},
    )
    assert r_old.status_code == 400

    # New password works
    r_new = await client.post(
        "/api/auth/login",
        json={"email": unique_email, "password": "Better1Pa55!"},
    )
    assert r_new.status_code == 200


@pytest.mark.asyncio
async def test_change_password_wrong_current(
    client: AsyncClient, unique_email: str
):
    body = await _signup(client, unique_email)
    r = await client.post(
        "/api/users/change-password",
        json={"current_password": "WrongPass!", "new_password": "Better1Pa55!"},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_change_password_revokes_old_refresh_tokens(
    client: AsyncClient, unique_email: str
):
    body = await _signup(client, unique_email)
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    r = await client.post(
        "/api/users/change-password",
        json={"current_password": "Pa55word!", "new_password": "Better1Pa55!"},
        headers=headers,
    )
    assert r.status_code == 200
    # The signup-issued refresh token must now be revoked.
    r_refresh = await client.post(
        "/api/auth/refresh-token",
        json={"refresh_token": body["refresh_token"]},
    )
    assert r_refresh.status_code == 401


@pytest.mark.asyncio
async def test_change_password_short_new_422(
    client: AsyncClient, unique_email: str
):
    body = await _signup(client, unique_email)
    r = await client.post(
        "/api/users/change-password",
        json={"current_password": "Pa55word!", "new_password": "short"},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_change_password_requires_auth(client: AsyncClient):
    r = await client.post(
        "/api/users/change-password",
        json={"current_password": "x", "new_password": "Pa55word!"},
    )
    assert r.status_code == 401
