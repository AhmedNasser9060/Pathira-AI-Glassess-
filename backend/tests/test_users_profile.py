import pytest
from httpx import AsyncClient


async def _signup(client: AsyncClient, email: str, name: str = "Profile"):
    r = await client.post(
        "/api/auth/signup",
        json={"name": name, "email": email, "password": "Pa55word!"},
    )
    assert r.status_code == 201, r.text
    return r.json()


@pytest.mark.asyncio
async def test_profile_update_name_phone(client: AsyncClient, unique_email: str):
    body = await _signup(client, unique_email)
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    r = await client.put(
        "/api/users/profile",
        json={"name": "New Name", "phone": "+15551234"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    u = r.json()
    assert u["name"] == "New Name"
    assert u["phone"] == "+15551234"
    assert u["email"] == unique_email
    # Confirm persisted via /me
    r2 = await client.get("/api/users/me", headers=headers)
    assert r2.json()["name"] == "New Name"


@pytest.mark.asyncio
async def test_profile_change_email_to_unique(
    client: AsyncClient, unique_email: str
):
    body = await _signup(client, unique_email)
    new_email = unique_email.replace("pytest+", "pytest+changed-")
    r = await client.put(
        "/api/users/profile",
        json={"email": new_email},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200
    assert r.json()["email"] == new_email


@pytest.mark.asyncio
async def test_profile_change_email_to_taken_400(
    client: AsyncClient, unique_email: str
):
    other_email = unique_email.replace("pytest+", "pytest+other-")
    await _signup(client, other_email, name="Other")
    body = await _signup(client, unique_email)
    r = await client.put(
        "/api/users/profile",
        json={"email": other_email},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 400
    assert "already" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_profile_empty_body_400(client: AsyncClient, unique_email: str):
    body = await _signup(client, unique_email)
    r = await client.put(
        "/api/users/profile",
        json={},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_profile_requires_auth(client: AsyncClient):
    r = await client.put("/api/users/profile", json={"name": "X"})
    assert r.status_code == 401
