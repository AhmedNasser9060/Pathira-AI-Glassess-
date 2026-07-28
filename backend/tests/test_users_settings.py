import pytest
from httpx import AsyncClient


async def _signup_and_auth(client: AsyncClient, email: str):
    r = await client.post(
        "/api/auth/signup",
        json={"name": "S", "email": email, "password": "Pa55word!"},
    )
    assert r.status_code == 201
    body = r.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body


@pytest.mark.asyncio
async def test_settings_partial_merge(client: AsyncClient, unique_email: str):
    headers, _ = await _signup_and_auth(client, unique_email)

    r1 = await client.put(
        "/api/users/settings",
        json={"notifications": True, "bluetooth": False},
        headers=headers,
    )
    assert r1.status_code == 200, r1.text
    s1 = r1.json()["settings"]
    assert s1 == {"notifications": True, "bluetooth": False}

    # Second update only sets one key — must merge with previous, not replace.
    r2 = await client.put(
        "/api/users/settings",
        json={"voice_commands": True},
        headers=headers,
    )
    assert r2.status_code == 200
    s2 = r2.json()["settings"]
    assert s2 == {
        "notifications": True,
        "bluetooth": False,
        "voice_commands": True,
    }


@pytest.mark.asyncio
async def test_settings_empty_body_no_change(
    client: AsyncClient, unique_email: str
):
    headers, _ = await _signup_and_auth(client, unique_email)
    await client.put(
        "/api/users/settings",
        json={"notifications": True},
        headers=headers,
    )
    r = await client.put("/api/users/settings", json={}, headers=headers)
    assert r.status_code == 200
    assert r.json()["settings"] == {"notifications": True}


@pytest.mark.asyncio
async def test_settings_unknown_key_ignored(
    client: AsyncClient, unique_email: str
):
    headers, _ = await _signup_and_auth(client, unique_email)
    r = await client.put(
        "/api/users/settings",
        json={"unknown_setting": True},
        headers=headers,
    )
    # Pydantic v2 default behavior is to ignore extra keys; we accept that policy.
    assert r.status_code == 200
    assert "unknown_setting" not in r.json()["settings"]


@pytest.mark.asyncio
async def test_settings_requires_auth(client: AsyncClient):
    r = await client.put("/api/users/settings", json={"notifications": True})
    assert r.status_code == 401
