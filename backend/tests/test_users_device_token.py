import pytest
from httpx import AsyncClient
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models.device_token import DeviceToken


async def _signup(client: AsyncClient, email: str):
    r = await client.post(
        "/api/auth/signup",
        json={"name": "DT", "email": email, "password": "Pa55word!"},
    )
    assert r.status_code == 201
    return r.json()


@pytest.mark.asyncio
async def test_device_token_register(
    client: AsyncClient, db: AsyncSession, unique_email: str
):
    body = await _signup(client, unique_email)
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    r = await client.post(
        "/api/users/device-token",
        json={"token": "expo-push-aaaa", "platform": "android"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    n = await db.execute(
        select(func.count())
        .select_from(DeviceToken)
        .where(DeviceToken.user_id == body["user"]["id"])
    )
    assert n.scalar() == 1


@pytest.mark.asyncio
async def test_device_token_idempotent_same_token(
    client: AsyncClient, db: AsyncSession, unique_email: str
):
    body = await _signup(client, unique_email)
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    payload = {"token": "expo-push-bbbb", "platform": "ios"}
    await client.post("/api/users/device-token", json=payload, headers=headers)
    r2 = await client.post(
        "/api/users/device-token", json=payload, headers=headers
    )
    assert r2.status_code == 200
    n = await db.execute(
        select(func.count())
        .select_from(DeviceToken)
        .where(DeviceToken.user_id == body["user"]["id"])
    )
    assert n.scalar() == 1  # still one row, not two


@pytest.mark.asyncio
async def test_device_token_two_distinct_tokens(
    client: AsyncClient, db: AsyncSession, unique_email: str
):
    body = await _signup(client, unique_email)
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    await client.post(
        "/api/users/device-token",
        json={"token": "tok-A", "platform": "ios"},
        headers=headers,
    )
    await client.post(
        "/api/users/device-token",
        json={"token": "tok-B", "platform": "android"},
        headers=headers,
    )
    n = await db.execute(
        select(func.count())
        .select_from(DeviceToken)
        .where(DeviceToken.user_id == body["user"]["id"])
    )
    assert n.scalar() == 2


@pytest.mark.asyncio
async def test_device_token_invalid_platform_422(
    client: AsyncClient, unique_email: str
):
    body = await _signup(client, unique_email)
    r = await client.post(
        "/api/users/device-token",
        json={"token": "x", "platform": "windows"},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_device_token_requires_auth(client: AsyncClient):
    r = await client.post(
        "/api/users/device-token",
        json={"token": "x", "platform": "ios"},
    )
    assert r.status_code == 401
