import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_forgot_password_always_200(client: AsyncClient):
    """Whether the email exists or not, return 200 to avoid user enumeration."""
    r = await client.post(
        "/api/auth/forgot-password", json={"email": "anyone@example.test"}
    )
    assert r.status_code == 200, r.text
    assert "message" in r.json()


@pytest.mark.asyncio
async def test_forgot_password_validates_email(client: AsyncClient):
    r = await client.post(
        "/api/auth/forgot-password", json={"email": "not-an-email"}
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_reset_password_returns_501_until_implemented(client: AsyncClient):
    r = await client.post(
        "/api/auth/reset-password",
        json={"token": "any", "new_password": "Pa55word!"},
    )
    assert r.status_code == 501
    assert "not implemented" in r.json()["detail"].lower()
