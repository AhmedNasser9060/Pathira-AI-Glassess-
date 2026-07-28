import pytest
from httpx import AsyncClient
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models.user import User
from backend.db.models.refresh_token import RefreshToken


async def _signup(client: AsyncClient, email: str):
    r = await client.post(
        "/api/auth/signup",
        json={"name": "D", "email": email, "password": "Pa55word!"},
    )
    assert r.status_code == 201
    return r.json()


@pytest.mark.asyncio
async def test_delete_account_succeeds(
    client: AsyncClient, db: AsyncSession, unique_email: str
):
    body = await _signup(client, unique_email)
    headers = {"Authorization": f"Bearer {body['access_token']}"}

    r = await client.request(
        "DELETE",
        "/api/users/account",
        json={"password": "Pa55word!"},
        headers=headers,
    )
    assert r.status_code == 200, r.text

    # User row gone
    n = await db.execute(
        select(func.count()).select_from(User).where(User.email == unique_email)
    )
    assert n.scalar() == 0

    # Refresh tokens gone via FK cascade
    user_id = body["user"]["id"]
    n2 = await db.execute(
        select(func.count())
        .select_from(RefreshToken)
        .where(RefreshToken.user_id == user_id)
    )
    assert n2.scalar() == 0


@pytest.mark.asyncio
async def test_delete_account_wrong_password_400(
    client: AsyncClient, unique_email: str
):
    body = await _signup(client, unique_email)
    r = await client.request(
        "DELETE",
        "/api/users/account",
        json={"password": "WrongPass!"},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_delete_account_requires_auth(client: AsyncClient):
    r = await client.request(
        "DELETE", "/api/users/account", json={"password": "Pa55word!"}
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_deleted_user_access_token_now_401(
    client: AsyncClient, unique_email: str
):
    """After delete-account, the deleted user's access token must no longer
    pass `get_current_user` because the user row is gone."""
    body = await _signup(client, unique_email)
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    r = await client.request(
        "DELETE",
        "/api/users/account",
        json={"password": "Pa55word!"},
        headers=headers,
    )
    assert r.status_code == 200
    r2 = await client.get("/api/users/me", headers=headers)
    assert r2.status_code == 401


@pytest.mark.asyncio
async def test_delete_account_cascades_alerts(
    client: AsyncClient, db: AsyncSession, unique_email: str
):
    """Regression for the C1 cascade fix: delete-account must wipe the user's
    alerts (and other FK-CASCADE'd children) instead of failing with FK
    violation."""
    from backend.db.models.alert import Alert
    body = await _signup(client, unique_email)
    user_id = body["user"]["id"]

    # Insert an alert directly via the db fixture
    import uuid
    db.add(Alert(
        id=uuid.uuid4(),
        user_id=user_id,
        type="info",
        title="hi",
        message="x",
    ))
    await db.commit()

    r = await client.request(
        "DELETE",
        "/api/users/account",
        json={"password": "Pa55word!"},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200, r.text

    # Alert row should be gone via CASCADE
    n = await db.execute(
        select(func.count()).select_from(Alert).where(Alert.user_id == user_id)
    )
    assert n.scalar() == 0
