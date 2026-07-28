import pytest
from httpx import AsyncClient


async def _signup(
    client: AsyncClient, email: str, name: str = "X", utype: str | None = "main"
):
    body = {"name": name, "email": email, "password": "Pa55word!"}
    if utype is not None:
        body["user_type"] = utype
    r = await client.post("/api/auth/signup", json=body)
    assert r.status_code == 201, r.text
    return r.json()


@pytest.mark.asyncio
async def test_send_to_self_happy(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    r = await client.post(
        "/api/notifications/send",
        json={"title": "Hi", "body": "Hello there"},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["target_kind"] == "self"
    assert out["status"] == "stub_logged"
    assert out["request_id"]
    assert out["notified_user_ids"] == [body["user"]["id"]]
    assert "stub" in out["message"].lower()


@pytest.mark.asyncio
async def test_send_to_companions_zero(client, unique_email):
    main = await _signup(client, unique_email, utype="main")
    r = await client.post(
        "/api/notifications/send-to-companions",
        json={"title": "Update", "body": "Just FYI"},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["target_kind"] == "companions"
    assert out["notified_user_ids"] == []
    assert out["status"] == "stub_logged"


@pytest.mark.asyncio
async def test_send_to_companions_one_active_link(client, unique_email):
    main = await _signup(client, unique_email, utype="main")
    cmp_email = unique_email.replace("pytest+", "pytest+cmp-")
    cmp_body = await _signup(client, cmp_email, utype="companion")
    await client.post(
        "/api/companions",
        json={"email": cmp_email, "permissions": {"alerts": True}},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )

    r = await client.post(
        "/api/notifications/send-to-companions",
        json={"title": "Update", "body": "Just FYI"},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["notified_user_ids"] == [cmp_body["user"]["id"]]


@pytest.mark.asyncio
async def test_emergency_push_to_known_user(client, unique_email):
    main = await _signup(client, unique_email, utype="main")
    other_email = unique_email.replace("pytest+", "pytest+other-")
    other = await _signup(client, other_email, utype="companion")

    r = await client.post(
        "/api/notifications/emergency",
        json={
            "title": "URGENT",
            "body": "help me",
            "target_user_id": other["user"]["id"],
        },
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["target_kind"] == "emergency"
    assert out["notified_user_ids"] == [other["user"]["id"]]


@pytest.mark.asyncio
async def test_emergency_push_to_unknown_user_404(client, unique_email):
    main = await _signup(client, unique_email, utype="main")
    fake_id = "00000000-0000-0000-0000-000000000000"
    r = await client.post(
        "/api/notifications/emergency",
        json={"title": "URGENT", "body": "help", "target_user_id": fake_id},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_send_to_companions_as_companion_403(client, unique_email):
    cmp_user = await _signup(client, unique_email, utype="companion")
    r = await client.post(
        "/api/notifications/send-to-companions",
        json={"title": "Hi", "body": "x"},
        headers={"Authorization": f"Bearer {cmp_user['access_token']}"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_send_to_companions_excludes_revoked_alerts_perm(client, unique_email):
    """Companions whose `permissions.alerts` was set to False must NOT receive
    the notification, even if the link is still active."""
    main = await _signup(client, unique_email, utype="main")
    cmp_email = unique_email.replace("pytest+", "pytest+revoked-")
    await _signup(client, cmp_email, utype="companion")
    await client.post(
        "/api/companions",
        json={"email": cmp_email, "permissions": {"alerts": False}},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    r = await client.post(
        "/api/notifications/send-to-companions",
        json={"title": "Hi", "body": "Hello"},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert r.status_code == 200
    assert r.json()["notified_user_ids"] == []  # filtered out by alerts=False
