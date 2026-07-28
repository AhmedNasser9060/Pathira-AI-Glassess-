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
async def test_sos_creates_event(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    r = await client.post(
        "/api/emergency/sos",
        json={"location": {"latitude": 30.0, "longitude": 31.0}, "type": "manual"},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["emergency"]["status"] == "active"
    assert out["emergency"]["type"] == "manual"
    assert out["emergency"]["location"] == {"latitude": 30.0, "longitude": 31.0}
    assert out["notified_companion_ids"] == []


@pytest.mark.asyncio
async def test_sos_notifies_companions_with_alerts_perm(client, unique_email):
    main_email = unique_email
    main = await _signup(client, main_email, name="MainUser", utype="main")
    cmp_email = unique_email.replace("pytest+", "pytest+cmp-")
    cmp_body = await _signup(client, cmp_email, utype="companion")
    await client.post(
        "/api/companions",
        json={"email": cmp_email, "permissions": {"alerts": True}},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )

    r = await client.post(
        "/api/emergency/sos",
        json={},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert r.status_code == 201
    notified = r.json()["notified_companion_ids"]
    assert len(notified) == 1
    assert notified[0] == cmp_body["user"]["id"]


@pytest.mark.asyncio
async def test_alert_companions(client, unique_email):
    main_email = unique_email
    main = await _signup(client, main_email, name="Bob", utype="main")
    cmp_email = unique_email.replace("pytest+", "pytest+cmp-")
    cmp_body = await _signup(client, cmp_email, utype="companion")
    await client.post(
        "/api/companions",
        json={"email": cmp_email, "permissions": {"alerts": True}},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    r = await client.post(
        "/api/emergency/alert-companions",
        json={"message": "I need help"},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["alerts_created"] == 1
    assert len(body["notified_companion_ids"]) == 1


@pytest.mark.asyncio
async def test_alert_companions_no_link_zero_alerts(client, unique_email):
    main = await _signup(client, unique_email, utype="main")
    r = await client.post(
        "/api/emergency/alert-companions",
        json={},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert r.status_code == 200
    assert r.json()["alerts_created"] == 0


@pytest.mark.asyncio
async def test_alert_companions_skips_link_without_perm(client, unique_email):
    main_email = unique_email
    main = await _signup(client, main_email, utype="main")
    cmp_email = unique_email.replace("pytest+", "pytest+cmp-")
    await _signup(client, cmp_email, utype="companion")
    # link WITHOUT alerts permission
    await client.post(
        "/api/companions",
        json={"email": cmp_email, "permissions": {"alerts": False}},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    r = await client.post(
        "/api/emergency/alert-companions",
        json={},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert r.status_code == 200
    assert r.json()["alerts_created"] == 0


@pytest.mark.asyncio
async def test_call_support_stub(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    r = await client.post(
        "/api/emergency/call-support",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "requested"
    assert "stub" in out["message"].lower()


@pytest.mark.asyncio
async def test_history_lists_events(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    h = {"Authorization": f"Bearer {body['access_token']}"}
    await client.post("/api/emergency/sos", json={}, headers=h)
    await client.post("/api/emergency/sos", json={}, headers=h)
    r = await client.get("/api/emergency/history", headers=h)
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_resolve_emergency(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    h = {"Authorization": f"Bearer {body['access_token']}"}
    sos = await client.post("/api/emergency/sos", json={}, headers=h)
    event_id = sos.json()["emergency"]["id"]

    r = await client.put(f"/api/emergency/{event_id}/resolve", headers=h)
    assert r.status_code == 200
    assert r.json()["status"] == "resolved"
    assert r.json()["resolved_at"] is not None


@pytest.mark.asyncio
async def test_resolve_not_mine_404(client, unique_email):
    a = await _signup(client, unique_email, utype="main")
    other_email = unique_email.replace("pytest+", "pytest+other-")
    b = await _signup(client, other_email, utype="main")
    sos = await client.post(
        "/api/emergency/sos", json={},
        headers={"Authorization": f"Bearer {a['access_token']}"},
    )
    event_id = sos.json()["emergency"]["id"]
    r = await client.put(
        f"/api/emergency/{event_id}/resolve",
        headers={"Authorization": f"Bearer {b['access_token']}"},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_companion_user_cannot_use_emergency_403(client, unique_email):
    body = await _signup(client, unique_email, utype="companion")
    r = await client.post(
        "/api/emergency/sos",
        json={},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_emergency_requires_auth(client):
    r = await client.post("/api/emergency/sos", json={})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_sos_alert_visible_via_companion_alerts(client, unique_email):
    """Alerts created by SOS should be visible to the companion via the
    companion-alerts endpoint (which queries alerts for monitored main users)."""
    main_email = unique_email
    main = await _signup(client, main_email, name="MainName", utype="main")
    cmp_email = unique_email.replace("pytest+", "pytest+cmp-")
    cmp_body = await _signup(client, cmp_email, utype="companion")
    await client.post(
        "/api/companions",
        json={"email": cmp_email, "permissions": {"alerts": True}},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )

    await client.post(
        "/api/emergency/sos", json={},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )

    r = await client.get(
        "/api/companion/alerts",
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["type"] == "danger"
    assert rows[0]["user_name"] == "MainName"
