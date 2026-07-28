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


# ---------- Main-user view ----------

@pytest.mark.asyncio
async def test_alerts_empty_list(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    r = await client.get(
        "/api/alerts",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_create_alert(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    h = {"Authorization": f"Bearer {body['access_token']}"}
    r = await client.post(
        "/api/alerts",
        json={"type": "warning", "title": "Low battery", "message": "10%"},
        headers=h,
    )
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["type"] == "warning"
    assert data["title"] == "Low battery"
    assert data["read"] is False


@pytest.mark.asyncio
async def test_invalid_alert_type_422(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    r = await client.post(
        "/api/alerts",
        json={"type": "critical", "title": "x"},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_mark_one_read(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    h = {"Authorization": f"Bearer {body['access_token']}"}
    r = await client.post(
        "/api/alerts", json={"type": "info", "title": "Hi"}, headers=h
    )
    alert_id = r.json()["id"]
    r2 = await client.put(f"/api/alerts/{alert_id}/read", headers=h)
    assert r2.status_code == 200
    assert r2.json()["read"] is True


@pytest.mark.asyncio
async def test_mark_one_read_not_mine_404(client, unique_email):
    a = await _signup(client, unique_email, utype="main")
    other_email = unique_email.replace("pytest+", "pytest+other-")
    b = await _signup(client, other_email, utype="main")
    r = await client.post(
        "/api/alerts", json={"type": "info"},
        headers={"Authorization": f"Bearer {a['access_token']}"},
    )
    alert_id = r.json()["id"]
    r2 = await client.put(
        f"/api/alerts/{alert_id}/read",
        headers={"Authorization": f"Bearer {b['access_token']}"},
    )
    assert r2.status_code == 404


@pytest.mark.asyncio
async def test_mark_all_read(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    h = {"Authorization": f"Bearer {body['access_token']}"}
    for i in range(3):
        await client.post(
            "/api/alerts", json={"type": "info", "title": f"a{i}"}, headers=h
        )

    r = await client.put("/api/alerts/read-all", headers=h)
    assert r.status_code == 200, r.text
    assert "3" in r.json()["message"]

    listing = await client.get("/api/alerts", headers=h)
    assert all(a["read"] is True for a in listing.json())


@pytest.mark.asyncio
async def test_delete_alert(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    h = {"Authorization": f"Bearer {body['access_token']}"}
    r = await client.post(
        "/api/alerts", json={"type": "info", "title": "x"}, headers=h
    )
    alert_id = r.json()["id"]
    r2 = await client.delete(f"/api/alerts/{alert_id}", headers=h)
    assert r2.status_code == 200
    listing = await client.get("/api/alerts", headers=h)
    assert listing.json() == []


@pytest.mark.asyncio
async def test_unread_only_filter(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    h = {"Authorization": f"Bearer {body['access_token']}"}
    r1 = await client.post(
        "/api/alerts", json={"type": "info", "title": "to-read"}, headers=h
    )
    await client.post(
        "/api/alerts", json={"type": "info", "title": "stays-unread"}, headers=h
    )
    await client.put(f"/api/alerts/{r1.json()['id']}/read", headers=h)

    r = await client.get("/api/alerts?unread_only=true", headers=h)
    titles = [a["title"] for a in r.json()]
    assert "stays-unread" in titles
    assert "to-read" not in titles


@pytest.mark.asyncio
async def test_companion_user_cannot_use_main_alerts_403(client, unique_email):
    body = await _signup(client, unique_email, utype="companion")
    r = await client.get(
        "/api/alerts",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 403


# ---------- Companion view ----------

@pytest.mark.asyncio
async def test_companion_alerts_empty_when_no_links(client, unique_email):
    body = await _signup(client, unique_email, utype="companion")
    r = await client.get(
        "/api/companion/alerts",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_companion_alerts_requires_alerts_permission(client, unique_email):
    """Without permissions.alerts=True, the companion sees nothing even if linked."""
    cmp_email = unique_email
    cmp_body = await _signup(client, cmp_email, utype="companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, utype="main")

    # main creates an alert
    await client.post(
        "/api/alerts", json={"type": "info", "title": "secret"},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )

    # main invites companion WITHOUT alerts permission
    await client.post(
        "/api/companions",
        json={"email": cmp_email, "permissions": {"alerts": False}},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )

    r = await client.get(
        "/api/companion/alerts",
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_companion_alerts_visible_with_permission(client, unique_email):
    cmp_email = unique_email
    cmp_body = await _signup(client, cmp_email, name="Cmp", utype="companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, name="MainName", utype="main")

    await client.post(
        "/api/companions",
        json={"email": cmp_email, "permissions": {"alerts": True}},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    await client.post(
        "/api/alerts",
        json={"type": "danger", "title": "Help", "message": "SOS"},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )

    r = await client.get(
        "/api/companion/alerts",
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["title"] == "Help"
    assert rows[0]["user_name"] == "MainName"


@pytest.mark.asyncio
async def test_companion_alerts_inactive_link_excluded(client, unique_email):
    cmp_email = unique_email
    cmp_body = await _signup(client, cmp_email, utype="companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, utype="main")
    inv = await client.post(
        "/api/companions",
        json={"email": cmp_email, "permissions": {"alerts": True}},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    link_id = inv.json()["id"]

    await client.post(
        "/api/alerts", json={"type": "info"},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )

    # companion leaves
    await client.delete(
        f"/api/companion/users/{link_id}",
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )

    r = await client.get(
        "/api/companion/alerts",
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.json() == []


@pytest.mark.asyncio
async def test_main_user_cannot_use_companion_alerts_403(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    r = await client.get(
        "/api/companion/alerts",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 403
