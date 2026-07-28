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
async def test_main_user_lists_their_companions_empty(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    r = await client.get(
        "/api/companions",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_main_user_invites_companion_by_email(client, unique_email):
    main = await _signup(client, unique_email, utype="main")
    cmp_email = unique_email.replace("pytest+", "pytest+cmp-")
    await _signup(client, cmp_email, name="Buddy", utype="companion")
    r = await client.post(
        "/api/companions",
        json={"email": cmp_email, "permissions": {"alerts": True}},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["companion_email"] == cmp_email
    assert body["permissions"]["alerts"] is True
    assert body["permissions"]["location"] is False
    assert body["is_active"] is True


@pytest.mark.asyncio
async def test_invite_unknown_email_404(client, unique_email):
    main = await _signup(client, unique_email, utype="main")
    r = await client.post(
        "/api/companions",
        json={"email": "ghost-pytest@example.test"},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_invite_self_400(client, unique_email):
    main = await _signup(client, unique_email, utype="main")
    r = await client.post(
        "/api/companions",
        json={"email": unique_email},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_invite_duplicate_400(client, unique_email):
    main = await _signup(client, unique_email, utype="main")
    cmp_email = unique_email.replace("pytest+", "pytest+dup-")
    await _signup(client, cmp_email, utype="companion")
    h = {"Authorization": f"Bearer {main['access_token']}"}
    r1 = await client.post("/api/companions", json={"email": cmp_email}, headers=h)
    assert r1.status_code == 201
    r2 = await client.post("/api/companions", json={"email": cmp_email}, headers=h)
    assert r2.status_code == 400


@pytest.mark.asyncio
async def test_companion_user_cannot_invite_403(client, unique_email):
    cmp_body = await _signup(client, unique_email, utype="companion")
    target_email = unique_email.replace("pytest+", "pytest+target-")
    await _signup(client, target_email, utype="main")
    r = await client.post(
        "/api/companions",
        json={"email": target_email},
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_update_permissions(client, unique_email):
    main = await _signup(client, unique_email, utype="main")
    cmp_email = unique_email.replace("pytest+", "pytest+pu-")
    await _signup(client, cmp_email, utype="companion")
    h = {"Authorization": f"Bearer {main['access_token']}"}
    r = await client.post("/api/companions", json={"email": cmp_email}, headers=h)
    link_id = r.json()["id"]

    r2 = await client.put(
        f"/api/companions/{link_id}/permissions",
        json={"camera": True, "location": True},
        headers=h,
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["permissions"]["camera"] is True
    assert r2.json()["permissions"]["location"] is True
    assert r2.json()["permissions"]["alerts"] is False  # untouched


@pytest.mark.asyncio
async def test_update_permissions_not_my_link_404(client, unique_email):
    main_a = await _signup(client, unique_email, utype="main")
    cmp_email = unique_email.replace("pytest+", "pytest+a-")
    await _signup(client, cmp_email, utype="companion")
    r = await client.post(
        "/api/companions", json={"email": cmp_email},
        headers={"Authorization": f"Bearer {main_a['access_token']}"},
    )
    link_id = r.json()["id"]

    other_email = unique_email.replace("pytest+", "pytest+b-")
    main_b = await _signup(client, other_email, utype="main")
    r2 = await client.put(
        f"/api/companions/{link_id}/permissions",
        json={"camera": True},
        headers={"Authorization": f"Bearer {main_b['access_token']}"},
    )
    assert r2.status_code == 404


@pytest.mark.asyncio
async def test_delete_companion(client, unique_email):
    main = await _signup(client, unique_email, utype="main")
    cmp_email = unique_email.replace("pytest+", "pytest+del-")
    await _signup(client, cmp_email, utype="companion")
    h = {"Authorization": f"Bearer {main['access_token']}"}
    r = await client.post("/api/companions", json={"email": cmp_email}, headers=h)
    link_id = r.json()["id"]

    r2 = await client.delete(f"/api/companions/{link_id}", headers=h)
    assert r2.status_code == 200
    r3 = await client.get("/api/companions", headers=h)
    assert r3.json() == []


@pytest.mark.asyncio
async def test_global_monitoring(client, unique_email):
    main = await _signup(client, unique_email, utype="main")
    h = {"Authorization": f"Bearer {main['access_token']}"}
    r = await client.post(
        "/api/companions/global-monitoring",
        json={"enabled": True}, headers=h,
    )
    assert r.status_code == 200
    me = await client.get("/api/users/me", headers=h)
    assert me.json().get("monitoring_globally") is True


# ---------- Companion view ----------

@pytest.mark.asyncio
async def test_companion_lists_users_empty(client, unique_email):
    body = await _signup(client, unique_email, utype="companion")
    r = await client.get(
        "/api/companion/users",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_companion_lists_users_after_main_invites(client, unique_email):
    cmp_email = unique_email
    cmp_body = await _signup(client, cmp_email, name="Buddy", utype="companion")
    main_email = unique_email.replace("pytest+", "pytest+main-")
    main = await _signup(client, main_email, name="Main", utype="main")

    # main invites companion
    await client.post(
        "/api/companions",
        json={"email": cmp_email, "permissions": {"alerts": True}},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )

    # companion lists their monitored users
    r = await client.get(
        "/api/companion/users",
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["main_user_email"] == main_email
    assert rows[0]["main_user_name"] == "Main"


@pytest.mark.asyncio
async def test_companion_confirms_inbound_link(client, unique_email):
    cmp_body = await _signup(client, unique_email, utype="companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, utype="main")
    await client.post(
        "/api/companions",
        json={"email": unique_email},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )

    r = await client.post(
        "/api/companion/users",
        json={"email": main_email},
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.status_code == 200
    assert r.json()["main_user_email"] == main_email


@pytest.mark.asyncio
async def test_companion_confirm_no_invite_404(client, unique_email):
    cmp_body = await _signup(client, unique_email, utype="companion")
    main_email = unique_email.replace("pytest+", "pytest+nm-")
    await _signup(client, main_email, utype="main")
    # main has NOT invited cmp
    r = await client.post(
        "/api/companion/users",
        json={"email": main_email},
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_companion_leaves_monitoring(client, unique_email):
    cmp_body = await _signup(client, unique_email, utype="companion")
    main_email = unique_email.replace("pytest+", "pytest+leave-")
    main = await _signup(client, main_email, utype="main")
    inv = await client.post(
        "/api/companions",
        json={"email": unique_email},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    link_id = inv.json()["id"]

    r = await client.delete(
        f"/api/companion/users/{link_id}",
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.status_code == 200

    # main side still sees the row, but is_active=False
    list_r = await client.get(
        "/api/companions",
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert len(list_r.json()) == 1
    assert list_r.json()[0]["is_active"] is False


@pytest.mark.asyncio
async def test_main_user_cannot_use_companion_endpoints_403(client, unique_email):
    main = await _signup(client, unique_email, utype="main")
    r = await client.get(
        "/api/companion/users",
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert r.status_code == 403
