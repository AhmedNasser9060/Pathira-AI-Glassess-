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


async def _link_with_perms(client, main, cmp_email, perms):
    return await client.post(
        "/api/companions",
        json={"email": cmp_email, "permissions": perms},
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )


@pytest.mark.asyncio
async def test_list_faces_empty(client, unique_email):
    cmp_email = unique_email
    cmp_body = await _signup(client, cmp_email, utype="companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, utype="main")
    await _link_with_perms(client, main, cmp_email, {"camera": True})

    r = await client.get(
        f"/api/companion/faces/{main['user']['id']}",
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_list_faces_no_camera_permission_403(client, unique_email):
    cmp_email = unique_email
    cmp_body = await _signup(client, cmp_email, utype="companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, utype="main")
    # link WITHOUT camera permission
    await _link_with_perms(client, main, cmp_email, {"camera": False})
    r = await client.get(
        f"/api/companion/faces/{main['user']['id']}",
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_list_faces_no_link_403(client, unique_email):
    cmp_body = await _signup(client, unique_email, utype="companion")
    other_email = unique_email.replace("pytest+", "pytest+other-")
    other = await _signup(client, other_email, utype="main")
    # no link at all
    r = await client.get(
        f"/api/companion/faces/{other['user']['id']}",
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_add_face(client, unique_email):
    cmp_email = unique_email
    cmp_body = await _signup(client, cmp_email, utype="companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, utype="main")
    await _link_with_perms(client, main, cmp_email, {"camera": True})

    r = await client.post(
        "/api/companion/faces",
        json={
            "user_id": main["user"]["id"],
            "name": "Mom",
            "relationship": "parent",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["name"] == "Mom"
    assert out["relationship"] == "parent"
    assert out["embedding"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert out["added_by"] == cmp_body["user"]["id"]

    # And it shows up in GET
    listing = await client.get(
        f"/api/companion/faces/{main['user']['id']}",
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert len(listing.json()) == 1


@pytest.mark.asyncio
async def test_delete_face(client, unique_email):
    cmp_email = unique_email
    cmp_body = await _signup(client, cmp_email, utype="companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, utype="main")
    await _link_with_perms(client, main, cmp_email, {"camera": True})

    cr = await client.post(
        "/api/companion/faces",
        json={"user_id": main["user"]["id"], "name": "Dad"},
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    face_id = cr.json()["id"]

    r = await client.delete(
        f"/api/companion/faces/{face_id}",
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.status_code == 200

    listing = await client.get(
        f"/api/companion/faces/{main['user']['id']}",
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert listing.json() == []


@pytest.mark.asyncio
async def test_delete_face_not_my_user_404(client, unique_email):
    cmp_a_email = unique_email
    cmp_a = await _signup(client, cmp_a_email, utype="companion")

    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, utype="main")
    await _link_with_perms(client, main, cmp_a_email, {"camera": True})

    # cmp_a adds a face
    cr = await client.post(
        "/api/companion/faces",
        json={"user_id": main["user"]["id"], "name": "X"},
        headers={"Authorization": f"Bearer {cmp_a['access_token']}"},
    )
    face_id = cr.json()["id"]

    # cmp_b (different companion, no link to main) tries to delete it.
    # Server collapses missing-face / no-access into a single 404 to avoid
    # leaking face existence to unauthorized companions.
    cmp_b_email = unique_email.replace("pytest+", "pytest+b-")
    cmp_b = await _signup(client, cmp_b_email, utype="companion")
    r = await client.delete(
        f"/api/companion/faces/{face_id}",
        headers={"Authorization": f"Bearer {cmp_b['access_token']}"},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_recognize_creates_alert_when_perms_allow(client, unique_email):
    cmp_email = unique_email
    cmp_body = await _signup(client, cmp_email, name="Cmp", utype="companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, utype="main")
    await _link_with_perms(client, main, cmp_email, {"camera": True, "alerts": True})

    cr = await client.post(
        "/api/companion/faces",
        json={"user_id": main["user"]["id"], "name": "Mom"},
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    face_id = cr.json()["id"]

    r = await client.post(
        "/api/companion/faces/recognize",
        json={
            "user_id": main["user"]["id"],
            "face_id": face_id,
            "confidence": 0.92,
        },
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["logged"] is True
    assert body["alert_created"] is True
    assert body["alert_id"] is not None

    # Main user sees the alert
    alerts = await client.get(
        "/api/alerts",
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert len(alerts.json()) == 1
    assert "Mom" in alerts.json()[0]["title"]


@pytest.mark.asyncio
async def test_recognize_no_alert_perm(client, unique_email):
    cmp_email = unique_email
    cmp_body = await _signup(client, cmp_email, utype="companion")
    main_email = unique_email.replace("pytest+", "pytest+m-")
    main = await _signup(client, main_email, utype="main")
    await _link_with_perms(client, main, cmp_email, {"camera": True, "alerts": False})

    cr = await client.post(
        "/api/companion/faces",
        json={"user_id": main["user"]["id"], "name": "Stranger"},
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    face_id = cr.json()["id"]

    r = await client.post(
        "/api/companion/faces/recognize",
        json={
            "user_id": main["user"]["id"],
            "face_id": face_id,
            "confidence": 0.85,
        },
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["logged"] is True
    assert body["alert_created"] is False
    assert body["alert_id"] is None


@pytest.mark.asyncio
async def test_recognize_invalid_confidence_422(client, unique_email):
    cmp_body = await _signup(client, unique_email, utype="companion")
    import uuid
    r = await client.post(
        "/api/companion/faces/recognize",
        json={
            "user_id": str(uuid.uuid4()),
            "face_id": str(uuid.uuid4()),
            "confidence": 1.5,  # > 1.0
        },
        headers={"Authorization": f"Bearer {cmp_body['access_token']}"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_main_user_cannot_use_face_endpoints_403(client, unique_email):
    main = await _signup(client, unique_email, utype="main")
    r = await client.get(
        f"/api/companion/faces/{main['user']['id']}",
        headers={"Authorization": f"Bearer {main['access_token']}"},
    )
    assert r.status_code == 403
