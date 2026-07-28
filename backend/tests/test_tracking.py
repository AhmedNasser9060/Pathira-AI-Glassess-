import pytest
from datetime import date
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
async def test_post_location(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    h = {"Authorization": f"Bearer {body['access_token']}"}
    r = await client.post(
        "/api/tracking/location",
        json={"latitude": 30.123456, "longitude": 31.987654, "speed": 1.5},
        headers=h,
    )
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["user_id"] == body["user"]["id"]
    assert float(out["latitude"]) == pytest.approx(30.123456)
    assert float(out["longitude"]) == pytest.approx(31.987654)


@pytest.mark.asyncio
async def test_invalid_latitude_422(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    r = await client.post(
        "/api/tracking/location",
        json={"latitude": 91.0, "longitude": 0.0},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_route_today_includes_recent_ping(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    h = {"Authorization": f"Bearer {body['access_token']}"}
    await client.post(
        "/api/tracking/location",
        json={"latitude": 1.0, "longitude": 2.0},
        headers=h,
    )
    r = await client.get("/api/tracking/route", headers=h)
    assert r.status_code == 200
    pts = r.json()["points"]
    assert any(float(p["latitude"]) == 1.0 for p in pts)


@pytest.mark.asyncio
async def test_route_specific_past_date_empty(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    h = {"Authorization": f"Bearer {body['access_token']}"}
    await client.post(
        "/api/tracking/location",
        json={"latitude": 1.0, "longitude": 2.0},
        headers=h,
    )
    # 2020-01-01 has no pings
    r = await client.get("/api/tracking/route?date=2020-01-01", headers=h)
    assert r.status_code == 200
    assert r.json()["points"] == []


@pytest.mark.asyncio
async def test_route_invalid_date_400(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    r = await client.get(
        "/api/tracking/route?date=not-a-date",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_history_descending(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    h = {"Authorization": f"Bearer {body['access_token']}"}
    for lat in [1.0, 2.0, 3.0]:
        await client.post(
            "/api/tracking/location",
            json={"latitude": lat, "longitude": 0.0},
            headers=h,
        )
    r = await client.get("/api/tracking/history", headers=h)
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 3
    # Newest first
    lats = [float(p["latitude"]) for p in rows]
    assert lats == [3.0, 2.0, 1.0]


@pytest.mark.asyncio
async def test_share_with_companion_sends_location_push(client, unique_email, monkeypatch):
    main = await _signup(client, unique_email, utype="main")
    cmp_email = unique_email.replace("pytest+", "pytest+cmp-")
    companion = await _signup(client, cmp_email, utype="companion")
    h = {"Authorization": f"Bearer {main['access_token']}"}

    companion_headers = {
        "Authorization": f"Bearer {companion['access_token']}"
    }
    token = "ExpoPushToken[test-location-share]"
    token_response = await client.post(
        "/api/users/device-token",
        json={"token": token, "platform": "android"},
        headers=companion_headers,
    )
    assert token_response.status_code == 200, token_response.text

    await client.post(
        "/api/tracking/location",
        json={"latitude": 30.114928, "longitude": 31.210000},
        headers=h,
    )

    sent = []

    async def fake_send(tokens, *, title, body, data):
        sent.append({
            "tokens": list(tokens),
            "title": title,
            "body": body,
            "data": data,
        })

    monkeypatch.setattr(
        "backend.api.endpoints.tracking.send_expo_push", fake_send
    )

    inv = await client.post(
        "/api/companions",
        json={"email": cmp_email},
        headers=h,
    )
    link_id = inv.json()["id"]
    assert inv.json()["permissions"]["location"] is False

    r = await client.post(
        "/api/tracking/share",
        json={"companion_id": link_id},
        headers=h,
    )
    assert r.status_code == 200, r.text
    assert r.json()["share_link"].endswith(link_id)
    assert sent[0]["tokens"] == [token]
    assert sent[0]["data"]["type"] == "location_share"
    assert sent[0]["data"]["linkId"] == link_id
    assert sent[0]["data"]["latitude"] == pytest.approx(30.114928)
    assert "30.114928" in sent[0]["body"]

    # Verify the link now has permissions.location = True
    listing = await client.get("/api/companions", headers=h)
    rows = listing.json()
    assert len(rows) == 1
    assert rows[0]["permissions"]["location"] is True


@pytest.mark.asyncio
async def test_share_unknown_link_404(client, unique_email):
    body = await _signup(client, unique_email, utype="main")
    import uuid
    r = await client.post(
        "/api/tracking/share",
        json={"companion_id": str(uuid.uuid4())},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_companion_user_cannot_post_location_403(client, unique_email):
    body = await _signup(client, unique_email, utype="companion")
    r = await client.post(
        "/api/tracking/location",
        json={"latitude": 0.0, "longitude": 0.0},
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_tracking_requires_auth(client):
    r = await client.post(
        "/api/tracking/location",
        json={"latitude": 0.0, "longitude": 0.0},
    )
    assert r.status_code == 401
