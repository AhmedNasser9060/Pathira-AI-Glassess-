import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.main import app


def _signup(c: TestClient, email: str) -> dict:
    r = c.post(
        "/api/auth/signup",
        json={"name": "WS", "email": email, "password": "Pa55word!"},
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_vision_stream_no_token_closes(unique_email):
    with TestClient(app) as c:
        _signup(c, unique_email)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with c.websocket_connect("/api/vision/stream") as ws:
                ws.receive_json()
        # Starlette surfaces the 4401 close as a WebSocketDisconnect with code attr
        assert getattr(exc_info.value, "code", None) == 4401


def test_vision_stream_bad_token_closes(unique_email):
    with TestClient(app) as c:
        _signup(c, unique_email)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with c.websocket_connect("/api/vision/stream?token=garbage") as ws:
                ws.receive_json()
        assert getattr(exc_info.value, "code", None) == 4401


def test_vision_stream_accepts_and_holds_open(unique_email):
    """After Phase 4 the vision_stream no longer echoes a placeholder
    detection. The endpoint just keeps the connection open; pushes are
    driven by REST handlers via vision_manager.send_to_user. We assert
    the WS opens cleanly, accepts a frame without errors, and can be
    closed by the client."""
    with TestClient(app) as c:
        body = _signup(c, unique_email)
        with c.websocket_connect(
            f"/api/vision/stream?token={body['access_token']}"
        ) as ws:
            ws.send_json({"type": "frame", "data": "abc"})
            # No placeholder echo any more — just make sure the server
            # didn't disconnect us. Closing the context exits cleanly.


def test_tracking_live_echoes(unique_email):
    with TestClient(app) as c:
        body = _signup(c, unique_email)
        with c.websocket_connect(
            f"/api/tracking/live?token={body['access_token']}"
        ) as ws:
            ws.send_json({"latitude": 0.0, "longitude": 0.0})
            response = ws.receive_json()
            assert response["type"] == "ack"
            assert response["echo"]["latitude"] == 0.0


def test_alerts_live_echoes(unique_email):
    with TestClient(app) as c:
        body = _signup(c, unique_email)
        with c.websocket_connect(
            f"/api/alerts/live?token={body['access_token']}"
        ) as ws:
            ws.send_json({"hello": "world"})
            response = ws.receive_json()
            assert response["type"] == "ack"


def test_alerts_live_with_refresh_token_rejected(unique_email):
    with TestClient(app) as c:
        body = _signup(c, unique_email)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with c.websocket_connect(
                f"/api/alerts/live?token={body['refresh_token']}"
            ) as ws:
                ws.receive_json()
        assert getattr(exc_info.value, "code", None) == 4401
