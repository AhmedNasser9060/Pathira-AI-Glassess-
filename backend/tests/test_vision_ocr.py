import pytest
from httpx import AsyncClient

from backend.api.endpoints import vision as vision_module


# 1x1 transparent PNG (smallest valid PNG payload). Content of the file is
# irrelevant because `run_ocr_sync` is mocked, but we send a real image
# header so the test exercises the multipart path realistically.
_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)


async def _signup(client: AsyncClient, email: str) -> dict:
    r = await client.post(
        "/api/auth/signup",
        json={
            "name": "X",
            "email": email,
            "password": "Pa55word!",
            "user_type": "main",
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


@pytest.mark.asyncio
async def test_ocr_returns_text_and_persists_row(client, unique_email, monkeypatch):
    # Patch the bound import in the endpoint module (the endpoint did
    # `from backend.ml.ocr import run_ocr_sync` at module load).
    monkeypatch.setattr(vision_module, "run_ocr_sync", lambda b, lang="eng+ara": "hello")

    body = await _signup(client, unique_email)
    h = {"Authorization": f"Bearer {body['access_token']}"}

    r = await client.post(
        "/api/vision/ocr",
        headers=h,
        files={"image": ("img.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["text"] == "hello"
    assert out["event_id"]

    # The row must show up in /history with summary='ocr' and the text we
    # mocked stored in detections[0].text.
    hist = await client.get("/api/vision/history", headers=h)
    assert hist.status_code == 200
    rows = hist.json()
    ocr_rows = [row for row in rows if row["summary"] == "ocr"]
    assert len(ocr_rows) == 1
    assert ocr_rows[0]["detections"] == [{"text": "hello"}]


@pytest.mark.asyncio
async def test_ocr_rejects_oversized_image(client, unique_email, monkeypatch):
    # OCR should never run for an oversized payload — fail fast for the test
    # if it does.
    def _boom(*_a, **_kw):  # pragma: no cover - shouldn't be called
        raise AssertionError("run_ocr_sync called for oversized payload")

    monkeypatch.setattr(vision_module, "run_ocr_sync", _boom)

    body = await _signup(client, unique_email)
    h = {"Authorization": f"Bearer {body['access_token']}"}

    big = b"\x00" * (6 * 1024 * 1024)  # 6 MB
    r = await client.post(
        "/api/vision/ocr",
        headers=h,
        files={"image": ("big.bin", big, "application/octet-stream")},
    )
    assert r.status_code == 413, r.text


@pytest.mark.asyncio
async def test_ocr_requires_auth(client):
    r = await client.post(
        "/api/vision/ocr",
        files={"image": ("img.png", _PNG_1x1, "image/png")},
    )
    assert r.status_code == 401
