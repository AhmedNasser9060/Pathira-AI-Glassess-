import time
from datetime import timedelta

import pytest
from jose import jwt

from backend.core import security
from backend.core.config import settings


def test_password_hash_roundtrip():
    h = security.get_password_hash("hunter2")
    assert security.verify_password("hunter2", h) is True
    assert security.verify_password("wrong", h) is False


def test_create_access_token_has_jti_and_type():
    token = security.create_access_token("user-123")
    payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.ALGORITHM])
    assert payload["sub"] == "user-123"
    assert payload["type"] == "access"
    assert "jti" in payload and len(payload["jti"]) >= 16
    assert "exp" in payload


def test_create_refresh_token_has_jti_and_type():
    token, jti = security.create_refresh_token("user-123")
    payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.ALGORITHM])
    assert payload["sub"] == "user-123"
    assert payload["type"] == "refresh"
    assert payload["jti"] == jti


def test_decode_token_returns_payload():
    token = security.create_access_token("abc")
    payload = security.decode_token(token)
    assert payload["sub"] == "abc"


def test_decode_token_rejects_garbage():
    with pytest.raises(security.InvalidTokenError):
        security.decode_token("not-a-token")


def test_decode_token_rejects_expired():
    token = security.create_access_token(
        "abc", expires_delta=timedelta(seconds=-1)
    )
    with pytest.raises(security.InvalidTokenError):
        security.decode_token(token)
