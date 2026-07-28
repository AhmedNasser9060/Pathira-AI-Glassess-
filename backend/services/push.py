"""Expo push notification helper.

Sends to https://exp.host/--/api/v2/push/send. Works for any device that has
been registered with `Notifications.getExpoPushTokenAsync()` on the client.

For dev (Expo Go) and production builds — Expo's service relays to FCM/APNS
under the hood. No Firebase setup required for Expo Go testing.

We use httpx (already a transitive dep). If httpx isn't available, falls back
to silent no-op so alert creation is never blocked by push delivery problems.
"""

from typing import Any, Iterable
import logging

logger = logging.getLogger(__name__)

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"

# Expo push tokens look like "ExponentPushToken[xxxxxxxxxxxxxxxxxxxxxx]"
# or "ExpoPushToken[xxxxxxxxxxxxxxxxxxxxxx]". Anything else is a placeholder
# / FCM-direct token and we skip it.
def _is_expo_push_token(token: str) -> bool:
    return token.startswith("ExponentPushToken[") or token.startswith("ExpoPushToken[")


async def send_expo_push(
    tokens: Iterable[str],
    *,
    title: str,
    body: str,
    data: dict[str, Any] | None = None,
) -> None:
    valid = [t for t in tokens if t and _is_expo_push_token(t)]
    if not valid:
        return

    messages = [
        {
            "to": t,
            "sound": "default",
            "title": title,
            "body": body,
            "data": data or {},
            "priority": "high",
        }
        for t in valid
    ]

    try:
        import httpx
    except ImportError:
        logger.warning("httpx not available; skipping push send")
        return

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                EXPO_PUSH_URL,
                json=messages,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate",
                    "Content-Type": "application/json",
                },
            )
            if r.status_code >= 400:
                logger.warning("Expo push HTTP %s: %s", r.status_code, r.text[:200])
    except Exception as e:
        # Never let push delivery problems block alert creation.
        logger.warning("Expo push failed: %s", e)
