"""Broadcast helpers — fan out a new alert to connected companion WebSockets
and to their device push tokens. Imported by the endpoints that create alerts.
"""

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.api.ws_manager import alerts_manager
from backend.db.models.companion import Companion as CompanionModel
from backend.db.models.device_token import DeviceToken
from backend.db.models.user import User as UserModel
from backend.services.push import send_expo_push


async def _companion_ids_with_alerts(
    main_user_id: UUID, db: AsyncSession
) -> list[UUID]:
    """Companion user_ids that have an active link AND alerts permission."""
    result = await db.execute(
        select(CompanionModel.companion_id).where(
            CompanionModel.main_user_id == main_user_id,
            CompanionModel.is_active == True,  # noqa: E712
            CompanionModel.permissions["alerts"].astext == "true",
        )
    )
    return [row[0] for row in result.all()]


async def broadcast_alert_to_companions(
    main_user: UserModel,
    alert_payload: dict[str, Any],
    db: AsyncSession,
) -> None:
    """Fan out a freshly-created alert to all companions of `main_user` who
    have alerts permission, both via WebSocket (live in-app refresh) and via
    Expo push notifications (banner even when app is closed).
    """
    companion_ids = await _companion_ids_with_alerts(main_user.id, db)
    if not companion_ids:
        return

    # 1) WebSocket push — drives instant in-app refresh on the dashboard /
    # alerts screen via the existing useWebSocket subscription.
    ws_message = {
        "type": "alert",
        "alert": alert_payload,
        "user_id": str(main_user.id),
        "user_name": main_user.name,
    }
    for cid in companion_ids:
        await alerts_manager.send_to_user(cid, ws_message)

    # 2) Push notifications — fetch all device tokens for these companions and
    # send via Expo's push service. Async/non-blocking inside send_expo_push.
    token_result = await db.execute(
        select(DeviceToken.token).where(DeviceToken.user_id.in_(companion_ids))
    )
    tokens = [row[0] for row in token_result.all() if row[0]]
    if not tokens:
        return

    title = alert_payload.get("title") or "Pathira alert"
    body_text = alert_payload.get("message") or f"From {main_user.name}"
    await send_expo_push(
        tokens,
        title=str(title),
        body=str(body_text),
        data={"type": "alert", "user_id": str(main_user.id)},
    )
