from typing import List, Literal, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class SendRequest(BaseModel):
    """Send a push to current user's own devices."""
    title: str = Field(min_length=1, max_length=255)
    body: str = Field(min_length=1)
    data: Optional[dict] = None


class SendToCompanionsRequest(SendRequest):
    """Send a push to every linked companion of the current main user."""
    pass  # same shape as SendRequest


class EmergencyPushRequest(SendRequest):
    target_user_id: UUID  # specific recipient (e.g. a particular companion)


class NotificationStubResponse(BaseModel):
    request_id: UUID
    target_kind: Literal["self", "user", "companions", "emergency"]
    notified_user_ids: List[UUID]  # who WOULD receive (informational; no real send)
    status: str  # "stub_logged"
    message: str
