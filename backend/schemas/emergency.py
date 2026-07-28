from typing import List, Literal, Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel


EmergencyKind = Literal["manual", "automatic"]
EmergencyStatus = Literal["active", "resolved", "cancelled"]


class LocationPoint(BaseModel):
    latitude: float
    longitude: float


class SosRequest(BaseModel):
    location: Optional[LocationPoint] = None
    type: EmergencyKind = "manual"


class EmergencyOut(BaseModel):
    id: UUID
    user_id: UUID
    type: EmergencyKind
    location: Optional[dict] = None
    status: EmergencyStatus
    created_at: datetime
    resolved_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SosResponse(BaseModel):
    emergency: EmergencyOut
    notified_companion_ids: List[UUID]


class AlertCompanionsRequest(BaseModel):
    message: Optional[str] = None


class AlertCompanionsResponse(BaseModel):
    notified_companion_ids: List[UUID]
    alerts_created: int


class CallSupportResponse(BaseModel):
    session_id: UUID
    status: str
    message: str
