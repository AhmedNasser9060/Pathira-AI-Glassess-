from typing import Literal, Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


AlertType = Literal["danger", "warning", "info", "success"]


class AlertCreate(BaseModel):
    type: AlertType
    title: Optional[str] = Field(default=None, max_length=255)
    message: Optional[str] = None


class AlertOut(BaseModel):
    id: UUID
    user_id: UUID
    type: AlertType
    title: Optional[str]
    message: Optional[str]
    read: bool
    created_at: datetime

    class Config:
        from_attributes = True


class CompanionAlertOut(AlertOut):
    user_name: str
    user_avatar: Optional[str] = None
