from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


class PermissionsModel(BaseModel):
    camera: bool = False
    location: bool = False
    alerts: bool = False


class CompanionInvite(BaseModel):
    email: EmailStr
    permissions: PermissionsModel = Field(default_factory=PermissionsModel)


class PermissionsUpdate(BaseModel):
    camera: Optional[bool] = None
    location: Optional[bool] = None
    alerts: Optional[bool] = None


class GlobalMonitoringRequest(BaseModel):
    enabled: bool


class CompanionLink(BaseModel):
    """A row in the main user's companions list (the people watching them)."""
    id: UUID
    companion_user_id: UUID
    companion_email: EmailStr
    companion_name: str
    companion_avatar: Optional[str] = None
    permissions: PermissionsModel
    is_active: bool
    created_at: datetime


class MonitoredUser(BaseModel):
    """A row in a companion's monitored-users list (people they watch)."""
    id: UUID
    main_user_id: UUID
    main_user_email: EmailStr
    main_user_name: str
    main_user_avatar: Optional[str] = None
    permissions: PermissionsModel
    created_at: datetime


class CompanionAddByUser(BaseModel):
    """Companion confirms an EXISTING inbound link by the main user's email."""
    email: EmailStr
