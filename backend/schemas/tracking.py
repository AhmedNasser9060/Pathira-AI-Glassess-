from decimal import Decimal
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class LocationPing(BaseModel):
    latitude: Decimal = Field(..., ge=-90, le=90)
    longitude: Decimal = Field(..., ge=-180, le=180)
    altitude: Optional[Decimal] = None
    speed: Optional[Decimal] = None
    accuracy: Optional[Decimal] = None
    heading: Optional[Decimal] = None


class LocationOut(BaseModel):
    id: UUID
    user_id: UUID
    latitude: Decimal
    longitude: Decimal
    altitude: Optional[Decimal] = None
    speed: Optional[Decimal] = None
    accuracy: Optional[Decimal] = None
    heading: Optional[Decimal] = None
    created_at: datetime

    class Config:
        from_attributes = True


class RoutePoint(BaseModel):
    latitude: Decimal
    longitude: Decimal
    timestamp: datetime


class RouteResponse(BaseModel):
    points: List[RoutePoint]


class ShareRequest(BaseModel):
    companion_id: UUID  # the companion-link row id


class ShareResponse(BaseModel):
    share_link: str
    message: str


class MarkerSet(BaseModel):
    latitude: Decimal = Field(..., ge=-90, le=90)
    longitude: Decimal = Field(..., ge=-180, le=180)
    label: Optional[str] = Field(default=None, max_length=120)


class MarkerOut(BaseModel):
    latitude: Decimal
    longitude: Decimal
    label: Optional[str] = None
    set_at: datetime
