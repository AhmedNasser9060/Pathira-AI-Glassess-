from typing import List, Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class DetectedObject(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[float]  # [x, y, w, h]


class VisionDetectRequest(BaseModel):
    """Device sends already-detected objects for storage/sync."""
    objects: List[DetectedObject] = Field(default_factory=list)
    text: Optional[str] = None


class VisionDetectResponse(BaseModel):
    logged: bool
    event_id: UUID


class VisionToggleRequest(BaseModel):
    is_active: bool


class VisionToggleResponse(BaseModel):
    is_active: bool
    message: str


class VisionHistoryItem(BaseModel):
    id: UUID
    is_active: bool
    summary: Optional[str]
    detections: list
    created_at: datetime

    class Config:
        from_attributes = True
