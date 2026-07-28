from typing import List, Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class FaceCreate(BaseModel):
    user_id: UUID
    name: str = Field(min_length=1, max_length=255)
    relationship: Optional[str] = Field(default=None, max_length=100)
    embedding: Optional[List[float]] = None


class FaceOut(BaseModel):
    id: UUID
    user_id: UUID
    name: str
    relationship: Optional[str] = None
    photo_url: Optional[str] = None
    embedding: Optional[List[float]] = None
    added_by: Optional[UUID] = None
    created_at: datetime

    class Config:
        from_attributes = True


class RecognitionEventRequest(BaseModel):
    user_id: UUID  # the main user being monitored
    face_id: UUID  # the registered Face that matched
    confidence: float = Field(ge=0.0, le=1.0)


class RecognitionEventResponse(BaseModel):
    logged: bool
    alert_created: bool
    alert_id: Optional[UUID] = None
    # Populated only by the multipart (server-side ML) path.
    matched_face_id: Optional[UUID] = None
    matched_name: Optional[str] = None
    similarity: Optional[float] = None
