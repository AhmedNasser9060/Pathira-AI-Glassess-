from typing import Optional
from uuid import UUID
from pydantic import BaseModel


class AvailabilityResponse(BaseModel):
    available: bool
    wait_time_minutes: int


class CallStartResponse(BaseModel):
    success: bool
    session_id: UUID
    connection_url: Optional[str] = None
    message: str


class ChatStartResponse(BaseModel):
    success: bool
    session_id: UUID
    agent_name: Optional[str] = None
    message: str
