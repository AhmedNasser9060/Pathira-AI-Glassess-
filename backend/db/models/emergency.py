import uuid
from sqlalchemy import Column, Enum, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from backend.db.base import Base

class EmergencyEvent(Base):
    __tablename__ = "emergency_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    type = Column(Enum("manual", "automatic", name="emergency_type_enum"), nullable=False)
    location = Column(JSONB, nullable=True) # {latitude, longitude}
    status = Column(Enum("active", "resolved", "cancelled", name="emergency_status_enum"), default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="emergency_events")
