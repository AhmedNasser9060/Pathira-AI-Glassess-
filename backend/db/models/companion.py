import uuid
from sqlalchemy import Column, String, Boolean, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from backend.db.base import Base

class Companion(Base):
    __tablename__ = "companions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    main_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    companion_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    permissions = Column(JSONB, server_default='{"camera": false, "location": false, "alerts": false}')
    is_active = Column(Boolean, nullable=False, server_default="true")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    main_user = relationship("User", foreign_keys=[main_user_id], back_populates="companions")
    companion = relationship("User", foreign_keys=[companion_id], back_populates="assigned_to")
