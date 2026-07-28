import uuid
from sqlalchemy import Column, String, ForeignKey, DateTime, ARRAY, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship as sa_relationship
from backend.db.base import Base

class Face(Base):
    __tablename__ = "faces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    relationship = Column(String(100), nullable=True)
    photo_url = Column(String, nullable=True)
    face_embedding = Column(ARRAY(Float), nullable=True) # Storing embeddings as float array
    added_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = sa_relationship("User", foreign_keys=[user_id], back_populates="faces")
    added_by_user = sa_relationship("User", foreign_keys=[added_by], back_populates="added_faces")

