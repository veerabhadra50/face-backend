"""SQLAlchemy ORM models.

Embeddings are stored as JSON arrays of 512 floats. This keeps the schema
portable across any PostgreSQL instance (including Supabase) without requiring
the pgvector extension. Similarity is computed in Python with numpy.
"""
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=False)
    email = Column(String(128), nullable=True)
    department = Column(String(128), nullable=True)
    # 512-dimensional facial embedding stored as a JSON list of floats.
    embedding = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    attendances = relationship(
        "Attendance", back_populates="employee", cascade="all, delete-orphan"
    )


class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    # "check_in" or "check_out"
    event_type = Column(String(16), default="check_in")
    confidence = Column(Float, nullable=True)

    employee = relationship("Employee", back_populates="attendances")


class Visitor(Base):
    __tablename__ = "visitors"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    # Best similarity score against known employees (below threshold = unknown).
    best_match_score = Column(Float, nullable=True)
    note = Column(Text, nullable=True)
    # Filename of the saved cropped face image (served from /visitor-faces/).
    image_file = Column(String(255), nullable=True)
