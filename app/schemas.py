"""Pydantic response/request schemas."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class EmployeeOut(BaseModel):
    id: int
    employee_id: str
    name: str
    email: Optional[str] = None
    department: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AttendanceOut(BaseModel):
    id: int
    employee_id: int
    employee_name: str
    employee_code: str
    timestamp: datetime
    event_type: str
    confidence: Optional[float] = None

    class Config:
        from_attributes = True


class RecognitionResult(BaseModel):
    recognized: bool
    employee: Optional[EmployeeOut] = None
    confidence: Optional[float] = None
    attendance_marked: bool = False
    event_type: Optional[str] = None
    message: str


class MultiRecognitionResponse(BaseModel):
    faces_detected: int
    results: list[RecognitionResult]


class DashboardStats(BaseModel):
    total_employees: int
    attendance_today: int
    visitors_today: int
    total_attendance: int
    weekly_attendance: list[dict]
    department_breakdown: list[dict]
