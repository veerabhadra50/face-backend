"""Data-access helpers and recognition/attendance business logic."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from . import face_service, models
from .config import get_settings

settings = get_settings()

# How long after a check-in before the next scan counts as a new event.
DEDUP_MINUTES = 1
# After this gap, a scan is treated as check_out instead of check_in.
CHECKOUT_AFTER_MINUTES = 60


def create_employee(
    db: Session,
    employee_id: str,
    name: str,
    embedding: np.ndarray,
    email: str | None = None,
    department: str | None = None,
) -> models.Employee:
    emp = models.Employee(
        employee_id=employee_id,
        name=name,
        email=email,
        department=department,
        embedding=[float(x) for x in embedding.tolist()],
    )
    db.add(emp)
    db.commit()
    db.refresh(emp)
    return emp


def get_employee_by_code(db: Session, employee_id: str) -> models.Employee | None:
    return (
        db.query(models.Employee)
        .filter(func.lower(models.Employee.employee_id) == employee_id.strip().lower())
        .first()
    )


def list_employees(db: Session) -> list[models.Employee]:
    return db.query(models.Employee).order_by(models.Employee.created_at.desc()).all()


# Prefix for auto-generated employee IDs, e.g. EMP001.
EMP_ID_PREFIX = "EMP"


def next_employee_id(db: Session) -> str:
    """Compute the next sequential employee ID like EMP001, EMP002, ...

    Scans existing IDs case-insensitively (so "emp001", "EMP001", "Emp001" all
    count) and returns the max + 1. Then guards against collisions in case an ID
    with that value already exists in a different format.
    """
    codes = [c[0] for c in db.query(models.Employee.employee_id).all()]
    prefix_lower = EMP_ID_PREFIX.lower()
    max_n = 0
    for code in codes:
        if not code:
            continue
        c = code.strip().lower()
        if c.startswith(prefix_lower):
            suffix = c[len(prefix_lower):]
            if suffix.isdigit():
                max_n = max(max_n, int(suffix))

    # Build the next ID and ensure it doesn't collide (case-insensitively).
    existing_lower = {c.strip().lower() for c in codes if c}
    n = max_n + 1
    while f"{EMP_ID_PREFIX}{n:03d}".lower() in existing_lower:
        n += 1
    return f"{EMP_ID_PREFIX}{n:03d}"


def load_gallery(db: Session) -> list[tuple[int, np.ndarray]]:
    """Load every employee embedding for matching."""
    rows = db.query(models.Employee.id, models.Employee.embedding).all()
    return [(rid, np.asarray(emb, dtype=np.float32)) for rid, emb in rows]


def _last_attendance(db: Session, employee_pk: int) -> models.Attendance | None:
    return (
        db.query(models.Attendance)
        .filter(models.Attendance.employee_id == employee_pk)
        .order_by(models.Attendance.timestamp.desc())
        .first()
    )


def mark_attendance(
    db: Session, employee_pk: int, confidence: float
) -> tuple[models.Attendance | None, str, bool]:
    """Record attendance with simple de-dup + check-in/out inference.

    Returns (attendance_row_or_None, event_type, newly_marked).
    """
    now = datetime.utcnow()
    last = _last_attendance(db, employee_pk)

    event_type = "check_in"
    if last is not None:
        delta = now - last.timestamp
        if delta < timedelta(minutes=DEDUP_MINUTES):
            # Too soon: treat as duplicate scan, don't create a new row.
            return None, last.event_type, False
        if last.event_type == "check_in" and delta < timedelta(minutes=CHECKOUT_AFTER_MINUTES):
            event_type = "check_out"

    row = models.Attendance(
        employee_id=employee_pk, confidence=confidence, event_type=event_type
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row, event_type, True


def log_visitor(
    db: Session, best_score: float | None, image_file: str | None = None
) -> models.Visitor:
    v = models.Visitor(
        best_match_score=best_score, note="Unrecognized face", image_file=image_file
    )
    db.add(v)
    db.commit()
    db.refresh(v)
    return v


def list_visitors(db: Session, limit: int = 50) -> list[dict]:
    rows = (
        db.query(models.Visitor)
        .order_by(models.Visitor.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": v.id,
            "timestamp": v.timestamp,
            "best_match_score": v.best_match_score,
            "image_file": v.image_file,
        }
        for v in rows
    ]


def list_attendance(
    db: Session, search: str | None = None, limit: int = 200
) -> list[dict]:
    q = (
        db.query(models.Attendance, models.Employee)
        .join(models.Employee, models.Attendance.employee_id == models.Employee.id)
        .order_by(models.Attendance.timestamp.desc())
    )
    if search:
        like = f"%{search}%"
        q = q.filter(
            (models.Employee.name.ilike(like))
            | (models.Employee.employee_id.ilike(like))
            | (models.Employee.department.ilike(like))
        )
    rows = q.limit(limit).all()
    return [
        {
            "id": att.id,
            "employee_id": emp.id,
            "employee_name": emp.name,
            "employee_code": emp.employee_id,
            "timestamp": att.timestamp,
            "event_type": att.event_type,
            "confidence": att.confidence,
        }
        for att, emp in rows
    ]


def dashboard_stats(db: Session) -> dict:
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    total_employees = db.query(func.count(models.Employee.id)).scalar() or 0
    total_attendance = db.query(func.count(models.Attendance.id)).scalar() or 0
    attendance_today = (
        db.query(func.count(models.Attendance.id))
        .filter(models.Attendance.timestamp >= today_start)
        .scalar()
        or 0
    )
    visitors_today = (
        db.query(func.count(models.Visitor.id))
        .filter(models.Visitor.timestamp >= today_start)
        .scalar()
        or 0
    )

    # Last 7 days attendance counts.
    weekly = []
    for i in range(6, -1, -1):
        day_start = today_start - timedelta(days=i)
        day_end = day_start + timedelta(days=1)
        count = (
            db.query(func.count(models.Attendance.id))
            .filter(
                models.Attendance.timestamp >= day_start,
                models.Attendance.timestamp < day_end,
            )
            .scalar()
            or 0
        )
        weekly.append({"date": day_start.strftime("%Y-%m-%d"), "count": count})

    dept_rows = (
        db.query(models.Employee.department, func.count(models.Employee.id))
        .group_by(models.Employee.department)
        .all()
    )
    department_breakdown = [
        {"department": dept or "Unassigned", "count": count} for dept, count in dept_rows
    ]

    return {
        "total_employees": total_employees,
        "attendance_today": attendance_today,
        "visitors_today": visitors_today,
        "total_attendance": total_attendance,
        "weekly_attendance": weekly,
        "department_breakdown": department_breakdown,
    }
