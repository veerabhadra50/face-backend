"""FastAPI application: FaceAttendance smart door system."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from . import crud, face_service, models, schemas
from .config import get_settings
from .database import Base, engine, get_db

settings = get_settings()

# Create tables on startup (safe: only creates missing tables).
Base.metadata.create_all(bind=engine)

app = FastAPI(title="FaceAttendance", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder where cropped visitor face thumbnails are saved, then served at
# /visitor-faces/<file>. NOTE: this stores images, unlike employees which
# store only embeddings.
VISITOR_FACES_DIR = Path(__file__).resolve().parent.parent / "visitor_faces"
VISITOR_FACES_DIR.mkdir(exist_ok=True)
app.mount(
    "/visitor-faces",
    StaticFiles(directory=str(VISITOR_FACES_DIR)),
    name="visitor-faces",
)


def _save_visitor_face(image_bytes: bytes, bbox: list[float]) -> str | None:
    """Crop the visitor's face and save it as a JPEG. Returns the filename."""
    jpeg = face_service.crop_face_jpeg(image_bytes, bbox)
    if not jpeg:
        return None
    filename = f"visitor_{uuid.uuid4().hex}.jpg"
    (VISITOR_FACES_DIR / filename).write_bytes(jpeg)
    return filename


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def public_config():
    """Public config for the frontend. Only the ANON key is exposed here.

    The service_role key is a secret and is never returned to clients.
    """
    return {
        "supabase_url": settings.supabase_url,
        "supabase_anon_key": settings.supabase_anon_key,
    }


@app.get("/api/next-employee-id")
def get_next_employee_id(db: Session = Depends(get_db)):
    """Return the next auto-generated employee ID (e.g. EMP001)."""
    return {"employee_id": crud.next_employee_id(db)}


@app.post("/api/employees", response_model=schemas.EmployeeOut)
async def register_employee(
    employee_id: str | None = Form(None),
    name: str = Form(...),
    email: str | None = Form(None),
    department: str | None = Form(None),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Register an employee. Extracts a single face embedding then discards the image.

    If employee_id is omitted or blank, a sequential ID is generated automatically.
    """
    employee_id = (employee_id or "").strip() or crud.next_employee_id(db)
    if crud.get_employee_by_code(db, employee_id):
        raise HTTPException(status_code=409, detail=f"Employee ID '{employee_id}' already exists.")

    image_bytes = await image.read()
    try:
        face = face_service.extract_single_embedding(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    emp = crud.create_employee(
        db,
        employee_id=employee_id,
        name=name,
        embedding=face.embedding,
        email=email,
        department=department,
    )
    return emp


@app.get("/api/employees", response_model=list[schemas.EmployeeOut])
def get_employees(db: Session = Depends(get_db)):
    return crud.list_employees(db)


@app.post("/api/recognize", response_model=schemas.MultiRecognitionResponse)
async def recognize(
    image: UploadFile = File(...),
    mark: bool = Form(True),
    db: Session = Depends(get_db),
):
    """Recognize all faces in a frame and optionally mark attendance."""
    image_bytes = await image.read()
    try:
        faces = face_service.detect_faces(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not faces:
        return schemas.MultiRecognitionResponse(
            faces_detected=0,
            results=[
                schemas.RecognitionResult(
                    recognized=False, message="No face detected in frame."
                )
            ],
        )

    gallery = crud.load_gallery(db)
    results: list[schemas.RecognitionResult] = []

    for face in faces:
        if not gallery:
            img_file = _save_visitor_face(image_bytes, face.bbox)
            crud.log_visitor(db, None, image_file=img_file)
            results.append(
                schemas.RecognitionResult(
                    recognized=False,
                    message="No employees registered yet.",
                )
            )
            continue

        emp_pk, score = face_service.best_match(face.embedding, gallery)

        if score >= settings.face_match_threshold and emp_pk is not None:
            emp = db.query(models.Employee).get(emp_pk)
            attendance_marked = False
            event_type = None
            if mark:
                row, event_type, attendance_marked = crud.mark_attendance(db, emp_pk, score)
            msg = (
                f"{emp.name} recognized"
                + (f" — {event_type} marked" if attendance_marked else " (already logged)")
            )
            results.append(
                schemas.RecognitionResult(
                    recognized=True,
                    employee=schemas.EmployeeOut.model_validate(emp),
                    confidence=round(score, 4),
                    attendance_marked=attendance_marked,
                    event_type=event_type,
                    message=msg,
                )
            )
        else:
            img_file = _save_visitor_face(image_bytes, face.bbox)
            crud.log_visitor(db, round(score, 4), image_file=img_file)
            results.append(
                schemas.RecognitionResult(
                    recognized=False,
                    confidence=round(score, 4),
                    message="Unknown face — visitor logged.",
                )
            )

    return schemas.MultiRecognitionResponse(faces_detected=len(faces), results=results)


@app.get("/api/attendance", response_model=list[schemas.AttendanceOut])
def get_attendance(search: str | None = None, limit: int = 200, db: Session = Depends(get_db)):
    return crud.list_attendance(db, search=search, limit=limit)


@app.get("/api/dashboard", response_model=schemas.DashboardStats)
def get_dashboard(db: Session = Depends(get_db)):
    return crud.dashboard_stats(db)


@app.get("/api/visitors")
def get_visitors(limit: int = 50, db: Session = Depends(get_db)):
    """Recent unknown visitors, including a URL to their saved face thumbnail."""
    rows = crud.list_visitors(db, limit=limit)
    for r in rows:
        r["image_url"] = f"/visitor-faces/{r['image_file']}" if r.get("image_file") else None
    return rows


# ---- Static frontend (served if FRONTEND_DIR is configured and exists) ----
def _resolve_frontend_dir() -> Path | None:
    if settings.frontend_dir:
        p = Path(settings.frontend_dir)
        return p if p.exists() else None
    # Fallback: sibling "face-login/frontend" next to the backend's parent drive layout.
    guess = Path(__file__).resolve().parent.parent.parent / "face-login" / "frontend"
    return guess if guess.exists() else None


FRONTEND_DIR = _resolve_frontend_dir()
if FRONTEND_DIR is not None:
    # Mount at root with html=True so index.html is served at "/" and sibling
    # assets (styles.css, app.js) resolve at their root paths. This mount is
    # added last, so the /api/* routes above take precedence.
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
