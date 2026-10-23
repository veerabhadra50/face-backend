"""Face detection & recognition service backed by InsightFace.

Loads the InsightFace model pack once (lazy singleton), extracts 512-dim
embeddings from images, and compares embeddings using cosine similarity.

The raw image is never persisted: callers extract embeddings and store only
the resulting vectors.
"""
from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

from .config import get_settings

settings = get_settings()

_model_lock = threading.Lock()
_face_app = None  # type: ignore


def _get_app():
    """Lazily initialise and cache the InsightFace FaceAnalysis app."""
    global _face_app
    if _face_app is None:
        with _model_lock:
            if _face_app is None:
                # Imported here so the module can be imported even if the heavy
                # dependency is missing (e.g. during lightweight tooling).
                from insightface.app import FaceAnalysis

                app = FaceAnalysis(
                    name=settings.insightface_model,
                    providers=["CPUExecutionProvider"],
                )
                # det_size is a good balance of speed/accuracy for webcam frames.
                app.prepare(ctx_id=0, det_size=(640, 640))
                _face_app = app
    return _face_app


class DetectedFace:
    """A single detected face: its embedding, bbox and detection score."""

    def __init__(self, embedding: np.ndarray, bbox: list[float], det_score: float):
        self.embedding = embedding
        self.bbox = bbox
        self.det_score = det_score


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR OpenCV image."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Unsupported or corrupt file.")
    return img


def detect_faces(image_bytes: bytes) -> list[DetectedFace]:
    """Detect all faces in an image and return their normalised embeddings."""
    img = _decode_image(image_bytes)
    app = _get_app()
    faces = app.get(img)

    results: list[DetectedFace] = []
    for f in faces:
        if float(f.det_score) < settings.face_det_threshold:
            continue
        # normed_embedding is L2-normalised -> cosine similarity == dot product.
        emb = np.asarray(f.normed_embedding, dtype=np.float32)
        results.append(
            DetectedFace(
                embedding=emb,
                bbox=[float(x) for x in f.bbox.tolist()],
                det_score=float(f.det_score),
            )
        )
    return results


def crop_face_jpeg(image_bytes: bytes, bbox: list[float], margin: float = 0.25) -> bytes | None:
    """Crop the face region (with a margin) from the image and return JPEG bytes.

    Returns None if the crop fails. Used to save a thumbnail of unknown visitors.
    """
    try:
        img = _decode_image(image_bytes)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = (x2 - x1), (y2 - y1)
        x1 = max(0, int(x1 - bw * margin))
        y1 = max(0, int(y1 - bh * margin))
        x2 = min(w, int(x2 + bw * margin))
        y2 = min(h, int(y2 + bh * margin))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img[y1:y2, x1:x2]
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buf.tobytes() if ok else None
    except Exception:
        return None


def extract_single_embedding(image_bytes: bytes) -> DetectedFace:
    """Extract exactly one face embedding (for registration).

    Raises ValueError if zero or more than one face is found so registration
    always maps to a single person.
    """
    faces = detect_faces(image_bytes)
    if not faces:
        raise ValueError("No face detected. Please face the camera clearly.")
    if len(faces) > 1:
        raise ValueError("Multiple faces detected. Please ensure only one person is in frame.")
    return faces[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)


def best_match(
    probe: np.ndarray, gallery: list[tuple[int, np.ndarray]]
) -> tuple[Optional[int], float]:
    """Return (employee_pk, score) of the closest gallery embedding.

    gallery is a list of (employee_id, embedding) tuples.
    """
    best_id: Optional[int] = None
    best_score = -1.0
    for emp_id, emb in gallery:
        score = cosine_similarity(probe, emb)
        if score > best_score:
            best_score = score
            best_id = emp_id
    return best_id, best_score
