"""
backend/app.py
---------------
Flask API for the face-enrollment module.

Endpoints
---------
POST /api/enroll/check-duplicate
    Body: { "image": "<base64 jpeg>" }
    Runs detection + embedding on a single frame and checks it against every
    already-enrolled face. Used by the frontend right after capture starts,
    so we can stop early and warn the operator instead of wasting 30-50
    frames on someone who's already enrolled.

POST /api/enroll
    multipart/form-data:
        name, register_number, department, year, section  (text fields)
        images[]  (one or more captured JPEG frames, each tagged with a
                   pose label via the matching `poses[]` field)
    Validates every frame server-side (quality_checks), extracts an
    embedding per frame, rejects the whole enrollment if a duplicate face is
    found, and otherwise persists everything to SQLite + disk.

GET /api/students
    Lists enrolled students (for admin / debugging use).
"""

import os
import sys
import base64
import uuid
import json
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import db
from utils.quality_checks import check_face_quality
from backend.face_utils import detect_faces, extract_embedding, find_duplicate

app = Flask(__name__)
CORS(app)  # allow the frontend (served separately / opened as a local file) to call this API

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "faces")
MIN_IMAGES_REQUIRED = 15   # backend floor even if the client claims more were captured


def _decode_base64_image(data_url_or_b64: str) -> np.ndarray:
    """Accepts either a raw base64 string or a data URL (data:image/jpeg;base64,...)."""
    if "," in data_url_or_b64:
        data_url_or_b64 = data_url_or_b64.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url_or_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def _error(message: str, status: int = 400):
    return jsonify({"success": False, "message": message}), status


@app.route("/api/enroll/check-duplicate", methods=["POST"])
def check_duplicate():
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image")
    if not image_b64:
        return _error("No image provided")

    frame = _decode_base64_image(image_b64)
    if frame is None:
        return _error("Could not decode image")

    boxes = detect_faces(frame)
    if len(boxes) != 1:
        return jsonify({"success": True, "duplicate": False,
                         "note": "Skipped duplicate check — need exactly one face"})

    embedding = extract_embedding(frame, boxes[0])
    if embedding is None:
        return jsonify({"success": True, "duplicate": False,
                         "note": "Could not extract embedding for duplicate check"})

    all_embeddings = db.get_all_embeddings()
    match = find_duplicate(embedding, all_embeddings)

    if match:
        return jsonify({
            "success": True,
            "duplicate": True,
            "matched_name": match["name"],
            "matched_register_number": match["register_number"],
            "distance": round(match["distance"], 4),
        })
    return jsonify({"success": True, "duplicate": False})


@app.route("/api/enroll", methods=["POST"])
def enroll_student():
    # ---- 1. Validate student details --------------------------------------
    name = (request.form.get("name") or "").strip()
    register_number = (request.form.get("register_number") or "").strip()
    department = (request.form.get("department") or "").strip()
    year = (request.form.get("year") or "").strip()
    section = (request.form.get("section") or "").strip()

    missing = [f for f, v in [("name", name), ("register_number", register_number),
                               ("department", department), ("year", year),
                               ("section", section)] if not v]
    if missing:
        return _error(f"Missing required field(s): {', '.join(missing)}")

    if db.get_student_by_register_number(register_number):
        return _error(f"Register number '{register_number}' is already enrolled", status=409)

    # ---- 2. Validate captured images ---------------------------------------
    images = request.files.getlist("images")
    poses = request.form.getlist("poses")  # parallel array: pose label per image
    if len(images) < MIN_IMAGES_REQUIRED:
        return _error(
            f"Not enough valid frames captured ({len(images)}/{MIN_IMAGES_REQUIRED} minimum). "
            "Please retry enrollment with better lighting and framing."
        )

    decoded_frames = []
    for idx, file_storage in enumerate(images):
        np_arr = np.frombuffer(file_storage.read(), np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        pose_label = poses[idx] if idx < len(poses) else "unknown"
        decoded_frames.append((frame, pose_label))

    # ---- 3. Per-frame quality + detection + embedding ----------------------
    accepted = []       # list of (frame, bbox, pose_label, embedding, quality_score)
    rejected_reasons = []

    for frame, pose_label in decoded_frames:
        boxes = detect_faces(frame)
        if len(boxes) != 1:
            rejected_reasons.append(f"{pose_label}: expected exactly 1 face, found {len(boxes)}")
            continue

        quality = check_face_quality(frame, boxes[0])
        if not quality.ok:
            rejected_reasons.append(f"{pose_label}: {quality.reason}")
            continue

        embedding = extract_embedding(frame, boxes[0])
        if embedding is None:
            rejected_reasons.append(f"{pose_label}: could not compute embedding")
            continue

        accepted.append((frame, boxes[0], pose_label, embedding, quality.blur_score))

    if len(accepted) < MIN_IMAGES_REQUIRED:
        return _error(
            f"Only {len(accepted)}/{len(decoded_frames)} frames passed quality checks "
            f"(need {MIN_IMAGES_REQUIRED}). First issues: {rejected_reasons[:5]}"
        )

    # ---- 4. Duplicate check against the whole enrolled population ----------
    all_embeddings = db.get_all_embeddings()
    representative_embedding = accepted[len(accepted) // 2][3]  # a "front-ish" frame
    match = find_duplicate(representative_embedding, all_embeddings)
    if match:
        return _error(
            f"This face appears to already be enrolled as "
            f"{match['name']} ({match['register_number']}, distance={match['distance']:.3f}). "
            "Enrollment rejected.",
            status=409,
        )

    # ---- 5. Persist: create student, save images, save embeddings ----------
    try:
        student_id = db.create_student(name, register_number, department, year, section)
    except Exception as e:
        return _error(f"Could not create student record: {e}", status=500)

    student_dir = os.path.join(ASSETS_DIR, register_number)
    os.makedirs(student_dir, exist_ok=True)

    saved_count = 0
    for frame, bbox, pose_label, embedding, quality_score in accepted:
        filename = f"{pose_label}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(student_dir, filename)
        cv2.imwrite(filepath, frame)

        relative_path = os.path.relpath(filepath, ASSETS_DIR)
        db.save_face_embedding(
            student_id=student_id,
            embedding=embedding,
            image_path=relative_path,
            pose_label=pose_label,
            quality_score=quality_score,
        )
        saved_count += 1

    return jsonify({
        "success": True,
        "message": f"Enrolled {name} ({register_number}) with {saved_count} face samples",
        "student_id": student_id,
        "images_saved": saved_count,
        "rejected_frames": len(rejected_reasons),
    })


@app.route("/api/students", methods=["POST"])
def create_student_api():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name") or "").strip()
    register_number = str(payload.get("register_number") or payload.get("regNo") or "").strip()
    department = str(payload.get("department") or "CSE").strip()
    year = str(payload.get("year") or "3").strip()
    section = str(payload.get("section") or "A").strip()

    if not name or not register_number:
        return _error("name and register_number are required")
    if not register_number.isdigit():
        return _error("register_number must contain digits only")
    if db.get_student_by_register_number(register_number):
        return _error(f"Register number '{register_number}' already exists", status=409)

    try:
        student_id = db.create_student(name, register_number, department, year, section)
    except Exception as e:
        return _error(f"Could not create student record: {e}", status=500)

    return jsonify({
        "success": True,
        "message": f"Student {name} added successfully",
        "student": {
            "id": student_id,
            "name": name,
            "regNo": register_number,
            "department": department,
            "year": year,
            "section": section
        }
    }), 201


@app.route("/api/students", methods=["GET"])
def list_students():
    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT s.id, s.name, s.register_number, s.department, s.year, s.section,
                   s.created_at, COUNT(fe.id) as face_count
            FROM students s
            LEFT JOIN face_embeddings fe ON fe.student_id = s.id
            GROUP BY s.id
            ORDER BY s.created_at DESC
        """).fetchall()
    return jsonify({"success": True, "students": [dict(r) for r in rows]})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"success": True, "status": "ok", "time": datetime.utcnow().isoformat()})


if __name__ == "__main__":
    os.makedirs(ASSETS_DIR, exist_ok=True)
    db.init_db()
    print("Face enrollment API running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
