# Face Enrollment Module — Digital Attendance System

Real-time face enrollment for a laptop-webcam attendance system: live
detection with a bounding box, guided multi-angle capture, liveness
(blink + head-turn) checks, server-side quality validation, duplicate-face
rejection, and storage of face embeddings in SQLite.

## Why the webcam logic lives in the browser

A laptop's camera is only reachable from JavaScript running in a browser
(`getUserMedia`) — a Python process can't reliably grab it through a web UI.
So the architecture is:

```
┌────────────────────────────┐        HTTP (multipart / JSON)        ┌─────────────────────────────┐
│  frontend/enrollment.html   │ ─────────────────────────────────────▶ │   backend/app.py (Flask)    │
│  + enrollment.js            │                                        │                              │
│  - getUserMedia webcam      │        30-50 captured JPEG frames      │  - MediaPipe re-detection    │
│  - face-api.js real-time    │        + student form fields           │  - Quality checks (blur,     │
│    detection & bounding box │ ◀───────────────────────────────────── │    brightness, size, center) │
│  - blink / head-turn        │        success / error JSON            │  - face_recognition embedding│
│    liveness signals         │                                        │  - duplicate-face check      │
│  - capture + progress UI    │                                        │  - SQLite (database/db.py)   │
└────────────────────────────┘                                        └─────────────────────────────┘
```

The browser layer is optimized for responsiveness (instant bounding box,
instant "no face / multiple faces" feedback, instant liveness cues). The
server layer is the one that's trusted: it never assumes the browser's
face detection or liveness result is correct, and re-runs its own checks
before writing anything to disk.

## Folder structure

```
frontend/     enrollment.html, enrollment.js, enrollment.css   (webcam enrollment UI, talks to the Flask API)
              attendance_register.html                          (standalone attendance-taking prototype, see below)
backend/      app.py (Flask API), face_utils.py (detection + embeddings)
database/     schema.sql, db.py (SQLite access layer)
utils/        quality_checks.py (blur / brightness / centering / EAR)
models/       README.md documenting which model files are used and where
assets/faces/ captured face images, one sub-folder per register number
```

## Setup

```bash
cd backend
pip install -r ../requirements.txt

# first run creates database/attendance.db and assets/faces/
python app.py
```

The API listens on `http://localhost:5000`.

Then just open `frontend/enrollment.html` directly in a browser (or serve
it with any static file server) — it talks to the API over CORS-enabled
localhost requests. Grant camera permission when prompted.

## Enrollment flow

1. Camera starts automatically; face-api.js runs `TinyFaceDetector` +
   `FaceLandmark68Net` every frame to draw the bounding box and status
   (✅ one face / ❌ none / ⚠️ multiple).
2. Fill in the student details form. The **Capture & Enroll** button only
   enables once the form is valid *and* exactly one face is being detected.
3. Clicking it first fires a quick duplicate pre-check against the backend,
   then walks the user through five head positions (front, left, right,
   slight up, slight down), grabbing ~8 frames per position while nudging
   for a natural blink and head turn (liveness).
4. All frames + form data are POSTed to `/api/enroll`. The backend:
   - re-detects the face with MediaPipe in every frame,
   - rejects frames that are blurry, too dark/bright, too small, or
     off-center,
   - computes a `face_recognition` embedding per accepted frame,
   - checks the embedding against every already-enrolled student
     (rejects the whole enrollment on a match — no duplicate people),
   - on success, saves the images to `assets/faces/<register_number>/`
     and writes rows to `students` and `face_embeddings` in SQLite.

## `frontend/attendance_register.html` — standalone attendance-taking prototype

This is a separate, self-contained page (own inline CSS/JS, no build step)
that covers the *day-to-day attendance* side rather than the one-time
enrollment side:

- Roster and subject management, with data kept in the browser
  (`localStorage`, or `window.storage` automatically when the page is
  opened inside a Claude artifact — same get/set shape either way, so no
  code changes are needed between the two environments).
- Its own **Enroll Face** flow per student, built on
  `face-api.js` (`TinyFaceDetector` + `FaceLandmark68Net` +
  `FaceRecognitionNet`), guiding the user through a few head poses and
  storing the resulting descriptors client-side.
- A live **recognition** mode for taking attendance: it matches each
  webcam frame's descriptor against the stored per-student descriptors
  and marks students present automatically, with manual present/absent/late
  overrides.
- **Export to Excel** (`.xlsx`, via SheetJS) of the attendance records.

Because it loads `face-api.js` and Google Fonts from a CDN, the camera and
face models only work when the file is served over `http://localhost` or
HTTPS (open it via `start_frontend.bat`, which serves the whole `frontend/`
folder on `http://localhost:5500/attendance_register.html`) — not when
opened directly as a `file://` path, and not inside a sandboxed preview
with no network access.

It runs entirely independently of `backend/app.py` — no server, Flask, or
SQLite required for this page to work by itself. If you want the two
enrollment flows to share one source of truth instead of keeping separate
face data:

- `database/db.py` and `database/schema.sql` are self-contained — point
  `DB_PATH` at your existing attendance database if you already have one,
  and the `students` / `face_embeddings` tables will sit alongside your
  attendance tables (add a `student_id` foreign key from your attendance
  table to `students.id`).
- `backend/face_utils.find_duplicate()` and the same embedding format can
  be reused for live recognition during attendance-taking (compare a live
  frame's embedding against `db.get_all_embeddings()`) instead of the
  `localStorage`/`window.storage` descriptors that `attendance_register.html`
  uses today — this would mean adding `fetch()` calls from that page to
  the Flask API (e.g. `POST /api/enroll`, and a new `GET/POST` endpoint
  for attendance records) in place of its current `appStorage` calls.

## Notes & tuning

- Thresholds for blur / brightness / face size / centering live at the top
  of `utils/quality_checks.py`.
- The liveness thresholds (blink EAR cutoff, head-turn sensitivity) are at
  the top of `frontend/enrollment.js`.
- `MIN_IMAGES_REQUIRED` in `backend/app.py` is a server-side floor (default
  15) independent of how many frames the client claims to have sent.
- Swapping the embedding model for InsightFace/ArcFace is documented at
  the bottom of `backend/face_utils.py`.
