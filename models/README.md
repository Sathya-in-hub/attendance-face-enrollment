# Models used in this pipeline

This project deliberately splits detection work across two layers so the
webcam UI stays fast (real-time, in-browser) while the backend does the
authoritative, harder-to-fool checks.

## 1. Client-side (frontend/enrollment.js) — `face-api.js`
Loaded from CDN at runtime, no local files needed:
- **TinyFaceDetector** — real-time bounding box + "how many faces" for the
  live overlay.
- **FaceLandmark68Net** — 68-point landmarks, used for:
  - Eye Aspect Ratio (blink detection / liveness)
  - Yaw estimation (head-turn liveness / pose labeling for capture angles)
- **FaceRecognitionNet** — optional client-side 128-d descriptor, used only
  to give the user instant "you look already enrolled" feedback before
  sending anything to the server. The server-side embedding is always the
  authoritative one used for the actual duplicate check and storage.

## 2. Server-side (backend/face_utils.py)
- **MediaPipe Face Detection** (`mediapipe.solutions.face_detection`,
  `model_selection=1`) — re-detects the face in every uploaded frame. Never
  trusts the browser's bounding box.
- **face_recognition** (dlib ResNet, 128-d embeddings) — the embedding
  stored in `face_embeddings.embedding` and used for duplicate detection.
  No model files to manage manually; `face_recognition` downloads its dlib
  models on first import.

### Swapping in InsightFace / ArcFace
If you have a GPU and want higher-accuracy 512-d embeddings, see the
commented-out block at the bottom of `backend/face_utils.py` — it's a
drop-in replacement for `extract_embedding()`.
