"""
backend/face_utils.py
----------------------
Wraps face detection + embedding extraction so the rest of the backend
doesn't need to know which underlying model is being used.

Detection : MediaPipe Face Detection (fast, CPU-friendly, good for a laptop
            webcam pipeline).
Embedding : face_recognition (dlib ResNet-based, 128-d vectors). This is the
            easiest production-grade FaceNet-style embedding to install
            without a GPU. Swap `extract_embedding()` for an InsightFace
            (ArcFace, 512-d) call if higher accuracy / GPU is available —
            see the comment at the bottom of this file for how.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional, Tuple

mp_face_detection = mp.solutions.face_detection

# Reused across calls — cheap to keep alive, expensive to reconstruct per frame.
_detector = mp_face_detection.FaceDetection(
    model_selection=1,          # 1 = full-range model, better for varied distances
    min_detection_confidence=0.6,
)

try:
    import face_recognition
    _HAS_FACE_RECOGNITION = True
except ImportError:
    _HAS_FACE_RECOGNITION = False


def detect_faces(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Run MediaPipe face detection on a BGR frame.
    Returns a list of (x, y, w, h) bounding boxes in pixel coordinates.
    """
    frame_h, frame_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _detector.process(frame_rgb)

    boxes = []
    if results.detections:
        for det in results.detections:
            rbb = det.location_data.relative_bounding_box
            x = int(rbb.xmin * frame_w)
            y = int(rbb.ymin * frame_h)
            w = int(rbb.width * frame_w)
            h = int(rbb.height * frame_h)
            boxes.append((x, y, w, h))
    return boxes


def extract_embedding(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[List[float]]:
    """
    Extract a 128-d face embedding for the face in `bbox`.
    Returns None if no embedding could be computed (e.g. crop too small).
    """
    if not _HAS_FACE_RECOGNITION:
        raise RuntimeError(
            "face_recognition is not installed. Run `pip install face_recognition` "
            "(requires dlib) or swap in an InsightFace backend — see file docstring."
        )

    x, y, w, h = bbox
    frame_h, frame_w = frame_bgr.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, frame_w - x), min(h, frame_h - y)
    if w <= 0 or h <= 0:
        return None

    # face_recognition expects (top, right, bottom, left) boxes, in RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    face_location = (y, x + w, y + h, x)  # top, right, bottom, left

    encodings = face_recognition.face_encodings(rgb, known_face_locations=[face_location])
    if not encodings:
        return None
    return encodings[0].tolist()


def euclidean_distance(vec_a: List[float], vec_b: List[float]) -> float:
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.linalg.norm(a - b))


def find_duplicate(new_embedding: List[float], all_embeddings: List[dict],
                    threshold: float = 0.45) -> Optional[dict]:
    """
    Compares `new_embedding` against every stored embedding.
    A distance below `threshold` (face_recognition's typical match cutoff)
    means the same person is very likely already enrolled.

    Returns the matching record (with student name/register_number) or None.
    """
    best_match = None
    best_distance = float("inf")
    for record in all_embeddings:
        dist = euclidean_distance(new_embedding, record["embedding"])
        if dist < best_distance:
            best_distance = dist
            best_match = record

    if best_match is not None and best_distance < threshold:
        best_match = dict(best_match)
        best_match["distance"] = best_distance
        return best_match
    return None


# ---------------------------------------------------------------------------
# To swap in InsightFace (ArcFace, 512-d embeddings, GPU-accelerated):
#
#   from insightface.app import FaceAnalysis
#   _app = FaceAnalysis(name="buffalo_l")
#   _app.prepare(ctx_id=0)   # -1 for CPU
#
#   def extract_embedding(frame_bgr, bbox):
#       faces = _app.get(frame_bgr)
#       if not faces:
#           return None
#       return faces[0].normed_embedding.tolist()
#
# and lower `threshold` in find_duplicate to ~0.35-0.40 for cosine-normalized
# ArcFace vectors (or switch euclidean_distance to a cosine-distance function).
# ---------------------------------------------------------------------------
