"""
utils/quality_checks.py
------------------------
Server-side re-validation of face image quality before anything is written
to disk or the database. The browser already does a first pass (see
frontend/enrollment.js) so the operator gets instant feedback, but the
server never trusts the client — every frame is re-checked here.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class QualityResult:
    ok: bool
    reason: str = ""
    blur_score: float = 0.0
    brightness: float = 0.0
    face_area_ratio: float = 0.0
    center_offset_ratio: float = 0.0


# ---- tunable thresholds ---------------------------------------------------
BLUR_THRESHOLD = 80.0          # Laplacian variance below this = too blurry
MIN_BRIGHTNESS = 60.0          # mean pixel intensity (0-255) below = too dark
MAX_BRIGHTNESS = 220.0         # above = blown out / overexposed
MIN_FACE_AREA_RATIO = 0.06     # face bbox area / frame area, below = too far away
MAX_CENTER_OFFSET_RATIO = 0.28 # face center distance from frame center / frame width


def compute_blur_score(gray_face: np.ndarray) -> float:
    """Variance of the Laplacian — a common, cheap sharpness proxy.
    Low variance means few sharp edges, i.e. a blurry image."""
    return cv2.Laplacian(gray_face, cv2.CV_64F).var()


def compute_brightness(gray_face: np.ndarray) -> float:
    return float(np.mean(gray_face))


def check_face_quality(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> QualityResult:
    """
    Validate a single frame + detected face bounding box.

    Args:
        frame_bgr: full camera frame, BGR (as read by cv2.imdecode)
        bbox: (x, y, w, h) of the detected face in pixel coordinates

    Returns:
        QualityResult with ok=True only if every check passes.
    """
    frame_h, frame_w = frame_bgr.shape[:2]
    x, y, w, h = bbox

    # Clamp bbox to frame bounds defensively
    x, y = max(0, x), max(0, y)
    w, h = min(w, frame_w - x), min(h, frame_h - y)
    if w <= 0 or h <= 0:
        return QualityResult(ok=False, reason="Invalid face region")

    face_crop = frame_bgr[y:y + h, x:x + w]
    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    blur_score = compute_blur_score(gray_face)
    brightness = compute_brightness(gray_face)
    face_area_ratio = (w * h) / float(frame_w * frame_h)

    face_center = (x + w / 2.0, y + h / 2.0)
    frame_center = (frame_w / 2.0, frame_h / 2.0)
    offset_px = ((face_center[0] - frame_center[0]) ** 2 +
                 (face_center[1] - frame_center[1]) ** 2) ** 0.5
    center_offset_ratio = offset_px / frame_w

    result = QualityResult(
        ok=True,
        blur_score=blur_score,
        brightness=brightness,
        face_area_ratio=face_area_ratio,
        center_offset_ratio=center_offset_ratio,
    )

    if blur_score < BLUR_THRESHOLD:
        result.ok = False
        result.reason = "Image is too blurry — hold still"
    elif brightness < MIN_BRIGHTNESS:
        result.ok = False
        result.reason = "Lighting is too dark"
    elif brightness > MAX_BRIGHTNESS:
        result.ok = False
        result.reason = "Lighting is too bright / overexposed"
    elif face_area_ratio < MIN_FACE_AREA_RATIO:
        result.ok = False
        result.reason = "Face is too small — move closer to the camera"
    elif center_offset_ratio > MAX_CENTER_OFFSET_RATIO:
        result.ok = False
        result.reason = "Face is not centered — move to the middle of the frame"

    return result


def eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    Standard EAR formula (Soukupová & Čech, 2016) used for blink / open-eye
    detection. `eye_landmarks` is a (6, 2) array of the 6 eye contour points
    in the conventional MediaPipe / dlib ordering:
        p1 --- p2  p3
       /              \
      p6                p4
       \\              /
        p5 --- --- ---
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p1, p2, p3, p4, p5, p6 = eye_landmarks
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)
