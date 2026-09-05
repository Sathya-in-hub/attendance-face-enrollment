/* ==========================================================================
   Face Enrollment — client logic
   Real-time detection runs entirely in the browser (face-api.js, via CDN
   models) for instant feedback. Captured frames are sent to the Python
   backend (backend/app.py) which re-validates everything and is the source
   of truth for storage.
   ========================================================================== */

// const API_BASE = "https://attendance-face-enrollment.onrender.com";
// const MODEL_URL = "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights";
const API_BASE = "https://attendance-face-enrollment.onrender.com";
const MODEL_URL = "https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/weights";

const TARGET_FRAMES = 40;           // total frames to capture across all poses
const POSES = ["front", "left", "right", "up", "down"];
const FRAMES_PER_POSE = Math.ceil(TARGET_FRAMES / POSES.length);

const EAR_BLINK_THRESHOLD = 0.21;   // below this = eyes considered closed
const YAW_TURN_THRESHOLD = 0.12;    // normalized horizontal nose offset for a "head turn"

// ---------------------------------------------------------------- DOM refs
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const camPlaceholder = document.getElementById("camPlaceholder");
const faceStatusBadge = document.getElementById("faceStatusBadge");
const livenessBadge = document.getElementById("livenessBadge");
const livenessLine = document.getElementById("livenessLine");
const captureBtn = document.getElementById("captureBtn");
const resetBtn = document.getElementById("resetBtn");
const progressFill = document.getElementById("progressFill");
const progressLabel = document.getElementById("progressLabel");
const progressCount = document.getElementById("progressCount");
const poseChips = Array.from(document.querySelectorAll(".pose-chip"));
const toastEl = document.getElementById("toast");

const formFields = {
  name: document.getElementById("inpName"),
  register_number: document.getElementById("inpRegNo"),
  department: document.getElementById("inpDept"),
  year: document.getElementById("inpYear"),
  section: document.getElementById("inpSection"),
};

// ---------------------------------------------------------------- state
let modelsReady = false;
let detectLoopHandle = null;
let currentFaceCount = 0;
let blinkObserved = false;      // liveness: has a blink happened this session?
let headTurnObserved = false;   // liveness: has a left/right turn happened this session?
let baselineNoseX = null;       // reference nose x for yaw comparisons
let earHistory = [];            // rolling EAR values to detect a blink (open->closed->open)

let capturing = false;
let captureQueue = [];          // { blob, pose }
let poseIndex = 0;
let framesForCurrentPose = 0;

// ============================================================== UTILITIES

function showToast(message, kind = "") {
  toastEl.textContent = message;
  toastEl.className = "toast show" + (kind ? " " + kind : "");
  setTimeout(() => { toastEl.className = "toast"; }, 3200);
}

function setBadge(el, text, level) {
  el.className = `status-badge ${level}`;
  el.innerHTML = `<span class="dot"></span> ${text}`;
}

function validateField(key, el) {
  const wrapper = document.getElementById(
    `field-${key === "register_number" ? "regno" : key}`
  );
  const value = el.value.trim();
  const isValid = value.length > 0;
  wrapper.classList.toggle("invalid", !isValid);
  wrapper.classList.toggle("valid", isValid);
  return isValid;
}

function formIsValid() {
  let ok = true;
  for (const [key, el] of Object.entries(formFields)) {
    if (!validateField(key, el)) ok = false;
  }
  return ok;
}

function getFormData() {
  return {
    name: formFields.name.value.trim(),
    register_number: formFields.register_number.value.trim(),
    department: formFields.department.value.trim(),
    year: formFields.year.value.trim(),
    section: formFields.section.value.trim(),
  };
}

Object.values(formFields).forEach(el => {
  el.addEventListener("input", updateCaptureButtonState);
  el.addEventListener("change", updateCaptureButtonState);
});

function updateCaptureButtonState() {
  const formOk = formIsValid();
  const faceOk = currentFaceCount === 1;
  captureBtn.disabled = !(formOk && faceOk && modelsReady) || capturing;
}

// ============================================================== CAMERA + MODELS
async function loadModels() {
  try {
    console.log("Loading Tiny Face Detector...");
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
    console.log("Tiny Face Detector loaded.");

    console.log("Loading Face Landmark 68...");
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
    console.log("Face Landmark 68 loaded.");

    modelsReady = true;
    console.log("All face-api models loaded successfully.");
  } catch (error) {
    modelsReady = false;
    console.error("FACE API MODEL ERROR:", error);
    throw error;
  }
}
async function startCamera() {
  try {
    // Check whether the browser supports camera access
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        "Camera access is not supported. Please use HTTPS or localhost."
      );
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "user" },
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    });

    video.srcObject = stream;

    video.onloadedmetadata = async () => {
      try {
        await video.play();

        camPlaceholder.style.display = "none";
        video.style.display = "block";
        overlay.style.display = "block";

        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;

        setBadge(faceStatusBadge, "Camera ready", "ok");

      } catch (playError) {
        console.error("Video play error:", playError);
        setBadge(faceStatusBadge, "Camera unavailable", "bad");
      }
    };

  } catch (err) {
    console.error("Camera error:", err);

    camPlaceholder.style.display = "flex";
    video.style.display = "none";
    overlay.style.display = "none";

    let message = "Camera access failed.";

    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      message =
        "Camera permission was denied. Please allow camera access in your browser settings and reload this page.";
    } else if (err.name === "NotFoundError") {
      message =
        "No camera was found. Please connect a camera and try again.";
    } else if (err.name === "NotReadableError") {
      message =
        "The camera is already being used by another application.";
    } else if (err.name === "SecurityError") {
      message =
        "Camera access requires HTTPS. Open this website using an HTTPS URL.";
    }

    camPlaceholder.innerHTML = `
      <strong>${message}</strong>
      <br><br>
      <small>
        Make sure your browser has permission to use the camera.
      </small>
    `;

    setBadge(faceStatusBadge, "Camera unavailable", "bad");
  }
}

// ============================================================== LIVENESS HELPERS

// Eye Aspect Ratio from 6 landmark points, in the dlib/face-api ordering.
function eyeAspectRatio(eye) {
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const vertical1 = dist(eye[1], eye[5]);
  const vertical2 = dist(eye[2], eye[4]);
  const horizontal = dist(eye[0], eye[3]);
  if (horizontal === 0) return 0;
  return (vertical1 + vertical2) / (2 * horizontal);
}

function updateLiveness(landmarks, boxWidth) {
  const leftEye = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();
  const nose = landmarks.getNose();

  const avgEAR = (eyeAspectRatio(leftEye) + eyeAspectRatio(rightEye)) / 2;
  earHistory.push(avgEAR);
  if (earHistory.length > 15) earHistory.shift();

  // Blink = EAR dips below threshold then recovers above it within the window.
  if (!blinkObserved && earHistory.length >= 3) {
    const min = Math.min(...earHistory);
    const last = earHistory[earHistory.length - 1];
    if (min < EAR_BLINK_THRESHOLD && last >= EAR_BLINK_THRESHOLD) {
      blinkObserved = true;
    }
  }

  // Head turn = nose tip x drifts far enough from the first-seen baseline.
  const noseX = nose[3].x; // tip of nose
  if (baselineNoseX === null) baselineNoseX = noseX;
  const normalizedOffset = (noseX - baselineNoseX) / boxWidth;
  if (Math.abs(normalizedOffset) > YAW_TURN_THRESHOLD) {
    headTurnObserved = true;
  }

  const parts = [];
  parts.push(blinkObserved ? "✅ blink detected" : "… waiting for a natural blink");
  parts.push(headTurnObserved ? "✅ head movement detected" : "… waiting for slight head turn");
  livenessLine.textContent = parts.join("   ");

  const liveOk = blinkObserved && headTurnObserved;
  setBadge(
    livenessBadge,
    liveOk ? "Liveness: verified" : "Liveness: checking…",
    liveOk ? "ok" : "warn"
  );
  return liveOk;
}

function resetLiveness() {
  blinkObserved = false;
  headTurnObserved = false;
  baselineNoseX = null;
  earHistory = [];
  setBadge(livenessBadge, "Liveness: waiting", "idle");
  livenessLine.textContent = "";
}

// ============================================================== DETECTION LOOP

async function detectionTick() {
  if (!modelsReady || video.paused || video.ended) {
    detectLoopHandle = requestAnimationFrame(detectionTick);
    return;
  }

  const displaySize = { width: overlay.clientWidth, height: overlay.clientHeight };
  overlay.width = displaySize.width;
  overlay.height = displaySize.height;

  const detections = await faceapi
    .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks();

  const resized = faceapi.resizeResults(detections, displaySize);
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  currentFaceCount = resized.length;

  if (currentFaceCount === 0) {
    setBadge(faceStatusBadge, "❌ No Face Detected", "bad");
    resetLiveness();
  } else if (currentFaceCount > 1) {
    setBadge(faceStatusBadge, "⚠️ Multiple Faces Detected", "warn");
    resized.forEach(det => drawBox(ctx, det.detection.box, "#a67c1e"));
    resetLiveness();
  } else {
    setBadge(faceStatusBadge, "✅ Face Detected", "ok");
    const det = resized[0];
    drawBox(ctx, det.detection.box, "#2f5233");
    updateLiveness(det.landmarks, det.detection.box.width);
  }

  updateCaptureButtonState();
  detectLoopHandle = requestAnimationFrame(detectionTick);
}

function drawBox(ctx, box, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(box.x, box.y, box.width, box.height);
}

// ============================================================== CAPTURE FLOW

function updatePoseChips() {
  poseChips.forEach((chip, idx) => {
    chip.classList.toggle("done", idx < poseIndex);
    chip.classList.toggle("active", idx === poseIndex && capturing);
  });
}

function updateProgress() {
  const pct = Math.min(100, Math.round((captureQueue.length / TARGET_FRAMES) * 100));
  progressFill.style.width = pct + "%";
  progressLabel.textContent = `${pct}% captured`;
  progressCount.textContent = `${captureQueue.length} / ${TARGET_FRAMES} frames`;
}

function poseInstruction(pose) {
  const map = {
    front: "Look straight at the camera",
    left: "Slowly turn your head to the LEFT",
    right: "Slowly turn your head to the RIGHT",
    up: "Tilt your chin slightly UP",
    down: "Tilt your chin slightly DOWN",
  };
  return map[pose] || "Hold still";
}

async function captureFrameBlob() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.9));
}

async function runCaptureSequence() {
  capturing = true;
  captureBtn.disabled = true;
  captureBtn.textContent = "Capturing…";
  captureQueue = [];
  poseIndex = 0;
  framesForCurrentPose = 0;
  updateProgress();

  for (poseIndex = 0; poseIndex < POSES.length; poseIndex++) {
    const pose = POSES[poseIndex];
    updatePoseChips();
    showToast(poseInstruction(pose));
    framesForCurrentPose = 0;

    // Small pause so the user can react to the instruction before we start grabbing frames.
    await sleep(900);

    while (framesForCurrentPose < FRAMES_PER_POSE && captureQueue.length < TARGET_FRAMES) {
      if (currentFaceCount !== 1) {
        // Pause capture if the face is lost / duplicated; wait and re-check.
        await sleep(300);
        continue;
      }
      const blob = await captureFrameBlob();
      captureQueue.push({ blob, pose });
      framesForCurrentPose++;
      updateProgress();
      await sleep(180); // ~5-6 frames/sec, gentle on the CPU and gives natural pose variety
    }
  }

  poseIndex = POSES.length;
  updatePoseChips();
  await submitEnrollment();
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ============================================================== BACKEND CALLS

async function checkDuplicateEarly() {
  try {
    const blob = await captureFrameBlob();
    const b64 = await blobToBase64(blob);
    const res = await fetch(`${API_BASE}/api/enroll/check-duplicate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: b64 }),
    });
    const data = await res.json();
    if (data.duplicate) {
      showToast(
        `This face looks already enrolled as ${data.matched_name} (${data.matched_register_number}).`,
        "error"
      );
      return true;
    }
  } catch (err) {
    console.warn("Duplicate pre-check failed (backend may be offline):", err);
  }
  return false;
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function submitEnrollment() {
  const formData = new FormData();
  const details = getFormData();
  Object.entries(details).forEach(([k, v]) => formData.append(k, v));
  captureQueue.forEach(item => {
    formData.append("images", item.blob, `${item.pose}.jpg`);
    formData.append("poses", item.pose);
  });

  try {
    const res = await fetch(`${API_BASE}/api/enroll`, { method: "POST", body: formData });
    const data = await res.json();
    if (data.success) {
      showToast(data.message, "success");
      resetAll();
    } else {
      showToast(data.message || "Enrollment failed.", "error");
      resetCaptureOnly();
    }
  } catch (err) {
    console.error(err);
    showToast("Could not reach the enrollment server. Is backend/app.py running?", "error");
    resetCaptureOnly();
  }
}

// ============================================================== RESET

function resetCaptureOnly() {
  capturing = false;
  captureQueue = [];
  poseIndex = 0;
  captureBtn.textContent = "Capture & Enroll";
  updateProgress();
  updatePoseChips();
  updateCaptureButtonState();
}

function resetAll() {
  resetCaptureOnly();
  Object.values(formFields).forEach(el => { el.value = ""; });
  document.querySelectorAll(".field").forEach(f => f.classList.remove("valid", "invalid"));
  resetLiveness();
}

// ============================================================== WIRE UP

captureBtn.addEventListener("click", async () => {
  if (capturing) return;
  const duplicate = await checkDuplicateEarly();
  if (duplicate) return;
  runCaptureSequence();
});

resetBtn.addEventListener("click", resetAll);

(async function init() {
  setBadge(faceStatusBadge, "Loading face models…", "idle");
  try {
    await loadModels();
  } catch (err) {
    console.error("Failed to load face-api models", err);
    setBadge(faceStatusBadge, "Model load failed — check connection", "bad");
    return;
  }
  await startCamera();
  setBadge(faceStatusBadge, "Looking for a face…", "idle");
  updateProgress();
  updatePoseChips();
  detectLoopHandle = requestAnimationFrame(detectionTick);
})();
