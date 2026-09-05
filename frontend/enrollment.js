
// Render backend
const API_BASE = "https://attendance-face-enrollment.onrender.com";

// IMPORTANT:
// The npm package does NOT contain the model files under /weights.
// Use the GitHub repository through jsDelivr.
const MODEL_URL =
  "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights";

const TARGET_FRAMES = 40;

const POSES = [
  "front",
  "left",
  "right",
  "up",
  "down"
];

const FRAMES_PER_POSE = Math.ceil(
  TARGET_FRAMES / POSES.length
);

const EAR_BLINK_THRESHOLD = 0.21;
const YAW_TURN_THRESHOLD = 0.12;


/* ========================================================================== 
   DOM REFERENCES
   ========================================================================== */

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const camPlaceholder = document.getElementById("camPlaceholder");

const faceStatusBadge =
  document.getElementById("faceStatusBadge");

const livenessBadge =
  document.getElementById("livenessBadge");

const livenessLine =
  document.getElementById("livenessLine");

const captureBtn =
  document.getElementById("captureBtn");

const resetBtn =
  document.getElementById("resetBtn");

const progressFill =
  document.getElementById("progressFill");

const progressLabel =
  document.getElementById("progressLabel");

const progressCount =
  document.getElementById("progressCount");

const poseChips =
  Array.from(document.querySelectorAll(".pose-chip"));

const toastEl =
  document.getElementById("toast");


/* ========================================================================== 
   FORM REFERENCES
   ========================================================================== */

const formFields = {
  name: document.getElementById("inpName"),
  register_number: document.getElementById("inpRegNo"),
  department: document.getElementById("inpDept"),
  year: document.getElementById("inpYear"),
  section: document.getElementById("inpSection")
};


/* ========================================================================== 
   STATE
   ========================================================================== */

let modelsReady = false;

let detectLoopHandle = null;

let currentFaceCount = 0;

let blinkObserved = false;

let headTurnObserved = false;

let baselineNoseX = null;

let earHistory = [];

let capturing = false;

let captureQueue = [];

let poseIndex = 0;

let framesForCurrentPose = 0;


/* ========================================================================== 
   BASIC SAFETY CHECKS
   ========================================================================== */

function checkRequiredElements() {

  const required = {
    video,
    overlay,
    camPlaceholder,
    faceStatusBadge,
    livenessBadge,
    livenessLine,
    captureBtn,
    resetBtn,
    progressFill,
    progressLabel,
    progressCount,
    toastEl
  };

  for (const [name, element] of Object.entries(required)) {

    if (!element) {
      console.error(
        `Required HTML element not found: ${name}`
      );
      return false;
    }
  }

  for (const [name, element] of Object.entries(formFields)) {

    if (!element) {
      console.error(
        `Required form field not found: ${name}`
      );
      return false;
    }
  }

  return true;
}


/* ========================================================================== 
   TOAST
   ========================================================================== */

function showToast(message, kind = "") {

  if (!toastEl) return;

  toastEl.textContent = message;

  toastEl.className =
    "toast show" +
    (kind ? ` ${kind}` : "");

  setTimeout(() => {

    toastEl.className = "toast";

  }, 3200);
}


/* ========================================================================== 
   STATUS BADGES
   ========================================================================== */

function setBadge(el, text, level) {

  if (!el) return;

  el.className =
    `status-badge ${level}`;

  el.innerHTML =
    `<span class="dot"></span> ${text}`;
}


/* ========================================================================== 
   FORM VALIDATION
   ========================================================================== */

function getFieldWrapper(key) {

  const wrapperIds = {

    name: "field-name",

    register_number: "field-regno",

    // IMPORTANT:
    // HTML uses field-dept, NOT field-department.
    department: "field-dept",

    year: "field-year",

    section: "field-section"
  };

  return document.getElementById(
    wrapperIds[key]
  );
}


function validateField(key, el) {

  if (!el) {
    return false;
  }

  const wrapper =
    getFieldWrapper(key);

  const value =
    String(el.value || "").trim();

  const isValid =
    value.length > 0;

  if (wrapper) {

    wrapper.classList.toggle(
      "invalid",
      !isValid
    );

    wrapper.classList.toggle(
      "valid",
      isValid
    );

  }

  return isValid;
}


function formIsValid() {

  let ok = true;

  for (
    const [key, el]
    of Object.entries(formFields)
  ) {

    if (!validateField(key, el)) {
      ok = false;
    }
  }

  return ok;
}


function getFormData() {

  return {

    name:
      formFields.name.value.trim(),

    register_number:
      formFields.register_number.value.trim(),

    department:
      formFields.department.value.trim(),

    year:
      formFields.year.value.trim(),

    section:
      formFields.section.value.trim()
  };
}


/* ========================================================================== 
   FORM EVENT LISTENERS
   ========================================================================== */

Object.values(formFields).forEach(el => {

  if (!el) return;

  el.addEventListener(
    "input",
    updateCaptureButtonState
  );

  el.addEventListener(
    "change",
    updateCaptureButtonState
  );

});


function updateCaptureButtonState() {

  if (!captureBtn) return;

  const formOk =
    formIsValid();

  const faceOk =
    currentFaceCount === 1;

  captureBtn.disabled =
    !(formOk && faceOk && modelsReady)
    || capturing;
}


/* ========================================================================== 
   FACE-API MODEL LOADING
   ========================================================================== */

async function loadModels() {

  try {

    console.log(
      "Loading Tiny Face Detector..."
    );

    await faceapi.nets.tinyFaceDetector.loadFromUri(
      MODEL_URL
    );

    console.log(
      "Tiny Face Detector loaded."
    );


    console.log(
      "Loading Face Landmark 68..."
    );

    await faceapi.nets.faceLandmark68Net.loadFromUri(
      MODEL_URL
    );

    console.log(
      "Face Landmark 68 loaded."
    );


    modelsReady = true;

    console.log(
      "All face-api models loaded successfully."
    );

  } catch (error) {

    modelsReady = false;

    console.error(
      "FACE API MODEL ERROR:",
      error
    );

    throw error;
  }
}


/* ========================================================================== 
   CAMERA
   ========================================================================== */

async function startCamera() {

  try {

    if (
      !navigator.mediaDevices ||
      !navigator.mediaDevices.getUserMedia
    ) {

      throw new Error(
        "Camera access is not supported. Please use HTTPS or localhost."
      );
    }


    const stream =
      await navigator.mediaDevices.getUserMedia({

        video: {

          facingMode: {
            ideal: "user"
          },

          width: {
            ideal: 640
          },

          height: {
            ideal: 480
          }

        },

        audio: false
      });


    video.srcObject = stream;


    video.onloadedmetadata =
      async () => {

        try {

          await video.play();


          camPlaceholder.style.display =
            "none";

          video.style.display =
            "block";

          overlay.style.display =
            "block";


          overlay.width =
            video.videoWidth;

          overlay.height =
            video.videoHeight;


          setBadge(
            faceStatusBadge,
            "Camera ready",
            "ok"
          );


        } catch (playError) {

          console.error(
            "Video play error:",
            playError
          );

          setBadge(
            faceStatusBadge,
            "Camera unavailable",
            "bad"
          );
        }

      };


  } catch (err) {

    console.error(
      "Camera error:",
      err
    );


    camPlaceholder.style.display =
      "flex";

    video.style.display =
      "none";

    overlay.style.display =
      "none";


    let message =
      "Camera access failed.";


    if (
      err.name === "NotAllowedError" ||
      err.name === "PermissionDeniedError"
    ) {

      message =
        "Camera permission was denied. Please allow camera access and reload the page.";

    } else if (
      err.name === "NotFoundError"
    ) {

      message =
        "No camera was found. Please connect a camera and try again.";

    } else if (
      err.name === "NotReadableError"
    ) {

      message =
        "The camera is already being used by another application.";

    } else if (
      err.name === "SecurityError"
    ) {

      message =
        "Camera access requires HTTPS.";
    }


    camPlaceholder.innerHTML = `
      <strong>${message}</strong>
      <br><br>
      <small>
        Make sure your browser has permission to use the camera.
      </small>
    `;


    setBadge(
      faceStatusBadge,
      "Camera unavailable",
      "bad"
    );
  }
}


/* ========================================================================== 
   LIVENESS
   ========================================================================== */

function eyeAspectRatio(eye) {

  const dist = (a, b) =>
    Math.hypot(
      a.x - b.x,
      a.y - b.y
    );


  const vertical1 =
    dist(eye[1], eye[5]);

  const vertical2 =
    dist(eye[2], eye[4]);

  const horizontal =
    dist(eye[0], eye[3]);


  if (horizontal === 0) {
    return 0;
  }


  return (
    (vertical1 + vertical2) /
    (2 * horizontal)
  );
}


function updateLiveness(
  landmarks,
  boxWidth
) {

  const leftEye =
    landmarks.getLeftEye();

  const rightEye =
    landmarks.getRightEye();

  const nose =
    landmarks.getNose();


  const avgEAR =
    (
      eyeAspectRatio(leftEye) +
      eyeAspectRatio(rightEye)
    ) / 2;


  earHistory.push(avgEAR);


  if (earHistory.length > 15) {
    earHistory.shift();
  }


  // Blink detection
  if (
    !blinkObserved &&
    earHistory.length >= 3
  ) {

    const min =
      Math.min(...earHistory);

    const last =
      earHistory[
      earHistory.length - 1
      ];


    if (
      min < EAR_BLINK_THRESHOLD &&
      last >= EAR_BLINK_THRESHOLD
    ) {

      blinkObserved = true;
    }
  }


  // Head turn detection
  const noseX =
    nose[3].x;


  if (
    baselineNoseX === null
  ) {

    baselineNoseX =
      noseX;
  }


  const normalizedOffset =
    (
      noseX - baselineNoseX
    ) / boxWidth;


  if (
    Math.abs(normalizedOffset) >
    YAW_TURN_THRESHOLD
  ) {

    headTurnObserved = true;
  }


  const parts = [];


  parts.push(
    blinkObserved
      ? "✅ blink detected"
      : "… waiting for a natural blink"
  );


  parts.push(
    headTurnObserved
      ? "✅ head movement detected"
      : "… waiting for slight head turn"
  );


  livenessLine.textContent =
    parts.join("   ");


  const liveOk =
    blinkObserved &&
    headTurnObserved;


  setBadge(
    livenessBadge,

    liveOk
      ? "Liveness: verified"
      : "Liveness: checking…",

    liveOk
      ? "ok"
      : "warn"
  );


  return liveOk;
}


function resetLiveness() {

  blinkObserved = false;

  headTurnObserved = false;

  baselineNoseX = null;

  earHistory = [];


  setBadge(
    livenessBadge,
    "Liveness: waiting",
    "idle"
  );


  livenessLine.textContent =
    "";
}


/* ========================================================================== 
   FACE DETECTION LOOP
   ========================================================================== */

async function detectionTick() {

  if (
    !modelsReady ||
    video.paused ||
    video.ended
  ) {

    detectLoopHandle =
      requestAnimationFrame(
        detectionTick
      );

    return;
  }


  try {

    const displaySize = {

      width:
        video.clientWidth ||
        video.videoWidth,

      height:
        video.clientHeight ||
        video.videoHeight
    };


    if (
      displaySize.width === 0 ||
      displaySize.height === 0
    ) {

      detectLoopHandle =
        requestAnimationFrame(
          detectionTick
        );

      return;
    }


    overlay.width =
      displaySize.width;

    overlay.height =
      displaySize.height;


    const detections =
      await faceapi
        .detectAllFaces(
          video,
          new faceapi.TinyFaceDetectorOptions()
        )
        .withFaceLandmarks();


    const resized =
      faceapi.resizeResults(
        detections,
        displaySize
      );


    const ctx =
      overlay.getContext("2d");


    ctx.clearRect(
      0,
      0,
      overlay.width,
      overlay.height
    );


    currentFaceCount =
      resized.length;


    /* No face */

    if (
      currentFaceCount === 0
    ) {

      setBadge(
        faceStatusBadge,
        "❌ No Face Detected",
        "bad"
      );

      resetLiveness();
    }


    /* Multiple faces */

    else if (
      currentFaceCount > 1
    ) {

      setBadge(
        faceStatusBadge,
        "⚠️ Multiple Faces Detected",
        "warn"
      );


      resized.forEach(
        det => {

          drawBox(
            ctx,
            det.detection.box,
            "#a67c1e"
          );

        }
      );


      resetLiveness();
    }


    /* Exactly one face */

    else {

      setBadge(
        faceStatusBadge,
        "✅ Face Detected",
        "ok"
      );


      const det =
        resized[0];


      drawBox(
        ctx,
        det.detection.box,
        "#2f5233"
      );


      updateLiveness(
        det.landmarks,
        det.detection.box.width
      );
    }


    updateCaptureButtonState();

  } catch (error) {

    console.error(
      "Face detection error:",
      error
    );

  }


  detectLoopHandle =
    requestAnimationFrame(
      detectionTick
    );
}


/* ========================================================================== 
   DRAW FACE BOX
   ========================================================================== */

function drawBox(
  ctx,
  box,
  color
) {

  ctx.strokeStyle =
    color;

  ctx.lineWidth =
    3;

  ctx.strokeRect(
    box.x,
    box.y,
    box.width,
    box.height
  );
}


/* ========================================================================== 
   POSE UI
   ========================================================================== */

function updatePoseChips() {

  poseChips.forEach(
    (chip, idx) => {

      chip.classList.toggle(
        "done",
        idx < poseIndex
      );


      chip.classList.toggle(
        "active",
        idx === poseIndex &&
        capturing
      );

    }
  );
}


/* ========================================================================== 
   PROGRESS
   ========================================================================== */

function updateProgress() {

  const pct =
    Math.min(
      100,
      Math.round(
        (
          captureQueue.length /
          TARGET_FRAMES
        ) * 100
      )
    );


  progressFill.style.width =
    `${pct}%`;


  progressLabel.textContent =
    `${pct}% captured`;


  progressCount.textContent =
    `${captureQueue.length} / ${TARGET_FRAMES} frames`;
}


/* ========================================================================== 
   POSE INSTRUCTIONS
   ========================================================================== */

function poseInstruction(pose) {

  const map = {

    front:
      "Look straight at the camera",

    left:
      "Slowly turn your head to the LEFT",

    right:
      "Slowly turn your head to the RIGHT",

    up:
      "Tilt your chin slightly UP",

    down:
      "Tilt your chin slightly DOWN"
  };


  return (
    map[pose] ||
    "Hold still"
  );
}


/* ========================================================================== 
   CAPTURE IMAGE
   ========================================================================== */

async function captureFrameBlob() {

  const canvas =
    document.createElement("canvas");


  canvas.width =
    video.videoWidth;

  canvas.height =
    video.videoHeight;


  const ctx =
    canvas.getContext(
      "2d",
      {
        willReadFrequently: true
      }
    );


  ctx.drawImage(
    video,
    0,
    0,
    canvas.width,
    canvas.height
  );


  return new Promise(
    resolve => {

      canvas.toBlob(
        resolve,
        "image/jpeg",
        0.9
      );

    }
  );
}


/* ========================================================================== 
   CAPTURE SEQUENCE
   ========================================================================== */

async function runCaptureSequence() {

  capturing = true;

  captureBtn.disabled = true;

  captureBtn.textContent =
    "Capturing…";


  captureQueue = [];

  poseIndex = 0;

  framesForCurrentPose = 0;


  updateProgress();


  try {

    for (
      poseIndex = 0;
      poseIndex < POSES.length;
      poseIndex++
    ) {

      const pose =
        POSES[poseIndex];


      updatePoseChips();


      showToast(
        poseInstruction(pose)
      );


      framesForCurrentPose = 0;


      await sleep(900);


      while (
        framesForCurrentPose <
        FRAMES_PER_POSE &&
        captureQueue.length <
        TARGET_FRAMES
      ) {

        if (
          currentFaceCount !== 1
        ) {

          await sleep(300);

          continue;
        }


        const blob =
          await captureFrameBlob();


        if (!blob) {

          throw new Error(
            "Could not capture camera frame."
          );
        }


        captureQueue.push({

          blob,

          pose
        });


        framesForCurrentPose++;


        updateProgress();


        await sleep(180);
      }
    }


    poseIndex =
      POSES.length;


    updatePoseChips();


    await submitEnrollment();

  } catch (error) {

    console.error(
      "Capture error:",
      error
    );


    showToast(
      "Capture failed. Please try again.",
      "error"
    );


    resetCaptureOnly();
  }
}


function sleep(ms) {

  return new Promise(
    resolve =>
      setTimeout(
        resolve,
        ms
      )
  );
}


/* ========================================================================== 
   BASE64
   ========================================================================== */

function blobToBase64(blob) {

  return new Promise(
    (resolve, reject) => {

      const reader =
        new FileReader();


      reader.onloadend =
        () => resolve(
          reader.result
        );


      reader.onerror =
        reject;


      reader.readAsDataURL(
        blob
      );
    }
  );
}


/* ========================================================================== 
   BACKEND HEALTH CHECK
   ========================================================================== */

async function checkBackendHealth() {

  try {

    const response =
      await fetch(
        `${API_BASE}/api/health`,
        {
          method: "GET"
        }
      );


    if (!response.ok) {

      throw new Error(
        `Backend returned ${response.status}`
      );
    }


    const data =
      await response.json();


    console.log(
      "Backend health:",
      data
    );


    return true;

  } catch (error) {

    console.error(
      "Backend health check failed:",
      error
    );


    return false;
  }
}


/* ========================================================================== 
   DUPLICATE CHECK
   ========================================================================== */

async function checkDuplicateEarly() {

  try {

    const backendOk =
      await checkBackendHealth();


    if (!backendOk) {

      showToast(
        "Backend server is unavailable. Please check Render.",
        "error"
      );

      return true;
    }


    const blob =
      await captureFrameBlob();


    if (!blob) {

      showToast(
        "Could not capture camera image.",
        "error"
      );

      return true;
    }


    const b64 =
      await blobToBase64(blob);


    const response =
      await fetch(
        `${API_BASE}/api/enroll/check-duplicate`,
        {

          method: "POST",

          headers: {
            "Content-Type":
              "application/json"
          },

          body:
            JSON.stringify({
              image: b64
            })
        }
      );


    if (!response.ok) {

      const text =
        await response.text();


      console.error(
        "Duplicate check failed:",
        response.status,
        text
      );


      showToast(
        `Backend duplicate check failed (${response.status}).`,
        "error"
      );


      return true;
    }


    const data =
      await response.json();


    if (data.duplicate) {

      showToast(

        `This face looks already enrolled as ` +
        `${data.matched_name} ` +
        `(${data.matched_register_number}).`,

        "error"
      );


      return true;
    }


    return false;


  } catch (error) {

    console.error(
      "Duplicate pre-check failed:",
      error
    );


    showToast(
      "Could not contact the backend server.",
      "error"
    );


    return true;
  }
}


/* ========================================================================== 
   SUBMIT ENROLLMENT
   ========================================================================== */

async function submitEnrollment() {

  const formData =
    new FormData();


  const details =
    getFormData();


  Object.entries(
    details
  ).forEach(
    ([key, value]) => {

      formData.append(
        key,
        value
      );

    }
  );


  captureQueue.forEach(
    (item, index) => {

      formData.append(
        "images",
        item.blob,
        `${item.pose}-${index + 1}.jpg`
      );


      formData.append(
        "poses",
        item.pose
      );

    }
  );


  try {

    showToast(
      "Uploading face data…"
    );


    const response =
      await fetch(
        `${API_BASE}/api/enroll`,
        {

          method: "POST",

          body: formData
        }
      );


    if (!response.ok) {

      const text =
        await response.text();


      console.error(
        "Enrollment backend error:",
        response.status,
        text
      );


      throw new Error(
        `Server returned ${response.status}`
      );
    }


    const data =
      await response.json();


    if (data.success) {

      showToast(
        data.message ||
        "Enrollment successful!",
        "success"
      );


      resetAll();

    } else {

      showToast(
        data.message ||
        "Enrollment failed.",
        "error"
      );


      resetCaptureOnly();
    }


  } catch (error) {

    console.error(
      "Enrollment error:",
      error
    );


    showToast(
      "Could not reach the enrollment server.",
      "error"
    );


    resetCaptureOnly();
  }
}


/* ========================================================================== 
   RESET
   ========================================================================== */

function resetCaptureOnly() {

  capturing = false;

  captureQueue = [];

  poseIndex = 0;

  framesForCurrentPose = 0;


  captureBtn.textContent =
    "Capture & Enroll";


  updateProgress();

  updatePoseChips();

  updateCaptureButtonState();
}


function resetAll() {

  resetCaptureOnly();


  Object.values(
    formFields
  ).forEach(
    el => {

      if (el) {
        el.value = "";
      }

    }
  );


  document
    .querySelectorAll(".field")
    .forEach(
      field => {

        field.classList.remove(
          "valid",
          "invalid"
        );

      }
    );


  resetLiveness();
}


/* ========================================================================== 
   BUTTON EVENTS
   ========================================================================== */

captureBtn.addEventListener(
  "click",
  async () => {

    if (capturing) {
      return;
    }


    const formOk =
      formIsValid();


    if (!formOk) {

      showToast(
        "Please complete all student details.",
        "error"
      );

      return;
    }


    if (
      currentFaceCount !== 1
    ) {

      showToast(
        "Exactly one face must be visible.",
        "error"
      );

      return;
    }


    if (!modelsReady) {

      showToast(
        "Face models are not ready yet.",
        "error"
      );

      return;
    }


    const duplicate =
      await checkDuplicateEarly();


    if (duplicate) {
      return;
    }


    await runCaptureSequence();
  }
);


resetBtn.addEventListener(
  "click",
  resetAll
);


/* ========================================================================== 
   INITIALIZATION
   ========================================================================== */

(async function init() {

  if (!checkRequiredElements()) {

    console.error(
      "Page initialization stopped because required HTML elements are missing."
    );

    return;
  }


  setBadge(
    faceStatusBadge,
    "Loading face models…",
    "idle"
  );


  try {

    await loadModels();

  } catch (error) {

    console.error(
      "Failed to load face-api models:",
      error
    );


    setBadge(
      faceStatusBadge,
      "Model load failed — check connection",
      "bad"
    );


    showToast(
      "Face models could not be loaded. Check the browser console.",
      "error"
    );


    return;
  }


  await startCamera();


  setBadge(
    faceStatusBadge,
    "Looking for a face…",
    "idle"
  );


  updateProgress();

  updatePoseChips();

  updateCaptureButtonState();


  detectLoopHandle =
    requestAnimationFrame(
      detectionTick
    );

})();

