/**
 * app.js — Spacecraft Attitude Estimation Demo
 *
 * Loads a TensorFlow.js model, runs quaternion regression on
 * synthetically rendered spacecraft images, and compares predictions
 * against ground-truth labels.
 */

// ── Constants ────────────────────────────────────────────────
const IMG_SIZE      = 224;
const N_BINS        = 16;
const EULER_LABELS  = ["Roll", "Pitch", "Yaw"];
const BIN_CENTERS_DEG = [];
for (let i = 0; i < N_BINS; i++) BIN_CENTERS_DEG.push(-180 + (360 / N_BINS) * (i + 0.5));

/**
 * Convert unit quaternion [qw, qx, qy, qz] to Euler angles in degrees [roll, pitch, yaw].
 * Uses ZYX (intrinsic) convention — matches the Python quat_to_euler_deg in the notebook.
 */
function quatToEulerDeg(q) {
  const [qw, qx, qy, qz] = q;

  // Roll (x-axis rotation)
  const sinrCosp = 2 * (qw * qx + qy * qz);
  const cosrCosp = 1 - 2 * (qx * qx + qy * qy);
  const roll = Math.atan2(sinrCosp, cosrCosp);

  // Pitch (y-axis rotation) — clamped to avoid NaN at gimbal lock
  const sinp = 2 * (qw * qy - qz * qx);
  const pitch = Math.asin(Math.max(-1, Math.min(1, sinp)));

  // Yaw (z-axis rotation)
  const sinyCosp = 2 * (qw * qz + qx * qy);
  const cosyCosp = 1 - 2 * (qy * qy + qz * qz);
  const yaw = Math.atan2(sinyCosp, cosyCosp);

  return [roll * (180 / Math.PI), pitch * (180 / Math.PI), yaw * (180 / Math.PI)];
}

/**
 * Convert UrsoNet softmax outputs to Euler angles in degrees.
 * Uses circular mean (atan2-based) to handle ±180° wraparound correctly.
 * Matches the Python bins_to_quaternion → circular_mean_rad logic.
 */
function softmaxToEuler(probs) {
  const euler = [];
  const centersRad = BIN_CENTERS_DEG.map(d => d * Math.PI / 180);

  for (let ax = 0; ax < 3; ax++) {
    let sinSum = 0, cosSum = 0;
    const start = ax * N_BINS;
    for (let b = 0; b < N_BINS; b++) {
      const p = probs[start + b];
      sinSum += p * Math.sin(centersRad[b]);
      cosSum += p * Math.cos(centersRad[b]);
    }
    euler.push(Math.atan2(sinSum, cosSum) * 180 / Math.PI);
  }
  return euler;
}

function eulerToQuat(rollDeg, pitchDeg, yawDeg) {
  const r = rollDeg * (Math.PI / 360);
  const p = pitchDeg * (Math.PI / 360);
  const y = yawDeg * (Math.PI / 360);
  const cr = Math.cos(r), sr = Math.sin(r);
  const cp = Math.cos(p), sp = Math.sin(p);
  const cy = Math.cos(y), sy = Math.sin(y);
  const qw = cr * cp * cy + sr * sp * sy;
  const qx = sr * cp * cy - cr * sp * sy;
  const qy = cr * sp * cy + sr * cp * sy;
  const qz = cr * cp * sy - sr * sp * cy;
  const norm = Math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
  return [qw/norm, qx/norm, qy/norm, qz/norm];
}

// ── Register custom Keras objects for TF.js ─────────────────
// The exported model may contain regularizers/initializers that
// TF.js doesn't recognize by default. Register them here.

class L2 {
  constructor(config) { this.l2 = config?.l2 ?? 0.01; }
  apply(x) { return tf.tidy(() => tf.mul(this.l2, tf.sum(tf.square(x)))); }
  getConfig() { return { l2: this.l2, __class__: "L2" }; }
  static get className() { return "L2"; }
}
tf.serialization.registerClass(L2);

class L1 {
  constructor(config) { this.l1 = config?.l1 ?? 0.01; }
  apply(x) { return tf.tidy(() => tf.mul(this.l1, tf.sum(tf.abs(x)))); }
  getConfig() { return { l1: this.l1, __class__: "L1" }; }
  static get className() { return "L1"; }
}
tf.serialization.registerClass(L1);

class L1L2 {
  constructor(config) {
    this.l1 = config?.l1 ?? 0.01;
    this.l2 = config?.l2 ?? 0.01;
  }
  apply(x) { return tf.tidy(() => {
    const l1Term = tf.mul(this.l1, tf.sum(tf.abs(x)));
    const l2Term = tf.mul(this.l2, tf.sum(tf.square(x)));
    return tf.add(l1Term, l2Term);
  }); }
  getConfig() { return { l1: this.l1, l2: this.l2, __class__: "L1L2" }; }
  static get className() { return "L1L2"; }
}
tf.serialization.registerClass(L1L2);

// ── State ────────────────────────────────────────────────────
let model       = null;
let demoData    = null;
let selectedItem = null;

// ── DOM refs ─────────────────────────────────────────────────
const modelUrlInput     = document.getElementById("model-url");
const loadModelBtn      = document.getElementById("load-model-btn");
const modelStatusDiv    = document.getElementById("model-status");
const modelStatusText   = document.getElementById("model-status-text");
const modelProgressFill = document.getElementById("model-progress-fill");
const galleryDiv        = document.getElementById("image-gallery");
const resultsSection    = document.getElementById("results-section");
const previewCanvas     = document.getElementById("preview-canvas");
const imageMetaDiv      = document.getElementById("image-meta");
const quatTbody         = document.getElementById("quat-tbody");
const angularErrorBox   = document.getElementById("angular-error-box");
const angularErrorValue = document.getElementById("angular-error-value");
const runInferenceBtn   = document.getElementById("run-inference-btn");
const runInferenceWrap  = document.getElementById("run-inference-btn-wrapper");
const preprocessCanvas  = document.getElementById("preprocess-canvas");

// ── Init ─────────────────────────────────────────────────────
(async function init() {
  demoData = await loadDemoData();
  renderGallery(demoData.images);
  bindEvents();
})();

// ── Data Loading ─────────────────────────────────────────────
async function loadDemoData() {
  const resp = await fetch("demo_data.json");
  if (!resp.ok) throw new Error("Failed to load demo_data.json");
  return resp.json();
}

// ── Gallery ──────────────────────────────────────────────────
function renderGallery(images) {
  galleryDiv.innerHTML = "";
  images.forEach((item, idx) => {
    const div = document.createElement("div");
    div.className = "gallery-item";
    div.dataset.index = idx;

    const img = document.createElement("img");
    img.src = `images/${item.file}`;
    img.alt = item.file;
    img.loading = "lazy";

    const badge = document.createElement("span");
    badge.className = "category-badge";
    badge.textContent = item.category;

    div.appendChild(img);
    div.appendChild(badge);
    galleryDiv.appendChild(div);
  });
}

// ── Event Binding ────────────────────────────────────────────
function bindEvents() {
  loadModelBtn.addEventListener("click", onLoadModel);
  galleryDiv.addEventListener("click", onGalleryClick);
  runInferenceBtn.addEventListener("click", onRunInference);
}

// ── Model Loading ────────────────────────────────────────────
async function onLoadModel() {
  const url = modelUrlInput.value.trim();
  if (!url) {
    setStatus("error", "Please enter a model.json URL.");
    return;
  }

  loadModelBtn.disabled = true;
  setStatus("loading", "Loading model...");
  setProgress(10);

  try {
    model = await tf.loadLayersModel(url, {
      onProgress: (fraction) => {
        setProgress(Math.round(fraction * 100));
        setStatus("loading", `Loading model... ${Math.round(fraction * 100)}%`);
      }
    });

    setProgress(100);
    setStatus("success", `Model loaded successfully (${model.params?.toLocaleString?.() ?? "?"} params)`);
    loadModelBtn.textContent = "Model Loaded";

    // If an image is already selected, show the run button
    if (selectedItem !== null) {
      runInferenceWrap.classList.remove("hidden");
    }
  } catch (err) {
    setStatus("error", `Failed to load model: ${err.message}`);
    loadModelBtn.disabled = false;
  }
}

// ── Gallery Click ────────────────────────────────────────────
function onGalleryClick(e) {
  const item = e.target.closest(".gallery-item");
  if (!item) return;

  // Deselect previous
  const prev = galleryDiv.querySelector(".selected");
  if (prev) prev.classList.remove("selected");

  item.classList.add("selected");

  const idx = parseInt(item.dataset.index, 10);
  selectedItem = demoData.images[idx];

  showSelectedImage(selectedItem);

  // If model is loaded, auto-run inference
  if (model) {
    runInference(selectedItem);
  } else {
    showGroundTruthOnly(selectedItem);
    runInferenceWrap.classList.remove("hidden");
  }
}

// ── Run Inference ────────────────────────────────────────────
function onRunInference() {
  if (!model || selectedItem === null) return;
  runInference(selectedItem);
}

async function runInference(item) {
  runInferenceBtn.disabled = true;
  runInferenceBtn.textContent = "Running...";

  try {
    const inputTensor = await preprocessImage(`images/${item.file}`);
    const prediction = model.predict(inputTensor);
    const raw = Array.from(await prediction.data());
    inputTensor.dispose();
    prediction.dispose();

    if (raw.length === 48) {
      const predEuler = softmaxToEuler(raw);
      const predQuat = eulerToQuat(predEuler[0], predEuler[1], predEuler[2]);
      const debug = `Output: ${raw.length} values | ` +
        `Roll softmax: [${raw.slice(0,16).map(v=>v.toFixed(3)).join(', ')}] | ` +
        `Euler: R=${predEuler[0].toFixed(1)}° P=${predEuler[1].toFixed(1)}° Y=${predEuler[2].toFixed(1)}°`;
      setStatus("success", debug);
      showResults(item.gt, predEuler, predQuat);
    } else if (raw.length === 4) {
      const canonical = enforceCanonical(raw);
      const predEuler = quatToEulerDeg(canonical);
      setStatus("success", `Output: ${raw.length} values | Quat: [${raw.map(v=>v.toFixed(4)).join(', ')}]`);
      showResults(item.gt, predEuler, canonical);
    } else {
      setStatus("error", `Unexpected model output size: ${raw.length}`);
    }
  } catch (err) {
    setStatus("error", `Inference failed: ${err.message}`);
  } finally {
    runInferenceBtn.disabled = false;
    runInferenceBtn.textContent = "Run Inference";
  }
}

// ── Image Preprocessing ─────────────────────────────────────
async function preprocessImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const ctx = preprocessCanvas.getContext("2d");
      ctx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
      ctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);

      const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
      const pixels = imageData.data; // Uint8ClampedArray [r,g,b,a, ...]

      // ResNet50V2 preprocessing: [0,255] → [-1,1]
      const float32 = new Float32Array(IMG_SIZE * IMG_SIZE * 3);
      for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
        float32[i * 3]     = (pixels[i * 4]     / 127.5) - 1.0;
        float32[i * 3 + 1] = (pixels[i * 4 + 1] / 127.5) - 1.0;
        float32[i * 3 + 2] = (pixels[i * 4 + 2] / 127.5) - 1.0;
      }

      const batched = tf.tensor3d(float32, [IMG_SIZE, IMG_SIZE, 3]).expandDims(0);

      resolve(batched);
    };
    img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
    img.src = src;
  });
}

// ── Display ──────────────────────────────────────────────────
function showSelectedImage(item) {
  resultsSection.classList.remove("hidden");

  // Draw on preview canvas
  const img = new Image();
  img.onload = () => {
    const ctx = previewCanvas.getContext("2d");
    ctx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
    ctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);
  };
  img.src = `images/${item.file}`;

  const gt = item.gt;
  imageMetaDiv.textContent =
    `${item.file}  |  Category: ${item.category}  |  Sun angle: ${item.sun_angle}°`;
}

function showGroundTruthOnly(item) {
  const gt = item.gt;
  const gtQuat = [gt.qw, gt.qx, gt.qy, gt.qz];
  if (gtQuat[0] < 0) gtQuat.forEach((v, i) => gtQuat[i] = -v);
  const gtEuler = quatToEulerDeg(gtQuat);

  quatTbody.innerHTML = "";

  EULER_LABELS.forEach((label, i) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${label}</td>
      <td>${gtEuler[i].toFixed(2)}°</td>
      <td class="text-muted">—</td>
      <td class="text-muted">—</td>
    `;
    quatTbody.appendChild(row);
  });

  angularErrorBox.classList.add("hidden");
}

function showResults(gt, predEuler, predQuat) {
  quatTbody.innerHTML = "";

  const gtArr = [gt.qw, gt.qx, gt.qy, gt.qz];
  if (gtArr[0] < 0) gtArr.forEach((v, i) => gtArr[i] = -v);
  const gtEuler = quatToEulerDeg(gtArr);

  EULER_LABELS.forEach((label, i) => {
    const gtVal   = gtEuler[i];
    const predVal = predEuler[i];
    const err     = Math.abs(gtVal - predVal);

    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${label}</td>
      <td>${gtVal.toFixed(2)}°</td>
      <td>${predVal.toFixed(2)}°</td>
      <td class="error-cell ${eulerErrClass(err)}">${err.toFixed(2)}°</td>
    `;
    quatTbody.appendChild(row);
  });

  const angErr = angularErrorDeg(gtArr, predQuat);
  angularErrorBox.classList.remove("hidden");
  angularErrorValue.textContent = `${angErr.toFixed(2)}°`;
  angularErrorValue.className = `error-value ${angErrClass(angErr)}`;
}

// ── Math Helpers ─────────────────────────────────────────────

/**
 * Enforce canonical quaternion form: qw >= 0.
 * Handles the double-cover property of quaternions.
 */
function enforceCanonical(q) {
  if (q[0] < 0) return q.map(v => -v);
  return q;
}

/**
 * Angular error between two unit quaternions in degrees.
 * Uses: angle = 2 * arccos(|q1 · q2|)
 */
function angularErrorDeg(gt, pred) {
  const dot = gt[0]*pred[0] + gt[1]*pred[1] + gt[2]*pred[2] + gt[3]*pred[3];
  const clamped = Math.min(Math.abs(dot), 1.0);
  return 2 * Math.acos(clamped) * (180 / Math.PI);
}

function eulerErrClass(err) {
  if (err < 5)  return "good";
  if (err > 15) return "bad";
  return "";
}

function angErrClass(deg) {
  if (deg < 5)  return "";
  if (deg < 15) return "moderate";
  return "poor";
}

// ── Status / Progress ────────────────────────────────────────
function setStatus(type, text) {
  modelStatusDiv.classList.remove("hidden", "loading", "success", "error");
  modelStatusDiv.classList.add(type);
  modelStatusText.textContent = text;
}

function setProgress(pct) {
  modelProgressFill.style.width = `${pct}%`;
}
