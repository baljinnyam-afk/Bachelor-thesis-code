/**
 * app.js — Spacecraft Attitude Estimation Demo
 *
 * Loads a TensorFlow.js model, runs quaternion regression on
 * synthetically rendered spacecraft images, and compares predictions
 * against ground-truth labels.
 */

// ── Constants ────────────────────────────────────────────────
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];
const IMG_SIZE      = 224;
const EULER_LABELS  = ["Roll", "Pitch", "Yaw"];

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
    const predQuat = await prediction.data();
    inputTensor.dispose();
    prediction.dispose();

    showResults(item.gt, Array.from(predQuat));
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

      // Convert to float32 tensor [224, 224, 3], normalize to [0, 1]
      const float32 = new Float32Array(IMG_SIZE * IMG_SIZE * 3);
      for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
        float32[i * 3]     = pixels[i * 4]     / 255.0; // R
        float32[i * 3 + 1] = pixels[i * 4 + 1] / 255.0; // G
        float32[i * 3 + 2] = pixels[i * 4 + 2] / 255.0; // B
      }

      let tensor = tf.tensor3d(float32, [IMG_SIZE, IMG_SIZE, 3]);

      // ImageNet normalization
      const mean = tf.tensor1d(IMAGENET_MEAN);
      const std  = tf.tensor1d(IMAGENET_STD);
      tensor = tensor.sub(mean).div(std);

      // Expand dims to [1, 224, 224, 3]
      const batched = tensor.expandDims(0);

      mean.dispose();
      std.dispose();
      tensor.dispose();

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

function showResults(gt, pred) {
  quatTbody.innerHTML = "";

  pred = enforceCanonical(pred);
  const gtArr = [gt.qw, gt.qx, gt.qy, gt.qz];
  if (gtArr[0] < 0) gtArr.forEach((v, i) => gtArr[i] = -v);

  const gtEuler   = quatToEulerDeg(gtArr);
  const predEuler = quatToEulerDeg(pred);

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

  const angErr = angularErrorDeg(gtArr, pred);
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
