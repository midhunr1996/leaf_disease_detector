/* =============================================================
   Leaf Health Detector — in-browser YOLOv12 inference
   Runs entirely on the visitor's device via ONNX Runtime Web.
   ============================================================= */

/* ---------- CONFIG — edit these after training ---------- */
// Class names IN THE SAME ORDER as your dataset's data.yaml.
// Read directly from the trained model metadata: {0: 'Nondefect', 1: 'defect'}
const CLASSES = ["Nondefect", "defect"];

const MODEL_URL = "model/best.onnx";
const INPUT_SIZE = 640;        // must match the imgsz used when exporting
let CONF_THRESHOLD = 0.35;     // controlled by the slider
const IOU_THRESHOLD = 0.45;    // NMS overlap threshold

// Colours per class (falls back to palette if a name isn't matched)
const CLASS_COLORS = {
  good: "#34d399", healthy: "#34d399", nondefect: "#34d399",   // green = healthy
  bad: "#f87171", diseased: "#f87171", defect: "#f87171",       // red = defective
};
const PALETTE = ["#34d399", "#f87171", "#60a5fa", "#fbbf24", "#c084fc", "#f472b6"];
/* -------------------------------------------------------- */

let session = null;
let inputName = "images";

const statusEl = document.getElementById("status");
const countsEl = document.getElementById("counts");
const fpsEl = document.getElementById("fps");

/* ---------------- Model loading ---------------- */
async function loadModel() {
  try {
    // Point ORT at the CDN so the WASM binaries resolve correctly.
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
    ort.env.wasm.numThreads = 1; // safest across browsers (no cross-origin isolation needed)

    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    inputName = session.inputNames[0];
    setStatus("ready", "✅ Model ready — pick an image or start the camera.");
  } catch (err) {
    console.error(err);
    setStatus("error",
      "❌ Could not load the model. Make sure <b>model/best.onnx</b> exists in the repo. (" + err.message + ")");
  }
}

function setStatus(kind, html) {
  statusEl.className = "status " + kind;
  statusEl.innerHTML = html;
}

/* ---------------- Preprocess (letterbox) ---------------- */
// Draws the source into a 640x640 canvas keeping aspect ratio, padded
// with grey (114). Returns the tensor + the scale/pad needed to map
// boxes back to the original image coordinates.
const letterboxCanvas = document.createElement("canvas");
letterboxCanvas.width = INPUT_SIZE;
letterboxCanvas.height = INPUT_SIZE;
const lbCtx = letterboxCanvas.getContext("2d", { willReadFrequently: true });

function preprocess(source, srcW, srcH) {
  const scale = Math.min(INPUT_SIZE / srcW, INPUT_SIZE / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = (INPUT_SIZE - newW) / 2;
  const padY = (INPUT_SIZE - newH) / 2;

  lbCtx.fillStyle = "rgb(114,114,114)";
  lbCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  lbCtx.drawImage(source, padX, padY, newW, newH);

  const { data } = lbCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const area = INPUT_SIZE * INPUT_SIZE;
  const floatData = new Float32Array(area * 3); // CHW, RGB, /255

  for (let i = 0; i < area; i++) {
    floatData[i]              = data[i * 4]     / 255; // R
    floatData[i + area]       = data[i * 4 + 1] / 255; // G
    floatData[i + area * 2]   = data[i * 4 + 2] / 255; // B
  }

  const tensor = new ort.Tensor("float32", floatData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  return { tensor, scale, padX, padY };
}

/* ---------------- Postprocess ---------------- */
// YOLOv12 ONNX output is [1, 4+nc, numAnchors]: rows 0-3 = cx,cy,w,h
// (in 640-space), remaining rows = per-class scores.
function postprocess(output, scale, padX, padY) {
  const dims = output.dims;          // [1, C, N]
  const data = output.data;
  const channels = dims[1];
  const numAnchors = dims[2];
  const numClasses = channels - 4;

  const boxes = [];
  for (let a = 0; a < numAnchors; a++) {
    // find best class for this anchor
    let bestScore = 0, bestClass = -1;
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numAnchors + a];
      if (score > bestScore) { bestScore = score; bestClass = c; }
    }
    if (bestScore < CONF_THRESHOLD) continue;

    const cx = data[a];
    const cy = data[numAnchors + a];
    const w  = data[2 * numAnchors + a];
    const h  = data[3 * numAnchors + a];

    // 640-space xywh -> original-image xyxy (undo letterbox)
    const x1 = (cx - w / 2 - padX) / scale;
    const y1 = (cy - h / 2 - padY) / scale;
    const x2 = (cx + w / 2 - padX) / scale;
    const y2 = (cy + h / 2 - padY) / scale;

    boxes.push({ x1, y1, x2, y2, score: bestScore, cls: bestClass });
  }
  return nms(boxes, IOU_THRESHOLD);
}

function iou(a, b) {
  const ix1 = Math.max(a.x1, b.x1), iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2), iy2 = Math.min(a.y2, b.y2);
  const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-6);
}

// Class-aware non-maximum suppression.
function nms(boxes, iouThr) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  const removed = new Array(boxes.length).fill(false);
  for (let i = 0; i < boxes.length; i++) {
    if (removed[i]) continue;
    keep.push(boxes[i]);
    for (let j = i + 1; j < boxes.length; j++) {
      if (removed[j]) continue;
      if (boxes[j].cls === boxes[i].cls && iou(boxes[i], boxes[j]) > iouThr) {
        removed[j] = true;
      }
    }
  }
  return keep;
}

/* ---------------- Full detect on a drawable source ---------------- */
async function detect(source, srcW, srcH) {
  const { tensor, scale, padX, padY } = preprocess(source, srcW, srcH);
  const results = await session.run({ [inputName]: tensor });
  const output = results[session.outputNames[0]];
  return postprocess(output, scale, padX, padY);
}

/* ---------------- Drawing ---------------- */
function colorFor(cls) {
  const name = (CLASSES[cls] || "").toLowerCase();
  return CLASS_COLORS[name] || PALETTE[cls % PALETTE.length];
}

function drawDetections(ctx, boxes, canvasW) {
  const lineW = Math.max(2, Math.round(canvasW / 320));
  const fontSize = Math.max(14, Math.round(canvasW / 40));
  ctx.lineWidth = lineW;
  ctx.font = `600 ${fontSize}px system-ui, sans-serif`;
  ctx.textBaseline = "top";

  for (const b of boxes) {
    const label = `${CLASSES[b.cls] ?? b.cls} ${(b.score * 100).toFixed(0)}%`;
    const color = colorFor(b.cls);
    const w = b.x2 - b.x1, h = b.y2 - b.y1;

    ctx.strokeStyle = color;
    ctx.strokeRect(b.x1, b.y1, w, h);

    const tw = ctx.measureText(label).width;
    const th = fontSize + 6;
    ctx.fillStyle = color;
    ctx.fillRect(b.x1 - lineW / 2, b.y1 - th, tw + 10, th);
    ctx.fillStyle = "#06231a";
    ctx.fillText(label, b.x1 + 5 - lineW / 2, b.y1 - th + 3);
  }
}

function updateCounts(boxes) {
  const tally = {};
  for (const b of boxes) {
    const name = CLASSES[b.cls] ?? String(b.cls);
    tally[name] = (tally[name] || 0) + 1;
  }
  const entries = Object.entries(tally);
  countsEl.innerHTML = entries.length
    ? entries.map(([n, c]) =>
        `<span class="count-pill" style="border-color:${colorFor(CLASSES.indexOf(n))}">${n}: ${c}</span>`).join("")
    : `<span class="count-pill">No leaves detected</span>`;
}

/* =========================================================
   UPLOAD MODE
   ========================================================= */
const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const uploadCanvas = document.getElementById("uploadCanvas");
const uCtx = uploadCanvas.getContext("2d");
const downloadBtn = document.getElementById("downloadBtn");

async function runOnImageFile(file) {
  if (!session) { alert("Model is still loading, please wait a moment."); return; }
  const img = new Image();
  img.onload = async () => {
    uploadCanvas.width = img.naturalWidth;
    uploadCanvas.height = img.naturalHeight;
    uCtx.drawImage(img, 0, 0);
    setStatus("ready", "🔎 Detecting…");
    const boxes = await detect(img, img.naturalWidth, img.naturalHeight);
    drawDetections(uCtx, boxes, uploadCanvas.width);
    updateCounts(boxes);
    downloadBtn.classList.remove("hidden");
    setStatus("ready", `✅ Found ${boxes.length} leaf region(s).`);
    URL.revokeObjectURL(img.src);
  };
  img.src = URL.createObjectURL(file);
}

fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) runOnImageFile(e.target.files[0]);
});
["dragover", "dragenter"].forEach(ev =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach(ev =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) runOnImageFile(file);
});
downloadBtn.addEventListener("click", () => {
  const link = document.createElement("a");
  link.download = "leaf-detection.png";
  link.href = uploadCanvas.toDataURL("image/png");
  link.click();
});

/* =========================================================
   WEBCAM MODE
   ========================================================= */
const video = document.getElementById("video");
const webcamCanvas = document.getElementById("webcamCanvas");
const wCtx = webcamCanvas.getContext("2d");
const startCam = document.getElementById("startCam");
const stopCam = document.getElementById("stopCam");
const switchCam = document.getElementById("switchCam");

let stream = null;
let rafId = null;
let facingMode = "environment"; // prefer rear camera on phones
let lastTime = 0;

async function startCamera() {
  if (!session) { alert("Model is still loading, please wait a moment."); return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    webcamCanvas.width = video.videoWidth;
    webcamCanvas.height = video.videoHeight;

    startCam.classList.add("hidden");
    stopCam.classList.remove("hidden");
    switchCam.classList.remove("hidden");
    loopWebcam();
  } catch (err) {
    setStatus("error", "❌ Camera access denied or unavailable. (" + err.message + ")");
  }
}

function stopCamera() {
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  startCam.classList.remove("hidden");
  stopCam.classList.add("hidden");
  switchCam.classList.add("hidden");
  fpsEl.textContent = "";
}

async function loopWebcam() {
  if (!stream) return;
  wCtx.drawImage(video, 0, 0, webcamCanvas.width, webcamCanvas.height);
  const boxes = await detect(video, video.videoWidth, video.videoHeight);
  drawDetections(wCtx, boxes, webcamCanvas.width);
  updateCounts(boxes);

  const now = performance.now();
  if (lastTime) fpsEl.textContent = (1000 / (now - lastTime)).toFixed(1) + " FPS";
  lastTime = now;

  rafId = requestAnimationFrame(loopWebcam);
}

startCam.addEventListener("click", startCamera);
stopCam.addEventListener("click", stopCamera);
switchCam.addEventListener("click", async () => {
  facingMode = facingMode === "environment" ? "user" : "environment";
  stopCamera();
  await startCamera();
});

/* =========================================================
   TABS + SLIDER
   ========================================================= */
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("panel-" + tab.dataset.tab).classList.add("active");
    if (tab.dataset.tab !== "webcam" && stream) stopCamera();
  });
});

const confSlider = document.getElementById("conf");
const confVal = document.getElementById("confVal");
confSlider.addEventListener("input", () => {
  CONF_THRESHOLD = parseFloat(confSlider.value);
  confVal.textContent = CONF_THRESHOLD.toFixed(2);
});

/* ---------------- Go ---------------- */
loadModel();
