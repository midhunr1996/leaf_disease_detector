/* Tea Leaf Inspection - industrial web interface.
   Detection runs in the browser with ONNX Runtime Web.
   The same page is served by the Raspberry Pi beside the line. */

const CLASSES = ["Nondefect", "defect"];
const COLOR   = { Nondefect: "#3fb950", defect: "#f85149" };
const INPUT   = 640;

const S = {
  session: null, running: false, estop: false, recording: false,
  conf: 0.35, iou: 0.45, motor: false, speed: 0.10,
  // How long a defect must stay in view before the motor is stopped. Counted
  // from the first frame it appears in and reset the moment it clears.
  holdMs: 5000, defectSince: 0, autoStopped: false,
  device: "pi5-line-01", maxRec: 2000,
  counts: { insp: 0, ok: 0, bad: 0 },
  t0: Date.now(), last: 0, stream: null, raf: null,
  backend: "", busy: false, boxes: [], inferMs: 0, lastFrame: 0, viewFps: 0,
  track: null, camDefaults: null, preset: 1,
};

// The C922 Pro Stream is specified for 1080p at 30 frames per second and
// 720p at 60. The lower modes are kept for slower machines and for the
// Raspberry Pi, where moving fewer pixels per frame matters more than detail.
const CAM_PRESETS = [
  { label: "1920 x 1080 @ 30  (most detail)", w: 1920, h: 1080, fps: 30 },
  { label: "1280 x 720 @ 60  (least blur)",   w: 1280, h: 720,  fps: 60 },
  { label: "1280 x 720 @ 30",                 w: 1280, h: 720,  fps: 30 },
  { label: "848 x 480 @ 30",                  w: 848,  h: 480,  fps: 30 },
  { label: "640 x 480 @ 30  (lightest)",      w: 640,  h: 480,  fps: 30 },
];

// Order matters here: the mode switches come before the values they unlock,
// so focus mode sits directly above focus distance in the panel.
const CAM_PROPS = [
  { k: "focusMode",            label: "Focus mode",        auto: "continuous" },
  { k: "focusDistance",        label: "Focus distance",    needs: "focusMode" },
  { k: "exposureMode",         label: "Exposure mode",     auto: "continuous" },
  { k: "exposureTime",         label: "Exposure time",     needs: "exposureMode" },
  { k: "exposureCompensation", label: "Exposure compensation" },
  { k: "whiteBalanceMode",     label: "White balance",     auto: "continuous" },
  { k: "colorTemperature",     label: "Colour temperature", unit: "K",
    needs: "whiteBalanceMode" },
  { k: "brightness",           label: "Brightness" },
  { k: "contrast",             label: "Contrast" },
  { k: "saturation",           label: "Saturation" },
  { k: "sharpness",            label: "Sharpness" },
  { k: "zoom",                 label: "Zoom" },
  { k: "pan",                  label: "Pan" },
  { k: "tilt",                 label: "Tilt" },
];

const $ = id => document.getElementById(id);
const recs = () => { try { return JSON.parse(localStorage.getItem("tli_recs") || "[]"); }
                     catch (e) { return []; } };
const saveRecs = r => { try { localStorage.setItem("tli_recs", JSON.stringify(r.slice(-S.maxRec))); }
                        catch (e) {} };

function setLed(k, state, text) {
  const led = document.querySelector('.status i[data-k="' + k + '"]');
  if (led) led.className = state || "";
  const map = { camera:"stCam", model:"stModel", motor:"stMotor", hold:"stHold", db:"stDb",
                queue:"stQueue", rate:"stRate", up:"stUp" };
  if (text !== undefined && $(map[k])) $(map[k]).textContent = text;
}
function runState(txt, cls) {
  const el = $("runState"); el.textContent = txt; el.className = "runstate " + (cls || "");
}

// One line telling the operator what the machine is doing and what to do next.
// The order of these tests is the order of the working procedure, so whichever
// step is outstanding is the one described.
function guide() {
  let cls = "info", msg;

  if (S.estop) {
    cls = "alarm";
    msg = "Emergency stop is active. Clear the hazard, then press RESET E-STOP.";
  } else if (S.autoStopped) {
    cls = "alarm";
    msg = "Motor stopped: a defect stayed in view for " + (S.holdMs/1000).toFixed(1)
        + " s. Remove the affected leaf, then press ACKNOWLEDGE and MOTOR START.";
  } else if (!S.session) {
    msg = "Loading the detector. Please wait.";
  } else if (!S.running) {
    msg = "Detector ready on " + (S.backend || "cpu") + ". Press START CAMERA to begin.";
  } else if (!S.recording) {
    msg = "Camera running. Press RECORD to start writing results.";
  } else if (!S.motor) {
    msg = "Recording. Press MOTOR START to run the conveyor.";
  } else if (S.defectSince) {
    cls = "warn";
    const held = (performance.now() - S.defectSince) / 1000;
    msg = "Defect in view for " + held.toFixed(1) + " s. The motor stops at "
        + (S.holdMs/1000).toFixed(1) + " s unless it clears.";
  } else {
    cls = "ok";
    msg = "Running and recording at " + S.speed.toFixed(2) + " m/s. No defect held in view.";
  }

  const el = $("guide"), t = $("guideText");
  if (t.textContent !== msg) t.textContent = msg;
  const want = "guide " + cls;
  if (el.className !== want) el.className = want;
}
setInterval(function () {
  const s = Math.floor((Date.now() - S.t0) / 1000);
  setLed("up", "ok", String(Math.floor(s/60)).padStart(2,"0") + ":" + String(s%60).padStart(2,"0"));
  if (!S.running) guide();   // the render loop keeps it current while running
}, 1000);

// The first run on any backend is slow because kernels and shaders are built
// then. Doing it once here keeps that cost out of the first camera frame.
async function warmup() {
  S.busy = true;
  try {
    const z = new ort.Tensor("float32", new Float32Array(3*INPUT*INPUT), [1,3,INPUT,INPUT]);
    const f = {}; f[S.session.inputNames[0]] = z;
    await S.session.run(f);
  } catch (e) {
    console.warn("warm-up run failed:", e.message);
  } finally {
    S.busy = false;
  }
}

async function loadModel() {
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
    // Threads need cross-origin isolation (COOP and COEP headers). Without
    // those the browser refuses to start workers, so fall back to one thread.
    ort.env.wasm.numThreads = self.crossOriginIsolated
      ? Math.min(navigator.hardwareConcurrency || 4, 8) : 1;
    ort.env.wasm.simd = true;
    setLed("model", "warn", "loading...");
    const path = ($("setModel") && $("setModel").value) || "model/best.onnx";

    // Prefer the GPU. WASM stays as the fallback for machines without WebGPU.
    const eps = [];
    if (navigator.gpu) eps.push("webgpu");
    eps.push("wasm");

    for (let i = 0; i < eps.length; i++) {
      try {
        S.session = await ort.InferenceSession.create(path,
          { executionProviders: [eps[i]], graphOptimizationLevel: "all" });
        S.backend = eps[i];
        break;
      } catch (e) { console.warn(eps[i] + " unavailable: " + e.message); }
    }
    if (!S.session) throw new Error("no execution provider could load the model");

    const th = S.backend === "wasm" ? " x" + ort.env.wasm.numThreads : "";
    setLed("model", "ok", "ready (" + S.backend + th + ")");
    await warmup();
  } catch (e) {
    console.error(e);
    setLed("model", "bad", "failed");
    $("viewHint").textContent = "Model failed to load: " + e.message;
  }
}

const lb = document.createElement("canvas");
lb.width = INPUT; lb.height = INPUT;
const lbx = lb.getContext("2d", { willReadFrequently: true });

function preprocess(src, w, h) {
  const r = Math.min(INPUT / w, INPUT / h);
  const nw = Math.round(w * r), nh = Math.round(h * r);
  const px = (INPUT - nw) / 2, py = (INPUT - nh) / 2;
  lbx.fillStyle = "rgb(114,114,114)";
  lbx.fillRect(0, 0, INPUT, INPUT);
  lbx.drawImage(src, px, py, nw, nh);
  const d = lbx.getImageData(0, 0, INPUT, INPUT).data;
  const a = INPUT * INPUT, f = new Float32Array(a * 3);
  for (let i = 0; i < a; i++) {
    f[i] = d[i*4] / 255; f[i+a] = d[i*4+1] / 255; f[i+2*a] = d[i*4+2] / 255;
  }
  return { tensor: new ort.Tensor("float32", f, [1,3,INPUT,INPUT]), r: r, px: px, py: py };
}

function iouOf(a, b) {
  const x1 = Math.max(a.x1,b.x1), y1 = Math.max(a.y1,b.y1);
  const x2 = Math.min(a.x2,b.x2), y2 = Math.min(a.y2,b.y2);
  const inter = Math.max(0,x2-x1) * Math.max(0,y2-y1);
  const ua = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter;
  return inter / (ua + 1e-6);
}

function postprocess(out, r, px, py) {
  const d = out.data, ch = out.dims[1], n = out.dims[2], nc = ch - 4;
  const boxes = [];
  for (let i = 0; i < n; i++) {
    let best = 0, cls = -1;
    for (let c = 0; c < nc; c++) {
      const v = d[(4+c)*n + i];
      if (v > best) { best = v; cls = c; }
    }
    if (best < S.conf) continue;
    const cx = d[i], cy = d[n+i], bw = d[2*n+i], bh = d[3*n+i];
    boxes.push({ x1:(cx-bw/2-px)/r, y1:(cy-bh/2-py)/r,
                 x2:(cx+bw/2-px)/r, y2:(cy+bh/2-py)/r, score:best, cls:cls });
  }
  boxes.sort(function (a,b) { return b.score - a.score; });
  const keep = [];
  for (const b of boxes) {
    let drop = false;
    for (const k of keep) { if (k.cls === b.cls && iouOf(k,b) > S.iou) { drop = true; break; } }
    if (!drop) keep.push(b);
  }
  return keep;
}

function draw(ctx, boxes, W) {
  const lw = Math.max(2, Math.round(W/380)), fs = Math.max(13, Math.round(W/48));
  ctx.lineWidth = lw;
  ctx.font = "600 " + fs + "px ui-monospace,monospace";
  ctx.textBaseline = "top";
  for (const b of boxes) {
    const name = CLASSES[b.cls] || b.cls, col = COLOR[name] || "#2f81f7";
    ctx.strokeStyle = col;
    ctx.strokeRect(b.x1, b.y1, b.x2-b.x1, b.y2-b.y1);
    const t = name + " " + (b.score*100).toFixed(0) + "%";
    const tw = ctx.measureText(t).width;
    ctx.fillStyle = col; ctx.fillRect(b.x1 - lw/2, b.y1 - fs - 6, tw + 10, fs + 6);
    ctx.fillStyle = "#05080c"; ctx.fillText(t, b.x1 + 5, b.y1 - fs - 3);
  }
}

function tally(boxes) {
  S.counts.insp += boxes.length;
  for (const b of boxes) {
    if (CLASSES[b.cls] === "defect") S.counts.bad++; else S.counts.ok++;
  }
  $("cInsp").textContent = S.counts.insp;
  $("cOk").textContent = S.counts.ok;
  $("cBad").textContent = S.counts.bad;
  $("cRate").textContent = S.counts.insp
      ? (100*S.counts.bad/S.counts.insp).toFixed(1) + "%" : "0.0%";
  if (boxes.length) $("cLast").textContent = new Date().toLocaleTimeString();

  if (S.recording && boxes.length) {
    const all = recs();
    for (const b of boxes) {
      all.push({ id: all.length + 1, ts: new Date().toISOString().slice(0,19),
                 cls: CLASSES[b.cls], conf: +b.score.toFixed(3),
                 box: [b.x1,b.y1,b.x2,b.y2].map(function (v) { return Math.round(v); }),
                 speed: S.motor ? S.speed : 0, device: S.device, event: "" });
    }
    saveRecs(all);
    setLed("queue", "ok", "0");
  }

  watchDefect(boxes);
}

// A single bad frame is not a fault. The motor is only stopped once a defect
// has been present continuously for the hold time, which rules out a one-off
// false positive while still catching a leaf that is genuinely sitting there.
function watchDefect(boxes) {
  const defects = boxes.filter(function (b) { return CLASSES[b.cls] === "defect"; });
  const now = performance.now();

  if (!defects.length) {
    S.defectSince = 0;
    setLed("hold", "", "--");
    return;
  }
  if (!S.defectSince) S.defectSince = now;

  const held = (now - S.defectSince) / 1000, limit = S.holdMs / 1000;
  const armed = S.recording && S.motor && !S.autoStopped && !S.estop;
  setLed("hold", armed ? "warn" : "", held.toFixed(1) + " / " + limit.toFixed(1) + " s");

  if (armed && now - S.defectSince >= S.holdMs) autoStop(defects, held);
}

function autoStop(defects, held) {
  S.motor = false;
  S.autoStopped = true;
  setLed("motor", "bad", "auto-stop");
  runState("DEFECT STOP", "stop");

  // Log the worst box that triggered it, so the record says what was seen.
  const worst = defects.slice().sort(function (a, b) { return b.score - a.score; })[0];
  const all = recs();
  all.push({ id: all.length + 1, ts: new Date().toISOString().slice(0,19),
             cls: "defect", conf: +worst.score.toFixed(3),
             box: [worst.x1,worst.y1,worst.x2,worst.y2].map(function (v) {
               return Math.round(v); }),
             speed: S.speed, device: S.device,
             event: "auto-stop after " + held.toFixed(1) + "s" });
  saveRecs(all);
  renderRecords();

  $("guideAck").hidden = false;
  guide();
}

// A session runs one inference at a time. The camera loop and the file upload
// can both reach this, so overlapping calls are dropped rather than queued: on
// a live view the newest frame is the one worth having.
async function infer(src, W, H) {
  if (!S.session || S.busy) return [];
  S.busy = true;
  try {
    const t0 = performance.now();
    const p = preprocess(src, W, H);
    const feeds = {}; feeds[S.session.inputNames[0]] = p.tensor;
    const out = await S.session.run(feeds);
    const ms = performance.now() - t0;
    const boxes = postprocess(out[S.session.outputNames[0]], p.r, p.px, p.py);
    S.boxes = boxes; S.inferMs = ms;
    tally(boxes);
    const now = performance.now();
    if (S.last) setLed("rate", "ok", (1000/(now - S.last)).toFixed(1) + " det/s");
    S.last = now;
    return boxes;
  } catch (e) {
    console.error("inference failed:", e);
    setLed("model", "bad", "run error");
    return [];
  } finally {
    S.busy = false;
  }
}

const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");

// Every camera exposes a different subset of adjustments, so the panel is
// built from what this device actually reports rather than from a fixed list.
// A control that is not offered is simply not drawn.
function buildCamControls() {
  const host = $("camCtls");
  host.innerHTML = "";
  const caps = S.track && S.track.getCapabilities ? S.track.getCapabilities() : null;
  const now  = S.track && S.track.getSettings ? S.track.getSettings() : {};

  if (!caps) {
    $("camNote").textContent = "This browser does not report camera capabilities, "
      + "so only the capture format can be set here.";
    return;
  }

  let shown = 0;
  for (let i = 0; i < CAM_PROPS.length; i++) {
    const p = CAM_PROPS[i], cap = caps[p.k];
    if (!cap) continue;
    shown++;

    const row = document.createElement("div");
    row.className = "camrow";
    row.dataset.prop = p.k;

    if (Array.isArray(cap)) {
      row.innerHTML = '<div class="lbl"><span>' + p.label + "</span></div>";
      const sel = document.createElement("select");
      sel.className = "sel";
      for (let j = 0; j < cap.length; j++) {
        const o = document.createElement("option");
        o.value = cap[j]; o.textContent = cap[j];
        if (cap[j] === now[p.k]) o.selected = true;
        sel.appendChild(o);
      }
      sel.onchange = function () { applyCam(p.k, sel.value); };
      row.appendChild(sel);
    } else if (typeof cap.min === "number") {
      const val = typeof now[p.k] === "number" ? now[p.k] : cap.min;
      const step = cap.step || (cap.max - cap.min) / 100 || 1;
      row.innerHTML =
        '<div class="lbl"><span>' + p.label + "</span><b>" + fmtCam(val, p) + "</b></div>";
      const r = document.createElement("input");
      r.type = "range"; r.min = cap.min; r.max = cap.max; r.step = step; r.value = val;
      r.oninput = function () {
        row.querySelector("b").textContent = fmtCam(+r.value, p);
        applyCam(p.k, +r.value);
      };
      row.appendChild(r);
    } else { shown--; continue; }

    host.appendChild(row);
  }

  refreshCamLocks();
  $("camNote").textContent = shown
    ? "Adjustments apply straight away and are not stored with the records."
    : "This camera reports no adjustable settings to the browser.";
}

function fmtCam(v, p) {
  const s = Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2).replace(/\.?0+$/, "");
  return s + (p.unit ? " " + p.unit : "");
}

// A manual value only has an effect once its mode is off automatic, so those
// rows are greyed out while the camera is still deciding for itself.
function refreshCamLocks() {
  const now = S.track && S.track.getSettings ? S.track.getSettings() : {};
  const rows = $("camCtls").querySelectorAll(".camrow");
  for (let i = 0; i < rows.length; i++) {
    const key = rows[i].dataset.prop;
    const p = CAM_PROPS.filter(function (x) { return x.k === key; })[0];
    if (p && p.needs) rows[i].classList.toggle("locked", now[p.needs] !== "manual");
  }
}

async function applyCam(prop, value) {
  if (!S.track) return;
  try {
    const c = {}; c[prop] = value;
    await S.track.applyConstraints({ advanced: [c] });
    refreshCamLocks();
  } catch (e) {
    console.warn("camera rejected " + prop + " = " + value + ": " + e.message);
    $("camNote").textContent = "The camera refused " + prop + ".";
  }
}

async function resetCam() {
  if (!S.track || !S.camDefaults) return;
  const caps = S.track.getCapabilities ? S.track.getCapabilities() : {};
  const wanted = [];
  for (let i = 0; i < CAM_PROPS.length; i++) {
    const p = CAM_PROPS[i];
    if (p.auto && Array.isArray(caps[p.k]) && caps[p.k].indexOf(p.auto) >= 0) {
      const c = {}; c[p.k] = p.auto; wanted.push(c);
    }
  }
  try { await S.track.applyConstraints({ advanced: wanted }); }
  catch (e) { console.warn("reset failed: " + e.message); }
  buildCamControls();
}

async function startCam() {
  if (S.estop) { alert("Emergency stop is active. Reset it first."); return; }
  const p = CAM_PRESETS[S.preset] || CAM_PRESETS[1];
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment",
               width:  { ideal: p.w },
               height: { ideal: p.h },
               frameRate: { ideal: p.fps } },
      audio: false });
    video.srcObject = S.stream; await video.play();
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    $("viewHint").style.display = "none";
    S.running = true;

    S.track = S.stream.getVideoTracks()[0];
    S.camDefaults = S.track.getSettings ? S.track.getSettings() : {};
    const got = S.camDefaults;
    $("camInfo").textContent = (got.width || video.videoWidth) + "x"
      + (got.height || video.videoHeight)
      + (got.frameRate ? " @" + Math.round(got.frameRate) : "");
    buildCamControls();

    setLed("camera","ok", video.videoWidth + "x" + video.videoHeight);
    runState("RUNNING","run"); loop(); detectLoop();
  } catch (e) { setLed("camera","bad","denied"); alert("Camera error: " + e.message); }
}

// Resolution and frame rate are fixed when the stream opens, so changing the
// format means taking the camera down and bringing it back up.
async function restartCam() {
  if (!S.running) return;
  stopCam();
  await new Promise(function (r) { setTimeout(r, 150); });
  await startCam();
}
function stopCam() {
  S.running = false;
  if (S.raf) cancelAnimationFrame(S.raf);
  if (S.stream) S.stream.getTracks().forEach(function (t) { t.stop(); });
  S.stream = null;
  S.boxes = []; S.lastFrame = 0; S.viewFps = 0; S.inferMs = 0;
  S.track = null;
  setLed("camera","","stopped"); runState("STANDBY",""); $("fps").textContent = "";
  setLed("rate","","0.0 /s");
  $("camInfo").textContent = "no device";
}
// Drawing and detection are kept apart on purpose. The camera delivers frames
// far faster than the model can process them, so the view redraws every frame
// with the most recent boxes overlaid, while detection runs at its own pace in
// the background. Waiting for the model before drawing froze the picture down
// to the detection rate.
function loop() {
  if (!S.running) return;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  if (S.boxes.length) draw(ctx, S.boxes, canvas.width);

  const now = performance.now();
  if (S.lastFrame) {
    S.viewFps = 0.9*(S.viewFps || 0) + 0.1*(1000/(now - S.lastFrame));
    $("fps").textContent = S.viewFps.toFixed(0) + " fps view / "
                         + (S.inferMs ? S.inferMs.toFixed(0) + " ms detect" : "detecting");
  }
  S.lastFrame = now;
  guide();
  S.raf = requestAnimationFrame(loop);
}

async function detectLoop() {
  while (S.running) {
    if (video.readyState >= 2) await infer(video, video.videoWidth, video.videoHeight);
    // Yield so the render loop and the interface stay responsive.
    await new Promise(function (r) { setTimeout(r, 0); });
  }
}

function renderRecords() {
  const f = $("fClass").value;
  const all = recs().filter(function (r) { return !f || r.cls === f; });
  $("recCount").textContent = all.length + " records";
  const rows = all.slice(-400).reverse().map(function (r) {
    return "<tr><td>" + r.id + "</td><td>" + r.ts.replace("T"," ") + "</td><td>" +
      '<span class="pill ' + (r.cls === "defect" ? "bad" : "ok") + '">' + r.cls + "</span></td><td>" +
      r.conf + "</td><td>" + r.box.join(", ") + "</td><td>" + r.speed + "</td><td>" +
      (r.event || "") + "</td><td>" + r.device + "</td></tr>";
  }).join("");
  document.querySelector("#recTable tbody").innerHTML = rows;
}
function exportCsv() {
  const all = recs();
  if (!all.length) { alert("No records yet."); return; }
  const head = "id,timestamp,class,confidence,x1,y1,x2,y2,belt_speed,event,device_id";
  const body = all.map(function (r) {
    return [r.id, r.ts, r.cls, r.conf].concat(r.box)
             .concat([r.speed, '"' + (r.event || "") + '"', r.device]).join(",");
  }).join("\n");
  const blob = new Blob([head + "\n" + body], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "detections.csv"; a.click();
}
function bar(cv, labels, values, colors) {
  const c = cv.getContext("2d");
  const W = cv.width = cv.clientWidth, H = cv.height;
  c.clearRect(0,0,W,H);
  const max = Math.max.apply(null, [1].concat(values));
  const pad = 26, bw = (W - pad*2) / (values.length || 1);
  c.strokeStyle = "#222c38";
  c.beginPath(); c.moveTo(pad,H-20); c.lineTo(W-pad,H-20); c.stroke();
  values.forEach(function (v,i) {
    const h = (H - 46) * v / max;
    c.fillStyle = colors[i] || "#2f81f7";
    c.fillRect(pad + i*bw + bw*0.18, H - 20 - h, bw*0.64, h);
    c.fillStyle = "#8b98a5"; c.font = "10px ui-monospace,monospace"; c.textAlign = "center";
    c.fillText(labels[i], pad + i*bw + bw/2, H - 7);
    if (v) c.fillText(v, pad + i*bw + bw/2, H - 26 - h);
  });
}
function renderStats() {
  const all = recs(), byHour = {};
  all.forEach(function (r) {
    const h = r.ts.slice(11,13) + ":00"; byHour[h] = (byHour[h] || 0) + 1;
  });
  const hk = Object.keys(byHour).sort();
  bar($("chartTime"), hk, hk.map(function (k) { return byHour[k]; }),
      hk.map(function () { return "#2f81f7"; }));
  const ok = all.filter(function (r) { return r.cls !== "defect"; }).length;
  bar($("chartShare"), ["Nondefect","defect"], [ok, all.length - ok],
      ["#3fb950","#f85149"]);
}

document.querySelectorAll(".tab").forEach(function (t) {
  t.onclick = function () {
    document.querySelectorAll(".tab").forEach(function (x) { x.classList.remove("active"); });
    document.querySelectorAll(".page").forEach(function (p) { p.classList.remove("active"); });
    t.classList.add("active");
    $("page-" + t.dataset.page).classList.add("active");
    if (t.dataset.page === "records") renderRecords();
    if (t.dataset.page === "stats") renderStats();
  };
});

$("btnStart").onclick = startCam;
$("btnStop").onclick  = stopCam;
$("btnEstop").onclick = function () {
  S.estop = !S.estop;
  if (S.estop) {
    stopCam(); S.motor = false; S.defectSince = 0;
    setLed("motor","bad","E-STOP"); setLed("hold","","--");
    runState("E-STOP","stop");
    $("btnEstop").textContent = "RESET E-STOP";
  } else {
    setLed("motor","","stopped"); runState("STANDBY","");
    $("btnEstop").textContent = "EMERGENCY STOP";
  }
  guide();
};
$("btnMotorStart").onclick = function () {
  if (S.estop) { alert("Emergency stop is active. Reset it first."); return; }
  if (S.autoStopped) {
    alert("The motor stopped on a defect. Press ACKNOWLEDGE first.");
    return;
  }
  S.motor = true;
  S.defectSince = 0;
  setLed("motor","ok", S.speed.toFixed(2) + " m/s");
  runState("RUNNING","run");
  guide();
};
$("btnMotorStop").onclick = function () {
  S.motor = false;
  setLed("motor","","stopped");
  if (!S.estop) runState(S.running ? "RUNNING" : "STANDBY", S.running ? "run" : "");
  guide();
};

// Clearing the alarm is deliberately separate from starting the motor, so the
// operator has to confirm the leaf was dealt with before the belt can move.
$("guideAck").onclick = function () {
  S.autoStopped = false;
  S.defectSince = 0;
  $("guideAck").hidden = true;
  setLed("motor","","stopped");
  setLed("hold","","--");
  runState(S.running ? "RUNNING" : "STANDBY", S.running ? "run" : "");
  guide();
};
$("speed").oninput = function (e) {
  S.speed = +e.target.value; $("speedOut").textContent = S.speed.toFixed(2);
  if (S.motor) setLed("motor","ok", S.speed.toFixed(2) + " m/s");
  guide();
};
$("setHold").oninput = function (e) {
  S.holdMs = +e.target.value * 1000;
  $("holdOut").textContent = (+e.target.value).toFixed(1);
  guide();
};
$("conf").oninput = function (e) {
  S.conf = +e.target.value; $("confOut").textContent = S.conf.toFixed(2);
};
$("iou").oninput = function (e) {
  S.iou = +e.target.value; $("iouOut").textContent = S.iou.toFixed(2);
};
$("btnRec").onclick = function () {
  S.recording = !S.recording;
  $("btnRec").classList.toggle("rec", S.recording);
  $("btnRec").textContent = S.recording ? "STOP REC" : "RECORD";
  setLed("db", S.recording ? "ok" : "", S.recording ? "recording" : "local");
  if (!S.recording) S.defectSince = 0;
  guide();
};
$("btnSnap").onclick = function () {
  const a = document.createElement("a");
  a.download = "snapshot.png"; a.href = canvas.toDataURL("image/png"); a.click();
};
$("btnReset").onclick = function () {
  S.counts = { insp:0, ok:0, bad:0 };
  ["cInsp","cOk","cBad"].forEach(function (i) { $(i).textContent = "0"; });
  $("cRate").textContent = "0.0%"; $("cLast").textContent = "--";
};
$("file").onchange = function (e) {
  const f = e.target.files[0]; if (!f) return;
  const img = new Image();
  img.onload = async function () {
    stopCam();
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    $("viewHint").style.display = "none";
    const boxes = await infer(img, img.naturalWidth, img.naturalHeight);
    draw(ctx, boxes, img.naturalWidth);
    $("fps").textContent = S.inferMs.toFixed(0) + " ms detect";
    URL.revokeObjectURL(img.src);
  };
  img.src = URL.createObjectURL(f);
};
$("btnCsv").onclick = exportCsv;
$("btnClear").onclick = function () {
  if (confirm("Delete all stored records?")) {
    localStorage.removeItem("tli_recs"); renderRecords();
  }
};
$("fClass").innerHTML = '<option value="">all classes</option>' +
  CLASSES.map(function (c) { return "<option>" + c + "</option>"; }).join("");
$("fClass").onchange = renderRecords;
(function initCamPanel() {
  const sel = $("camPreset");
  for (let i = 0; i < CAM_PRESETS.length; i++) {
    const o = document.createElement("option");
    o.value = i; o.textContent = CAM_PRESETS[i].label;
    if (i === S.preset) o.selected = true;
    sel.appendChild(o);
  }
  sel.onchange = function (e) { S.preset = +e.target.value; restartCam(); };
  $("btnCamAuto").onclick = resetCam;
})();

$("setDev").onchange = function (e) { S.device = e.target.value; };
$("setMax").onchange = function (e) { S.maxRec = +e.target.value || 2000; };
$("setModel").onchange = loadModel;

setLed("db","","local"); setLed("queue","","0");
loadModel();
