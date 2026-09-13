"""Tea leaf inspection server for the Raspberry Pi 5.

Runs the detector on the Pi itself rather than in the viewer's browser. That
matters for machine control: the automatic stop has to work whether or not
anybody has the page open, and only this process owns the GPIO pins.

    python server.py                     # defaults to the int8 model
    python server.py --model ../pi5_export/YOLO26s_640_fp32.onnx
    python server.py --no-motor          # bench test without the L298N wired

Endpoints:
    GET  /                  minimal viewer
    GET  /video             MJPEG stream with boxes drawn
    GET  /api/status        machine and detector state
    POST /api/motor         {"action": "start|stop|brake|speed", "duty": 0.5}
    GET  /api/records       recent detections
    GET  /api/records.csv   full record export
    POST /api/settings      {"conf": 0.35, "iou": 0.45, "hold_s": 5.0}
"""

import argparse
import csv
import io
import json
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, jsonify, request

HERE = os.path.dirname(os.path.abspath(__file__))

# fp32, not int8. The int8 export in pi5_export/ is broken: its classification
# head returns exactly 0.0 for every candidate on every image, so it detects
# nothing at all, while its box head still produces sensible coordinates. Until
# it is re-exported with proper calibration, fp32 is the only usable file.
DEFAULT_MODEL = os.path.join(HERE, "..", "pi5_export", "YOLO26s_640_fp32.onnx")
RECORDS_PATH = os.path.join(HERE, "records.jsonl")

INPUT = 640
PAD = 114
CLASSES = ["Nondefect", "defect"]
COLOURS = {"Nondefect": (80, 185, 63), "defect": (73, 81, 248)}  # BGR


# --------------------------------------------------------------------- state
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None            # latest annotated JPEG bytes
        self.boxes = []
        self.infer_ms = 0.0
        self.conf = 0.35
        self.iou = 0.45
        self.hold_s = 5.0
        self.speed_mps = 0.10
        self.recording = False
        self.estop = False
        self.auto_stopped = False
        self.defect_since = 0.0
        self.counts = {"insp": 0, "ok": 0, "bad": 0}
        self.device = "pi5-line-01"
        self.running = True
        self.backend = ""


S = State()
MOTOR = None
ENCODER = None


def motor_state():
    if MOTOR is None:
        return {"running": False, "duty": 0.0, "reverse": False}
    return MOTOR.state


def belt_speed():
    """Measured belt speed if an encoder is fitted, otherwise the value the
    operator set. Records should carry what the belt did, not what was asked
    of it, because the motion-blur limit is a function of real speed."""
    if ENCODER is not None:
        return round(abs(ENCODER.speed_mps), 4)
    return S.speed_mps if motor_state()["running"] else 0.0


# ------------------------------------------------------------------ records
_rec_lock = threading.Lock()
_next_id = None


def next_record_id():
    if not os.path.exists(RECORDS_PATH):
        return 1
    with open(RECORDS_PATH, "rb") as fh:
        try:
            fh.seek(-4096, os.SEEK_END)
        except OSError:
            fh.seek(0)
        text = fh.read().decode("utf-8", "replace")
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return 1
    try:
        return json.loads(lines[-1])["id"] + 1
    except Exception:
        return 1


def write_records(rows):
    """Append detection rows. One JSON object per line, so a crash mid-write
    costs at most the last line rather than the whole file."""
    global _next_id
    with _rec_lock:
        if _next_id is None:
            _next_id = next_record_id()
        with open(RECORDS_PATH, "a", encoding="utf-8") as fh:
            for r in rows:
                r["id"] = _next_id
                _next_id += 1
                fh.write(json.dumps(r) + "\n")


def read_records(limit=None):
    if not os.path.exists(RECORDS_PATH):
        return []
    out = []
    with open(RECORDS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out[-limit:] if limit else out


# ------------------------------------------------------------- preprocessing
def letterbox(img):
    """Scale by the smaller ratio and pad with grey, so the leaf keeps its
    shape. Must match the training preprocessing exactly, or the boxes land in
    the wrong place."""
    h, w = img.shape[:2]
    r = min(INPUT / w, INPUT / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((INPUT, INPUT, 3), PAD, dtype=np.uint8)
    px, py = (INPUT - nw) // 2, (INPUT - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(rgb.transpose(2, 0, 1), 0), r, px, py


def nms(boxes, scores, thr):
    """Greedy non-maximum suppression."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def postprocess(out, r, px, py, conf_thr, iou_thr):
    """Output is [1, 4+nc, 8400]: cx, cy, w, h then one score per class."""
    pred = out[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T                       # -> [8400, 4+nc]
    xywh, scores = pred[:, :4], pred[:, 4:]
    cls_id = scores.argmax(1)
    cls_score = scores.max(1)
    keep = cls_score >= conf_thr
    if not keep.any():
        return []
    xywh, cls_id, cls_score = xywh[keep], cls_id[keep], cls_score[keep]

    cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    boxes = np.stack([(cx - w / 2 - px) / r, (cy - h / 2 - py) / r,
                      (cx + w / 2 - px) / r, (cy + h / 2 - py) / r], 1)

    out_boxes = []
    for c in np.unique(cls_id):               # class-aware NMS
        m = cls_id == c
        sub_boxes, sub_scores = boxes[m], cls_score[m]
        for i in nms(sub_boxes, sub_scores, iou_thr):
            out_boxes.append({"cls": int(c), "name": CLASSES[int(c)],
                              "score": float(sub_scores[i]),
                              "box": [float(v) for v in sub_boxes[i]]})
    return out_boxes


def draw(img, boxes):
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b["box"]]
        col = COLOURS.get(b["name"], (247, 129, 47))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        label = "%s %.0f%%" % (b["name"], b["score"] * 100)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), col, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (12, 8, 5), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- decisions
def tally(boxes):
    S.counts["insp"] += len(boxes)
    for b in boxes:
        if b["name"] == "defect":
            S.counts["bad"] += 1
        else:
            S.counts["ok"] += 1

    if S.recording and boxes:
        now = datetime.now().isoformat(timespec="seconds")
        speed = belt_speed()
        write_records([{
            "ts": now, "cls": b["name"], "conf": round(b["score"], 3),
            "box": [round(v) for v in b["box"]],
            "speed": speed,
            "event": "", "device": S.device,
        } for b in boxes])


def watch_defect(boxes):
    """Stop the belt when a defect has been present continuously for hold_s.

    The timer resets the moment the defect clears, so one false-positive frame
    cannot halt the line. This runs here rather than in the browser, so it
    keeps working with nobody watching the page.
    """
    defects = [b for b in boxes if b["name"] == "defect"]
    now = time.time()

    if not defects:
        S.defect_since = 0.0
        return
    if not S.defect_since:
        S.defect_since = now

    armed = (S.recording and motor_state()["running"]
             and not S.auto_stopped and not S.estop)
    if armed and (now - S.defect_since) >= S.hold_s:
        held = now - S.defect_since
        if MOTOR:
            MOTOR.brake()
        S.auto_stopped = True
        worst = max(defects, key=lambda b: b["score"])
        write_records([{
            "ts": datetime.now().isoformat(timespec="seconds"),
            "cls": "defect", "conf": round(worst["score"], 3),
            "box": [round(v) for v in worst["box"]],
            "speed": belt_speed(),
            "event": "auto-stop after %.1fs" % held,
            "device": S.device,
        }])


# ------------------------------------------------------------- capture loop
def capture_loop(model_path, cam_index, width, height, fps):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    S.backend = "onnxruntime cpu"
    inp = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    # MJPG lets the camera do the compression, which a Pi appreciates.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise SystemExit("camera %d did not open" % cam_index)

    while S.running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        t0 = time.time()
        tensor, r, px, py = letterbox(frame)
        out = sess.run(None, {inp: tensor})[0]
        boxes = postprocess(out, r, px, py, S.conf, S.iou)
        infer_ms = (time.time() - t0) * 1000.0

        annotated = draw(frame.copy(), boxes)
        encoded, jpg = cv2.imencode(".jpg", annotated,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        with S.lock:
            S.boxes = boxes
            S.infer_ms = infer_ms
            if encoded:
                S.frame = jpg.tobytes()
            tally(boxes)
            watch_defect(boxes)

    cap.release()


# --------------------------------------------------------------------- http
app = Flask(__name__)

VIEWER = """<!doctype html><meta charset=utf-8><title>Tea Leaf Inspection</title>
<style>body{background:#0d1117;color:#e6edf3;font:14px system-ui;margin:0;padding:16px}
img{max-width:100%;border:1px solid #222c38;border-radius:4px}
button{background:#1d2530;color:#e6edf3;border:1px solid #222c38;border-radius:4px;
padding:9px 14px;margin:4px 2px;cursor:pointer;font-weight:600}
pre{background:#151b24;padding:10px;border-radius:4px;font-size:12px}</style>
<h2>Tea Leaf Inspection</h2><img src="/video">
<div><button onclick=m('start')>MOTOR START</button>
<button onclick=m('stop')>MOTOR STOP</button>
<button onclick=m('brake')>BRAKE</button>
<button onclick=m('acknowledge')>ACKNOWLEDGE</button>
<button onclick=rec()>TOGGLE RECORD</button></div><pre id=s>loading</pre>
<script>
const m=a=>fetch('/api/motor',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({action:a,duty:0.5})});
const rec=()=>fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({recording:'toggle'})});
setInterval(async()=>{const r=await fetch('/api/status');
 document.getElementById('s').textContent=JSON.stringify(await r.json(),null,1)},500);
</script>"""


@app.get("/")
def index():
    return Response(VIEWER, mimetype="text/html")


@app.get("/video")
def video():
    def gen():
        while S.running:
            with S.lock:
                frame = S.frame
            if frame:
                yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=f")


@app.get("/api/status")
def status():
    with S.lock:
        held = (time.time() - S.defect_since) if S.defect_since else 0.0
        return jsonify({
            "motor": motor_state(), "estop": S.estop,
            "auto_stopped": S.auto_stopped, "recording": S.recording,
            "detections": len(S.boxes), "infer_ms": round(S.infer_ms, 1),
            "fps": round(1000.0 / S.infer_ms, 1) if S.infer_ms else 0,
            "defect_held_s": round(held, 1), "hold_s": S.hold_s,
            "conf": S.conf, "iou": S.iou, "counts": S.counts,
            "backend": S.backend, "device": S.device,
            "belt_mps": belt_speed(),
            "belt_measured": ENCODER is not None,
            "encoder_counts": ENCODER.counts if ENCODER else None,
        })


@app.post("/api/motor")
def motor_api():
    body = request.get_json(silent=True) or {}
    action = body.get("action", "")
    duty = float(body.get("duty", 0.5))

    if action == "estop":
        S.estop = True
        if MOTOR:
            MOTOR.brake()
        return jsonify(motor_state())
    if action == "reset_estop":
        S.estop = False
        return jsonify(motor_state())
    if action == "acknowledge":
        S.auto_stopped = False
        S.defect_since = 0.0
        return jsonify(motor_state())

    if S.estop and action in ("start", "speed"):
        return jsonify({"error": "emergency stop active"}), 409
    if S.auto_stopped and action in ("start", "speed"):
        return jsonify({"error": "acknowledge the defect stop first"}), 409
    if MOTOR is None:
        return jsonify({"error": "motor disabled (--no-motor)"}), 503

    if action == "start":
        S.defect_since = 0.0
        MOTOR.start(duty, bool(body.get("reverse")))
    elif action == "stop":
        MOTOR.stop()
    elif action == "brake":
        MOTOR.brake()
    elif action == "speed":
        MOTOR.set_speed(duty)
    else:
        return jsonify({"error": "unknown action"}), 400
    return jsonify(motor_state())


@app.post("/api/settings")
def settings():
    body = request.get_json(silent=True) or {}
    if "conf" in body:
        S.conf = max(0.05, min(0.95, float(body["conf"])))
    if "iou" in body:
        S.iou = max(0.1, min(0.9, float(body["iou"])))
    if "hold_s" in body:
        S.hold_s = max(0.5, min(60.0, float(body["hold_s"])))
    if "speed_mps" in body:
        S.speed_mps = float(body["speed_mps"])
    if "device" in body:
        S.device = str(body["device"])[:64]
    if "recording" in body:
        S.recording = ((not S.recording) if body["recording"] == "toggle"
                       else bool(body["recording"]))
    return jsonify({"conf": S.conf, "iou": S.iou, "hold_s": S.hold_s,
                    "recording": S.recording, "device": S.device})


@app.get("/api/records")
def records():
    return jsonify(read_records(limit=int(request.args.get("limit", 200))))


@app.get("/api/records.csv")
def records_csv():
    rows = read_records()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "timestamp", "class", "confidence",
                "x1", "y1", "x2", "y2", "belt_speed", "event", "device_id"])
    for r in rows:
        w.writerow([r.get("id"), r.get("ts"), r.get("cls"), r.get("conf")]
                   + list(r.get("box", [0, 0, 0, 0]))
                   + [r.get("speed"), r.get("event", ""), r.get("device")])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition":
                             "attachment; filename=detections.csv"})


# --------------------------------------------------------------------- main
def main():
    global MOTOR, ENCODER
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-motor", action="store_true")
    p.add_argument("--no-encoder", action="store_true",
                   help="skip the encoder and record the requested speed")
    p.add_argument("--counts-per-rev", type=float, default=None,
                   help="encoder counts per output shaft revolution")
    p.add_argument("--pulley-mm", type=float, default=None,
                   help="drive pulley diameter in millimetres")
    a = p.parse_args()

    if not os.path.exists(a.model):
        raise SystemExit("model not found: %s" % a.model)

    if not a.no_motor:
        from motor import Motor
        MOTOR = Motor()

    if not a.no_encoder:
        from motor import Encoder, COUNTS_PER_REV, PULLEY_DIAMETER_M
        ENCODER = Encoder(
            counts_per_rev=a.counts_per_rev or COUNTS_PER_REV,
            pulley_d=(a.pulley_mm / 1000.0) if a.pulley_mm else PULLEY_DIAMETER_M)

    t = threading.Thread(target=capture_loop,
                         args=(a.model, a.camera, a.width, a.height, a.fps),
                         daemon=True)
    t.start()

    print("model  :", os.path.basename(a.model))
    print("motor  :", "disabled" if a.no_motor else "GPIO12 ENA, 23 IN1, 24 IN2")
    if ENCODER:
        print("encoder: GPIO5 C1, GPIO6 C2, %.0f counts/rev, %.1f mm pulley"
              % (ENCODER.counts_per_rev, ENCODER.pulley_d * 1000))
    else:
        print("encoder: disabled, records will carry the requested speed")
    print("open   : http://<pi-address>:%d/" % a.port)
    try:
        app.run(host="0.0.0.0", port=a.port, threaded=True)
    finally:
        S.running = False
        if MOTOR:
            MOTOR.close()
        if ENCODER:
            ENCODER.close()


if __name__ == "__main__":
    main()
