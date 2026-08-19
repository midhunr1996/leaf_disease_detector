# 🍃 Leaf Health Detector — Good vs Bad Leaf (YOLOv12)

Detect **good vs bad leaves** in real time. Upload an image or use your webcam —
inference runs **100% in the browser** with YOLOv12 + ONNX Runtime Web, so
nothing is ever uploaded to a server. Free to host, works for everyone with a
browser.

**Live demo:** _enable GitHub Pages (see below), then your URL appears here_ →
`https://<your-username>.github.io/<repo-name>/`

![screenshot placeholder](https://via.placeholder.com/800x400?text=Leaf+Health+Detector)

---

## What's in here

| File | Purpose |
|------|---------|
| `train_yolov12.ipynb` | Colab notebook: trains YOLOv12 on your Roboflow dataset, exports `best.onnx` |
| `index.html` / `style.css` / `app.js` | The website (image upload + live webcam detection) |
| `model/best.onnx` | Your trained model — **you add this** after training |

---

## Step 1 — Train the model (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/) → **File ▸ Upload notebook** → pick `train_yolov12.ipynb`.
2. Set **Runtime ▸ Change runtime type ▸ T4 GPU**.
3. **Run all cells.** It will download your Roboflow dataset, train YOLOv12-nano, and export ONNX.
4. In **Step 4**, note the printed **class names** — you'll need them.
5. The last cell downloads **`best.onnx`** and `best.pt`.

## Step 2 — Add your model + class names to the website

1. Put `best.onnx` into the `model/` folder → `model/best.onnx`.
2. Open `app.js` and set the class list to match your dataset (same order as Step 4):
   ```js
   const CLASSES = ["bad", "good"];   // ← use YOUR class names, in order
   ```

## Step 3 — Test locally (optional but recommended)

Browsers block webcams/model loading from `file://`, so run a tiny local server:

```bash
# from the project folder
python -m http.server 8000
```
Then open <http://localhost:8000>.

## Step 4 — Publish to GitHub Pages (free, public)

```bash
git init
git add .
git commit -m "Leaf health detector"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Then on GitHub: **Settings ▸ Pages ▸ Build and deployment ▸ Source = `Deploy from a branch`**,
pick **`main`** / **`/ (root)`**, Save. After ~1 minute your site is live at
`https://<your-username>.github.io/<repo-name>/` — share it with anyone.

---

## How it works

- **Training:** Ultralytics YOLOv12-nano fine-tuned on your Roboflow images, exported to ONNX (opset 12).
- **In-browser inference:** `app.js` letterboxes each frame to 640×640, runs the ONNX model via ONNX Runtime Web (WASM), then decodes boxes and applies non-maximum suppression — all on the user's device.
- **Privacy:** images and webcam frames never leave the browser.

## Tips & troubleshooting

- **"Could not load the model"** → make sure `model/best.onnx` exists and you're serving over http(s), not opening the file directly.
- **Boxes look shifted** → confirm the model was exported at `imgsz=640` (matches `INPUT_SIZE` in `app.js`).
- **Wrong labels** → the `CLASSES` order in `app.js` must match `data.yaml` exactly.
- **Slow on phones** → keep the model at **nano**; lower the confidence slider if you miss detections.
- **Too many/few boxes** → adjust the confidence slider, or `IOU_THRESHOLD` in `app.js`.

---

Built with [Ultralytics YOLOv12](https://github.com/ultralytics/ultralytics) and
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/).
