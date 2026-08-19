# Put your trained model here

After running `train_yolov12.ipynb` in Colab, download **`best.onnx`** and
place it in this folder so the path is:

```
model/best.onnx
```

The website (`app.js`) loads this file. If you rename it, update `MODEL_URL`
in `app.js`.

> ⚠️ GitHub blocks files larger than 100 MB. A YOLOv12-nano ONNX model is
> only ~6 MB, so you're fine. If you trained a larger variant and the file is
> big, either use nano or add the model via [Git LFS](https://git-lfs.com/).
