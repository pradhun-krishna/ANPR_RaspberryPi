## ANPR-India: Real-time Automatic Number Plate Recognition (Indian plates)

This repository provides a complete real-time ANPR system optimized for Indian number plates. It uses YOLOv8 for plate detection and a fast OCR backend, with a threaded OpenCV pipeline for real-time inference from webcams or network streams. It supports GPU acceleration (CUDA) and ONNX Runtime inference.

### Features
- YOLOv8 plate detector (PyTorch) with optional ONNX Runtime inference
- OCR recognizer via EasyOCR (PyTorch) or custom CRNN+CTC (optional) with ONNX fallback
- Real-time, threaded capture → detect+OCR → display pipeline with FPS overlay
- Indian plate rules: regex validation, normalization, confusion correction (O↔0, I↔1, B↔8, S↔5)
- CSV and JSONL logging of recognized plates
- FastAPI server for HTTP frame inference
- Dockerfile with GPU support (nvidia-docker)
- Synthetic Indian plate data generator and demo notebook

### Repo Tree
```
models/
datasets/
src/
  detector.py
  recognizer.py
  postprocess.py
  video_infer.py
  fastapi_server.py
configs/config.yaml
tools/export_to_onnx.py
tools/generate_synthetic_plates.py
logs/
notebooks/demo.ipynb
Dockerfile
README.md
```

### Installation
1) Python 3.9+ recommended. Create a venv.
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```
2) Install dependencies.
```bash
pip install --upgrade pip
pip install ultralytics onnxruntime-gpu onnx opencv-python easyocr pyyaml fastapi uvicorn numpy pillow
# CPU-only ONNX: pip install onnxruntime
```

### Models
- Detector: default uses a YOLOv8 plate model path set in `configs/config.yaml`. You can use a community-trained Indian plate model or fine-tune your own.
- Recognizer: default uses EasyOCR (English). For best performance, train a plate-focused CRNN/PARSeq and point `configs/config.yaml` to its weights or ONNX.

To export YOLO detector and custom OCR to ONNX:
```bash
python tools/export_to_onnx.py --detector models/plate_yolov8.pt --ocr models/ocr_crnn.pt
```

### Real-time Inference
**Optimized for speed and accuracy:**
```bash
python -m src.video_infer --source 0 --conf 0.4 --skip 3 --mode both
```

**Other useful commands:**
```bash
# Webcam with higher confidence
python -m src.video_infer --source 0 --conf 0.5 --skip 2 --mode overlay

# Video file
python -m src.video_infer --source path/to/video.mp4 --conf 0.4 --skip 2 --mode both

# RTSP/HTTP stream
python -m src.video_infer --source rtsp://user:pass@host/stream --conf 0.4 --skip 3 --mode both

# JSON only (no display)
python -m src.video_infer --source 0 --conf 0.4 --skip 3 --mode json
```

**Controls:**
- Press **'q'** to quit the video window

CLI flags (overrides config):
- `--source`: 0, 1, path.mp4, or rtsp/http url
- `--conf`: detection confidence threshold
- `--skip`: process every Nth frame (default 2)
- `--mode`: overlay | json | both
- `--detector-only`: run detection without OCR

Outputs:
- Overlays: bounding boxes, plate text, confidence, FPS, plates per frame
- JSONL stream to stdout when `--mode json` or `both`
- CSV log at `logs/plates.csv` and JSONL at `logs/plates.jsonl`

Example JSON line:
```json
{"timestamp":"2025-01-01T12:00:00.000Z","plate_text":"KA01AB1234","confidence":0.94,"bbox":[x1,y1,x2,y2]}
```

### Indian Plate Rules
- Normalization: uppercase, remove spaces/dashes
- Confusion correction: O↔0, I↔1, B↔8, S↔5 (context-aware)
- Regex validation examples (common formats):
  - `^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{3,4}$`
  - `^[A-Z]{2}[0-9]{2}[A-Z]{0,2}[0-9]{4}$`
  - `^[A-Z]{2}[0-9]{1,2}[A-Z][0-9]{4}$`
See `src/postprocess.py` for full patterns and logic.

### Training (High-level)
Detector (YOLOv8):
1) Prepare dataset in YOLO format with a `license_plate` class.
2) Train:
```bash
pip install ultralytics
yolo detect train data=path/to/data.yaml model=yolov8n.pt imgsz=640 epochs=100 batch=16 name=plates
# Export best model to models/plate_yolov8.pt
```

Recognizer (CRNN/PARSeq):
1) Prepare cropped plate images and labels.
2) Train your recognizer (e.g., CRNN+CTC) and export weights.
3) Update `configs/config.yaml` and optionally export to ONNX with `tools/export_to_onnx.py`.

### FastAPI Server
Run the API:
```bash
uvicorn src.fastapi_server:app --host 0.0.0.0 --port 8000
```
POST `/infer` with `multipart/form-data` field `image`.

### Docker (GPU)
Build and run:
```bash
docker build -t anpr-india .
docker run --gpus all -it --rm -p 8000:8000 anpr-india
```

### Synthetic Data and Samples
Generate 10 sample images and a short demo video:
```bash
python tools/generate_synthetic_plates.py --out datasets/samples --num 10 --video demo.mp4
```

### Example Benchmarks (reference on RTX 3060 / i7-11800H)
- GPU (YOLOv8n ONNX + EasyOCR): 22–28 FPS
- GPU (PyTorch): 18–22 FPS
- CPU (ONNXRuntime CPU): 8–12 FPS

Your results may vary based on models and input resolution.

### Configuration
See `configs/config.yaml` for all options:
- Model paths (detector PyTorch/ONNX; OCR backend easyocr/crnn and optional ONNX)
- Confidence thresholds
- Frame skip interval
- Output mode: overlay/json/both
- Input: camera index, file, or RTSP/HTTP
- Preprocessing: CLAHE on crops for low-light

### License
For research and internal use. Ensure compliance with local laws and privacy regulations when deploying ANPR.


