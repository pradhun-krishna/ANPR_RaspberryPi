# Automatic Number Plate Recognition (ANPR) System

A robust, real-time ANPR system built with YOLOv8 for detection and EasyOCR/PaddleOCR for character recognition. Designed for high accuracy on Indian license plates.

## Features
- **Real-time Detection**: Uses YOLOv8 for efficient plate detection.
- **High Accuracy OCR**: Integrated with EasyOCR and PaddleOCR with custom post-processing for Indian plate formats.
- **Multi-Camera Support**: Handle multiple entry/exit streams.
- **Tracking**: IoU-based object tracking to prevent duplicate reads.
- **Logging**: diverse logging options (CSV, JSONL).

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ANPR_PI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch**
   Ensure you have PyTorch installed compatible with your CUDA version (for GPU support).
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## Configuration

Edit `configs/config.yaml` to customize the system:

```yaml
inference:
  device: cuda  # or 'cpu'
  conf_threshold: 0.25

io:
  source: 0  # Webcam ID or RTSP URL or Video file path
```

## Usage

### Single Camera Mode
Run the inference script directly:
```bash
python -m src.video_infer --source 0
```

### Multi-Camera Mode
Enable `multi_camera: true` in `configs/config.yaml` and configure your camera sources.
```bash
python -m src.multi_camera_infer
```

## Database Integration

The system uses CSV files for checking authorized vehicles, but you can easily integrate your own database (SQL, NoSQL, etc.).

- **Default Behavior**: The system looks for `logs/parking_database.csv` (not included) to validate plates (configurable in `configs/config.yaml`).
- **Testing**: If you just want to test the ANPR detection, you can run the system without any database file. The logs will simply show the detected plates in the terminal.
- **Customization**: To use your own database, modify `src/postprocess.py` to connect to your data source instead of loading the CSV.

## Models
Place your YOLOv8 model (`.pt` or `.onnx`) in the `models/` directory and update `configs/config.yaml` accordingly.

## License
MIT License
