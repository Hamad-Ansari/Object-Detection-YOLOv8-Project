# Object Detection with YOLOv8

A comprehensive object detection system using YOLOv8 for real-time detection, training, and deployment.

## Features

- **Multiple Detection Modes**: Real-time webcam, video file, image, and batch processing
- **Custom Training**: Train on your own datasets with easy configuration
- **Pre-trained Models**: Use COCO-trained models out of the box
- **Performance Metrics**: Precision, Recall, F1-Score, mAP tracking
- **Deployment Ready**: REST API and batch processing scripts included

## Project Structure

```
Object-Detection-YOLOv8-Project/
├── src/
│   ├── train.py              # Model training script
│   ├── detect.py             # Detection inference
│   ├── evaluate.py           # Model evaluation
│   └── utils.py              # Helper functions
├── models/                   # Trained model weights
├── data/
│   ├── images/              # Input images
│   ├── videos/              # Input videos
│   └── datasets/            # Training datasets
├── configs/
│   ├── config.yaml          # Main configuration
│   └── dataset.yaml         # Dataset configuration
├── outputs/                 # Detection results
├── notebooks/
│   └── demo.ipynb          # Jupyter demo notebook
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Hamad-Ansari/Object-Detection-YOLOv8-Project.git
cd Object-Detection-YOLOv8-Project
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### Detection on Images
```bash
python src/detect.py --source data/images/sample.jpg --weights yolov8n.pt
```

### Real-time Webcam Detection
```bash
python src/detect.py --source 0 --weights yolov8n.pt
```

### Video Detection
```bash
python src/detect.py --source data/videos/sample.mp4 --weights yolov8n.pt
```

## Training Custom Model

### 1. Prepare Dataset
Organize your dataset in YOLO format:
```
datasets/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### 2. Configure Dataset
Edit `configs/dataset.yaml`:
```yaml
path: ./data/datasets/custom
train: images/train
val: images/val
names:
  0: person
  1: car
  2: dog
```

### 3. Train Model
```bash
python src/train.py --data configs/dataset.yaml --epochs 100 --batch 16
```

## Evaluation

```bash
python src/evaluate.py --weights models/best.pt --data configs/dataset.yaml
```

## Model Performance

| Model | Size | mAP@0.5 | Speed (ms) |
|-------|------|---------|------------|
| YOLOv8n | 640 | 37.3 | 1.2 |
| YOLOv8s | 640 | 44.9 | 2.3 |
| YOLOv8m | 640 | 50.2 | 4.5 |

## Configuration

Edit `configs/config.yaml` to customize:
- Model architecture
- Confidence threshold
- IoU threshold
- Input image size
- Device (CPU/GPU)

## API Deployment

Start the REST API server:
```bash
python src/api.py
```

Test the API:
```bash
curl -X POST -F "image=@test.jpg" http://localhost:5000/detect
```

## Advanced Usage

### Batch Processing
```bash
python src/detect.py --source data/images/ --save-txt --save-conf
```

### Export Model
```bash
python src/export.py --weights models/best.pt --format onnx
```

## Supported Object Classes (COCO)

80 classes including: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, and more.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU)
- 8GB RAM minimum
- GPU recommended for training

## Troubleshooting

**Issue**: CUDA out of memory
**Solution**: Reduce batch size or image size

**Issue**: Low FPS on CPU
**Solution**: Use smaller model (yolov8n) or enable GPU

**Issue**: Poor detection accuracy
**Solution**: Train longer, use larger model, or improve dataset quality

## Contributing

Contributions welcome! Please open an issue or submit a PR.

## License

MIT License

## Acknowledgments

- Ultralytics YOLOv8
- PyTorch Team
- COCO Dataset

## Contact

**Hammad Zahid**
- GitHub: [@Hamad-Ansari](https://github.com/Hamad-Ansari)
- Email: mrhammadzahid24@gmail.com

---

⭐ Star this repo if you find it helpful!
