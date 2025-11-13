# Usage Examples

Comprehensive examples for using the Object Detection system.

## Table of Contents
1. [Basic Detection](#basic-detection)
2. [Advanced Detection](#advanced-detection)
3. [Training Examples](#training-examples)
4. [API Usage](#api-usage)
5. [Batch Processing](#batch-processing)
6. [Custom Scenarios](#custom-scenarios)

---

## Basic Detection

### Detect Objects in Single Image
```bash
python src/detect.py --source data/images/sample.jpg --weights yolov8n.pt
```

### Detect with Custom Confidence
```bash
python src/detect.py \
    --source data/images/sample.jpg \
    --weights yolov8n.pt \
    --conf 0.5 \
    --iou 0.5
```

### Save Detection Results
```bash
python src/detect.py \
    --source data/images/sample.jpg \
    --weights yolov8n.pt \
    --save outputs/result.jpg \
    --show
```

---

## Advanced Detection

### Real-time Webcam Detection
```bash
# Default webcam (camera 0)
python src/detect.py --source 0 --weights yolov8n.pt

# External webcam (camera 1)
python src/detect.py --source 1 --weights yolov8n.pt
```

### Video File Detection
```bash
python src/detect.py \
    --source data/videos/traffic.mp4 \
    --weights yolov8n.pt \
    --save outputs/traffic_detected.mp4
```

### Batch Image Processing
```bash
python src/detect.py \
    --source data/images/ \
    --weights yolov8n.pt \
    --batch \
    --save outputs/batch_results/
```

### Using Different Model Sizes
```bash
# Nano (fastest, least accurate)
python src/detect.py --source image.jpg --weights yolov8n.pt

# Small
python src/detect.py --source image.jpg --weights yolov8s.pt

# Medium
python src/detect.py --source image.jpg --weights yolov8m.pt

# Large
python src/detect.py --source image.jpg --weights yolov8l.pt

# XLarge (slowest, most accurate)
python src/detect.py --source image.jpg --weights yolov8x.pt
```

---

## Training Examples

### Train on Custom Dataset (Quick Start)
```bash
python src/train.py \
    --data configs/dataset.yaml \
    --model n \
    --epochs 50 \
    --batch 16
```

### Train with Full Configuration
```bash
python src/train.py \
    --data configs/dataset.yaml \
    --model s \
    --epochs 100 \
    --batch 32 \
    --img-size 640 \
    --patience 20 \
    --save-dir runs/train/custom_model
```

### Resume Training
```bash
python src/train.py \
    --data configs/dataset.yaml \
    --model runs/train/exp/weights/last.pt \
    --epochs 100 \
    --batch 16
```

### Transfer Learning (Fine-tuning)
```bash
# Start from pre-trained COCO weights
python src/train.py \
    --data configs/custom_dataset.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --batch 16 \
    --img-size 640
```

---

## Evaluation Examples

### Basic Evaluation
```bash
python src/evaluate.py \
    --weights runs/train/exp/weights/best.pt \
    --data configs/dataset.yaml
```

### Evaluation with Benchmarking
```bash
python src/evaluate.py \
    --weights runs/train/exp/weights/best.pt \
    --data configs/dataset.yaml \
    --benchmark \
    --img-size 640
```

### Compare Multiple Models
```python
from src.utils import ModelUtils

model_paths = [
    'yolov8n.pt',
    'yolov8s.pt',
    'yolov8m.pt'
]

ModelUtils.compare_models(
    model_paths=model_paths,
    data_yaml='configs/dataset.yaml',
    save_path='outputs/model_comparison.png'
)
```

---

## API Usage

### Start API Server
```bash
python src/api.py --weights yolov8n.pt --host 0.0.0.0 --port 5000
```

### Test API with cURL

**Single Image Detection:**
```bash
curl -X POST \
    -F "file=@test.jpg" \
    -F "conf=0.25" \
    -F "iou=0.45" \
    http://localhost:5000/detect
```

**Batch Detection:**
```bash
curl -X POST \
    -F "files=@image1.jpg" \
    -F "files=@image2.jpg" \
    -F "files=@image3.jpg" \
    http://localhost:5000/detect_batch
```

**Detection from URL:**
```bash
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"url": "https://example.com/image.jpg", "conf": 0.3}' \
    http://localhost:5000/detect_url
```

### Python API Client
```python
import requests

# Single image detection
url = 'http://localhost:5000/detect'
files = {'file': open('test.jpg', 'rb')}
data = {'conf': 0.25, 'return_image': 'true'}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Detected {result['count']} objects")
for detection in result['detections']:
    print(f"  - {detection['class_name']}: {detection['confidence']:.2f}")
```

### JavaScript API Client
```javascript
// Single image detection
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('conf', '0.25');

fetch('http://localhost:5000/detect', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log(`Detected ${data.count} objects`);
    data.detections.forEach(det => {
        console.log(`${det.class_name}: ${det.confidence}`);
    });
});
```

---

## Batch Processing

### Process Directory of Images
```bash
python src/detect.py \
    --source data/images/ \
    --weights yolov8n.pt \
    --batch \
    --save outputs/batch/
```

### Process with Python Script
```python
from pathlib import Path
from src.detect import ObjectDetector

detector = ObjectDetector(weights='yolov8n.pt')

image_dir = Path('data/images')
output_dir = Path('outputs/batch')

detector.detect_batch(image_dir, output_dir)
```

### Parallel Batch Processing
```python
from concurrent.futures import ThreadPoolExecutor
from src.detect import ObjectDetector
from pathlib import Path

def process_image(image_path):
    detector = ObjectDetector(weights='yolov8n.pt')
    return detector.detect_image(image_path)

image_files = list(Path('data/images').glob('*.jpg'))

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_image, image_files))

print(f"Processed {len(results)} images")
```

---

## Custom Scenarios

### Detect Specific Classes Only
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Detect only persons and cars (class 0 and 2)
results = model.predict(
    source='image.jpg',
    classes=[0, 2],
    conf=0.25
)
```

### Custom Visualization
```python
import cv2
from src.utils import VisualizationUtils

image = cv2.imread('image.jpg')
results = model.predict(image)

# Extract detections
boxes = results[0].boxes.xyxy.cpu().numpy()
labels = results[0].boxes.cls.cpu().numpy()
scores = results[0].boxes.conf.cpu().numpy()

# Custom drawing
annotated = VisualizationUtils.draw_boxes(
    image, boxes, labels, scores,
    class_names=model.names,
    color=(0, 255, 0)
)

cv2.imwrite('output.jpg', annotated)
```

### Track Objects in Video
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Track objects with ByteTrack
results = model.track(
    source='video.mp4',
    conf=0.3,
    iou=0.5,
    tracker='bytetrack.yaml',
    save=True
)
```

### Export Model for Deployment
```python
from ultralytics import YOLO

model = YOLO('runs/train/exp/weights/best.pt')

# Export to ONNX
model.export(format='onnx', imgsz=640)

# Export to TensorRT (requires TensorRT)
model.export(format='engine', imgsz=640, half=True)

# Export to CoreML (for iOS)
model.export(format='coreml', imgsz=640)

# Export to TFLite (for mobile)
model.export(format='tflite', imgsz=640)
```

### Custom Training Loop
```python
from ultralytics import YOLO
import torch

model = YOLO('yolov8n.yaml')  # Build from scratch
model.load('yolov8n.pt')      # Load pretrained weights

# Custom training with callbacks
def on_train_epoch_end(trainer):
    print(f"Epoch {trainer.epoch} completed")

results = model.train(
    data='configs/dataset.yaml',
    epochs=100,
    batch=16,
    callbacks={'on_train_epoch_end': on_train_epoch_end}
)
```

### Real-time Detection with Custom Processing
```python
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model.predict(frame, verbose=False)
    
    # Custom processing
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        # Custom logic
        if model.names[cls] == 'person' and conf > 0.5:
            # Trigger alert or action
            print(f"Person detected with confidence {conf:.2f}")
    
    # Display
    annotated = results[0].plot()
    cv2.imshow('Detection', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Performance Optimization

### GPU Optimization
```python
# Use half precision (FP16)
model = YOLO('yolov8n.pt')
results = model.predict('image.jpg', half=True)

# Batch processing for speed
results = model.predict(
    source='data/images/',
    batch=32,  # Process 32 images at once
    stream=True  # Memory efficient
)
```

### CPU Optimization
```bash
# Use smaller model
python src/detect.py --source image.jpg --weights yolov8n.pt

# Reduce image size
python src/detect.py --source image.jpg --weights yolov8n.pt --img-size 320

# Use OpenVINO (Intel CPUs)
model.export(format='openvino')
```

---

## Integration Examples

### Flask Web App
```python
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('yolov8n.pt')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    results = model.predict(file)
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
```

### Streamlit Dashboard
```python
import streamlit as st
from ultralytics import YOLO
import cv2

st.title('Object Detection Dashboard')

model = YOLO('yolov8n.pt')
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png'])

if uploaded_file:
    results = model.predict(uploaded_file)
    st.image(results[0].plot(), caption='Detection Results')
```

---

For more examples, check the [notebooks/demo.ipynb](notebooks/demo.ipynb) file!
