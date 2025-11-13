# Setup Guide - Object Detection with YOLOv8

Complete setup instructions for the object detection project.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Dataset Preparation](#dataset-preparation)
4. [Quick Start](#quick-start)
5. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum
- **Storage**: 10GB free space
- **CPU**: Multi-core processor

### Recommended for Training
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **CUDA**: 11.8 or higher
- **cuDNN**: Compatible version
- **RAM**: 16GB+
- **Storage**: 50GB+ SSD

### Check Your System
```bash
# Check Python version
python --version

# Check CUDA availability (if GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU info
nvidia-smi
```

---

## Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/Hamad-Ansari/Object-Detection-YOLOv8-Project.git
cd Object-Detection-YOLOv8-Project
```

### Step 2: Create Virtual Environment

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

**For CPU only:**
```bash
pip install -r requirements.txt
```

**For GPU (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**For GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "from ultralytics import YOLO; print('Installation successful!')"
```

---

## Dataset Preparation

### Option 1: Use Pre-trained COCO Model
No dataset needed! Use pre-trained weights:
```bash
# Download happens automatically on first use
python src/detect.py --source data/images/sample.jpg --weights yolov8n.pt
```

### Option 2: Prepare Custom Dataset

#### Dataset Structure
```
data/datasets/custom/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/
        ├── image1.txt
        └── ...
```

#### Label Format (YOLO)
Each `.txt` file contains one line per object:
```
class_id center_x center_y width height
```

Example (`image1.txt`):
```
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.2
```

Values are normalized (0-1):
- `class_id`: Integer class index
- `center_x, center_y`: Object center coordinates
- `width, height`: Object dimensions

#### Create Dataset Configuration
Edit `configs/dataset.yaml`:
```yaml
path: ./data/datasets/custom
train: images/train
val: images/val

nc: 3  # number of classes
names:
  0: class1
  1: class2
  2: class3
```

#### Validate Dataset
```bash
python -c "from src.utils import DatasetUtils; DatasetUtils.validate_dataset_structure('data/datasets/custom')"
```

### Option 3: Use Public Datasets

**COCO Dataset:**
```bash
# Download COCO (118K images, ~18GB)
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

**Pascal VOC:**
```bash
# Download VOC
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```

**Convert to YOLO format:**
```python
from ultralytics.data.converter import convert_coco
convert_coco(labels_dir='path/to/coco/annotations')
```

---

## Quick Start

### 1. Image Detection
```bash
# Using pre-trained model
python src/detect.py --source data/images/sample.jpg --weights yolov8n.pt

# Save results
python src/detect.py --source data/images/sample.jpg --weights yolov8n.pt --save outputs/result.jpg

# Adjust confidence threshold
python src/detect.py --source data/images/sample.jpg --weights yolov8n.pt --conf 0.5
```

### 2. Video Detection
```bash
python src/detect.py --source data/videos/sample.mp4 --weights yolov8n.pt --save outputs/result.mp4
```

### 3. Webcam Detection
```bash
python src/detect.py --source 0 --weights yolov8n.pt
```

### 4. Batch Processing
```bash
python src/detect.py --source data/images/ --weights yolov8n.pt --batch
```

### 5. Train Custom Model
```bash
python src/train.py \
    --data configs/dataset.yaml \
    --model n \
    --epochs 100 \
    --batch 16 \
    --img-size 640
```

### 6. Evaluate Model
```bash
python src/evaluate.py \
    --weights runs/train/exp/weights/best.pt \
    --data configs/dataset.yaml
```

### 7. Start API Server
```bash
python src/api.py --weights yolov8n.pt --host 0.0.0.0 --port 5000
```

Test API:
```bash
curl -X POST -F "file=@test.jpg" http://localhost:5000/detect
```

---

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution:**
```bash
# Reduce batch size
python src/train.py --data configs/dataset.yaml --batch 8

# Use smaller model
python src/detect.py --source image.jpg --weights yolov8n.pt

# Reduce image size
python src/train.py --data configs/dataset.yaml --img-size 416
```

### Issue: Slow CPU Inference
**Solution:**
```bash
# Use nano model
python src/detect.py --source image.jpg --weights yolov8n.pt

# Reduce image size
python src/detect.py --source image.jpg --weights yolov8n.pt --img-size 320
```

### Issue: Import Errors
**Solution:**
```bash
# Reinstall dependencies
pip uninstall ultralytics torch torchvision
pip install --upgrade ultralytics torch torchvision

# Clear cache
pip cache purge
```

### Issue: Permission Denied
**Solution:**
```bash
# On Linux/macOS
chmod +x src/*.py

# Run with sudo if needed
sudo python src/detect.py --source image.jpg
```

### Issue: Model Download Fails
**Solution:**
```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Place in project root or specify path
python src/detect.py --source image.jpg --weights ./yolov8n.pt
```

### Issue: Dataset Not Found
**Solution:**
```bash
# Check paths in dataset.yaml
cat configs/dataset.yaml

# Use absolute paths
path: /absolute/path/to/dataset
```

### Issue: Low Detection Accuracy
**Solutions:**
1. **Lower confidence threshold:**
   ```bash
   python src/detect.py --source image.jpg --conf 0.1
   ```

2. **Use larger model:**
   ```bash
   python src/detect.py --source image.jpg --weights yolov8x.pt
   ```

3. **Train longer:**
   ```bash
   python src/train.py --data configs/dataset.yaml --epochs 300
   ```

4. **Improve dataset quality:**
   - Add more training images
   - Balance class distribution
   - Improve annotation quality

---

## Additional Resources

### Documentation
- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [PyTorch Docs](https://pytorch.org/docs/)
- [Project Wiki](https://github.com/Hamad-Ansari/Object-Detection-YOLOv8-Project/wiki)

### Tutorials
- [Training Custom Models](docs/training.md)
- [API Usage](docs/api.md)
- [Performance Optimization](docs/optimization.md)

### Community
- [GitHub Issues](https://github.com/Hamad-Ansari/Object-Detection-YOLOv8-Project/issues)
- [Discussions](https://github.com/Hamad-Ansari/Object-Detection-YOLOv8-Project/discussions)

---

## Next Steps

1. ✅ Complete installation
2. ✅ Run detection on sample images
3. ✅ Prepare your dataset
4. ✅ Train custom model
5. ✅ Deploy API server

**Need help?** Open an issue on GitHub!
