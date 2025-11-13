"""
REST API for Object Detection
Flask-based API for serving object detection predictions
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import torch
import time
import os


app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs/api'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# Global model variable
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(weights_path='yolov8n.pt'):
    """Load YOLOv8 model"""
    global model
    print(f"Loading model: {weights_path} on {device}")
    model = YOLO(weights_path)
    print("Model loaded successfully!")


def encode_image(image):
    """Encode image to base64"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64


def decode_image(img_base64):
    """Decode base64 image"""
    img_data = base64.b64decode(img_base64)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Object Detection API',
        'version': '1.0',
        'endpoints': {
            '/detect': 'POST - Detect objects in image',
            '/detect_batch': 'POST - Batch detection',
            '/health': 'GET - Health check',
            '/model_info': 'GET - Model information'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': device,
        'timestamp': time.time()
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_name': 'YOLOv8',
        'device': device,
        'task': model.task,
        'classes': len(model.names),
        'class_names': model.names
    })


@app.route('/detect', methods=['POST'])
def detect():
    """
    Detect objects in uploaded image
    
    Request:
        - file: Image file
        - conf: Confidence threshold (optional, default: 0.25)
        - iou: IoU threshold (optional, default: 0.45)
        - return_image: Return annotated image (optional, default: true)
    
    Response:
        - detections: List of detected objects
        - image: Base64 encoded annotated image (if return_image=true)
        - inference_time: Inference time in milliseconds
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Get parameters
        conf_threshold = float(request.form.get('conf', 0.25))
        iou_threshold = float(request.form.get('iou', 0.45))
        return_image = request.form.get('return_image', 'true').lower() == 'true'
        
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Perform detection
        start_time = time.time()
        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            verbose=False
        )[0]
        inference_time = (time.time() - start_time) * 1000
        
        # Extract detections
        detections = []
        boxes = results.boxes
        
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            
            detections.append({
                'class_id': cls,
                'class_name': model.names[cls],
                'confidence': conf,
                'bbox': {
                    'x1': float(box[0]),
                    'y1': float(box[1]),
                    'x2': float(box[2]),
                    'y2': float(box[3])
                }
            })
        
        response = {
            'success': True,
            'detections': detections,
            'count': len(detections),
            'inference_time_ms': round(inference_time, 2),
            'image_size': {
                'width': image.shape[1],
                'height': image.shape[0]
            }
        }
        
        # Add annotated image if requested
        if return_image:
            annotated_image = results.plot()
            response['image'] = encode_image(annotated_image)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect_batch', methods=['POST'])
def detect_batch():
    """
    Batch detection on multiple images
    
    Request:
        - files: Multiple image files
        - conf: Confidence threshold (optional)
        - iou: IoU threshold (optional)
    
    Response:
        - results: List of detection results for each image
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if len(files) == 0:
        return jsonify({'error': 'No files provided'}), 400
    
    try:
        conf_threshold = float(request.form.get('conf', 0.25))
        iou_threshold = float(request.form.get('iou', 0.45))
        
        results_list = []
        
        for file in files:
            if not allowed_file(file.filename):
                continue
            
            # Read image
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                continue
            
            # Perform detection
            start_time = time.time()
            results = model.predict(
                source=image,
                conf=conf_threshold,
                iou=iou_threshold,
                device=device,
                verbose=False
            )[0]
            inference_time = (time.time() - start_time) * 1000
            
            # Extract detections
            detections = []
            boxes = results.boxes
            
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                detections.append({
                    'class_id': cls,
                    'class_name': model.names[cls],
                    'confidence': conf,
                    'bbox': {
                        'x1': float(box[0]),
                        'y1': float(box[1]),
                        'x2': float(box[2]),
                        'y2': float(box[3])
                    }
                })
            
            results_list.append({
                'filename': file.filename,
                'detections': detections,
                'count': len(detections),
                'inference_time_ms': round(inference_time, 2)
            })
        
        return jsonify({
            'success': True,
            'total_images': len(results_list),
            'results': results_list
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect_url', methods=['POST'])
def detect_url():
    """
    Detect objects from image URL
    
    Request:
        - url: Image URL
        - conf: Confidence threshold (optional)
        - iou: IoU threshold (optional)
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    
    if 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        import requests
        
        url = data['url']
        conf_threshold = float(data.get('conf', 0.25))
        iou_threshold = float(data.get('iou', 0.45))
        
        # Download image
        response = requests.get(url, timeout=10)
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to load image from URL'}), 400
        
        # Perform detection
        start_time = time.time()
        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            verbose=False
        )[0]
        inference_time = (time.time() - start_time) * 1000
        
        # Extract detections
        detections = []
        boxes = results.boxes
        
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            
            detections.append({
                'class_id': cls,
                'class_name': model.names[cls],
                'confidence': conf,
                'bbox': {
                    'x1': float(box[0]),
                    'y1': float(box[1]),
                    'x2': float(box[2]),
                    'y2': float(box[3])
                }
            })
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections),
            'inference_time_ms': round(inference_time, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Object Detection API Server')
    parser.add_argument('--weights', type=str, default='yolov8n.pt',
                        help='Model weights path')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port number')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load model
    load_model(args.weights)
    
    # Start server
    print(f"\n{'='*60}")
    print("Object Detection API Server")
    print(f"{'='*60}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Model: {args.weights}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
