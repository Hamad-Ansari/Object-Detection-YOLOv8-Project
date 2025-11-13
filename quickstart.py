#!/usr/bin/env python3
"""
Quick Start Script for Object Detection
Demonstrates basic usage of the detection system
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from ultralytics import YOLO
import cv2
import torch
import urllib.request
import numpy as np


def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def check_environment():
    """Check system environment"""
    print_header("System Information")
    
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Running on CPU")


def download_sample_image():
    """Download a sample image for testing"""
    print_header("Downloading Sample Image")
    
    url = "https://ultralytics.com/images/bus.jpg"
    output_path = "sample_image.jpg"
    
    try:
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Sample image saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Failed to download image: {e}")
        return None


def run_detection_demo():
    """Run basic detection demo"""
    print_header("Object Detection Demo")
    
    # Download sample image
    image_path = download_sample_image()
    if not image_path:
        print("Cannot proceed without sample image")
        return
    
    # Load model
    print("\nLoading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    print("✓ Model loaded successfully!")
    
    # Run detection
    print(f"\nRunning detection on: {image_path}")
    results = model.predict(
        source=image_path,
        conf=0.25,
        save=True,
        project='outputs',
        name='quickstart'
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("Detection Results")
    print(f"{'='*60}")
    
    detections = results[0].boxes
    print(f"\nTotal objects detected: {len(detections)}")
    
    if len(detections) > 0:
        print(f"\n{'Class':<20} {'Confidence':<12} {'Bounding Box'}")
        print(f"{'-'*60}")
        
        for box in detections:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()
            
            class_name = model.names[cls]
            print(f"{class_name:<20} {conf:<12.2f} {bbox}")
    
    print(f"\n✓ Results saved to: outputs/quickstart/")
    print(f"{'='*60}\n")


def show_available_models():
    """Show available YOLOv8 models"""
    print_header("Available YOLOv8 Models")
    
    models = [
        ("yolov8n.pt", "Nano", "Fastest, least accurate", "3.2M", "6.2ms"),
        ("yolov8s.pt", "Small", "Fast, good accuracy", "11.2M", "9.2ms"),
        ("yolov8m.pt", "Medium", "Balanced", "25.9M", "15.9ms"),
        ("yolov8l.pt", "Large", "Slow, high accuracy", "43.7M", "25.2ms"),
        ("yolov8x.pt", "XLarge", "Slowest, best accuracy", "68.2M", "36.3ms"),
    ]
    
    print(f"{'Model':<15} {'Size':<10} {'Description':<25} {'Params':<10} {'Speed'}")
    print(f"{'-'*80}")
    
    for model, size, desc, params, speed in models:
        print(f"{model:<15} {size:<10} {desc:<25} {params:<10} {speed}")
    
    print(f"\nNote: Speed measured on NVIDIA V100 GPU")


def show_usage_examples():
    """Show usage examples"""
    print_header("Usage Examples")
    
    examples = [
        ("Image Detection", "python src/detect.py --source image.jpg --weights yolov8n.pt"),
        ("Video Detection", "python src/detect.py --source video.mp4 --weights yolov8n.pt"),
        ("Webcam Detection", "python src/detect.py --source 0 --weights yolov8n.pt"),
        ("Train Model", "python src/train.py --data configs/dataset.yaml --epochs 100"),
        ("Evaluate Model", "python src/evaluate.py --weights best.pt --data configs/dataset.yaml"),
        ("Start API", "python src/api.py --weights yolov8n.pt --port 5000"),
    ]
    
    for i, (name, command) in enumerate(examples, 1):
        print(f"{i}. {name}")
        print(f"   {command}\n")


def interactive_menu():
    """Interactive menu for quick start"""
    while True:
        print_header("Object Detection Quick Start Menu")
        
        print("1. Check System Environment")
        print("2. Run Detection Demo")
        print("3. Show Available Models")
        print("4. Show Usage Examples")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            check_environment()
        elif choice == '2':
            run_detection_demo()
        elif choice == '3':
            show_available_models()
        elif choice == '4':
            show_usage_examples()
        elif choice == '5':
            print("\nThank you for using Object Detection System!")
            print("For more information, visit: https://github.com/Hamad-Ansari/Object-Detection-YOLOv8-Project\n")
            break
        else:
            print("\n✗ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")


def main():
    """Main function"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║        Object Detection with YOLOv8 - Quick Start        ║
    ║                                                           ║
    ║              Created by: Hammad Zahid                     ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check if running with arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            check_environment()
            run_detection_demo()
        elif sys.argv[1] == '--help':
            print("Usage:")
            print("  python quickstart.py           # Interactive menu")
            print("  python quickstart.py --demo    # Run demo directly")
            print("  python quickstart.py --help    # Show this help")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Interactive mode
        interactive_menu()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
