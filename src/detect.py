"""
Object Detection Inference Script
Performs detection on images, videos, and real-time webcam feeds
"""

import argparse
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
import time
import numpy as np


class ObjectDetector:
    def __init__(self, weights='yolov8n.pt', conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize object detector
        
        Args:
            weights: Path to model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading model on {self.device}...")
        
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        print(f"Model loaded successfully!")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"IoU threshold: {iou_threshold}")
    
    def detect_image(self, image_path, save_path=None, show=False):
        """
        Detect objects in a single image
        
        Args:
            image_path: Path to input image
            save_path: Path to save output image
            show: Whether to display the result
        """
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            save=save_path is not None,
            show=show
        )
        
        return results
    
    def detect_video(self, video_path, save_path=None, show=False):
        """
        Detect objects in video file
        
        Args:
            video_path: Path to input video
            save_path: Path to save output video
            show: Whether to display the result
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Info:")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total Frames: {total_frames}")
        
        # Initialize video writer
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            # Draw results on frame
            annotated_frame = results[0].plot()
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time
            
            # Add FPS text
            cv2.putText(annotated_frame, f'FPS: {current_fps:.1f}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save frame
            if save_path:
                out.write(annotated_frame)
            
            # Display frame
            if show:
                cv2.imshow('Object Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({current_fps:.1f} FPS)")
        
        cap.release()
        if save_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Average FPS: {frame_count / elapsed_time:.2f}")
    
    def detect_webcam(self, camera_id=0):
        """
        Real-time detection from webcam
        
        Args:
            camera_id: Camera device ID (0 for default)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\nStarting webcam detection...")
        print("Press 'q' to quit")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            
            # Add FPS and detection count
            detections = len(results[0].boxes)
            cv2.putText(annotated_frame, f'FPS: {fps:.1f}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Detections: {detections}', 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Webcam Object Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nAverage FPS: {fps:.2f}")
    
    def detect_batch(self, input_dir, output_dir):
        """
        Batch detection on directory of images
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"\nProcessing {len(image_files)} images...")
        
        for i, image_file in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            results = self.model.predict(
                source=str(image_file),
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                save=True,
                project=str(output_path),
                name='',
                exist_ok=True
            )
        
        print(f"\nBatch processing complete! Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection')
    parser.add_argument('--source', type=str, required=True,
                        help='Input source (image path, video path, 0 for webcam, or directory)')
    parser.add_argument('--weights', type=str, default='yolov8n.pt',
                        help='Model weights path')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save output')
    parser.add_argument('--show', action='store_true',
                        help='Display results')
    parser.add_argument('--batch', action='store_true',
                        help='Batch processing mode for directories')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ObjectDetector(
        weights=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Determine source type and run detection
    if args.source == '0' or args.source.isdigit():
        # Webcam
        detector.detect_webcam(int(args.source))
    elif Path(args.source).is_dir() or args.batch:
        # Batch processing
        output_dir = args.save if args.save else 'outputs/batch'
        detector.detect_batch(args.source, output_dir)
    elif Path(args.source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video
        detector.detect_video(args.source, args.save, args.show)
    else:
        # Image
        detector.detect_image(args.source, args.save, args.show)


if __name__ == '__main__':
    main()
