"""
Utility Functions for Object Detection Project
Helper functions for data processing, visualization, and model operations
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import torch


class DatasetUtils:
    """Utilities for dataset management"""
    
    @staticmethod
    def create_dataset_yaml(dataset_path, class_names, train_path='images/train', 
                           val_path='images/val', test_path=None):
        """
        Create dataset YAML configuration file
        
        Args:
            dataset_path: Root path of dataset
            class_names: List of class names
            train_path: Relative path to training images
            val_path: Relative path to validation images
            test_path: Relative path to test images (optional)
        """
        config = {
            'path': str(Path(dataset_path).absolute()),
            'train': train_path,
            'val': val_path,
            'nc': len(class_names),
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        if test_path:
            config['test'] = test_path
        
        yaml_path = Path(dataset_path) / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Dataset YAML created: {yaml_path}")
        return yaml_path
    
    @staticmethod
    def validate_dataset_structure(dataset_path):
        """
        Validate YOLO dataset structure
        
        Args:
            dataset_path: Path to dataset root
        """
        dataset_path = Path(dataset_path)
        required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
        
        print("Validating dataset structure...")
        all_valid = True
        
        for dir_path in required_dirs:
            full_path = dataset_path / dir_path
            if full_path.exists():
                file_count = len(list(full_path.iterdir()))
                print(f"✓ {dir_path}: {file_count} files")
            else:
                print(f"✗ {dir_path}: NOT FOUND")
                all_valid = False
        
        if all_valid:
            print("\n✓ Dataset structure is valid!")
        else:
            print("\n✗ Dataset structure is invalid!")
        
        return all_valid
    
    @staticmethod
    def split_dataset(image_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Split dataset into train/val/test sets
        
        Args:
            image_dir: Directory containing images and labels
            output_dir: Output directory for split dataset
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
        """
        import shutil
        import random
        
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        
        # Get all image files
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        random.shuffle(image_files)
        
        total = len(image_files)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        
        splits = {
            'train': image_files[:train_count],
            'val': image_files[train_count:train_count + val_count],
            'test': image_files[train_count + val_count:]
        }
        
        for split_name, files in splits.items():
            # Create directories
            img_dir = output_dir / 'images' / split_name
            lbl_dir = output_dir / 'labels' / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            for img_file in files:
                # Copy image
                shutil.copy(img_file, img_dir / img_file.name)
                
                # Copy label
                lbl_file = img_file.with_suffix('.txt')
                if lbl_file.exists():
                    shutil.copy(lbl_file, lbl_dir / lbl_file.name)
            
            print(f"{split_name}: {len(files)} images")
        
        print(f"\nDataset split complete! Output: {output_dir}")


class VisualizationUtils:
    """Utilities for visualization"""
    
    @staticmethod
    def draw_boxes(image, boxes, labels, scores, class_names, color=(0, 255, 0)):
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image (numpy array)
            boxes: List of bounding boxes [x1, y1, x2, y2]
            labels: List of class labels
            scores: List of confidence scores
            class_names: Dictionary of class names
            color: Box color (BGR)
        """
        img = image.copy()
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            class_name = class_names.get(label, f'Class {label}')
            text = f'{class_name}: {score:.2f}'
            
            # Text background
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(img, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Text
            cv2.putText(img, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img
    
    @staticmethod
    def plot_training_history(results_csv, save_path=None):
        """
        Plot training history from results CSV
        
        Args:
            results_csv: Path to results.csv file
            save_path: Path to save plot
        """
        import pandas as pd
        
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss')
        axes[0, 0].plot(df['epoch'], df['train/cls_loss'], label='Class Loss')
        axes[0, 0].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # mAP plots
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].set_title('Mean Average Precision')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision/Recall
        axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
        axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision and Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        if 'lr/pg0' in df.columns:
            axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def create_detection_grid(images, predictions, class_names, grid_size=(3, 3)):
        """
        Create grid of detection results
        
        Args:
            images: List of images
            predictions: List of predictions
            class_names: Dictionary of class names
            grid_size: Grid dimensions (rows, cols)
        """
        rows, cols = grid_size
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx, (img, pred) in enumerate(zip(images[:rows*cols], predictions[:rows*cols])):
            if len(pred) > 0:
                img_with_boxes = VisualizationUtils.draw_boxes(
                    img, pred['boxes'], pred['labels'], 
                    pred['scores'], class_names
                )
            else:
                img_with_boxes = img
            
            axes[idx].imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
            axes[idx].axis('off')
            axes[idx].set_title(f'Image {idx+1}')
        
        plt.tight_layout()
        plt.show()


class ModelUtils:
    """Utilities for model operations"""
    
    @staticmethod
    def export_model(model_path, export_format='onnx', img_size=640):
        """
        Export model to different formats
        
        Args:
            model_path: Path to model weights
            export_format: Export format (onnx, torchscript, coreml, etc.)
            img_size: Input image size
        """
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        export_path = model.export(format=export_format, imgsz=img_size)
        
        print(f"Model exported to: {export_path}")
        return export_path
    
    @staticmethod
    def compare_models(model_paths, data_yaml, save_path=None):
        """
        Compare multiple models
        
        Args:
            model_paths: List of model weight paths
            data_yaml: Dataset configuration
            save_path: Path to save comparison plot
        """
        from ultralytics import YOLO
        
        results = []
        
        for model_path in model_paths:
            print(f"\nEvaluating {model_path}...")
            model = YOLO(model_path)
            metrics = model.val(data=data_yaml, verbose=False)
            
            results.append({
                'model': Path(model_path).stem,
                'mAP50': float(metrics.box.map50),
                'mAP50-95': float(metrics.box.map),
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr)
            })
        
        # Create comparison plot
        import pandas as pd
        df = pd.DataFrame(results)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df))
        width = 0.2
        
        ax.bar(x - 1.5*width, df['mAP50'], width, label='mAP@0.5')
        ax.bar(x - 0.5*width, df['mAP50-95'], width, label='mAP@0.5:0.95')
        ax.bar(x + 0.5*width, df['precision'], width, label='Precision')
        ax.bar(x + 1.5*width, df['recall'], width, label='Recall')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(df['model'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nComparison plot saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return df
    
    @staticmethod
    def get_model_info(model_path):
        """
        Get model information
        
        Args:
            model_path: Path to model weights
        """
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        
        info = {
            'model_path': model_path,
            'task': model.task,
            'model_name': model.model_name if hasattr(model, 'model_name') else 'Unknown',
            'parameters': sum(p.numel() for p in model.model.parameters()),
            'layers': len(list(model.model.modules())),
        }
        
        print(f"\n{'='*60}")
        print("MODEL INFORMATION")
        print(f"{'='*60}")
        for key, value in info.items():
            print(f"{key}: {value}")
        print(f"{'='*60}\n")
        
        return info


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config, config_path):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_directories(base_dir='outputs'):
    """Create necessary directories"""
    dirs = [
        f'{base_dir}/images',
        f'{base_dir}/videos',
        f'{base_dir}/models',
        f'{base_dir}/logs',
        f'{base_dir}/plots'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"Directories created in: {base_dir}")
