"""
Object Detection Model Training Script
Trains YOLOv8 model on custom datasets with comprehensive logging and evaluation
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime
import json


class ObjectDetectionTrainer:
    def __init__(self, config_path='configs/config.yaml'):
        """Initialize trainer with configuration"""
        self.config = self.load_config(config_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def train(self, data_yaml, model_size='n', epochs=100, batch_size=16, 
              img_size=640, patience=50, save_dir='runs/train'):
        """
        Train YOLOv8 model
        
        Args:
            data_yaml: Path to dataset configuration
            model_size: Model size (n, s, m, l, x)
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
            patience: Early stopping patience
            save_dir: Directory to save results
        """
        
        # Initialize model
        model_name = f'yolov8{model_size}.pt'
        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Dataset: {data_yaml}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Image Size: {img_size}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Load model
        model = YOLO(model_name)
        
        # Training parameters
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=patience,
            save=True,
            device=self.device,
            project=save_dir,
            name=f'exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=True,
            close_mosaic=10,
            resume=False,
            amp=True,
            fraction=1.0,
            profile=False,
            # Data augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0
        )
        
        # Save training summary
        self.save_training_summary(results, save_dir)
        
        print(f"\n{'='*60}")
        print("Training completed successfully!")
        print(f"Best model saved to: {save_dir}")
        print(f"{'='*60}\n")
        
        return results
    
    def save_training_summary(self, results, save_dir):
        """Save training summary to JSON"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'results': str(results)
        }
        
        summary_path = Path(save_dir) / 'training_summary.json'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"Training summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 Object Detection Model')
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='n', 
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, 
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, 
                        help='Input image size')
    parser.add_argument('--patience', type=int, default=50, 
                        help='Early stopping patience')
    parser.add_argument('--save-dir', type=str, default='runs/train', 
                        help='Directory to save results')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                        help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ObjectDetectionTrainer(config_path=args.config)
    
    # Train model
    trainer.train(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        patience=args.patience,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
