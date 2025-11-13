"""
Model Evaluation Script
Evaluates trained model performance with comprehensive metrics
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ModelEvaluator:
    def __init__(self, weights_path, data_yaml):
        """
        Initialize model evaluator
        
        Args:
            weights_path: Path to trained model weights
            data_yaml: Path to dataset configuration
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading model on {self.device}...")
        
        self.model = YOLO(weights_path)
        self.data_yaml = data_yaml
        
        with open(data_yaml, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        print(f"Model loaded: {weights_path}")
        print(f"Dataset: {data_yaml}")
    
    def evaluate(self, conf_threshold=0.001, iou_threshold=0.6, save_dir='runs/eval'):
        """
        Evaluate model on validation set
        
        Args:
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            save_dir: Directory to save results
        """
        print(f"\n{'='*60}")
        print("Starting Model Evaluation")
        print(f"{'='*60}\n")
        
        # Run validation
        results = self.model.val(
            data=self.data_yaml,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            save_json=True,
            save_hybrid=True,
            plots=True,
            project=save_dir,
            name=f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        
        # Extract metrics
        metrics = self.extract_metrics(results)
        
        # Print results
        self.print_metrics(metrics)
        
        # Save metrics
        self.save_metrics(metrics, save_dir)
        
        # Generate plots
        self.generate_plots(results, save_dir)
        
        return metrics
    
    def extract_metrics(self, results):
        """Extract key metrics from validation results"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1_score': 2 * (float(results.box.mp) * float(results.box.mr)) / 
                       (float(results.box.mp) + float(results.box.mr) + 1e-6),
        }
        
        # Per-class metrics
        if hasattr(results.box, 'maps'):
            class_names = self.data_config.get('names', {})
            metrics['per_class'] = {}
            
            for i, map_val in enumerate(results.box.maps):
                class_name = class_names.get(i, f'class_{i}')
                metrics['per_class'][class_name] = {
                    'mAP50-95': float(map_val),
                    'precision': float(results.box.p[i]) if i < len(results.box.p) else 0.0,
                    'recall': float(results.box.r[i]) if i < len(results.box.r) else 0.0
                }
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print metrics in formatted table"""
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"mAP@0.5:      {metrics['mAP50']:.4f}")
        print(f"mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"F1-Score:     {metrics['f1_score']:.4f}")
        print(f"{'='*60}")
        
        if 'per_class' in metrics:
            print(f"\nPER-CLASS METRICS:")
            print(f"{'='*60}")
            print(f"{'Class':<20} {'mAP50-95':<12} {'Precision':<12} {'Recall':<12}")
            print(f"{'-'*60}")
            
            for class_name, class_metrics in metrics['per_class'].items():
                print(f"{class_name:<20} "
                      f"{class_metrics['mAP50-95']:<12.4f} "
                      f"{class_metrics['precision']:<12.4f} "
                      f"{class_metrics['recall']:<12.4f}")
            print(f"{'='*60}\n")
    
    def save_metrics(self, metrics, save_dir):
        """Save metrics to JSON file"""
        save_path = Path(save_dir) / 'metrics.json'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Metrics saved to: {save_path}")
    
    def generate_plots(self, results, save_dir):
        """Generate visualization plots"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix plot
        if hasattr(results, 'confusion_matrix'):
            plt.figure(figsize=(12, 10))
            sns.heatmap(results.confusion_matrix.matrix, 
                       annot=True, fmt='g', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(save_path / 'confusion_matrix.png', dpi=300)
            plt.close()
            print(f"Confusion matrix saved to: {save_path / 'confusion_matrix.png'}")
        
        # Metrics comparison plot
        metrics_data = {
            'Precision': float(results.box.mp),
            'Recall': float(results.box.mr),
            'mAP@0.5': float(results.box.map50),
            'mAP@0.5:0.95': float(results.box.map)
        }
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_data.keys(), metrics_data.values(), 
                      color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        plt.ylabel('Score')
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path / 'metrics_comparison.png', dpi=300)
        plt.close()
        print(f"Metrics comparison saved to: {save_path / 'metrics_comparison.png'}")
    
    def benchmark_speed(self, img_size=640, batch_size=1, iterations=100):
        """
        Benchmark inference speed
        
        Args:
            img_size: Input image size
            batch_size: Batch size for inference
            iterations: Number of iterations for benchmarking
        """
        print(f"\n{'='*60}")
        print("SPEED BENCHMARK")
        print(f"{'='*60}")
        print(f"Image Size: {img_size}")
        print(f"Batch Size: {batch_size}")
        print(f"Iterations: {iterations}")
        print(f"Device: {self.device}")
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
        
        # Warmup
        print("\nWarming up...")
        for _ in range(10):
            _ = self.model.predict(dummy_input, verbose=False)
        
        # Benchmark
        print("Benchmarking...")
        import time
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        for _ in range(iterations):
            _ = self.model.predict(dummy_input, verbose=False)
            if self.device == 'cuda':
                torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / iterations
        fps = 1 / avg_time
        
        print(f"\n{'='*60}")
        print(f"Total Time:    {total_time:.2f}s")
        print(f"Average Time:  {avg_time*1000:.2f}ms")
        print(f"FPS:           {fps:.2f}")
        print(f"{'='*60}\n")
        
        return {
            'total_time': total_time,
            'avg_time_ms': avg_time * 1000,
            'fps': fps
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8 Model')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset YAML')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6,
                        help='IoU threshold')
    parser.add_argument('--save-dir', type=str, default='runs/eval',
                        help='Directory to save results')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run speed benchmark')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for benchmark')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.weights, args.data)
    
    # Run evaluation
    metrics = evaluator.evaluate(
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save_dir=args.save_dir
    )
    
    # Run benchmark if requested
    if args.benchmark:
        speed_metrics = evaluator.benchmark_speed(img_size=args.img_size)


if __name__ == '__main__':
    main()
