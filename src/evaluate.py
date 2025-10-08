"""
Evaluation script for video anomaly detection.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import yaml
import json
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.causal_model import CausalAnomalyDetector
from data.dataset import get_dataloader
from utils.metrics import AnomalyMetrics, compute_auc_roc
from utils.visualization import plot_anomaly_scores, save_visualization


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate Causal Video Anomaly Detection Model'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to test video dataset directory'
    )
    parser.add_argument(
        '--annotation_file',
        type=str,
        required=True,
        help='Path to annotation file with ground truth labels'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (optional, will use from checkpoint dir)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for evaluation'
    )
    
    return parser.parse_args()


def load_config(config_path: str = None, checkpoint_dir: str = None) -> dict:
    """Load configuration."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Try to load from checkpoint directory
    if checkpoint_dir:
        config_file = os.path.join(checkpoint_dir, 'config.yaml')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
    
    # Default config
    return {
        'model': {
            'input_channels': 3,
            'latent_dim': 512,
            'num_temporal_layers': 2
        },
        'data': {
            'sequence_length': 16,
            'frame_size': [256, 256],
            'num_workers': 4
        }
    }


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    visualize: bool = False
) -> dict:
    """
    Evaluate the model on test data.
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    all_scores = []
    all_labels = []
    all_reconstructions = []
    all_originals = []
    
    print('Evaluating model...')
    
    with torch.no_grad():
        for batch_idx, (frames, labels) in enumerate(tqdm(dataloader)):
            frames = frames.to(device)
            
            # Forward pass
            outputs = model(frames)
            
            # Collect scores and labels
            all_scores.append(outputs['anomaly_scores'].cpu().numpy())
            all_labels.append(labels.numpy())
            
            # Collect samples for visualization
            if visualize and batch_idx < 5:
                all_reconstructions.append(outputs['recon'].cpu().numpy())
                all_originals.append(frames.cpu().numpy())
    
    # Concatenate all results
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Flatten to frame-level
    scores_flat = all_scores.flatten()
    labels_flat = all_labels.flatten()
    
    # Compute metrics
    metrics_calculator = AnomalyMetrics()
    metrics = metrics_calculator.compute(scores_flat, labels_flat)
    
    # Additional metrics
    metrics['auc_roc'] = compute_auc_roc(scores_flat, labels_flat)
    
    # Print results
    print('\n' + '='*50)
    print('Evaluation Results:')
    print('='*50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f'{key:20s}: {value:.4f}')
        else:
            print(f'{key:20s}: {value}')
    print('='*50)
    
    # Save results
    results = {
        'metrics': metrics,
        'scores': scores_flat.tolist(),
        'labels': labels_flat.tolist()
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualizations
    if visualize and all_reconstructions:
        print('\nGenerating visualizations...')
        
        # Plot anomaly scores
        plot_anomaly_scores(
            scores_flat,
            labels_flat,
            save_path=str(output_path / 'anomaly_scores.png')
        )
        
        # Save reconstruction examples
        for i, (orig, recon) in enumerate(zip(all_originals, all_reconstructions)):
            save_visualization(
                orig[0],  # First sample in batch
                recon[0],
                save_path=str(output_path / f'reconstruction_{i}.png')
            )
    
    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Load configuration
    checkpoint_dir = os.path.dirname(args.checkpoint)
    config = load_config(args.config, checkpoint_dir)
    
    # Create model
    model = CausalAnomalyDetector(
        input_channels=config['model']['input_channels'],
        latent_dim=config['model']['latent_dim'],
        num_temporal_layers=config['model']['num_temporal_layers']
    ).to(device)
    
    # Load checkpoint
    print(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create data loader
    test_loader = get_dataloader(
        dataset_name='default',
        data_dir=args.data_dir,
        annotation_file=args.annotation_file,
        batch_size=8,
        num_workers=config['data']['num_workers'],
        sequence_length=config['data']['sequence_length'],
        frame_size=tuple(config['data']['frame_size']),
        is_training=False
    )
    
    # Evaluate
    metrics = evaluate(
        model,
        test_loader,
        device,
        args.output_dir,
        args.visualize
    )
    
    print(f'\nResults saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
