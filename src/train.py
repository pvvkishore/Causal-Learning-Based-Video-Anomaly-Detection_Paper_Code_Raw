"""
Training script for causal representation learning video anomaly detection.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import yaml
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.causal_model import CausalAnomalyDetector
from data.dataset import get_dataloader
from utils.metrics import AnomalyMetrics
from utils.visualization import visualize_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Causal Representation Learning for Video Anomaly Detection'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to video dataset directory'
    )
    parser.add_argument(
        '--annotation_file',
        type=str,
        default=None,
        help='Path to annotation file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/default',
        help='Output directory for models and logs'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'model': {
                'input_channels': 3,
                'latent_dim': 512,
                'num_temporal_layers': 2
            },
            'training': {
                'batch_size': 8,
                'num_epochs': 100,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'beta': 1.0,
                'gradient_clip': 1.0
            },
            'data': {
                'sequence_length': 16,
                'frame_size': [256, 256],
                'num_workers': 4
            }
        }
    
    return config


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    writer: SummaryWriter
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_losses = {
        'total_loss': 0.0,
        'recon_loss': 0.0,
        'kld_loss': 0.0,
        'anomaly_loss': 0.0
    }
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (frames, labels) in enumerate(pbar):
        frames = frames.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(frames, return_components=True)
        
        # Compute losses
        losses = model.compute_loss(
            x=frames,
            x_recon=outputs['recon'],
            mu=outputs['mu'],
            log_var=outputs['log_var'],
            anomaly_scores=outputs['anomaly_scores'],
            labels=labels,
            beta=config['training']['beta']
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # Gradient clipping
        if config['training'].get('gradient_clip'):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient_clip']
            )
        
        optimizer.step()
        
        # Update metrics
        for key in total_losses.keys():
            if key in losses:
                total_losses[key] += losses[key].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': losses['total_loss'].item(),
            'recon': losses['recon_loss'].item()
        })
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        if batch_idx % 10 == 0:
            for key, value in losses.items():
                writer.add_scalar(f'train/{key}', value.item(), global_step)
    
    # Average losses
    num_batches = len(dataloader)
    for key in total_losses.keys():
        total_losses[key] /= num_batches
    
    return total_losses


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict
) -> dict:
    """Validate the model."""
    model.eval()
    
    metrics = AnomalyMetrics()
    
    total_losses = {
        'total_loss': 0.0,
        'recon_loss': 0.0,
        'kld_loss': 0.0
    }
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc='Validating'):
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(frames, return_components=True)
            
            # Compute losses
            losses = model.compute_loss(
                x=frames,
                x_recon=outputs['recon'],
                mu=outputs['mu'],
                log_var=outputs['log_var'],
                anomaly_scores=outputs['anomaly_scores'],
                labels=labels,
                beta=config['training']['beta']
            )
            
            # Update metrics
            for key in total_losses.keys():
                if key in losses:
                    total_losses[key] += losses[key].item()
            
            # Collect scores and labels
            all_scores.append(outputs['anomaly_scores'].cpu())
            all_labels.append(labels.cpu())
    
    # Average losses
    num_batches = len(dataloader)
    for key in total_losses.keys():
        total_losses[key] /= num_batches
    
    # Compute metrics
    all_scores = torch.cat(all_scores, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    metrics_dict = metrics.compute(all_scores, all_labels)
    total_losses.update(metrics_dict)
    
    return total_losses


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    output_dir: str,
    is_best: bool = False
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(output_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
    
    # Save epoch checkpoint
    epoch_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, epoch_path)


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Setup device
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Create model
    model = CausalAnomalyDetector(
        input_channels=config['model']['input_channels'],
        latent_dim=config['model']['latent_dim'],
        num_temporal_layers=config['model']['num_temporal_layers']
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_metric = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'metrics' in checkpoint:
            best_metric = checkpoint['metrics'].get('total_loss', float('inf'))
    
    # Create data loaders
    train_loader = get_dataloader(
        dataset_name='default',
        data_dir=args.data_dir,
        annotation_file=args.annotation_file,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        sequence_length=config['data']['sequence_length'],
        frame_size=tuple(config['data']['frame_size']),
        is_training=True
    )
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    
    # Training loop
    print('Starting training...')
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, config, writer
        )
        
        # Log metrics
        print(f'\nEpoch {epoch}:')
        for key, value in train_metrics.items():
            print(f'  {key}: {value:.4f}')
            writer.add_scalar(f'epoch/{key}', value, epoch)
        
        # Save checkpoint
        is_best = train_metrics['total_loss'] < best_metric
        if is_best:
            best_metric = train_metrics['total_loss']
        
        save_checkpoint(
            model, optimizer, epoch, train_metrics,
            str(output_dir), is_best
        )
        
        # Update learning rate
        scheduler.step(train_metrics['total_loss'])
    
    writer.close()
    print('Training complete!')


if __name__ == '__main__':
    main()
