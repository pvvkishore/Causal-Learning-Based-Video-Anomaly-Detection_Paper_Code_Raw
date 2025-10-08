"""
Demo script to test the model with sample data.
"""

import torch
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.causal_model import CausalAnomalyDetector


def create_sample_data(batch_size=2, seq_len=16, channels=3, height=256, width=256):
    """
    Create random sample video data for testing.
    
    Returns:
        frames: Random video frames
        labels: Random binary labels
    """
    frames = torch.randn(batch_size, seq_len, channels, height, width)
    labels = torch.randint(0, 2, (batch_size, seq_len))
    
    return frames, labels


def test_model():
    """Test the causal anomaly detection model."""
    print("="*60)
    print("Testing Causal Anomaly Detection Model")
    print("="*60)
    
    # Model parameters
    input_channels = 3
    latent_dim = 512
    num_temporal_layers = 2
    
    # Create model
    print("\nCreating model...")
    model = CausalAnomalyDetector(
        input_channels=input_channels,
        latent_dim=latent_dim,
        num_temporal_layers=num_temporal_layers
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters ({num_params/1e6:.2f}M)")
    
    # Create sample data
    print("\nCreating sample data...")
    frames, labels = create_sample_data(batch_size=2, seq_len=16)
    print(f"Input shape: {frames.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(frames, return_components=True)
    
    print("\nModel outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Compute loss
    print("\nComputing loss...")
    losses = model.compute_loss(
        x=frames,
        x_recon=outputs['recon'],
        mu=outputs['mu'],
        log_var=outputs['log_var'],
        anomaly_scores=outputs['anomaly_scores'],
        labels=labels,
        beta=1.0
    )
    
    print("\nLoss values:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test individual components
    print("\n" + "="*60)
    print("Testing Individual Components")
    print("="*60)
    
    # Test encoder
    print("\nTesting encoder...")
    batch_size, seq_len, C, H, W = frames.shape
    frames_flat = frames.view(batch_size * seq_len, C, H, W)
    mu, log_var = model.encoder(frames_flat)
    print(f"  Encoder output - mu: {mu.shape}, log_var: {log_var.shape}")
    
    # Test decoder
    print("\nTesting decoder...")
    z = model.reparameterize(mu, log_var)
    recon = model.decoder(z)
    print(f"  Decoder output: {recon.shape}")
    
    # Test temporal module
    print("\nTesting temporal module...")
    z_seq = z.view(batch_size, seq_len, -1)
    temporal_out, attn_weights = model.temporal_module(z_seq, return_attention=True)
    print(f"  Temporal output: {temporal_out.shape}")
    if attn_weights is not None:
        print(f"  Attention weights: {attn_weights.shape}")
    
    print("\n" + "="*60)
    print("All tests passed successfully!")
    print("="*60)


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_model()
