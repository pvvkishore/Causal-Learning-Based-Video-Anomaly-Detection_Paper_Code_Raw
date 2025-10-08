"""
Causal Representation Learning Module for Video Anomaly Detection

This module implements the causal representation learning framework for
discovering temporal anomaly patterns in videos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class CausalEncoder(nn.Module):
    """
    Causal encoder that learns disentangled representations
    of video frames considering causal relationships.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 512,
        hidden_dims: Optional[list] = None
    ):
        super(CausalEncoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
        
        self.latent_dim = latent_dim
        
        # Convolutional encoder
        modules = []
        in_channels = input_channels
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Flatten and project to latent space
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        
        return mu, log_var


class CausalDecoder(nn.Module):
    """
    Decoder that reconstructs frames from causal latent representations.
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        output_channels: int = 3,
        hidden_dims: Optional[list] = None
    ):
        super(CausalDecoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        self.latent_dim = latent_dim
        
        # Project from latent space
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 4 * 4)
        
        # Transpose convolutional decoder
        modules = []
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        
        # Final layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.Tanh()
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to image.
        
        Args:
            z: Latent tensor of shape (B, latent_dim)
            
        Returns:
            Reconstructed image of shape (B, C, H, W)
        """
        h = self.decoder_input(z)
        h = h.view(-1, 512, 4, 4)
        h = self.decoder(h)
        x_recon = self.final_layer(h)
        
        return x_recon


class TemporalCausalModule(nn.Module):
    """
    Temporal module that models causal relationships across time.
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        num_layers: int = 2,
        bidirectional: bool = False
    ):
        super(TemporalCausalModule, self).__init__()
        
        self.latent_dim = latent_dim
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Attention mechanism for temporal dependencies
        self.attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Causal graph learner
        self.causal_weight = nn.Parameter(torch.randn(latent_dim, latent_dim))
        
    def forward(
        self,
        z_seq: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Model temporal causal relationships.
        
        Args:
            z_seq: Sequence of latent representations (B, T, D)
            return_attention: Whether to return attention weights
            
        Returns:
            temporal_features: Temporally enhanced features
            attention_weights: Optional attention weights
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(z_seq)
        
        # Self-attention for temporal dependencies
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Apply causal relationships
        causal_out = torch.matmul(attn_out, self.causal_weight)
        
        if return_attention:
            return causal_out, attn_weights
        
        return causal_out, None


class CausalAnomalyDetector(nn.Module):
    """
    Complete causal representation learning model for video anomaly detection.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 512,
        num_temporal_layers: int = 2
    ):
        super(CausalAnomalyDetector, self).__init__()
        
        self.encoder = CausalEncoder(input_channels, latent_dim)
        self.decoder = CausalDecoder(latent_dim, input_channels)
        self.temporal_module = TemporalCausalModule(latent_dim, num_temporal_layers)
        
        # Anomaly score predictor
        self.anomaly_predictor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Args:
            x: Input video sequence (B, T, C, H, W)
            return_components: Whether to return intermediate components
            
        Returns:
            Dictionary containing:
                - recon: Reconstructed frames
                - anomaly_scores: Anomaly scores per frame
                - mu, log_var: Latent distribution parameters (if return_components)
        """
        batch_size, seq_len, C, H, W = x.shape
        
        # Reshape for encoding
        x_flat = x.view(batch_size * seq_len, C, H, W)
        
        # Encode
        mu, log_var = self.encoder(x_flat)
        z = self.reparameterize(mu, log_var)
        
        # Reshape for temporal modeling
        z_seq = z.view(batch_size, seq_len, -1)
        
        # Temporal causal modeling
        temporal_features, _ = self.temporal_module(z_seq)
        
        # Decode
        temporal_flat = temporal_features.view(batch_size * seq_len, -1)
        x_recon = self.decoder(temporal_flat)
        x_recon = x_recon.view(batch_size, seq_len, C, H, W)
        
        # Predict anomaly scores
        anomaly_scores = self.anomaly_predictor(temporal_flat)
        anomaly_scores = anomaly_scores.view(batch_size, seq_len)
        
        output = {
            'recon': x_recon,
            'anomaly_scores': anomaly_scores
        }
        
        if return_components:
            output.update({
                'mu': mu.view(batch_size, seq_len, -1),
                'log_var': log_var.view(batch_size, seq_len, -1),
                'z': z_seq,
                'temporal_features': temporal_features
            })
        
        return output
    
    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        anomaly_scores: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        beta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the complete loss function.
        
        Args:
            x: Original frames
            x_recon: Reconstructed frames
            mu: Latent mean
            log_var: Latent log variance
            anomaly_scores: Predicted anomaly scores
            labels: Ground truth labels (1 for anomaly, 0 for normal)
            beta: Weight for KL divergence
            
        Returns:
            Dictionary of loss components
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence
        kld_loss = -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )
        
        # Total VAE loss
        vae_loss = recon_loss + beta * kld_loss
        
        losses = {
            'recon_loss': recon_loss,
            'kld_loss': kld_loss,
            'vae_loss': vae_loss
        }
        
        # Anomaly detection loss (if labels provided)
        if labels is not None:
            anomaly_loss = F.binary_cross_entropy(
                anomaly_scores,
                labels.float(),
                reduction='mean'
            )
            losses['anomaly_loss'] = anomaly_loss
            losses['total_loss'] = vae_loss + anomaly_loss
        else:
            losses['total_loss'] = vae_loss
        
        return losses
