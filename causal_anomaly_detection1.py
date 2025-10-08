"""
Robust Video Anomaly Detection for UCSDped2 Dataset
Memory-based approach with comprehensive NaN safety
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score
import warnings
import random
warnings.filterwarnings('ignore')

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def init_weights(m):
    """Initialize weights properly to prevent NaN"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=0.5)  # Smaller gain for stability
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def safe_normalize(tensor, dim=-1, eps=1e-8):
    """Safe normalization to prevent NaN"""
    norm = torch.norm(tensor, dim=dim, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return tensor / norm

def check_and_fix_nan(tensor, name="tensor"):
    """Check for NaN and replace with zeros"""
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}! Replacing with zeros.")
        return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    return tensor

class UCSDped2Dataset(Dataset):
    def __init__(self, root_dir, split='Train', transform=None, sequence_length=16):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.sequence_length = sequence_length
        self.sequences = []
        self.labels = []
        
        folders = sorted([f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, f))])
        
        for folder in folders:
            folder_path = os.path.join(self.root_dir, folder)
            frames = sorted([f for f in os.listdir(folder_path) 
                           if f.endswith(('.jpg', '.png', '.tif'))])
            
            # Create overlapping sequences
            step_size = max(1, self.sequence_length // 4)
            for i in range(0, len(frames) - self.sequence_length + 1, step_size):
                frame_sequence = frames[i:i + self.sequence_length]
                if len(frame_sequence) == self.sequence_length:
                    self.sequences.append((folder_path, frame_sequence, i, folder))
                    
                    if split == 'Train':
                        self.labels.append(0)  # All training is normal
                    else:
                        folder_num = int(folder.replace('Test', '').replace('Train', ''))
                        frame_progress = i / max(len(frames) - self.sequence_length, 1)
                        
                        anomaly_videos = {1, 2, 4, 5, 6, 9, 10, 11, 12}
                        
                        if folder_num in anomaly_videos:
                            if 0.2 <= frame_progress <= 0.8:
                                self.labels.append(1)  # Anomaly
                            else:
                                self.labels.append(0)  # Normal
                        else:
                            self.labels.append(0)  # Normal
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        folder_path, frame_names, start_frame, folder_name = self.sequences[idx]
        frames = []
        
        for frame_name in frame_names:
            frame_path = os.path.join(folder_path, frame_name)
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                frame = np.zeros((240, 360), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (64, 64))
            frames.append(frame)
        
        frames = np.stack(frames).astype(np.float32) / 255.0
        frames = torch.FloatTensor(frames).unsqueeze(1)  # [T, 1, H, W]
        
        # Clamp to prevent extreme values
        frames = torch.clamp(frames, 0.001, 0.999)
        
        if self.transform:
            transformed_frames = []
            for frame in frames:
                transformed_frames.append(self.transform(frame))
            frames = torch.stack(transformed_frames)
        
        return frames, torch.tensor(self.labels[idx], dtype=torch.long)

class VideoAutoEncoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=64):  # Smaller latent dim
        super().__init__()
        
        # Spatial encoder - simplified
        self.encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),  # Smaller negative slope
            
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(128, 128, 4, stride=2, padding=1),  # Reduced channels
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
            nn.Tanh()
        )
        
        # Decoder - simplified
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Unflatten(1, (128, 4, 4)),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, input_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        # Simple temporal encoder
        self.temporal_encoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=1,  # Single layer for stability
            batch_first=True,
            dropout=0.0
        )
        
        # Memory bank for normal patterns
        self.register_buffer('normal_memory', torch.zeros(500, latent_dim))  # Smaller memory
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        self.memory_size = 500
        
        # Initialize weights
        self.apply(init_weights)
        
        # Temperature for distance scaling
        self.register_buffer('temperature', torch.tensor(1.0))
        
    def update_memory(self, features):
        """Update memory bank with normal features"""
        features = check_and_fix_nan(features, "memory_features")
        
        batch_size = features.shape[0]
        ptr = int(self.memory_ptr)
        
        if ptr + batch_size <= self.memory_size:
            self.normal_memory[ptr:ptr + batch_size] = features.detach()
            self.memory_ptr[0] = (ptr + batch_size) % self.memory_size
        else:
            # Wrap around
            remaining = self.memory_size - ptr
            self.normal_memory[ptr:] = features[:remaining].detach()
            if batch_size - remaining > 0:
                self.normal_memory[:batch_size - remaining] = features[remaining:].detach()
                self.memory_ptr[0] = batch_size - remaining
            else:
                self.memory_ptr[0] = 0
        
    def encode_sequence(self, frames):
        """Encode a sequence of frames"""
        B, T, C, H, W = frames.shape
        
        # Encode each frame
        frame_features = []
        for t in range(T):
            frame = frames[:, t]
            latent = self.encoder(frame)
            latent = check_and_fix_nan(latent, f"frame_latent_{t}")
            frame_features.append(latent)
        
        frame_features = torch.stack(frame_features, dim=1)  # [B, T, latent_dim]
        frame_features = check_and_fix_nan(frame_features, "frame_features_stack")
        
        # Temporal encoding with error handling
        try:
            temporal_out, (h_n, c_n) = self.temporal_encoder(frame_features)
            sequence_feature = h_n[-1]
            sequence_feature = check_and_fix_nan(sequence_feature, "sequence_feature")
        except Exception as e:
            print(f"LSTM error: {e}")
            sequence_feature = frame_features.mean(dim=1)  # Fallback to mean
            sequence_feature = check_and_fix_nan(sequence_feature, "fallback_sequence_feature")
        
        return sequence_feature, frame_features
    
    def decode_sequence(self, sequence_feature, sequence_length):
        """Decode sequence feature back to frames"""
        B = sequence_feature.shape[0]
        
        # Decode each frame using the sequence feature
        reconstructed_frames = []
        for t in range(sequence_length):
            reconstructed_frame = self.decoder(sequence_feature)
            reconstructed_frame = check_and_fix_nan(reconstructed_frame, f"recon_frame_{t}")
            reconstructed_frames.append(reconstructed_frame)
        
        reconstructed_frames = torch.stack(reconstructed_frames, dim=1)  # [B, T, C, H, W]
        return reconstructed_frames
    
    def compute_anomaly_score(self, sequence_feature):
        """Compute anomaly score based on distance to normal memory"""
        sequence_feature = check_and_fix_nan(sequence_feature, "sequence_feature_anomaly")
        
        memory_filled = int(self.memory_ptr[0])
        if memory_filled < 10:  # Need some memory to compare
            return torch.zeros(sequence_feature.shape[0], device=sequence_feature.device)
        
        # Get populated memory
        populated_memory = self.normal_memory[:memory_filled]
        populated_memory = check_and_fix_nan(populated_memory, "populated_memory")
        
        if populated_memory.shape[0] == 0:
            return torch.zeros(sequence_feature.shape[0], device=sequence_feature.device)
        
        try:
            # Normalize features for stable distance computation
            sequence_norm = safe_normalize(sequence_feature, dim=-1)
            memory_norm = safe_normalize(populated_memory, dim=-1)
            
            # Compute cosine distances (more stable than euclidean)
            similarities = torch.mm(sequence_norm, memory_norm.t())  # [B, memory_size]
            similarities = torch.clamp(similarities, -1, 1)  # Clamp for numerical stability
            
            # Convert similarity to distance (higher similarity = lower anomaly)
            distances = 1 - similarities
            
            # Anomaly score is minimum distance to normal patterns
            min_distances, _ = torch.min(distances, dim=1)
            min_distances = torch.clamp(min_distances, 0, 2)  # Clamp distances
            
            # Scale to [0, 1] range
            anomaly_scores = min_distances / 2.0
            anomaly_scores = check_and_fix_nan(anomaly_scores, "anomaly_scores")
            
            return anomaly_scores
            
        except Exception as e:
            print(f"Error in anomaly score computation: {e}")
            return torch.zeros(sequence_feature.shape[0], device=sequence_feature.device)
    
    def forward(self, frames):
        frames = check_and_fix_nan(frames, "input_frames")
        B, T, C, H, W = frames.shape
        
        # Encode sequence
        sequence_feature, frame_features = self.encode_sequence(frames)
        
        # Decode sequence
        reconstructed_frames = self.decode_sequence(sequence_feature, T)
        
        # Compute anomaly score
        anomaly_scores = self.compute_anomaly_score(sequence_feature)
        
        return {
            'reconstructed': reconstructed_frames,
            'sequence_feature': sequence_feature,
            'frame_features': frame_features,
            'anomaly_score': anomaly_scores
        }

def safe_mse_loss(pred, target, eps=1e-8):
    """Safe MSE loss with NaN protection"""
    pred = check_and_fix_nan(pred, "pred_loss")
    target = check_and_fix_nan(target, "target_loss")
    
    diff = pred - target
    loss = torch.mean(diff * diff)
    
    if torch.isnan(loss) or torch.isinf(loss):
        print("NaN/Inf in MSE loss! Using L1 fallback.")
        loss = torch.mean(torch.abs(diff))
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN/Inf in L1 fallback! Using zero loss.")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    return loss

def reconstruction_loss(original, reconstructed):
    """Simple and stable reconstruction loss"""
    # Just use MSE for now - keep it simple
    mse_loss = safe_mse_loss(reconstructed, original)
    return mse_loss

def train_model(model, train_loader, val_loader, num_epochs=30, lr=5e-7):
    model.to(device)
    
    # Conservative optimizer settings
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, 
                                weight_decay=1e-6, eps=1e-8, betas=(0.9, 0.999))
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=3, verbose=True, min_lr=1e-7
    )
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    print("Training memory-based autoencoder...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        valid_batches = 0
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            # Only train on normal videos
            normal_mask = labels == 0
            if not normal_mask.any():
                continue
                
            videos = videos[normal_mask].to(device)
            B = videos.shape[0]
            
            if B == 0:
                continue
            
            # Check input validity
            if torch.isnan(videos).any() or torch.isinf(videos).any():
                print(f"Invalid input at batch {batch_idx}!")
                continue
            
            optimizer.zero_grad()
            
            try:
                outputs = model(videos)
                
                # Check outputs
                if torch.isnan(outputs['reconstructed']).any():
                    print(f"NaN in reconstruction at batch {batch_idx}!")
                    continue
                
                # Simple reconstruction loss
                recon_loss = reconstruction_loss(videos, outputs['reconstructed'])
                
                if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                    print(f"Invalid loss at batch {batch_idx}!")
                    continue
                
                # Update memory with normal patterns
                model.update_memory(outputs['sequence_feature'])
                
                total_loss = recon_loss
                
                total_loss.backward()
                
                # Check gradients
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"Invalid gradient in {name}")
                            has_nan_grad = True
                            break
                
                if not has_nan_grad:
                    # Conservative gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    optimizer.step()
                    
                    train_loss += total_loss.item()
                    valid_batches += 1
                
                num_batches += 1
                
                if batch_idx % 20 == 0:
                    memory_filled = int(model.memory_ptr[0])
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, '
                          f'Loss: {total_loss.item():.4f}, Memory: {memory_filled}/{model.memory_size}')
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        if valid_batches == 0:
            print("No valid batches in this epoch!")
            break
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_anomaly_scores = []
        val_labels = []
        valid_val_batches = 0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                try:
                    videos = videos.to(device)
                    
                    if torch.isnan(videos).any():
                        continue
                    
                    outputs = model(videos)
                    recon_loss = reconstruction_loss(videos, outputs['reconstructed'])
                    
                    if not torch.isnan(recon_loss):
                        val_loss += recon_loss.item()
                        val_anomaly_scores.extend(outputs['anomaly_score'].cpu().numpy())
                        val_labels.extend(labels.numpy())
                        valid_val_batches += 1
                        
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue
        
        if valid_val_batches == 0:
            print("No valid validation batches!")
            continue
        
        avg_train_loss = train_loss / max(valid_batches, 1)
        avg_val_loss = val_loss / max(valid_val_batches, 1)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Calculate separation
        normal_scores = [s for i, s in enumerate(val_anomaly_scores) if val_labels[i] == 0]
        abnormal_scores = [s for i, s in enumerate(val_anomaly_scores) if val_labels[i] == 1]
        
        separation = 0.0
        if normal_scores and abnormal_scores:
            separation = np.mean(abnormal_scores) - np.mean(normal_scores)
        
        print(f'\n=== Epoch {epoch+1}/{num_epochs} Summary ===')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'Valid batches: {valid_batches}/{num_batches}')
        print(f'Memory filled: {int(model.memory_ptr[0])}/{model.memory_size}')
        if normal_scores:
            print(f'Normal scores: {np.mean(normal_scores):.3f} ± {np.std(normal_scores):.3f}')
        if abnormal_scores:
            print(f'Abnormal scores: {np.mean(abnormal_scores):.3f} ± {np.std(abnormal_scores):.3f}')
        print(f'Separation: {separation:.4f}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print('=' * 50)
        
        # Early stopping
        if not np.isnan(avg_val_loss) and avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_robust_autoencoder.pth')
            print("New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    
    # Load best model
    try:
        model.load_state_dict(torch.load('best_robust_autoencoder.pth', map_location=device))
        print("Loaded best model checkpoint")
    except:
        print("Using final model")
    
    return model, train_losses, val_losses

def calculate_anomaly_scores(model, test_loader):
    """Calculate anomaly scores"""
    model.eval()
    all_scores = []
    all_labels = []
    all_recon_errors = []
    all_memory_scores = []
    
    with torch.no_grad():
        for videos, labels in test_loader:
            try:
                videos = videos.to(device)
                
                if torch.isnan(videos).any():
                    continue
                
                outputs = model(videos)
                
                # Reconstruction error
                recon_error = F.mse_loss(outputs['reconstructed'], videos, reduction='none')
                recon_error = recon_error.view(recon_error.shape[0], -1).mean(dim=1)
                
                # Memory-based anomaly score
                memory_scores = outputs['anomaly_score']
                
                # Combine both signals
                combined_scores = 0.7 * recon_error + 0.3 * memory_scores
                
                all_scores.extend(combined_scores.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_recon_errors.extend(recon_error.cpu().numpy())
                all_memory_scores.extend(memory_scores.cpu().numpy())
                
            except Exception as e:
                print(f"Error in test batch: {e}")
                continue
    
    return (np.array(all_scores), np.array(all_labels), 
            np.array(all_recon_errors), np.array(all_memory_scores))

def visualize_results(test_scores, test_labels, train_losses, val_losses):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training curves
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Score distribution
    normal_scores = test_scores[test_labels == 0]
    abnormal_scores = test_scores[test_labels == 1]
    
    axes[0, 1].hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    if len(abnormal_scores) > 0:
        axes[0, 1].hist(abnormal_scores, bins=50, alpha=0.7, label='Abnormal', color='red', density=True)
    axes[0, 1].set_xlabel('Anomaly Score')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Score Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # ROC Curve
    if len(np.unique(test_labels)) > 1:
        fpr, tpr, _ = roc_curve(test_labels, test_scores)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 2].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        axes[0, 2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 2].set_xlim([0.0, 1.0])
        axes[0, 2].set_ylim([0.0, 1.05])
        axes[0, 2].set_xlabel('False Positive Rate')
        axes[0, 2].set_ylabel('True Positive Rate')
        axes[0, 2].set_title('ROC Curve')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Score timeline
    colors = ['blue' if label == 0 else 'red' for label in test_labels]
    axes[1, 0].scatter(range(len(test_scores)), test_scores, c=colors, alpha=0.6, s=20)
    threshold = np.percentile(normal_scores, 95)
    axes[1, 0].axhline(y=threshold, color='black', linestyle='--', alpha=0.8, label=f'Threshold (95%)')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Anomaly Score')
    axes[1, 0].set_title('Anomaly Scores Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plots
    data_to_plot = [normal_scores]
    labels_plot = ['Normal']
    if len(abnormal_scores) > 0:
        data_to_plot.append(abnormal_scores)
        labels_plot.append('Abnormal')
    
    axes[1, 1].boxplot(data_to_plot, labels=labels_plot)
    axes[1, 1].set_ylabel('Anomaly Score')
    axes[1, 1].set_title('Score Distribution (Box Plot)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Statistics
    stats_text = f"Statistics:\n\n"
    stats_text += f"Normal samples: {len(normal_scores)}\n"
    stats_text += f"Normal mean: {np.mean(normal_scores):.4f}\n"
    stats_text += f"Normal std: {np.std(normal_scores):.4f}\n\n"
    
    if len(abnormal_scores) > 0:
        stats_text += f"Abnormal samples: {len(abnormal_scores)}\n"
        stats_text += f"Abnormal mean: {np.mean(abnormal_scores):.4f}\n"
        stats_text += f"Abnormal std: {np.std(abnormal_scores):.4f}\n\n"
        stats_text += f"Separation: {np.mean(abnormal_scores) - np.mean(normal_scores):.4f}\n\n"
        
        if len(np.unique(test_labels)) > 1:
            auc_score = roc_auc_score(test_labels, test_scores)
            stats_text += f"AUC: {auc_score:.4f}\n"
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig('robust_anomaly_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    data_root = "UCSDped2"
    
    print("=" * 60)
    print("ROBUST MEMORY-BASED VIDEO ANOMALY DETECTION")
    print("=" * 60)
    print("Strategy: Memory bank + comprehensive NaN safety")
    print("=" * 60)
    
    # Load datasets
    train_dataset = UCSDped2Dataset(data_root, split='Train', sequence_length=8)
    test_dataset = UCSDped2Dataset(data_root, split='Test', sequence_length=8)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Check labels
    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    test_labels = [test_dataset[i][1].item() for i in range(len(test_dataset))]
    
    print(f"Train - Normal: {train_labels.count(0)}, Abnormal: {train_labels.count(1)}")
    print(f"Test - Normal: {test_labels.count(0)}, Abnormal: {test_labels.count(1)}")
    
    # Data loaders - very small batches for stability
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, num_workers=1, pin_memory=False)
    val_loader = DataLoader(val_subset, batch_size=2, shuffle=False, num_workers=1, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=1, pin_memory=False)
    
    # Initialize model with smaller dimensions
    model = VideoAutoEncoder(input_channels=1, latent_dim=64)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Train model with very conservative settings
    trained_model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, num_epochs=25, lr=1e-6  # Very low LR
    )
    
    # Test model
    print("\nCalculating anomaly scores...")
    test_scores, test_labels, recon_errors, memory_scores = calculate_anomaly_scores(trained_model, test_loader)
    
    if len(test_scores) == 0:
        print("No valid test scores obtained!")
        return None, None, None
    
    # Evaluate
    normal_scores = test_scores[test_labels == 0]
    abnormal_scores = test_scores[test_labels == 1]
    
    if len(normal_scores) > 0:
        threshold = np.percentile(normal_scores, 95)
        binary_preds = (test_scores > threshold).astype(int)
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Test Samples: {len(test_labels)}")
        print(f"True - Normal: {(test_labels == 0).sum()}, Abnormal: {(test_labels == 1).sum()}")
        print(f"Predicted - Normal: {(binary_preds == 0).sum()}, Abnormal: {(binary_preds == 1).sum()}")
        print(f"Threshold (95th percentile): {threshold:.4f}")
        
        if len(np.unique(test_labels)) > 1:
            auc_score = roc_auc_score(test_labels, test_scores)
            accuracy = accuracy_score(test_labels, binary_preds)
            precision = precision_score(test_labels, binary_preds, zero_division=0)
            recall = recall_score(test_labels, binary_preds, zero_division=0)
            f1 = f1_score(test_labels, binary_preds, zero_division=0)
            
            print(f"\nPerformance Metrics:")
            print(f"AUC: {auc_score:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
        
        print(f"\nScore Statistics:")
        print(f"Normal - Mean: {np.mean(normal_scores):.4f}, Std: {np.std(normal_scores):.4f}")
        if len(abnormal_scores) > 0:
            separation = np.mean(abnormal_scores) - np.mean(normal_scores)
            print(f"Abnormal - Mean: {np.mean(abnormal_scores):.4f}, Std: {np.std(abnormal_scores):.4f}")
            print(f"Separation: {separation:.4f}")
            
            if separation > 0.0001:
                print("✓ Good separation achieved!")
            elif separation > 0.00003:
                print("⚠ Moderate separation")
            else:
                print("⚠ Limited separation")
        
        # Show component breakdown
        print(f"\nComponent Analysis:")
        if len(recon_errors) > 0:
            print(f"Reconstruction Error - Normal: {np.mean(recon_errors[test_labels == 0]):.4f}")
            if len(abnormal_scores) > 0:
                print(f"Reconstruction Error - Abnormal: {np.mean(recon_errors[test_labels == 1]):.4f}")
        if len(memory_scores) > 0:
            print(f"Memory Score - Normal: {np.mean(memory_scores[test_labels == 0]):.4f}")
            if len(abnormal_scores) > 0:
                print(f"Memory Score - Abnormal: {np.mean(memory_scores[test_labels == 1]):.4f}")
        
        # Show sample predictions
        print(f"\nSample Predictions (threshold={threshold:.4f}):")
        for i in range(min(15, len(test_labels))):
            status = "✓" if (test_labels[i] == binary_preds[i]) else "✗"
            print(f"{status} Sample {i+1}: True={test_labels[i]}, Score={test_scores[i]:.4f}, Pred={binary_preds[i]}")
        
        # Visualize results
        visualize_results(test_scores, test_labels, train_losses, val_losses)
        
        # Save model
        torch.save(trained_model.state_dict(), 'robust_video_autoencoder.pth')
        print("\nModel saved successfully!")
        
        return trained_model, test_scores, test_labels
    else:
        print("No normal scores to calculate threshold!")
        return trained_model, test_scores, test_labels

if __name__ == "__main__":
    try:
        trained_model, test_scores, test_labels = main()
        print("\n✅ Training and testing completed!")
        
        # Additional debugging info
        if trained_model is not None:
            print(f"\nModel Memory Status:")
            print(f"Memory filled: {int(trained_model.memory_ptr[0])}/{trained_model.memory_size}")
            print(f"Memory utilization: {int(trained_model.memory_ptr[0])/trained_model.memory_size*100:.1f}%")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()