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
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class UCSDped2Dataset(Dataset):
    def __init__(self, root_dir, split='Train', transform=None, sequence_length=16):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.sequence_length = sequence_length
        self.sequences = []
        self.labels = []
        
        # Load video sequences
        for folder in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                frames = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.tif'))])
                
                # Create sequences of frames
                for i in range(0, len(frames) - self.sequence_length + 1, self.sequence_length // 2):
                    frame_sequence = frames[i:i + self.sequence_length]
                    if len(frame_sequence) == self.sequence_length:
                        self.sequences.append((folder_path, frame_sequence))
                        # Labels: Train=0 (normal), Test=1 (may contain anomalies)
                        self.labels.append(0 if split == 'Train' else 1)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        folder_path, frame_names = self.sequences[idx]
        frames = []
        
        for frame_name in frame_names:
            frame_path = os.path.join(folder_path, frame_name)
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            frame = cv2.resize(frame, (360, 240))
            frames.append(frame)
        
        frames = np.stack(frames)  # (T, H, W)
        frames = torch.FloatTensor(frames).unsqueeze(1)  # (T, 1, H, W)
        
        if self.transform:
            # Apply transform to each frame
            transformed_frames = []
            for frame in frames:
                transformed_frames.append(self.transform(frame))
            frames = torch.stack(transformed_frames)
        
        return frames, torch.tensor(self.labels[idx], dtype=torch.long)

class ResNetBackbone(nn.Module):
    def __init__(self, input_channels=1, output_dim=256):  # Reduced output dim
        super().__init__()
        # Simplified ResNet-like backbone for lightweight processing
        self.conv1 = nn.Conv2d(input_channels, 32, 7, stride=2, padding=3)  # Reduced channels
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, output_dim, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 6))  # Smaller output
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(B, T, -1)  # (B, T, feature_dim)
        
        return x

class PedestrianDetector(nn.Module):
    def __init__(self, feature_dim, num_anchors=3, max_detections=10):
        super().__init__()
        self.num_anchors = num_anchors
        self.max_detections = max_detections
        
        # Detection heads
        self.bbox_head = nn.Linear(feature_dim, num_anchors * 4)  # x, y, w, h
        self.conf_head = nn.Linear(feature_dim, num_anchors)      # confidence scores
        
    def forward(self, features):
        # features: (B, T, feature_dim)
        B, T, _ = features.shape
        
        # Flatten spatial features for detection
        features_flat = features.mean(dim=-1) if len(features.shape) > 3 else features
        
        # Generate bounding boxes and confidence scores
        bbox_pred = self.bbox_head(features_flat)  # (B, T, num_anchors * 4)
        conf_pred = self.conf_head(features_flat)   # (B, T, num_anchors)
        
        # Reshape and apply sigmoid/softmax
        bbox_pred = bbox_pred.view(B, T, self.num_anchors, 4)
        conf_pred = torch.sigmoid(conf_pred).view(B, T, self.num_anchors)
        
        # Simulate pedestrian detection (simplified)
        detections = []
        for b in range(B):
            batch_detections = []
            for t in range(T):
                # Select top detections based on confidence
                conf_scores = conf_pred[b, t]
                top_indices = torch.topk(conf_scores, min(self.max_detections, self.num_anchors))[1]
                
                frame_detections = bbox_pred[b, t, top_indices]
                frame_conf = conf_scores[top_indices]
                
                # Filter by confidence threshold
                valid_mask = frame_conf > 0.5
                if valid_mask.sum() > 0:
                    frame_detections = frame_detections[valid_mask]
                    batch_detections.append(frame_detections)
                else:
                    # Add dummy detection if none found
                    batch_detections.append(torch.zeros(1, 4, device=features.device))
            
            detections.append(batch_detections)
        
        return detections

class TrajectoryTracker(nn.Module):
    def __init__(self, max_tracks=20, reid_dim=64):  # Reduced reid_dim
        super().__init__()
        self.max_tracks = max_tracks
        self.reid_dim = reid_dim
        
        # ReID feature extractor
        self.reid_net = nn.Sequential(
            nn.Linear(4, 32),  # bbox features
            nn.ReLU(),
            nn.Linear(32, reid_dim),
            nn.ReLU(),
            nn.Linear(reid_dim, reid_dim)
        )
        
    def forward(self, detections_sequence):
        # Simplified tracking - convert detections to trajectories
        # In practice, this would use Kalman filtering and Hungarian algorithm
        trajectories = []
        
        for batch_detections in detections_sequence:
            batch_trajectories = []
            for frame_detections in batch_detections:
                if len(frame_detections) > 0:
                    # Generate ReID features
                    reid_features = self.reid_net(frame_detections)
                    
                    # Create trajectory data (bbox + reid features)
                    traj_data = torch.cat([frame_detections, reid_features], dim=-1)
                    batch_trajectories.append(traj_data)
            
            if batch_trajectories:
                # Pad trajectories to same length
                max_len = max(len(traj) for traj in batch_trajectories)
                padded_trajs = []
                
                for traj in batch_trajectories:
                    if len(traj) < max_len:
                        padding = torch.zeros(max_len - len(traj), traj.shape[-1], device=traj.device)
                        traj = torch.cat([traj, padding], dim=0)
                    padded_trajs.append(traj[:max_len])
                
                trajectories.append(torch.stack(padded_trajs))
            else:
                # Create dummy trajectory
                dummy_traj = torch.zeros(1, 1, 4 + self.reid_dim, device=detections_sequence[0][0].device)
                trajectories.append(dummy_traj)
        
        return trajectories

class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dim=64):  # Reduced dimensions
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=False)  # Removed bidirectional
        self.encoder = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, trajectories):
        # trajectories: list of (T, N_detections, feature_dim)
        encoded_trajs = []
        
        for traj_batch in trajectories:
            if len(traj_batch.shape) == 3:
                T, N, D = traj_batch.shape
                # Reshape for GRU: (N, T, D)
                traj_reshaped = traj_batch.permute(1, 0, 2)
                
                # Encode each trajectory
                encoded_batch = []
                for n in range(N):
                    traj_seq = traj_reshaped[n].unsqueeze(0)  # (1, T, D)
                    gru_out, _ = self.gru(traj_seq)
                    # Use last output
                    encoded_traj = self.encoder(gru_out[:, -1, :])  # (1, latent_dim)
                    encoded_batch.append(encoded_traj)
                
                if encoded_batch:
                    encoded_trajs.append(torch.cat(encoded_batch, dim=0))
                else:
                    encoded_trajs.append(torch.zeros(1, self.latent_dim, device=traj_batch.device))
            else:
                encoded_trajs.append(torch.zeros(1, self.latent_dim, device=traj_batch.device))
        
        return encoded_trajs

class CausalFactorExtractor(nn.Module):
    def __init__(self, input_dim, num_factors=6, hidden_dim=32):  # Reduced factors and hidden_dim
        super().__init__()
        self.num_factors = num_factors
        
        # VAE-style disentangling
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu_head = nn.Linear(hidden_dim, num_factors)
        self.logvar_head = nn.Linear(hidden_dim, num_factors)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, encoded_trajectories):
        causal_factors = []
        kl_losses = []
        
        for encoded_traj in encoded_trajectories:
            if encoded_traj.numel() > 0:
                h = self.encoder(encoded_traj)
                mu = self.mu_head(h)
                logvar = self.logvar_head(h)
                
                z = self.reparameterize(mu, logvar)
                
                # KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
                
                causal_factors.append(z)
                kl_losses.append(kl_loss.mean())
            else:
                causal_factors.append(torch.zeros(1, self.num_factors, device=encoded_traj.device))
                kl_losses.append(torch.tensor(0.0, device=encoded_traj.device))
        
        return causal_factors, kl_losses

class CausalStructureLearner(nn.Module):
    def __init__(self, num_factors, hidden_dim=32):
        super().__init__()
        self.num_factors = num_factors
        
        # Graph neural network for learning causal structure
        self.node_encoder = nn.Linear(num_factors, hidden_dim)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Structural equation parameters
        self.structure_params = nn.Parameter(torch.randn(num_factors, num_factors))
        
    def forward(self, causal_factors_batch):
        adjacency_matrices = []
        structural_equations = []
        
        for factors in causal_factors_batch:
            if factors.numel() > 0:
                # Encode nodes
                node_features = self.node_encoder(factors)  # (N, hidden_dim)
                N = node_features.shape[0]
                
                # Predict edges
                adjacency = torch.zeros(self.num_factors, self.num_factors, device=factors.device)
                
                for i in range(min(N, self.num_factors)):
                    for j in range(min(N, self.num_factors)):
                        if i != j:
                            edge_input = torch.cat([node_features[i], node_features[j]], dim=0)
                            edge_prob = self.edge_predictor(edge_input.unsqueeze(0))
                            adjacency[i, j] = edge_prob.squeeze()
                
                # Apply acyclicity constraint (simplified)
                adjacency = adjacency * (1 - torch.eye(self.num_factors, device=factors.device))
                
                adjacency_matrices.append(adjacency)
                structural_equations.append(self.structure_params)
            else:
                adjacency_matrices.append(torch.zeros(self.num_factors, self.num_factors, device=factors.device))
                structural_equations.append(torch.zeros(self.num_factors, self.num_factors, device=factors.device))
        
        return adjacency_matrices, structural_equations

class DynamicsPredictor(nn.Module):
    def __init__(self, num_factors, hidden_dim=32):
        super().__init__()
        self.num_factors = num_factors
        
        self.dynamics_net = nn.Sequential(
            nn.Linear(num_factors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_factors)
        )
        
    def forward(self, causal_factors, adjacency_matrix):
        predictions = []
        
        for factors, adj in zip(causal_factors, adjacency_matrix):
            if factors.numel() > 0:
                # Apply causal structure
                structured_factors = torch.matmul(adj, factors.T).T
                
                # Predict next state
                pred_factors = self.dynamics_net(structured_factors)
                predictions.append(pred_factors)
            else:
                predictions.append(torch.zeros_like(factors))
        
        return predictions

class AnomalyScorer(nn.Module):
    def __init__(self, num_factors):
        super().__init__()
        self.num_factors = num_factors
        
        self.score_net = nn.Sequential(
            nn.Linear(num_factors * 3, 32),  # current, predicted, difference
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, current_factors, predicted_factors):
        anomaly_scores = []
        
        for current, predicted in zip(current_factors, predicted_factors):
            if current.numel() > 0 and predicted.numel() > 0:
                diff = torch.abs(current - predicted)
                
                # Concatenate features
                score_input = torch.cat([current, predicted, diff], dim=-1)
                score = self.score_net(score_input)
                anomaly_scores.append(score.mean())
            else:
                anomaly_scores.append(torch.tensor(0.0, device=current.device))
        
        return torch.stack(anomaly_scores)

class CausalAnomalyDetector(nn.Module):
    def __init__(self, num_factors=6, reid_dim=64):
        super().__init__()
        
        # Initialize all modules with reduced dimensions
        self.backbone = ResNetBackbone(input_channels=1, output_dim=256)
        self.detector = PedestrianDetector(256 * 4 * 6)  # Adjusted for feature map size
        self.tracker = TrajectoryTracker(reid_dim=reid_dim)
        self.traj_encoder = TrajectoryEncoder(4 + reid_dim, latent_dim=32)
        self.causal_extractor = CausalFactorExtractor(32, num_factors=num_factors)
        self.structure_learner = CausalStructureLearner(num_factors)
        self.dynamics_predictor = DynamicsPredictor(num_factors)
        self.anomaly_scorer = AnomalyScorer(num_factors)
        
    def forward(self, video_frames):
        # 1. Feature extraction
        features = self.backbone(video_frames)
        B, T, _ = features.shape
        
        # Reshape for detection
        features_reshaped = features.view(B, T, -1)
        
        # 2. Detection
        detections = self.detector(features_reshaped)
        
        # 3. Tracking
        trajectories = self.tracker(detections)
        
        # 4. Trajectory encoding
        encoded_trajs = self.traj_encoder(trajectories)
        
        # 5. Causal factor extraction
        causal_factors, kl_losses = self.causal_extractor(encoded_trajs)
        
        # 6. Causal structure learning
        adjacency_matrices, structural_equations = self.structure_learner(causal_factors)
        
        # 7. Dynamics prediction
        predicted_factors = self.dynamics_predictor(causal_factors, adjacency_matrices)
        
        # 8. Anomaly scoring
        anomaly_scores = self.anomaly_scorer(causal_factors, predicted_factors)
        
        return {
            'anomaly_scores': anomaly_scores,
            'causal_factors': causal_factors,
            'adjacency_matrices': adjacency_matrices,
            'kl_losses': kl_losses,
            'detections': detections
        }

def apply_memory_efficient_training(model):
    """Apply memory-efficient training techniques"""
    
    # Freeze some early layers to reduce memory usage
    for name, param in model.named_parameters():
        if 'backbone.conv1' in name or 'backbone.bn1' in name:
            param.requires_grad = False
            
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    return model

def train_model(model, train_loader, val_loader, num_epochs=15, lr=1e-4):
    """Memory-efficient training loop"""
    
    # Apply memory-efficient training techniques
    model = apply_memory_efficient_training(model)
    model.to(device)
    
    # Use AdamW with lower learning rate for stability
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        weight_decay=1e-5,
        eps=1e-8
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Enable mixed precision training for memory efficiency
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    train_losses = []
    val_losses = []
    
    print(f"Starting training with mixed precision: {scaler is not None}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            try:
                videos = videos.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(videos)
                        
                        # Loss computation with error handling
                        anomaly_loss = F.mse_loss(outputs['anomaly_scores'], labels.float())
                        
                        # Handle KL losses safely
                        kl_losses = outputs.get('kl_losses', [])
                        if kl_losses and len(kl_losses) > 0:
                            valid_kl_losses = [kl for kl in kl_losses if torch.isfinite(kl)]
                            kl_loss = sum(valid_kl_losses) / len(valid_kl_losses) if valid_kl_losses else torch.tensor(0.0, device=device)
                        else:
                            kl_loss = torch.tensor(0.0, device=device)
                        
                        total_loss = anomaly_loss + 0.001 * kl_loss  # Further reduced KL weight
                    
                    # Backward pass with gradient scaling
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard precision training
                    outputs = model(videos)
                    
                    anomaly_loss = F.mse_loss(outputs['anomaly_scores'], labels.float())
                    
                    kl_losses = outputs.get('kl_losses', [])
                    if kl_losses and len(kl_losses) > 0:
                        valid_kl_losses = [kl for kl in kl_losses if torch.isfinite(kl)]
                        kl_loss = sum(valid_kl_losses) / len(valid_kl_losses) if valid_kl_losses else torch.tensor(0.0, device=device)
                    else:
                        kl_loss = torch.tensor(0.0, device=device)
                    
                    total_loss = anomaly_loss + 0.001 * kl_loss
                    
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                train_loss += total_loss.item()
                num_batches += 1
                
                if batch_idx % 5 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {total_loss.item():.6f}')
                
                # Clear cache periodically
                if batch_idx % 10 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CUDA out of memory at batch {batch_idx}. Skipping batch...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                try:
                    videos = videos.to(device)
                    labels = labels.to(device)
                    
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = model(videos)
                            anomaly_loss = F.mse_loss(outputs['anomaly_scores'], labels.float())
                            
                            kl_losses = outputs.get('kl_losses', [])
                            if kl_losses and len(kl_losses) > 0:
                                valid_kl_losses = [kl for kl in kl_losses if torch.isfinite(kl)]
                                kl_loss = sum(valid_kl_losses) / len(valid_kl_losses) if valid_kl_losses else torch.tensor(0.0, device=device)
                            else:
                                kl_loss = torch.tensor(0.0, device=device)
                            
                            total_loss = anomaly_loss + 0.001 * kl_loss
                    else:
                        outputs = model(videos)
                        anomaly_loss = F.mse_loss(outputs['anomaly_scores'], labels.float())
                        
                        kl_losses = outputs.get('kl_losses', [])
                        if kl_losses and len(kl_losses) > 0:
                            valid_kl_losses = [kl for kl in kl_losses if torch.isfinite(kl)]
                            kl_loss = sum(valid_kl_losses) / len(valid_kl_losses) if valid_kl_losses else torch.tensor(0.0, device=device)
                        else:
                            kl_loss = torch.tensor(0.0, device=device)
                        
                        total_loss = anomaly_loss + 0.001 * kl_loss
                    
                    val_loss += total_loss.item()
                    val_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("CUDA out of memory during validation. Skipping batch...")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        scheduler.step()
        
        avg_train_loss = train_loss / max(num_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # Clear cache at end of epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return model, train_losses, val_losses

def test_model(model, test_loader):
    """Test the model and return predictions"""
    model.eval()
    all_scores = []
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for videos, labels in test_loader:
            videos = videos.to(device)
            
            outputs = model(videos)
            scores = outputs['anomaly_scores'].cpu().numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())
            all_outputs.append(outputs)
    
    return np.array(all_scores), np.array(all_labels), all_outputs

def visualize_results(test_scores, test_labels, test_outputs, test_dataset):
    """Create comprehensive visualizations"""
    
    # 1. Plot anomaly score curves for 5 videos
    plt.figure(figsize=(15, 10))
    
    for i in range(min(5, len(test_scores))):
        plt.subplot(2, 3, i+1)
        plt.plot(test_scores[i:i+1], 'b-', label='Anomaly Score')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
        plt.title(f'Video {i+1} (Label: {"Abnormal" if test_labels[i] else "Normal"})')
        plt.xlabel('Frame')
        plt.ylabel('Anomaly Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anomaly_score_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. ROC curve and metrics
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 3. Score distribution
    plt.subplot(1, 3, 2)
    normal_scores = test_scores[test_labels == 0]
    abnormal_scores = test_scores[test_labels == 1]
    
    plt.hist(normal_scores, bins=20, alpha=0.7, label='Normal', color='blue', density=True)
    plt.hist(abnormal_scores, bins=20, alpha=0.7, label='Abnormal', color='red', density=True)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Correlation matrix for causal factors
    plt.subplot(1, 3, 3)
    
    # Extract causal factors from test outputs
    all_factors = []
    for output_batch in test_outputs[:10]:  # Use first 10 batches
        for factors in output_batch['causal_factors']:
            if factors.numel() > 0:
                all_factors.append(factors.cpu().numpy())
    
    if all_factors:
        factors_matrix = np.vstack(all_factors)
        correlation_matrix = np.corrcoef(factors_matrix.T)
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': .8})
        plt.title('Causal Factors Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_bounding_boxes(model, test_dataset, num_videos=5):
    """Visualize bounding boxes on test videos"""
    
    plt.figure(figsize=(20, 12))
    
    for i in range(min(num_videos, len(test_dataset))):
        video_frames, label = test_dataset[i]
        
        # Get model predictions
        model.eval()
        with torch.no_grad():
            video_batch = video_frames.unsqueeze(0).to(device)
            outputs = model(video_batch)
            detections = outputs['detections'][0]  # First batch
            anomaly_score = outputs['anomaly_scores'][0].cpu().numpy()
        
        # Visualize first frame with detections
        frame = video_frames[0, 0].cpu().numpy()  # First frame, first channel
        
        plt.subplot(2, 3, i+1)
        plt.imshow(frame, cmap='gray')
        
        # Draw bounding boxes if available
        if len(detections) > 0 and len(detections[0]) > 0:
            bboxes = detections[0].cpu().numpy()  # First frame detections
            
            for bbox in bboxes:
                if bbox.sum() > 0:  # Valid detection
                    x, y, w, h = bbox
                    # Convert to corner coordinates
                    x1, y1 = max(0, x - w/2), max(0, y - h/2)
                    x2, y2 = min(frame.shape[1], x + w/2), min(frame.shape[0], y + h/2)
                    
                    # Draw rectangle
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       fill=False, color='red', linewidth=2)
                    plt.gca().add_patch(rect)
        
        plt.title(f'Video {i+1}\nLabel: {"Abnormal" if label else "Normal"}\n'
                 f'Anomaly Score: {anomaly_score:.3f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('bounding_box_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_additional_plots(test_outputs, test_labels, test_scores):
    """Create additional analysis plots"""
    
    plt.figure(figsize=(20, 15))
    
    # 1. Causal factor evolution over time
    plt.subplot(3, 4, 1)
    sample_factors = []
    for i, output_batch in enumerate(test_outputs[:10]):
        for factors in output_batch['causal_factors']:
            if factors.numel() > 0:
                sample_factors.append(factors.cpu().numpy().mean(axis=0))
                break
        if len(sample_factors) >= 10:
            break
    
    if sample_factors:
        factors_matrix = np.array(sample_factors)
        for i in range(min(6, factors_matrix.shape[1])):
            plt.plot(factors_matrix[:, i], label=f'Factor {i+1}', alpha=0.7)
        plt.xlabel('Sample')
        plt.ylabel('Factor Value')
        plt.title('Causal Factor Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 2. Adjacency matrix visualization
    plt.subplot(3, 4, 2)
    sample_adj_matrices = []
    for output_batch in test_outputs[:5]:
        for adj_matrix in output_batch['adjacency_matrices']:
            if adj_matrix.numel() > 0:
                sample_adj_matrices.append(adj_matrix.cpu().numpy())
                break
        if len(sample_adj_matrices) >= 5:
            break
    
    if sample_adj_matrices:
        avg_adj_matrix = np.mean(sample_adj_matrices, axis=0)
        sns.heatmap(avg_adj_matrix, annot=True, cmap='Blues', square=True, fmt='.2f')
        plt.title('Average Causal Adjacency Matrix')
    
    # 3. Score distribution by class
    plt.subplot(3, 4, 3)
    normal_scores = test_scores[test_labels == 0]
    abnormal_scores = test_scores[test_labels == 1] if np.any(test_labels == 1) else []
    
    plt.boxplot([normal_scores, abnormal_scores] if len(abnormal_scores) > 0 else [normal_scores], 
                labels=['Normal', 'Abnormal'] if len(abnormal_scores) > 0 else ['Normal'])
    plt.ylabel('Anomaly Score')
    plt.title('Score Distribution by Class')
    plt.grid(True, alpha=0.3)
    
    # 4. Temporal anomaly pattern
    plt.subplot(3, 4, 4)
    window_size = min(50, len(test_scores) // 4)
    if window_size > 0:
        moving_avg = np.convolve(test_scores, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(len(moving_avg)), moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
        plt.plot(test_scores, 'b-', alpha=0.3, label='Raw Scores')
        plt.xlabel('Sample Index')
        plt.ylabel('Anomaly Score')
        plt.title('Temporal Anomaly Pattern')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 5. KL divergence evolution
    plt.subplot(3, 4, 5)
    kl_values = []
    for output_batch in test_outputs:
        batch_kl = [kl.item() if hasattr(kl, 'item') else float(kl) for kl in output_batch['kl_losses']]
        kl_values.extend(batch_kl)
    
    if kl_values:
        plt.plot(kl_values[:100], 'g-', alpha=0.7)
        plt.xlabel('Sample')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence (Disentanglement Loss)')
        plt.grid(True, alpha=0.3)
    
    # 6. Feature space visualization (t-SNE)
    plt.subplot(3, 4, 6)
    try:
        from sklearn.manifold import TSNE
        
        # Collect causal factors for t-SNE
        tsne_features = []
        tsne_labels = []
        for i, output_batch in enumerate(test_outputs[:20]):
            for factors in output_batch['causal_factors']:
                if factors.numel() > 0:
                    tsne_features.append(factors.cpu().numpy().mean(axis=0))
                    tsne_labels.append(test_labels[min(i, len(test_labels)-1)])
                    break
        
        if len(tsne_features) > 5:
            tsne_features = np.array(tsne_features)
            tsne_labels = np.array(tsne_labels)
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(tsne_features)-1))
            tsne_result = tsne.fit_transform(tsne_features)
            
            scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                c=tsne_labels, cmap='RdYlBu', alpha=0.7)
            plt.colorbar(scatter)
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title('Causal Factor Space (t-SNE)')
    except ImportError:
        plt.text(0.5, 0.5, 'scikit-learn not available\nfor t-SNE visualization', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('t-SNE Not Available')
    
    # 7. Anomaly score histogram
    plt.subplot(3, 4, 7)
    plt.hist(test_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(test_scores), color='red', linestyle='--', label=f'Mean: {np.mean(test_scores):.3f}')
    plt.axvline(np.median(test_scores), color='green', linestyle='--', label=f'Median: {np.median(test_scores):.3f}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Cumulative anomaly detection
    plt.subplot(3, 4, 8)
    sorted_indices = np.argsort(test_scores)[::-1]  # Sort by score descending
    sorted_labels = test_labels[sorted_indices]
    cumulative_tp = np.cumsum(sorted_labels)
    total_positives = np.sum(test_labels) if np.sum(test_labels) > 0 else 1
    
    recall = cumulative_tp / total_positives
    precision = cumulative_tp / (np.arange(len(sorted_labels)) + 1)
    
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    # 9. Sample trajectory visualization
    plt.subplot(3, 4, 9)
    # Simulate trajectory data for visualization
    sample_trajectories = []
    for output_batch in test_outputs[:3]:
        for detection_batch in output_batch['detections']:
            for frame_detections in detection_batch[:5]:  # First 5 frames
                if len(frame_detections) > 0:
                    bbox = frame_detections[0].cpu().numpy()  # First detection
                    sample_trajectories.append([bbox[0], bbox[1]])  # x, y coordinates
    
    if len(sample_trajectories) > 1:
        trajectories = np.array(sample_trajectories)
        plt.plot(trajectories[:, 0], trajectories[:, 1], 'ro-', alpha=0.6, markersize=4)
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title('Sample Pedestrian Trajectories')
        plt.grid(True, alpha=0.3)
    
    # 10. Model confidence analysis
    plt.subplot(3, 4, 10)
    confidence_bins = np.linspace(0, 1, 11)
    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    bin_counts = np.histogram(test_scores, bins=confidence_bins)[0]
    
    plt.bar(bin_centers, bin_counts, width=0.08, alpha=0.7, color='lightcoral')
    plt.xlabel('Confidence Score Range')
    plt.ylabel('Number of Samples')
    plt.title('Model Confidence Distribution')
    plt.grid(True, alpha=0.3)
    
    # 11. Detection count per frame
    plt.subplot(3, 4, 11)
    detection_counts = []
    for output_batch in test_outputs:
        for detection_batch in output_batch['detections']:
            frame_counts = [len(frame_det) for frame_det in detection_batch]
            detection_counts.extend(frame_counts)
    
    if detection_counts:
        plt.hist(detection_counts, bins=range(max(detection_counts)+2), alpha=0.7, color='lightgreen')
        plt.xlabel('Number of Detections per Frame')
        plt.ylabel('Frequency')
        plt.title('Detection Count Distribution')
        plt.grid(True, alpha=0.3)
    
    # 12. Correlation between factors and anomaly scores
    plt.subplot(3, 4, 12)
    factor_score_correlations = []
    for i, output_batch in enumerate(test_outputs):
        if i < len(test_scores):
            for factors in output_batch['causal_factors']:
                if factors.numel() > 0:
                    factor_mean = factors.cpu().numpy().mean()
                    factor_score_correlations.append([factor_mean, test_scores[i]])
                    break
    
    if len(factor_score_correlations) > 5:
        correlations = np.array(factor_score_correlations)
        plt.scatter(correlations[:, 0], correlations[:, 1], alpha=0.6)
        
        # Add trend line
        z = np.polyfit(correlations[:, 0], correlations[:, 1], 1)
        p = np.poly1d(z)
        plt.plot(correlations[:, 0], p(correlations[:, 0]), "r--", alpha=0.8)
        
        corr_coef = np.corrcoef(correlations[:, 0], correlations[:, 1])[0, 1]
        plt.xlabel('Mean Causal Factor Value')
        plt.ylabel('Anomaly Score')
        plt.title(f'Factor-Score Correlation (r={corr_coef:.3f})')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
def main():
    # Data preparation
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    # Dataset paths - update these to your actual paths
    data_root = "UCSDped2"  # Main folder containing Train and Test subfolders
    
    print("Loading datasets...")
    train_dataset = UCSDped2Dataset(data_root, split='Train', transform=transform, sequence_length=16)
    test_dataset = UCSDped2Dataset(data_root, split='Test', transform=transform, sequence_length=16)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loaders with smaller batch sizes for memory efficiency
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Split train set for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=2)
    train_loader = DataLoader(train_subset, batch_size=2, shuffle=True, num_workers=2)
    
    print("Initializing model...")
    model = CausalAnomalyDetector(num_factors=6, reid_dim=64)  # Reduced parameters
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training
    print("Starting training...")
    trained_model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=10, lr=1e-4  # Reduced epochs for faster training
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Testing
    print("Testing model...")
    test_scores, test_labels, test_outputs = test_model(trained_model, test_loader)
    
    # Calculate metrics
    threshold = 0.5
    predictions = (test_scores > threshold).astype(int)
    accuracy = accuracy_score(test_labels, predictions)
    
    try:
        auc_score = roc_auc_score(test_labels, test_scores)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {auc_score:.4f}")
    except:
        print(f"Test Accuracy: {accuracy:.4f}")
        print("AUC calculation failed (possibly due to single class in test set)")
    
    # Visualizations
    print("Creating visualizations...")
    
    # 1. Training loss plot
    plt.subplot(1, 2, 2)
    plt.plot(range(len(test_scores)), test_scores, 'b-', alpha=0.7, label='Anomaly Scores')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    colors = ['green' if label == 0 else 'red' for label in test_labels]
    plt.scatter(range(len(test_scores)), test_scores, c=colors, alpha=0.6, s=20)
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    plt.title('Test Anomaly Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_and_test_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Comprehensive result visualization
    visualize_results(test_scores, test_labels, test_outputs, test_dataset)
    
    # 3. Bounding box visualization
    visualize_bounding_boxes(trained_model, test_dataset, num_videos=5)
    
    # 4. Additional analysis plots
    create_additional_plots(test_outputs, test_labels, test_scores)
    
    # Save model
    torch.save(trained_model.state_dict(), 'causal_anomaly_detector.pth')
    print("Model saved as 'causal_anomaly_detector.pth'")
    
    return trained_model, test_scores, test_labels

def load_and_test_pretrained():
    """Function to load a pretrained model and test it"""
    print("Loading pretrained model...")
    
    model = CausalAnomalyDetector(num_factors=6, reid_dim=64)
    
    try:
        model.load_state_dict(torch.load('causal_anomaly_detector.pth', map_location=device))
        model.to(device)
        print("Pretrained model loaded successfully!")
        return model
    except FileNotFoundError:
        print("No pretrained model found. Please train the model first.")
        return None

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Enable mixed precision training for memory efficiency
    torch.backends.cudnn.benchmark = True
    
    print("=" * 60)
    print("CAUSAL STRUCTURE LEARNING FOR ANOMALY DETECTION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)
    
    # Run main training and testing
    try:
        trained_model, test_scores, test_labels = main()
        print("\nTraining and testing completed successfully!")
        
        # Print final statistics
        print(f"\nFinal Results:")
        print(f"Mean anomaly score: {np.mean(test_scores):.4f}")
        print(f"Std anomaly score: {np.std(test_scores):.4f}")
        print(f"Max anomaly score: {np.max(test_scores):.4f}")
        print(f"Min anomaly score: {np.min(test_scores):.4f}")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        print("Please check your data path and ensure UCSDped2 dataset is properly structured.")
        print("\nExpected directory structure:")
        print("UCSDped2/")
        print("├── Train/")
        print("│   ├── Train001/")
        print("│   ├── Train002/")
        print("│   └── ...")
        print("└── Test/")
        print("    ├── Test001/")
        print("    ├── Test002/")
        print("    └── ...")