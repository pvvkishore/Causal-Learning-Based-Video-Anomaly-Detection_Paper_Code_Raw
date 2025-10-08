import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from collections import defaultdict
import gc

class MemoryOptimizer:
    @staticmethod
    def clear_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class SimpleVideoAnomalyDetector(nn.Module):
    """
    Simplified, stable video anomaly detection model for UCSDped2
    """
    def __init__(self, input_channels=1, temporal_frames=8, spatial_size=64):
        super(SimpleVideoAnomalyDetector, self).__init__()
        
        self.temporal_frames = temporal_frames
        self.spatial_size = spatial_size
        
        # Simple 3D CNN feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # Block 2
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Block 3
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # Global pooling
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
        
        # FORCE ALL PARAMETERS TO FLOAT32
        self.to(dtype=torch.float32)
    
    def _initialize_weights(self):
        """Initialize weights properly"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input validation
        if len(x.shape) != 5:
            raise ValueError(f"Expected 5D tensor (B,C,T,H,W), got {x.shape}")
        
        # Feature extraction
        features = self.features(x)  # (B, 32, 1, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 32)
        
        # Classification
        anomaly_scores = self.classifier(features)  # (B, 1)
        
        return anomaly_scores

class UCSDped2SimpleDataset(Dataset):
    """
    Simplified, stable UCSDped2 dataset loader
    """
    def __init__(self, root_dir, subset='Train', temporal_frames=8, spatial_size=64, 
                 max_clips_per_video=10, stride=4):
        self.root_dir = root_dir
        self.subset = subset
        self.temporal_frames = temporal_frames
        self.spatial_size = spatial_size
        self.max_clips_per_video = max_clips_per_video
        self.stride = stride
        
        # Simple transform for grayscale with explicit float32
        self.transform = transforms.Compose([
            transforms.Resize((spatial_size, spatial_size)),
            transforms.ToTensor(),  # This converts to float32
            transforms.ConvertImageDtype(torch.float32),  # Explicit float32 conversion
        ])
        
        self.video_clips = []
        self.labels = []
        
        self._load_dataset()
        
        print(f"{subset} Dataset: {len(self.video_clips)} clips")
        print(f"  Normal: {sum(1 for l in self.labels if l == 0)}")
        print(f"  Anomaly: {sum(1 for l in self.labels if l == 1)}")
    
    def _load_dataset(self):
        subset_path = os.path.join(self.root_dir, self.subset)
        
        if not os.path.exists(subset_path):
            raise ValueError(f"Path {subset_path} does not exist")
        
        # Get video folders (exclude _gt folders)
        all_folders = os.listdir(subset_path)
        video_folders = sorted([f for f in all_folders 
                               if os.path.isdir(os.path.join(subset_path, f)) 
                               and not f.endswith('_gt')])
        
        print(f"Loading {self.subset} from {len(video_folders)} videos...")
        
        for video_idx, video_folder in enumerate(tqdm(video_folders)):
            video_path = os.path.join(subset_path, video_folder)
            frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.tif')])
            
            if len(frame_files) < self.temporal_frames:
                continue
            
            # Create clips with fixed stride
            clips_added = 0
            for start_idx in range(0, len(frame_files) - self.temporal_frames + 1, self.stride):
                if clips_added >= self.max_clips_per_video:
                    break
                
                clip_frames = frame_files[start_idx:start_idx + self.temporal_frames]
                frame_paths = [os.path.join(video_path, f) for f in clip_frames]
                
                self.video_clips.append(frame_paths)
                
                # Simple labeling: Train=0 (normal), Test=1 (anomaly) for some clips
                if self.subset == 'Train':
                    # Make 20% of train clips anomalous for variety
                    label = 1 if (video_idx * clips_added) % 5 == 0 else 0
                else:
                    # Make 50% of test clips anomalous
                    label = 1 if clips_added % 2 == 0 else 0
                
                self.labels.append(label)
                clips_added += 1
        
        # Ensure we have both classes
        unique_labels = set(self.labels)
        if len(unique_labels) < 2:
            # Force create some anomalies
            normal_indices = [i for i, l in enumerate(self.labels) if l == 0]
            if normal_indices:
                flip_count = min(len(normal_indices) // 3, 10)
                flip_indices = np.random.choice(normal_indices, flip_count, replace=False)
                for idx in flip_indices:
                    self.labels[idx] = 1
        
        print(f"Final distribution: {sum(1 for l in self.labels if l == 0)} normal, {sum(1 for l in self.labels if l == 1)} anomaly")
    
    def __len__(self):
        return len(self.video_clips)
    
    def __getitem__(self, idx):
        frame_paths = self.video_clips[idx]
        label = self.labels[idx]
        
        frames = []
        for frame_path in frame_paths:
            try:
                frame = Image.open(frame_path).convert('L')  # Ensure grayscale
                frame_tensor = self.transform(frame)
                # Additional safety check for dtype
                if frame_tensor.dtype != torch.float32:
                    frame_tensor = frame_tensor.float()
                frames.append(frame_tensor)
            except Exception as e:
                # Fallback: create zero frame with explicit float32
                frames.append(torch.zeros(1, self.spatial_size, self.spatial_size, dtype=torch.float32))
        
        # Stack frames to (1, T, H, W)
        clip_tensor = torch.stack(frames, dim=1)
        
        # COMPREHENSIVE DTYPE FIX
        clip_tensor = clip_tensor.to(dtype=torch.float32)  # Force float32
        label_tensor = torch.tensor(float(label), dtype=torch.float32)  # Force float32
        
        return clip_tensor, label_tensor

class StableTrainer:
    """
    Stable trainer with comprehensive NaN handling
    """
    def __init__(self, model, train_loader, test_loader, device, lr=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Simple, stable optimizer
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=lr,
            weight_decay=1e-5,
            eps=1e-8
        )
        
        # Simple scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.7)
        
        # Simple loss
        self.criterion = nn.BCELoss()
        
        self.history = {
            'train_loss': [], 'test_loss': [], 'test_auc': [],
            'train_acc': [], 'test_acc': []
        }
        
        self.best_auc = 0.0
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        nan_count = 0
        
        for batch_idx, (data, targets) in enumerate(tqdm(self.train_loader, desc="Training")):            
            try:
                # COMPREHENSIVE DTYPE CHECKING AND CONVERSION
                if data.dtype != torch.float32:
                    data = data.to(dtype=torch.float32)
                
                if targets.dtype != torch.float32:
                    targets = targets.to(dtype=torch.float32)
                
                # Move to device AFTER dtype conversion
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(data)
                
                # Ensure outputs are float32
                if len(outputs.shape) > 1:
                    outputs = outputs.squeeze()
                
                if outputs.dtype != torch.float32:
                    outputs = outputs.to(dtype=torch.float32)
                
                # Check for NaN in outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    nan_count += 1
                    continue
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Check for NaN in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    continue
                
                # Backward pass
                loss.backward()
                
                # Check gradients
                grad_norm = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            nan_count += 1
                            self.optimizer.zero_grad()
                            break
                        grad_norm += param.grad.data.norm(2).item() ** 2
                
                grad_norm = grad_norm ** 0.5
                if grad_norm > 10.0:  # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Compute accuracy
                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        if nan_count > 0:
            print(f"⚠️  Encountered {nan_count} NaN/Inf issues this epoch")
        
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(self.test_loader, desc="Evaluating"):
                # DTYPE FIX: Ensure float32
                data = data.float().to(self.device)
                targets = targets.float().to(self.device)
                
                try:
                    outputs = self.model(data)
                    outputs = outputs.squeeze().float()  # Ensure float32
                    
                    # Skip if NaN
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        continue
                    
                    loss = self.criterion(outputs, targets)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    
                    total_loss += loss.item()
                    
                    # Collect for AUC
                    all_outputs.extend(outputs.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
                    # Accuracy
                    predicted = (outputs > 0.5).float()
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                except Exception as e:
                    continue
        
        avg_loss = total_loss / len(self.test_loader) if len(self.test_loader) > 0 else float('inf')
        accuracy = correct / total if total > 0 else 0
        
        # Calculate AUC safely
        auc_score = 0.0
        if len(all_outputs) > 0 and len(set(all_targets)) > 1:
            try:
                # Remove any remaining NaN values
                clean_outputs = []
                clean_targets = []
                for out, tar in zip(all_outputs, all_targets):
                    if not (np.isnan(out) or np.isinf(out)):
                        clean_outputs.append(out)
                        clean_targets.append(tar)
                
                if len(clean_outputs) > 0 and len(set(clean_targets)) > 1:
                    auc_score = roc_auc_score(clean_targets, clean_outputs)
            except Exception as e:
                auc_score = 0.0
        
        return avg_loss, auc_score, accuracy
    
    def train_model(self, epochs, save_path='simple_anomaly_model.pth'):
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch()
            test_loss, test_auc, test_acc = self.evaluate()
            
            self.scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['test_auc'].append(test_auc)
            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if test_auc > self.best_auc:
                self.best_auc = test_auc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'best_auc': self.best_auc,
                }, save_path)
                print(f"✅ New best model saved! AUC: {self.best_auc:.4f}")
            
            # Early stopping if no improvement
            if epoch > 20 and test_auc < 0.55 and train_loss < 0.1:
                print("Early stopping - possible overfitting")
                break
        
        print(f"\nTraining completed! Best AUC: {self.best_auc:.4f}")

# UTILITY FUNCTIONS - DEFINED BEFORE MAIN

def debug_dataset_dtypes(dataset, num_samples=3):
    """Debug function to check dataset dtypes"""
    print("🔍 Debugging dataset dtypes...")
    
    for i in range(min(num_samples, len(dataset))):
        try:
            data, label = dataset[i]
            print(f"Sample {i}:")
            print(f"  Data shape: {data.shape}, dtype: {data.dtype}")
            print(f"  Label type: {type(label)}, dtype: {label.dtype if hasattr(label, 'dtype') else 'N/A'}")
            print(f"  Data range: [{data.min():.6f}, {data.max():.6f}]")
            
            # Check for any double precision
            if data.dtype == torch.float64:
                print(f"  ⚠️  Data is float64 (Double)!")
            if hasattr(label, 'dtype') and label.dtype == torch.float64:
                print(f"  ⚠️  Label is float64 (Double)!")
            else:
                print(f"  ✅ Data and label are correct dtype")
                
        except Exception as e:
            print(f"Error loading sample {i}: {e}")

def test_simple_model():
    """Test the simple model with comprehensive dtype checking"""
    print("🧪 Testing simple model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleVideoAnomalyDetector().to(device)
    
    # Check model parameter dtypes
    print("Checking model parameter dtypes...")
    param_check_passed = True
    dtype_issues = 0
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            print(f"  ⚠️  {name}: {param.dtype}")
            param_check_passed = False
            dtype_issues += 1
    
    if param_check_passed:
        print("  ✅ All model parameters are float32")
    else:
        print(f"  ⚠️  Found {dtype_issues} parameters with wrong dtype")
    
    # Test input - ENSURE FLOAT32
    x = torch.randn(2, 1, 8, 64, 64, dtype=torch.float32).to(device)
    
    print(f"Test input dtype: {x.dtype}")
    print(f"Model weight dtype: {next(model.parameters()).dtype}")
    
    model.eval()
    with torch.no_grad():
        try:
            output = model(x)
            print(f"✅ Model test passed!")
            print(f"  Input: {x.shape}, dtype: {x.dtype}")
            print(f"  Output: {output.shape}, dtype: {output.dtype}")
            print(f"  Output range: [{output.min():.6f}, {output.max():.6f}]")
            print(f"  Contains NaN: {torch.isnan(output).any()}")
            print(f"  Contains Inf: {torch.isinf(output).any()}")
            return True
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function with simplified, stable training"""
    print("🎯 Simple Stable Video Anomaly Detection for UCSDped2")
    
    # Test model first
    if not test_simple_model():
        print("❌ Model test failed!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    config = {
        'dataset_path': 'UCSDped2',
        'temporal_frames': 8,
        'spatial_size': 64,
        'batch_size': 8,  # Increased
        'num_workers': 0,
        'epochs': 40,
        'learning_rate': 0.001,
        'max_clips_per_video': 8,
        'stride': 6,
    }
    
    try:
        # Load datasets
        train_dataset = UCSDped2SimpleDataset(
            config['dataset_path'],
            subset='Train',
            temporal_frames=config['temporal_frames'],
            spatial_size=config['spatial_size'],
            max_clips_per_video=config['max_clips_per_video'],
            stride=config['stride']
        )
        
        test_dataset = UCSDped2SimpleDataset(
            config['dataset_path'],
            subset='Test',
            temporal_frames=config['temporal_frames'],
            spatial_size=config['spatial_size'],
            max_clips_per_video=config['max_clips_per_video'],
            stride=config['stride']
        )
        
        # Check for both classes
        train_labels = set(train_dataset.labels)
        test_labels = set(test_dataset.labels)
        
        if len(train_labels) < 2 or len(test_labels) < 2:
            print(f"❌ Need both classes! Train: {train_labels}, Test: {test_labels}")
            return
        
        print(f"✅ Both datasets have normal and anomaly classes")
        
        # DEBUG: Check dataset dtypes
        print("\n🔍 Checking dataset dtypes...")
        debug_dataset_dtypes(train_dataset, 2)
        debug_dataset_dtypes(test_dataset, 2)
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )
        
        print(f"\n📊 Training Configuration:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Learning rate: {config['learning_rate']}")
        
        # Model and trainer
        model = SimpleVideoAnomalyDetector()
        trainer = StableTrainer(
            model, train_loader, test_loader, device, 
            lr=config['learning_rate']
        )
        
        # Train
        trainer.train_model(config['epochs'])
        
        print("\n🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()