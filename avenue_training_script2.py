import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from pathlib import Path
import time
import json
from datetime import datetime

# ============================================================================
# MODEL DEFINITION (Complete - no imports needed)
# ============================================================================

class CompactFeatureExtractor(nn.Module):
    """Lightweight 3D CNN for video feature extraction - Memory efficient"""
    def __init__(self, input_channels=3, feature_dim=64):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(input_channels, 16, (3,3,3), stride=(1,2,2), padding=1)
        self.conv3d_2 = nn.Conv3d(16, 32, (3,3,3), stride=(2,2,2), padding=1)
        self.conv3d_3 = nn.Conv3d(32, 64, (3,3,3), stride=(2,2,2), padding=1)
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.fc = nn.Linear(64 * 4 * 4 * 4, feature_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.conv3d_1(x))
        x = F.relu(self.conv3d_2(x))  
        x = F.relu(self.conv3d_3(x))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc(x))
        return x

class DifferentiableCausalDiscovery(nn.Module):
    """Lightweight NOTEARS-based causal discovery - GPU memory optimized"""
    def __init__(self, num_variables=16, hidden_dim=32):
        super().__init__()
        self.num_variables = num_variables
        
        self.causal_net = nn.Sequential(
            nn.Linear(num_variables, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_variables * num_variables),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        batch_size = features.size(0)
        
        adj_vec = self.causal_net(features)
        adj_matrix = adj_vec.view(batch_size, self.num_variables, self.num_variables)
        
        # Ensure no self-loops
        mask = torch.eye(self.num_variables, device=features.device)
        adj_matrix = adj_matrix * (1 - mask)
        
        return adj_matrix
    
    def acyclicity_constraint(self, adj_matrix):
        """NOTEARS acyclicity constraint - prevents cycles in causal graph"""
        batch_mean_adj = adj_matrix.mean(dim=0)
        adj_squared = torch.matrix_power(batch_mean_adj + 1e-8, 2)
        trace = torch.trace(adj_squared)
        return trace

class CausalAnomalyDetector(nn.Module):
    """Main anomaly detection using causal graph embeddings"""
    def __init__(self, feature_dim=64, causal_dim=16, hidden_dim=128):
        super().__init__()
        
        self.feature_extractor = CompactFeatureExtractor(feature_dim=causal_dim)
        self.causal_discovery = DifferentiableCausalDiscovery(num_variables=causal_dim)
        
        self.graph_encoder = nn.Sequential(
            nn.Linear(causal_dim * causal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64)
        )
        
        self.anomaly_predictor = nn.Sequential(
            nn.Linear(causal_dim + 64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, video_clips):
        features = self.feature_extractor(video_clips)
        causal_adj = self.causal_discovery(features)
        
        batch_size = causal_adj.size(0)
        graph_features = self.graph_encoder(causal_adj.view(batch_size, -1))
        
        combined_features = torch.cat([features, graph_features], dim=1)
        anomaly_scores = self.anomaly_predictor(combined_features)
        
        return anomaly_scores, causal_adj, features

# ============================================================================
# IMPROVED TRAINING CLASS
# ============================================================================

class ImprovedMiniCausalVAD:
    """Improved training pipeline addressing the identified issues"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model = CausalAnomalyDetector().to(device)
        
        # Improved optimizer with better learning rates
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.0005,  # Reduced learning rate for stability
            weight_decay=0.001  # Reduced weight decay
        )
        
        # FIXED: Better balanced loss weights
        self.anomaly_weight = 1.0      # Primary task
        self.causal_weight = 0.01      # Reduced - was causing dominance
        self.sparsity_weight = 0.001   # Much reduced - was killing causal learning
        self.consistency_weight = 0.01 # Reduced
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        print(f"ImprovedMiniCausalVAD initialized on {device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def compute_improved_loss(self, anomaly_scores, causal_adj, targets, features):
        """Improved multi-objective loss function"""
        
        # 1. Anomaly detection loss with pseudo-labels
        with torch.no_grad():
            # Create pseudo-labels: 5% random anomalies for training
            pseudo_targets = (torch.rand_like(targets) > 0.95).float()
        
        # Use focal loss to handle class imbalance
        alpha = 0.25
        gamma = 2.0
        ce_loss = F.binary_cross_entropy(anomaly_scores.squeeze(), pseudo_targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1-pt)**gamma * ce_loss
        anomaly_loss = focal_loss.mean()
        
        # 2. Modified acyclicity constraint (more stable)
        batch_mean_adj = causal_adj.mean(dim=0)
        acyclicity_loss = torch.trace(torch.mm(batch_mean_adj, batch_mean_adj))
        
        # 3. Adaptive sparsity regularization
        target_sparsity = 0.3  # Want 30% of edges to be active
        current_sparsity = torch.mean((causal_adj > 0.1).float())
        sparsity_loss = torch.abs(current_sparsity - target_sparsity)
        
        # 4. Improved consistency loss
        normal_mask = (pseudo_targets == 0)
        if normal_mask.sum() > 1:
            normal_adj = causal_adj[normal_mask]
            pairwise_distances = []
            
            for i in range(len(normal_adj)):
                for j in range(i+1, len(normal_adj)):
                    dist = torch.mean(torch.abs(normal_adj[i] - normal_adj[j]))
                    pairwise_distances.append(dist)
            
            if pairwise_distances:
                avg_distance = torch.stack(pairwise_distances).mean()
                consistency_loss = torch.abs(avg_distance - 0.1)
            else:
                consistency_loss = torch.tensor(0.0, device=self.device)
        else:
            consistency_loss = torch.tensor(0.0, device=self.device)
        
        # 5. Structure encouragement (NEW)
        edge_count = torch.sum(causal_adj > 0.1)
        min_edges = 10
        max_edges = 40
        
        if edge_count < min_edges:
            structure_loss = (min_edges - edge_count) * 0.01
        elif edge_count > max_edges:
            structure_loss = (edge_count - max_edges) * 0.01
        else:
            structure_loss = torch.tensor(0.0, device=self.device)
        
        total_loss = (self.anomaly_weight * anomaly_loss + 
                     self.causal_weight * acyclicity_loss +
                     self.sparsity_weight * sparsity_loss +
                     self.consistency_weight * consistency_loss +
                     0.01 * structure_loss)
        
        return total_loss, {
            'anomaly_loss': anomaly_loss.item(),
            'acyclicity_loss': acyclicity_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'structure_loss': structure_loss.item(),
            'edge_count': edge_count.item(),
            'sparsity_ratio': current_sparsity.item()
        }
    
    def train_epoch_improved(self, dataloader):
        """Improved training loop with better monitoring"""
        self.model.train()
        total_loss = 0
        loss_components = {
            'anomaly_loss': 0, 'acyclicity_loss': 0, 'sparsity_loss': 0, 
            'consistency_loss': 0, 'structure_loss': 0, 'edge_count': 0,
            'sparsity_ratio': 0
        }
        
        for batch_idx, (videos, labels) in enumerate(dataloader):
            videos = videos.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()
            
            self.optimizer.zero_grad()
            
            # Forward pass
            anomaly_scores, causal_adj, features = self.model(videos)
            
            # Compute improved loss
            loss, components = self.compute_improved_loss(anomaly_scores, causal_adj, labels, features)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"NaN loss detected at batch {batch_idx}, skipping...")
                continue
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in components.items():
                loss_components[key] += value
            
            # Debug print every 20 batches
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}: Loss={loss.item():.6f}, "
                      f"Edges={components['edge_count']:.0f}, "
                      f"Sparsity={components['sparsity_ratio']:.3f}")
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # Average losses
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_components = {k: v/num_batches for k, v in loss_components.items()}
        
        # Update learning rate
        self.scheduler.step(avg_loss)
        
        return avg_loss, avg_components
    
    def evaluate_improved(self, dataloader):
        """Improved evaluation with better statistics"""
        self.model.eval()
        predictions = []
        causal_graphs = []
        
        with torch.no_grad():
            for videos, _ in dataloader:
                videos = videos.to(self.device, non_blocking=True)
                anomaly_scores, causal_adj, features = self.model(videos)
                
                predictions.extend(anomaly_scores.squeeze().cpu().numpy())
                causal_graphs.append(causal_adj.cpu().numpy())
                
                del videos, anomaly_scores, causal_adj, features
                torch.cuda.empty_cache()
        
        predictions = np.array(predictions)
        causal_graphs = np.vstack(causal_graphs)
        
        # Better evaluation metrics
        eval_metrics = {
            'mean_score': float(np.mean(predictions)),
            'std_score': float(np.std(predictions)),
            'min_score': float(np.min(predictions)),
            'max_score': float(np.max(predictions)),
            'score_range': float(np.max(predictions) - np.min(predictions)),
            'avg_edges': float(np.mean(np.sum(causal_graphs > 0.1, axis=(1,2)))),
            'avg_sparsity': float(np.mean(np.sum(causal_graphs > 0.1, axis=(1,2)) / 256)),
            'unique_graphs': len(np.unique(causal_graphs.reshape(len(causal_graphs), -1), axis=0))
        }
        
        return predictions, causal_graphs, eval_metrics

# ============================================================================
# JSON UTILITIES (Inline to avoid import issues)
# ============================================================================

def convert_to_json_serializable(obj):
    """Convert numpy types to JSON-compatible types"""
    if isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

def safe_json_save(data, filepath):
    """Safely save data to JSON file"""
    try:
        serializable_data = convert_to_json_serializable(data)
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"✅ Data saved to {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save JSON: {e}")
        return False

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_improved_minicausal_vad(dataset_path: str, 
                                 num_epochs: int = 100,
                                 batch_size: int = 4,
                                 save_interval: int = 20):
    """
    Complete improved training pipeline
    """
    print("=" * 80)
    print("🔧 IMPROVED MINICAUSAL-VAD TRAINING")
    print("=" * 80)
    
    # Setup environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Create dataloaders
    print("📊 Loading dataset...")
    try:
        from avenue_dataset_usage import create_avenue_dataloaders
        
        train_loader, test_loader = create_avenue_dataloaders(
            dataset_path=dataset_path,
            batch_size=batch_size,
            num_workers=2,
            clip_length=8,
            frame_size=(64, 64)
        )
        
        print(f"✅ Dataset loaded:")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Testing batches: {len(test_loader)}")
        
    except ImportError:
        print("❌ Cannot import avenue_dataset_usage. Please ensure the file exists.")
        return None, None
    
    # Initialize improved model
    print("🧠 Initializing Improved MiniCausal-VAD...")
    model = ImprovedMiniCausalVAD(device=device)
    
    # Training tracking
    training_history = {
        'train_losses': [],
        'loss_components': [],
        'evaluation_metrics': [],
        'epochs': [],
        'learning_rates': []
    }
    
    # Create output directory
    output_dir = Path("improved_avenue_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Results will be saved to: {output_dir}")
    
    # Training loop
    print("\n🚀 Starting improved training...")
    best_score_range = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        print("🏃 Training phase...")
        train_loss, loss_components = model.train_epoch_improved(train_loader)
        
        current_lr = model.optimizer.param_groups[0]['lr']
        
        print(f"📊 Training Results:")
        print(f"   Total Loss: {train_loss:.6f}")
        print(f"   Anomaly Loss: {loss_components['anomaly_loss']:.6f}")
        print(f"   Acyclicity Loss: {loss_components['acyclicity_loss']:.6f}")
        print(f"   Sparsity Loss: {loss_components['sparsity_loss']:.6f}")
        print(f"   Structure Loss: {loss_components['structure_loss']:.6f}")
        print(f"   Avg Edges: {loss_components['edge_count']:.1f}")
        print(f"   Learning Rate: {current_lr:.2e}")
        
        # Store training history
        training_history['train_losses'].append(train_loss)
        training_history['loss_components'].append(loss_components)
        training_history['epochs'].append(epoch + 1)
        training_history['learning_rates'].append(current_lr)
        
        # Evaluation phase (every 5 epochs)
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print("🔍 Evaluation phase...")
            predictions, causal_graphs, eval_metrics = model.evaluate_improved(test_loader)
            
            print(f"📈 Evaluation Results:")
            for key, value in eval_metrics.items():
                print(f"   {key}: {value:.6f}")
            
            training_history['evaluation_metrics'].append(eval_metrics)
            
            # Save best model
            if eval_metrics['score_range'] > best_score_range:
                best_score_range = eval_metrics['score_range']
                best_model_path = output_dir / "best_improved_model.pth"
                torch.save({
                    'model_state_dict': model.model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'epoch': epoch,
                    'eval_metrics': eval_metrics
                }, str(best_model_path))
                print(f"💾 Saved best model (score range: {best_score_range:.6f})")
        
        # Save checkpoint
        if epoch % save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'scheduler_state_dict': model.scheduler.state_dict(),
                'epoch': epoch,
                'training_history': training_history
            }, str(checkpoint_path))
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save training history
        history_path = output_dir / "improved_training_history.json"
        safe_json_save(training_history, history_path)
        
        # Memory cleanup
        torch.cuda.empty_cache()
    
    print("\n🎉 Improved training completed!")
    print(f"💾 Results saved to: {output_dir}")
    
    return model, training_history

# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def diagnose_training_issues(history_file: str):
    """Analyze training history to identify issues"""
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"❌ History file not found: {history_file}")
        return
    except Exception as e:
        print(f"❌ Error loading history: {e}")
        return
    
    print("🔍 TRAINING DIAGNOSIS")
    print("=" * 50)
    
    # Check loss progression
    losses = history.get('train_losses', [])
    if len(losses) > 10:
        initial_loss = np.mean(losses[:5])
        final_loss = np.mean(losses[-5:])
        reduction = (initial_loss - final_loss) / initial_loss * 100
        
        print(f"📉 Loss Reduction: {reduction:.1f}%")
        if reduction < 10:
            print("⚠️ WARNING: Minimal loss reduction detected")
    
    # Check component balance
    if history.get('loss_components'):
        latest = history['loss_components'][-1]
        total = sum(latest.values())
        
        print(f"📊 Final Loss Components:")
        for component, value in latest.items():
            percentage = (value / total) * 100 if total > 0 else 0
            print(f"   {component}: {percentage:.1f}%")
            
            if component == 'sparsity_loss' and percentage > 80:
                print("⚠️ WARNING: Sparsity loss dominates - reduce sparsity_weight")
    
    # Check evaluation metrics
    if history.get('evaluation_scores'):
        eval_scores = history['evaluation_scores']
        if len(eval_scores) > 0 and all(score == 0.0 for score in eval_scores):
            print("⚠️ WARNING: All evaluation scores are zero - model not learning properly")

if __name__ == "__main__":
    # First diagnose previous training
    print("🔍 Diagnosing previous training issues...")
    diagnose_training_issues("avenue_training_results/training_history.json")
    
    print("\n" + "="*60)
    print("Starting improved training with fixes...")
    
    # Configuration
    DATASET_PATH = "avenue"  # Update this to your actual path
    
    if not Path(DATASET_PATH).exists():
        print(f"❌ Dataset path not found: {DATASET_PATH}")
        print("Please update DATASET_PATH to your avenue folder location")
        exit(1)
    
    try:
        model, history = train_improved_minicausal_vad(
            dataset_path=DATASET_PATH,
            num_epochs=80,
            batch_size=4,
            save_interval=10
        )
        
        if model is not None:
            print("✅ Improved training completed successfully!")
        else:
            print("❌ Training failed - check error messages above")
            
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()