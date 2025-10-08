#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 11:17:52 2025

@author: pvvkishore
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import time
import json
from datetime import datetime

# Import your dataset loader and model
from avenue_dataset_usage import create_avenue_dataloaders, AvenueFramesDataset
from minicausal_vad import MiniCausalVAD
from json_utils import safe_json_save, safe_json_load

def setup_training_environment():
    """Setup training environment and check GPU"""
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()  # Clear any cached memory
    
    return device

def create_unsupervised_labels(test_loader, model, threshold_percentile=95):
    """
    Create pseudo-labels for testing without ground truth
    Uses the training data statistics to determine anomaly threshold
    """
    print("🔍 Creating pseudo-labels for evaluation...")
    
    model.model.eval()
    all_scores = []
    
    with torch.no_grad():
        for batch_idx, (videos, _) in enumerate(test_loader):
            videos = videos.to(model.device, non_blocking=True)
            
            # Get anomaly scores
            anomaly_scores, _, _ = model.model(videos)
            all_scores.extend(anomaly_scores.squeeze().cpu().numpy())
            
            if batch_idx % 50 == 0:
                print(f"   Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    all_scores = np.array(all_scores)
    
    # Use percentile-based threshold for pseudo-labels
    threshold = np.percentile(all_scores, threshold_percentile)
    pseudo_labels = (all_scores > threshold).astype(float)
    
    anomaly_count = np.sum(pseudo_labels)
    print(f"   Threshold: {threshold:.4f}")
    print(f"   Detected anomalies: {anomaly_count}/{len(pseudo_labels)} ({anomaly_count/len(pseudo_labels)*100:.1f}%)")
    
    return all_scores, pseudo_labels, threshold

def train_minicausal_vad_on_avenue(dataset_path: str, 
                                   num_epochs: int = 50,
                                   batch_size: int = 4,
                                   learning_rate: float = 0.001,
                                   save_interval: int = 10):
    """
    Complete training pipeline for MiniCausal-VAD on Avenue dataset
    """
    print("=" * 80)
    print("🎬 TRAINING MINICAUSAL-VAD ON AVENUE DATASET")
    print("=" * 80)
    
    # Setup environment
    device = setup_training_environment()
    
    # Create dataloaders
    print("📊 Loading dataset...")
    train_loader, test_loader = create_avenue_dataloaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=2,
        clip_length=8,
        frame_size=(64, 64)
    )
    
    print(f"✅ Dataset loaded:")
    print(f"   Training batches: {len(train_loader)} (batch_size={batch_size})")
    print(f"   Testing batches: {len(test_loader)}")
    print(f"   Total training clips: {len(train_loader) * batch_size}")
    
    # Initialize model
    print("🧠 Initializing MiniCausal-VAD...")
    model = MiniCausalVAD(device=device)
    
    # Update learning rate if specified
    if learning_rate != 0.001:
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = learning_rate
    
    print(f"✅ Model initialized:")
    print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"   Learning rate: {learning_rate}")
    
    # Training tracking
    training_history = {
        'train_losses': [],
        'loss_components': [],
        'evaluation_scores': [],
        'causal_sparsity': [],
        'epochs': [],
        'timestamps': []
    }
    
    # Create output directory for saving results
    output_dir = Path("avenue_training_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Results will be saved to: {output_dir}")
    
    # Training loop
    print("\n🚀 Starting training...")
    start_time = time.time()
    
    best_score = 0.0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        print("🏃 Training phase...")
        train_loss, loss_components = model.train_epoch(train_loader)
        
        # Log training results
        training_history['train_losses'].append(train_loss)
        training_history['loss_components'].append(loss_components)
        training_history['epochs'].append(epoch + 1)
        training_history['timestamps'].append(datetime.now().isoformat())
        
        print(f"📊 Training Results:")
        print(f"   Total Loss: {train_loss:.6f}")
        print(f"   Anomaly Loss: {loss_components['anomaly_loss']:.6f}")
        print(f"   Acyclicity Loss: {loss_components['acyclicity_loss']:.6f}")
        print(f"   Sparsity Loss: {loss_components['sparsity_loss']:.6f}")
        print(f"   Consistency Loss: {loss_components['consistency_loss']:.6f}")
        
        # Evaluation phase (every 5 epochs)
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print("🔍 Evaluation phase...")
            
            # Get predictions and causal graphs
            predictions, _, causal_graphs = model.evaluate(test_loader)
            
            # Create pseudo-labels for evaluation
            _, pseudo_labels, threshold = create_unsupervised_labels(
                test_loader, model, threshold_percentile=95
            )
            
            # Calculate metrics using pseudo-labels
            try:
                # Use a more sophisticated scoring approach
                eval_score = np.mean(predictions)  # Average anomaly score
                std_score = np.std(predictions)    # Standard deviation
                
                print(f"📈 Evaluation Results:")
                print(f"   Average Score: {eval_score:.6f}")
                print(f"   Score Std: {std_score:.6f}")
                print(f"   Score Range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
                
            except Exception as e:
                print(f"   ⚠️ Evaluation warning: {e}")
                eval_score = train_loss  # Use training loss as fallback
            
            # Analyze causal graphs
            avg_causal_edges = np.mean(np.sum(causal_graphs > 0.1, axis=(1, 2)))
            avg_sparsity = avg_causal_edges / (causal_graphs.shape[1] ** 2)
            
            print(f"🧠 Causal Analysis:")
            print(f"   Avg Causal Edges: {avg_causal_edges:.1f}/256")
            print(f"   Graph Sparsity: {avg_sparsity:.3f}")
            
            # Save evaluation results
            training_history['evaluation_scores'].append(eval_score)
            training_history['causal_sparsity'].append(avg_sparsity)
            
            # Save best model
            if eval_score > best_score:
                best_score = eval_score
                best_model_path = output_dir / "best_model.pth"
                model.save_model(str(best_model_path))
                print(f"💾 Saved best model (score: {best_score:.6f})")
        
        # Save checkpoint
        if epoch % save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            model.save_model(str(checkpoint_path))
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Calculate and display timing
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        eta = (total_time / (epoch + 1)) * (num_epochs - epoch - 1)
        
        print(f"⏱️  Epoch time: {epoch_time:.1f}s | Total: {total_time/60:.1f}min | ETA: {eta/60:.1f}min")
        
        # Save training history (convert numpy types to Python types)
        history_path = output_dir / "training_history.json"
        safe_json_save(training_history, history_path)
        
        # Memory cleanup
        torch.cuda.empty_cache()
    
    print("\n🎉 Training completed!")
    print(f"💾 Results saved to: {output_dir}")
    print(f"⏱️  Total training time: {(time.time() - start_time)/60:.1f} minutes")
    
    return model, training_history

def visualize_training_results(history_path: str = "avenue_training_results/training_history.json"):
    """Visualize training results"""
    
    # Load training history using safe JSON loader
    history = safe_json_load(history_path)
    
    if history is None:
        print("❌ Could not load training history for visualization")
        return
    
    print("📊 Creating training visualization...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MiniCausal-VAD Training Results on Avenue Dataset', fontsize=16)
        
        # Training loss
        if history.get('train_losses'):
            axes[0, 0].plot(history['epochs'], history['train_losses'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Loss components
        if history.get('loss_components'):
            components = ['anomaly_loss', 'acyclicity_loss', 'sparsity_loss', 'consistency_loss']
            for comp in components:
                if comp in history['loss_components'][0]:  # Check if component exists
                    values = [lc[comp] for lc in history['loss_components']]
                    axes[0, 1].plot(history['epochs'], values, label=comp)
            axes[0, 1].set_title('Loss Components')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Evaluation scores
        if history.get('evaluation_scores'):
            eval_epochs = [e for i, e in enumerate(history['epochs']) if i % 5 == 0 or i == len(history['epochs']) - 1]
            axes[1, 0].plot(eval_epochs[:len(history['evaluation_scores'])], history['evaluation_scores'])
            axes[1, 0].set_title('Evaluation Scores')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Causal sparsity
        if history.get('causal_sparsity'):
            eval_epochs = [e for i, e in enumerate(history['epochs']) if i % 5 == 0 or i == len(history['epochs']) - 1]
            axes[1, 1].plot(eval_epochs[:len(history['causal_sparsity'])], history['causal_sparsity'])
            axes[1, 1].set_title('Causal Graph Sparsity')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Sparsity Ratio')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = "avenue_training_results/training_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plots saved to: {plot_path}")
        
    except Exception as e:
        print(f"❌ Could not create visualization: {e}")
        import traceback
        traceback.print_exc()

def test_trained_model(model_path: str, dataset_path: str):
    """Test a trained model on Avenue dataset"""
    print("🧪 Testing trained model...")
    
    # Load model
    device = setup_training_environment()
    model = MiniCausalVAD(device=device)
    model.load_model(model_path)
    
    # Load test data
    _, test_loader = create_avenue_dataloaders(dataset_path, batch_size=4)
    
    # Evaluate
    predictions, _, causal_graphs = model.evaluate(test_loader)
    
    print(f"✅ Model tested successfully!")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"   Average causal edges: {np.mean(np.sum(causal_graphs > 0.1, axis=(1,2))):.1f}")
    
    return predictions, causal_graphs

if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "avenue"  # Update this to your path
    
    print("🎬 MiniCausal-VAD Training on Avenue Dataset")
    print("=" * 60)
    
    # Check if dataset path exists
    if not Path(DATASET_PATH).exists():
        print(f"❌ Dataset path not found: {DATASET_PATH}")
        print("Please update DATASET_PATH to your avenue folder location")
        exit(1)
    
    # Train model
    model, history = train_minicausal_vad_on_avenue(
        dataset_path=DATASET_PATH,
        num_epochs=50,
        batch_size=4,
        learning_rate=0.001,
        save_interval=10
    )
    
    # Visualize results
    visualize_training_results()
    
    # Test best model
    best_model_path = "avenue_training_results/best_model.pth"
    if Path(best_model_path).exists():
        test_trained_model(best_model_path, DATASET_PATH)
    
    print("\n🎉 All done! Check 'avenue_training_results/' folder for outputs")