"""
Visualization utilities for video anomaly detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import cv2


def plot_anomaly_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """
    Plot anomaly scores over time with ground truth labels.
    
    Args:
        scores: Anomaly scores
        labels: Ground truth labels
        save_path: Path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot scores
    plt.plot(scores, label='Anomaly Score', linewidth=1.5, alpha=0.8)
    
    # Highlight anomalous regions
    anomaly_regions = np.where(labels == 1)[0]
    if len(anomaly_regions) > 0:
        plt.scatter(
            anomaly_regions,
            scores[anomaly_regions],
            color='red',
            s=10,
            alpha=0.5,
            label='Ground Truth Anomaly'
        )
    
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Anomaly Score', fontsize=12)
    plt.title('Anomaly Detection Results', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8)
):
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        save_path: Path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    auc_score: float,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8)
):
    """
    Plot Precision-Recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        auc_score: AUC-PR score
        save_path: Path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    plt.plot(recall, precision, linewidth=2, 
             label=f'PR curve (AUC = {auc_score:.3f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_visualization(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_path: str,
    num_frames: int = 8
):
    """
    Save visualization comparing original and reconstructed frames.
    
    Args:
        original: Original frames (T, C, H, W)
        reconstructed: Reconstructed frames (T, C, H, W)
        save_path: Path to save the visualization
        num_frames: Number of frames to display
    """
    # Select frames to display
    T = min(original.shape[0], num_frames)
    indices = np.linspace(0, original.shape[0] - 1, T, dtype=int)
    
    fig, axes = plt.subplots(2, T, figsize=(T * 2, 4))
    
    for i, idx in enumerate(indices):
        # Original frame
        orig_frame = original[idx].transpose(1, 2, 0)
        orig_frame = (orig_frame + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        orig_frame = np.clip(orig_frame, 0, 1)
        
        axes[0, i].imshow(orig_frame)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10, fontweight='bold')
        
        # Reconstructed frame
        recon_frame = reconstructed[idx].transpose(1, 2, 0)
        recon_frame = (recon_frame + 1) / 2  # Denormalize
        recon_frame = np.clip(recon_frame, 0, 1)
        
        axes[1, i].imshow(recon_frame)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_attention_weights(
    attention_weights: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights matrix (T, T)
        save_path: Path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        attention_weights,
        cmap='viridis',
        annot=False,
        square=True,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Time Step', fontsize=12)
    plt.title('Temporal Attention Weights', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_results(
    frames: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: str
):
    """
    Create a comprehensive visualization of results.
    
    Args:
        frames: Video frames (T, C, H, W)
        scores: Anomaly scores (T,)
        labels: Ground truth labels (T,)
        save_path: Path to save the visualization
    """
    num_frames = min(8, len(frames))
    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, num_frames + 1, width_ratios=[1]*num_frames + [0.5])
    
    # Display frames
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(gs[0, i])
        
        frame = frames[idx].transpose(1, 2, 0)
        frame = (frame + 1) / 2  # Denormalize
        frame = np.clip(frame, 0, 1)
        
        ax.imshow(frame)
        ax.axis('off')
        
        # Color border based on label
        color = 'red' if labels[idx] == 1 else 'green'
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        
        ax.set_title(f'Frame {idx}\nScore: {scores[idx]:.3f}',
                    fontsize=8)
    
    # Plot anomaly scores
    ax_score = fig.add_subplot(gs[1, :])
    ax_score.plot(scores, linewidth=2, label='Anomaly Score')
    
    # Highlight anomalous regions
    anomaly_regions = np.where(labels == 1)[0]
    if len(anomaly_regions) > 0:
        ax_score.scatter(
            anomaly_regions,
            scores[anomaly_regions],
            color='red',
            s=20,
            alpha=0.5,
            label='Ground Truth Anomaly'
        )
    
    # Mark displayed frames
    ax_score.scatter(indices, scores[indices], color='blue', s=50,
                    marker='v', label='Displayed Frames', zorder=5)
    
    ax_score.set_xlabel('Frame Index', fontsize=10)
    ax_score.set_ylabel('Anomaly Score', fontsize=10)
    ax_score.legend(fontsize=8)
    ax_score.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
