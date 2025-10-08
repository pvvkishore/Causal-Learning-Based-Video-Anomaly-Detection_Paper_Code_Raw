"""
Utility functions for computing evaluation metrics.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix
)
from typing import Dict, Tuple


class AnomalyMetrics:
    """
    Compute various metrics for anomaly detection.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Threshold for binary classification
        """
        self.threshold = threshold
    
    def compute(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            scores: Anomaly scores (higher means more anomalous)
            labels: Ground truth labels (1 for anomaly, 0 for normal)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # ROC AUC
        try:
            metrics['auc_roc'] = roc_auc_score(labels, scores)
        except ValueError:
            metrics['auc_roc'] = 0.0
        
        # Average Precision (PR AUC)
        try:
            metrics['auc_pr'] = average_precision_score(labels, scores)
        except ValueError:
            metrics['auc_pr'] = 0.0
        
        # Binary predictions
        predictions = (scores >= self.threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        # Accuracy
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        
        # Precision
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 Score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                  (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        # False Positive Rate
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # False Negative Rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return metrics
    
    def find_optimal_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, float]:
        """
        Find optimal threshold based on a metric.
        
        Args:
            scores: Anomaly scores
            labels: Ground truth labels
            metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            optimal_threshold: Best threshold
            optimal_value: Best metric value
        """
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        
        if metric == 'f1':
            # Compute F1 score for each threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_value = f1_scores[optimal_idx]
        elif metric == 'precision':
            optimal_idx = np.argmax(precision)
            optimal_value = precision[optimal_idx]
        elif metric == 'recall':
            optimal_idx = np.argmax(recall)
            optimal_value = recall[optimal_idx]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        return optimal_threshold, optimal_value


def compute_auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute ROC AUC score.
    
    Args:
        scores: Anomaly scores
        labels: Ground truth labels
        
    Returns:
        ROC AUC score
    """
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return 0.0


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER).
    
    Args:
        scores: Anomaly scores
        labels: Ground truth labels
        
    Returns:
        EER value
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Find where FPR and FNR are closest
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    return eer


def compute_frame_level_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    video_lengths: list
) -> float:
    """
    Compute frame-level AUC for video anomaly detection.
    
    Args:
        scores: Frame-level anomaly scores
        labels: Frame-level ground truth labels
        video_lengths: List of video lengths
        
    Returns:
        Frame-level AUC
    """
    # Flatten if needed
    scores_flat = scores.flatten()
    labels_flat = labels.flatten()
    
    return roc_auc_score(labels_flat, scores_flat)


def compute_video_level_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    video_lengths: list,
    aggregation: str = 'max'
) -> float:
    """
    Compute video-level AUC.
    
    Args:
        scores: Frame-level anomaly scores
        labels: Frame-level ground truth labels
        video_lengths: List of video lengths
        aggregation: How to aggregate frame scores ('max', 'mean', 'median')
        
    Returns:
        Video-level AUC
    """
    video_scores = []
    video_labels = []
    
    start_idx = 0
    for length in video_lengths:
        end_idx = start_idx + length
        
        # Get scores and labels for this video
        vid_scores = scores[start_idx:end_idx]
        vid_labels = labels[start_idx:end_idx]
        
        # Aggregate frame scores
        if aggregation == 'max':
            video_scores.append(np.max(vid_scores))
        elif aggregation == 'mean':
            video_scores.append(np.mean(vid_scores))
        elif aggregation == 'median':
            video_scores.append(np.median(vid_scores))
        
        # Video is anomalous if any frame is anomalous
        video_labels.append(int(np.any(vid_labels)))
        
        start_idx = end_idx
    
    return roc_auc_score(video_labels, video_scores)
