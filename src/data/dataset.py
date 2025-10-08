"""
Data loaders for video anomaly detection datasets.

Supports common datasets:
- UCF-Crime
- ShanghaiTech
- Avenue
- UCSD Ped1/Ped2
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import json


class VideoAnomalyDataset(Dataset):
    """
    Generic video anomaly detection dataset.
    """
    
    def __init__(
        self,
        video_dir: str,
        annotation_file: Optional[str] = None,
        sequence_length: int = 16,
        frame_size: Tuple[int, int] = (256, 256),
        stride: int = 1,
        transform=None,
        is_training: bool = True
    ):
        """
        Args:
            video_dir: Directory containing video files
            annotation_file: JSON file with annotations (frame-level labels)
            sequence_length: Number of frames per sequence
            frame_size: Target frame size (H, W)
            stride: Stride for sequence sampling
            transform: Optional transform to apply to frames
            is_training: Whether in training mode
        """
        self.video_dir = video_dir
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.stride = stride
        self.transform = transform
        self.is_training = is_training
        
        # Load video files
        self.video_files = self._load_video_files()
        
        # Load annotations if provided
        self.annotations = {}
        if annotation_file and os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)
        
        # Create sequence indices
        self.sequences = self._create_sequences()
        
    def _load_video_files(self) -> List[str]:
        """Load all video files from directory."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for root, _, files in os.walk(self.video_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(root, file))
        
        return sorted(video_files)
    
    def _create_sequences(self) -> List[dict]:
        """Create sequence indices for all videos."""
        sequences = []
        
        for video_idx, video_path in enumerate(self.video_files):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Create sequences with stride
            for start_frame in range(0, total_frames - self.sequence_length + 1, self.stride):
                sequences.append({
                    'video_idx': video_idx,
                    'video_path': video_path,
                    'start_frame': start_frame,
                    'end_frame': start_frame + self.sequence_length
                })
        
        return sequences
    
    def _read_frames(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int
    ) -> np.ndarray:
        """
        Read frames from video file.
        
        Returns:
            frames: Array of shape (T, H, W, C)
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, self.frame_size)
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # Pad if necessary
        while len(frames) < self.sequence_length:
            frames.append(np.zeros_like(frames[0]) if frames else 
                         np.zeros((*self.frame_size, 3), dtype=np.uint8))
        
        return np.array(frames)
    
    def _get_labels(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int
    ) -> np.ndarray:
        """
        Get frame-level anomaly labels.
        
        Returns:
            labels: Binary labels (1 for anomaly, 0 for normal)
        """
        video_name = os.path.basename(video_path)
        
        if video_name in self.annotations:
            frame_labels = self.annotations[video_name]
            labels = [
                frame_labels.get(str(i), 0)
                for i in range(start_frame, end_frame)
            ]
        else:
            # Default to normal if no annotations
            labels = [0] * self.sequence_length
        
        return np.array(labels, dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence of frames and corresponding labels.
        
        Returns:
            frames: Tensor of shape (T, C, H, W)
            labels: Tensor of shape (T,)
        """
        seq_info = self.sequences[idx]
        
        # Read frames
        frames = self._read_frames(
            seq_info['video_path'],
            seq_info['start_frame'],
            seq_info['end_frame']
        )
        
        # Get labels
        labels = self._get_labels(
            seq_info['video_path'],
            seq_info['start_frame'],
            seq_info['end_frame']
        )
        
        # Apply transforms
        if self.transform:
            frames = self.transform(frames)
        
        # Normalize to [-1, 1]
        frames = frames.astype(np.float32) / 127.5 - 1.0
        
        # Convert to tensor (T, H, W, C) -> (T, C, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        labels = torch.from_numpy(labels)
        
        return frames, labels


class UCFCrimeDataset(VideoAnomalyDataset):
    """
    UCF-Crime dataset for weakly supervised anomaly detection.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ShanghaiTechDataset(VideoAnomalyDataset):
    """
    ShanghaiTech dataset for video anomaly detection.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def get_dataloader(
    dataset_name: str,
    data_dir: str,
    annotation_file: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    sequence_length: int = 16,
    frame_size: Tuple[int, int] = (256, 256),
    is_training: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a data loader for the specified dataset.
    
    Args:
        dataset_name: Name of dataset ('ucf-crime', 'shanghaitech', 'avenue', etc.)
        data_dir: Directory containing video files
        annotation_file: Path to annotation file
        batch_size: Batch size
        num_workers: Number of data loading workers
        sequence_length: Number of frames per sequence
        frame_size: Target frame size
        is_training: Whether in training mode
        
    Returns:
        DataLoader instance
    """
    dataset_map = {
        'ucf-crime': UCFCrimeDataset,
        'shanghaitech': ShanghaiTechDataset,
        'default': VideoAnomalyDataset
    }
    
    dataset_class = dataset_map.get(dataset_name.lower(), VideoAnomalyDataset)
    
    dataset = dataset_class(
        video_dir=data_dir,
        annotation_file=annotation_file,
        sequence_length=sequence_length,
        frame_size=frame_size,
        is_training=is_training,
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_training
    )
    
    return dataloader
