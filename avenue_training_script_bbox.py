#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 09:54:14 2025

@author: pvvkishore
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed MiniCausal-VAD Anomaly Visualization System
Created on Fri Aug 29 09:46:00 2025

@author: pvvkishore
"""
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import json
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Global variable for detection backend
DETECTION_BACKEND = 'motion'  # Default fallback

# Try different person detection backends
try:
    # Option 1: YOLOv5 (best performance)
    import yolov5
    DETECTION_BACKEND = 'yolov5'
    print("✅ Using YOLOv5 for person detection")
except ImportError:
    try:
        # Option 2: OpenCV DNN
        import cv2.dnn
        DETECTION_BACKEND = 'opencv'
        print("✅ Using OpenCV DNN for person detection")
    except:
        # Option 3: Fallback to simple motion-based detection
        DETECTION_BACKEND = 'motion'
        print("⚠️ Using motion-based detection (install yolov5 for better results)")

# Simple model architecture for loading (you'll need to replace this with your actual model)
class CausalAnomalyDetector(nn.Module):
    """Simplified model architecture - replace with your actual model"""
    def __init__(self, input_channels=3, hidden_dim=64, num_frames=8):
        super().__init__()
        self.num_frames = num_frames
        
        # Simple convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 4, 4))
        )
        
        # Feature dimension
        self.feature_dim = 64 * 16  # 64 channels * 4*4 spatial
        
        # Causal discovery module (simplified)
        self.causal_net = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 16 * 16)  # 16x16 adjacency matrix
        )
        
        # Anomaly classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch, channels, frames, height, width)
        batch_size = x.size(0)
        
        # Encode features
        features = self.encoder(x)
        features = features.view(batch_size, -1)
        
        # Causal adjacency matrix
        causal_logits = self.causal_net(features)
        causal_adj = torch.sigmoid(causal_logits).view(batch_size, 16, 16)
        
        # Anomaly prediction
        anomaly_score = self.classifier(features)
        
        return anomaly_score.squeeze(), causal_adj, features

class AnomalyVisualizer:
    """Complete system for visualizing video anomalies with bounding boxes"""
    
    def __init__(self, model_path: str, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = self.load_trained_model(model_path)
        self.setup_person_detector()
        
        # Visualization settings
        self.colors = {
            'normal': (0, 255, 0),      # Green
            'anomaly': (0, 0, 255),     # Red
            'suspicious': (0, 255, 255), # Yellow
            'bbox': (255, 0, 0)         # Blue for bounding boxes
        }
        
        self.font_scale = 0.7
        self.thickness = 2
        
    def load_trained_model(self, model_path: str):
        """Load the trained MiniCausal-VAD model"""
        print(f"📂 Loading trained model from {model_path}")
        
        try:
            # Initialize model with simplified architecture
            model = CausalAnomalyDetector().to(self.device)
            
            # Try to load checkpoint
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                print("✅ Model loaded successfully")
                return model
            else:
                print(f"⚠️ Model file not found: {model_path}")
                print("🔄 Using randomly initialized model for demonstration")
                return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Using randomly initialized model for demonstration")
            # Return randomly initialized model for demo purposes
            model = CausalAnomalyDetector().to(self.device)
            return model
    
    def setup_person_detector(self):
        """Initialize person detection system"""
        global DETECTION_BACKEND
        
        if DETECTION_BACKEND == 'yolov5':
            try:
                self.detector = yolov5.load('yolov5s', pretrained=True)
                self.detector.conf = 0.2  # Confidence threshold
                self.detector.classes = [0]  # Only detect persons (class 0)
            except Exception as e:
                print(f"⚠️ YOLOv5 setup failed: {e}, falling back to motion detection")
                DETECTION_BACKEND = 'motion'
                
        elif DETECTION_BACKEND == 'opencv':
            # Download YOLO weights if not present
            try:
                self.download_yolo_weights()
                
                # Load YOLO
                config_path = "yolo_configs/yolov4.cfg"
                weights_path = "yolo_configs/yolov4.weights"
                
                if Path(weights_path).exists():
                    self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
                    self.output_layers = self.net.getUnconnectedOutLayersNames()
                else:
                    print("⚠️ YOLO weights not found, using motion detection")
                    DETECTION_BACKEND = 'motion'
            except Exception as e:
                print(f"⚠️ OpenCV DNN setup failed: {e}, falling back to motion detection")
                DETECTION_BACKEND = 'motion'
        
        print(f"🎯 Person detection backend: {DETECTION_BACKEND}")
    
    def download_yolo_weights(self):
        """Download YOLO configuration and weights"""
        import urllib.request
        
        config_dir = Path("yolo_configs")
        config_dir.mkdir(exist_ok=True)
        
        # Only download config file (weights are too large)
        config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
        config_path = config_dir / "yolov4.cfg"
        
        if not config_path.exists():
            try:
                print(f"📥 Downloading yolov4.cfg...")
                urllib.request.urlretrieve(config_url, config_path)
                print("✅ Config downloaded successfully")
            except Exception as e:
                print(f"⚠️ Could not download config: {e}")
    
    def detect_persons_yolov5(self, frame):
        """Detect persons using YOLOv5"""
        try:
            results = self.detector(frame)
            detections = results.pandas().xyxy[0]
            
            persons = []
            for _, detection in detections.iterrows():
                if detection['name'] == 'person' and detection['confidence'] > 0.2:
                    bbox = [
                        int(detection['xmin']), int(detection['ymin']),
                        int(detection['xmax']), int(detection['ymax'])
                    ]
                    persons.append({
                        'bbox': bbox,
                        'confidence': detection['confidence'],
                        'center': ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    })
            
            return persons
        except Exception as e:
            print(f"⚠️ YOLOv5 detection failed: {e}")
            return []
    
    def detect_persons_opencv(self, frame):
        """Detect persons using OpenCV DNN"""
        try:
            height, width = frame.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)
            
            boxes, confidences, class_ids = [], [], []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if class_id == 0 and confidence > 0.2:  # Person class
                        center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                        w, h = int(detection[2] * width), int(detection[3] * height)
                        
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)
            
            persons = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    persons.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': confidences[i],
                        'center': (x + w//2, y + h//2)
                    })
            
            return persons
        except Exception as e:
            print(f"⚠️ OpenCV detection failed: {e}")
            return []
    
    def detect_persons_motion(self, frame, background=None):
        """Fallback: detect motion areas as potential persons"""
        try:
            # Simple contour-based detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Simple edge detection as proxy for "interesting" regions
            edges = cv2.Canny(blur, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            persons = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area for person
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = h / w if w > 0 else 0
                    
                    # Filter for person-like shapes
                    if w > 10 and h > 20 and 1.2 < aspect_ratio < 4.0:
                        persons.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': 0.5,
                            'center': (x + w//2, y + h//2)
                        })
            
            return persons[:3]  # Limit to 3 detections to avoid noise
        except Exception as e:
            print(f"⚠️ Motion detection failed: {e}")
            return []
    
    def detect_persons(self, frame, background=None):
        """Unified person detection interface"""
        if frame is None:
            return []
            
        global DETECTION_BACKEND
        
        if DETECTION_BACKEND == 'yolov5':
            return self.detect_persons_yolov5(frame)
        elif DETECTION_BACKEND == 'opencv':
            return self.detect_persons_opencv(frame)
        else:
            return self.detect_persons_motion(frame, background)
    
    def predict_anomaly_for_clip(self, video_clip):
        """Predict anomaly score for a video clip"""
        if self.model is None:
            return 0.2, np.random.rand(16, 16), np.random.rand(16)
        
        try:
            with torch.no_grad():
                # Convert numpy to tensor
                if isinstance(video_clip, np.ndarray):
                    video_tensor = torch.from_numpy(video_clip).float()
                else:
                    video_tensor = video_clip.float()
                
                # Ensure correct shape: (1, 3, 8, 64, 64)
                if len(video_tensor.shape) == 4:  # (3, 8, 64, 64)
                    video_tensor = video_tensor.unsqueeze(0)
                
                video_tensor = video_tensor.to(self.device)
                
                # Get predictions
                anomaly_scores, causal_adj, features = self.model(video_tensor)
                
                return (
                    float(anomaly_scores.squeeze().cpu().numpy()),
                    causal_adj.squeeze().cpu().numpy(),
                    features.squeeze().cpu().numpy()
                )
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return 0.5, np.random.rand(16, 16), np.random.rand(16)
    
    def extract_anomalous_frames(self, video_dir: str, threshold: float = 0.3):
        """Extract frames with anomaly scores above threshold"""
        video_dir = Path(video_dir)
        anomalous_clips = []
        
        print(f"🔍 Scanning for anomalous clips in {video_dir}")
        
        if not video_dir.exists():
            print(f"❌ Directory not found: {video_dir}")
            return []
        
        # Get all video directories
        video_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir()])
        
        if not video_dirs:
            print(f"⚠️ No video directories found in {video_dir}")
            return []
        
        for video_idx, video_path in enumerate(video_dirs):
            print(f"🎹 Processing video {video_idx + 1}/{len(video_dirs)}: {video_path.name}")
            
            # Look for image files
            frame_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                frame_files.extend(list(video_path.glob(ext)))
            
            frame_files = sorted(frame_files)
            
            if len(frame_files) < 8:
                print(f"  ⚠️ Insufficient frames ({len(frame_files)}) in {video_path.name}")
                continue
            
            # Process in clips of 8 frames
            for start_idx in range(0, len(frame_files) - 8, 4):  # Stride of 4
                clip_frames = []
                clip_paths = frame_files[start_idx:start_idx + 8]
                
                # Load frames
                for frame_path in clip_paths:
                    try:
                        frame = cv2.imread(str(frame_path))
                        if frame is not None:
                            frame = cv2.resize(frame, (64, 64))
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            clip_frames.append(frame)
                    except Exception as e:
                        print(f"    ⚠️ Error loading frame {frame_path}: {e}")
                
                if len(clip_frames) == 8:
                    # Convert to tensor format
                    clip_array = np.array(clip_frames)  # (8, 64, 64, 3)
                    clip_array = clip_array.transpose(3, 0, 1, 2)  # (3, 8, 64, 64)
                    clip_array = clip_array / 255.0  # Normalize
                    
                    # Predict anomaly
                    anomaly_score, causal_graph, features = self.predict_anomaly_for_clip(clip_array)
                    
                    if anomaly_score > threshold:
                        anomalous_clips.append({
                            'video_id': video_path.name,
                            'start_frame': start_idx,
                            'end_frame': start_idx + 8,
                            'frame_paths': clip_paths,
                            'anomaly_score': float(anomaly_score),
                            'causal_graph': causal_graph,
                            'features': features
                        })
                        
                        print(f"  🚨 Anomaly detected! Score: {anomaly_score:.3f}, Frames: {start_idx}-{start_idx+8}")
        
        print(f"✅ Found {len(anomalous_clips)} anomalous clips")
        return anomalous_clips
    
    def visualize_anomalous_clip(self, clip_info: dict, output_dir: str):
        """Visualize a single anomalous clip with bounding boxes"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        clip_id = f"video_{clip_info['video_id']}_frames_{clip_info['start_frame']}_{clip_info['end_frame']}"
        anomaly_score = clip_info['anomaly_score']
        
        print(f"🎨 Visualizing clip: {clip_id} (Score: {anomaly_score:.3f})")
        
        # Create visualization grid
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Anomalous Clip: {clip_id}\nAnomaly Score: {anomaly_score:.3f}', 
                     fontsize=16, fontweight='bold')
        
        visualized_frames = []
        
        for i, frame_path in enumerate(clip_info['frame_paths']):
            try:
                # Load original frame
                original_frame = cv2.imread(str(frame_path))
                if original_frame is None:
                    print(f"⚠️ Could not load frame: {frame_path}")
                    continue
                    
                original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                
                # Detect persons in frame
                persons = self.detect_persons(original_frame)
                
                # Create annotated frame
                annotated_frame = original_frame.copy()
                
                # Draw bounding boxes
                for person in persons:
                    bbox = person['bbox']
                    confidence = person['confidence']
                    
                    # Determine box color based on anomaly context
                    if anomaly_score > 0.4:
                        color = self.colors['anomaly']
                        label = f"ANOMALY: {confidence:.2f}"
                    elif anomaly_score > 0.3:
                        color = self.colors['suspicious']
                        label = f"SUSPICIOUS: {confidence:.2f}"
                    else:
                        color = self.colors['normal']
                        label = f"NORMAL: {confidence:.2f}"
                    
                    # Draw rectangle
                    cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    
                    # Draw label
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_frame, 
                                 (bbox[0], bbox[1] - label_size[1] - 10), 
                                 (bbox[0] + label_size[0], bbox[1]), 
                                 color, -1)
                    cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Plot in grid
                if i < 8:  # Safety check
                    row, col = divmod(i, 4)
                    axes[row, col].imshow(annotated_frame)
                    axes[row, col].set_title(f'Frame {clip_info["start_frame"] + i}\nPersons: {len(persons)}')
                    axes[row, col].axis('off')
                    
                    visualized_frames.append(annotated_frame)
                    
            except Exception as e:
                print(f"⚠️ Error processing frame {i}: {e}")
                continue
        
        # Hide unused subplots
        for i in range(len(visualized_frames), 8):
            row, col = divmod(i, 4)
            axes[row, col].axis('off')
        
        # Save grid visualization
        grid_path = output_dir / f'{clip_id}_grid.png'
        plt.tight_layout()
        plt.savefig(grid_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create video from frames if we have any
        if visualized_frames:
            video_path = output_dir / f'{clip_id}_annotated.mp4'
            self.create_video_from_frames(visualized_frames, video_path, fps=2)
        
        # Save causal graph visualization
        self.visualize_causal_graph(clip_info['causal_graph'], 
                                   output_dir / f'{clip_id}_causal_graph.png')
        
        # Save clip information
        info_path = output_dir / f'{clip_id}_info.json'
        clip_data = {
            'clip_id': clip_id,
            'anomaly_score': float(anomaly_score),
            'video_id': clip_info['video_id'],
            'frame_range': [clip_info['start_frame'], clip_info['end_frame']],
            'detection_backend': DETECTION_BACKEND
        }
        
        with open(info_path, 'w') as f:
            json.dump(clip_data, f, indent=2)
        
        print(f"✅ Saved visualizations to {output_dir}")
        
        return {
            'grid_image': grid_path,
            'annotated_video': video_path if visualized_frames else None,
            'causal_graph': output_dir / f'{clip_id}_causal_graph.png',
            'info_file': info_path
        }
    
    def create_video_from_frames(self, frames: List[np.ndarray], output_path: str, fps: int = 2):
        """Create MP4 video from list of frames"""
        if not frames:
            return
        
        try:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            
            writer.release()
            print(f"🎬 Video saved: {output_path}")
        except Exception as e:
            print(f"⚠️ Error creating video: {e}")
    
    def visualize_causal_graph(self, causal_graph: np.ndarray, output_path: str):
        """Visualize the learned causal graph"""
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot adjacency matrix
            im = ax.imshow(causal_graph, cmap='Reds', vmin=0, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Causal Strength', rotation=270, labelpad=20)
            
            # Add labels
            ax.set_xlabel('Target Variables')
            ax.set_ylabel('Source Variables')
            ax.set_title('Learned Causal Graph\n(Red = Strong Causal Relationship)')
            
            # Add grid
            ax.set_xticks(range(16))
            ax.set_yticks(range(16))
            ax.grid(True, alpha=0.3)
            
            # Highlight strongest relationships
            strong_edges = np.where(causal_graph > 0.5)
            for i, j in zip(strong_edges[0], strong_edges[1]):
                ax.text(j, i, f'{causal_graph[i,j]:.2f}', 
                       ha='center', va='center', color='white', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"⚠️ Error creating causal graph: {e}")
    
    def create_anomaly_report(self, anomalous_clips: List[dict], output_dir: str):
        """Create comprehensive anomaly detection report"""
        output_dir = Path(output_dir)
        
        print(f"📊 Creating comprehensive anomaly report...")
        
        # Process all clips
        all_results = []
        for i, clip_info in enumerate(anomalous_clips):
            print(f"Processing clip {i+1}/{len(anomalous_clips)}")
            
            clip_results = self.visualize_anomalous_clip(
                clip_info, 
                output_dir / f"clip_{i+1:03d}"
            )
            
            clip_results['clip_info'] = clip_info
            all_results.append(clip_results)
        
        # Create summary report
        summary = {
            'total_clips_analyzed': len(anomalous_clips),
            'detection_backend': DETECTION_BACKEND,
            'anomaly_scores': [clip['anomaly_score'] for clip in anomalous_clips],
            'video_distribution': {},
            'clips': []
        }
        
        # Video distribution
        for clip in anomalous_clips:
            video_id = clip['video_id']
            summary['video_distribution'][video_id] = summary['video_distribution'].get(video_id, 0) + 1
        
        # Clip summaries
        for i, (clip_info, results) in enumerate(zip(anomalous_clips, all_results)):
            summary['clips'].append({
                'clip_id': f"clip_{i+1:03d}",
                'anomaly_score': clip_info['anomaly_score'],
                'video_id': clip_info['video_id'],
                'frame_range': [clip_info['start_frame'], clip_info['end_frame']],
                'files': {
                    'grid_image': str(results['grid_image'].name),
                    'annotated_video': str(results['annotated_video'].name) if results['annotated_video'] else None,
                    'causal_graph': str(results['causal_graph'].name)
                }
            })
        
        # Save summary
        summary_path = output_dir / 'anomaly_detection_report.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create HTML report
        self.create_html_report(summary, output_dir)
        
        print(f"✅ Complete anomaly report created in {output_dir}")
        print(f"📋 Summary: {summary_path}")
        
        return summary