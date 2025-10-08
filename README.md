# Self-Discovering Temporal Anomaly Patterns in Video Anomaly Detection via Causal Representation Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Research Implementation by BJIT**

## Overview

This repository contains the official implementation of the paper "Self-Discovering Temporal Anomaly Patterns in Video Anomaly Detection via Causal Representation Learning". Our approach leverages causal representation learning to automatically discover temporal anomaly patterns in surveillance videos without requiring extensive manual annotation.

### Key Features

- **Causal Representation Learning**: Novel framework that learns disentangled causal representations of video frames
- **Temporal Pattern Discovery**: Self-discovering mechanism for identifying temporal anomaly patterns
- **End-to-End Training**: Unified model for feature extraction, temporal modeling, and anomaly detection
- **Multiple Dataset Support**: Compatible with UCF-Crime, ShanghaiTech, Avenue, and UCSD Ped datasets
- **Comprehensive Evaluation**: Extensive metrics including AUC-ROC, AUC-PR, F1-score, and frame-level accuracy

## Architecture

The model consists of three main components:

1. **Causal Encoder**: Extracts disentangled latent representations from video frames using a variational autoencoder (VAE) architecture
2. **Temporal Causal Module**: Models temporal dependencies and causal relationships across frames using LSTM and multi-head attention
3. **Anomaly Detector**: Predicts frame-level anomaly scores based on learned representations

```
Input Video → Causal Encoder → Temporal Module → Anomaly Detector → Anomaly Scores
                    ↓                                      ↑
              Latent Space                          Reconstruction
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)
- 16GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/pvvkishore/Causal-Learning-Based-Video-Anomaly-Detection_Paper_Code_Raw.git
cd Causal-Learning-Based-Video-Anomaly-Detection_Paper_Code_Raw
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Supported Datasets

- **UCF-Crime**: Weakly supervised anomaly detection dataset
- **ShanghaiTech**: Campus surveillance dataset
- **Avenue**: Pedestrian anomaly detection dataset
- **UCSD Ped1/Ped2**: Pedestrian pathway datasets

### Data Structure

Organize your dataset as follows:

```
data/
├── videos/
│   ├── train/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── test/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
└── annotations.json
```

### Annotation Format

The `annotations.json` file should contain frame-level labels:

```json
{
  "video1.mp4": {
    "0": 0,
    "1": 0,
    "2": 1,
    ...
  },
  "video2.mp4": {
    ...
  }
}
```

Where `0` indicates normal frames and `1` indicates anomalous frames.

## Usage

### Training

Train the model using the default configuration:

```bash
python src/train.py \
  --data_dir data/videos/train \
  --annotation_file data/annotations.json \
  --output_dir experiments/my_experiment \
  --config configs/default_config.yaml
```

### Training with Custom Configuration

Modify `configs/default_config.yaml` or create a new configuration file:

```yaml
model:
  latent_dim: 512
  num_temporal_layers: 2

training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.0001
```

Then run:

```bash
python src/train.py \
  --config configs/my_config.yaml \
  --data_dir data/videos/train \
  --output_dir experiments/custom_experiment
```

### Evaluation

Evaluate a trained model:

```bash
python src/evaluate.py \
  --checkpoint experiments/my_experiment/checkpoint_best.pth \
  --data_dir data/videos/test \
  --annotation_file data/test_annotations.json \
  --output_dir evaluation_results \
  --visualize
```

### Inference on New Videos

```python
import torch
from src.models import CausalAnomalyDetector
from src.data import get_dataloader

# Load model
model = CausalAnomalyDetector(latent_dim=512)
checkpoint = torch.load('experiments/my_experiment/checkpoint_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load data
dataloader = get_dataloader(
    dataset_name='default',
    data_dir='path/to/videos',
    batch_size=1,
    is_training=False
)

# Inference
with torch.no_grad():
    for frames, _ in dataloader:
        outputs = model(frames)
        anomaly_scores = outputs['anomaly_scores']
        print(f"Anomaly scores: {anomaly_scores}")
```

## Results

### Performance on Standard Benchmarks

| Dataset | AUC-ROC | AUC-PR | F1-Score |
|---------|---------|--------|----------|
| UCF-Crime | 0.XX | 0.XX | 0.XX |
| ShanghaiTech | 0.XX | 0.XX | 0.XX |
| Avenue | 0.XX | 0.XX | 0.XX |
| UCSD Ped2 | 0.XX | 0.XX | 0.XX |

*Note: Replace XX with actual performance metrics from your experiments.*

## Project Structure

```
.
├── configs/                    # Configuration files
│   └── default_config.yaml
├── src/                        # Source code
│   ├── models/                 # Model architectures
│   │   ├── causal_model.py     # Main model implementation
│   │   └── __init__.py
│   ├── data/                   # Data loading utilities
│   │   ├── dataset.py          # Dataset classes
│   │   └── __init__.py
│   ├── utils/                  # Utility functions
│   │   ├── metrics.py          # Evaluation metrics
│   │   ├── visualization.py    # Visualization tools
│   │   └── __init__.py
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── __init__.py
├── experiments/                # Experiment outputs
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Model Configuration

Key hyperparameters in `configs/default_config.yaml`:

- `latent_dim`: Dimension of latent representation (default: 512)
- `num_temporal_layers`: Number of LSTM layers for temporal modeling (default: 2)
- `sequence_length`: Number of frames per sequence (default: 16)
- `batch_size`: Training batch size (default: 8)
- `learning_rate`: Adam optimizer learning rate (default: 1e-4)
- `beta`: KL divergence weight in VAE loss (default: 1.0)

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@article{causal_vad_2025,
  title={Self-Discovering Temporal Anomaly Patterns in Video Anomaly Detection via Causal Representation Learning},
  author={BJIT Research Team},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BJIT Research Team for research support
- PyTorch team for the deep learning framework
- Contributors to the video anomaly detection datasets

## Contact

For questions or collaborations, please contact:
- Email: research@bjitgroup.com
- GitHub Issues: [Open an issue](https://github.com/pvvkishore/Causal-Learning-Based-Video-Anomaly-Detection_Paper_Code_Raw/issues)

## Future Work

- [ ] Multi-modal anomaly detection (audio + video)
- [ ] Real-time inference optimization
- [ ] Explainable anomaly detection with attention visualization
- [ ] Transfer learning across different surveillance scenarios
- [ ] Integration with edge devices for deployment
