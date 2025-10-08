# Quick Start Guide

This guide will help you get started with the Causal Video Anomaly Detection system quickly.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works too)
- 16GB+ RAM recommended

## Installation

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/pvvkishore/Causal-Learning-Based-Video-Anomaly-Detection_Paper_Code_Raw.git
cd Causal-Learning-Based-Video-Anomaly-Detection_Paper_Code_Raw

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Option 2: Manual installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Test

Run the demo script to verify installation:

```bash
python src/demo.py
```

This will test the model with synthetic data. You should see output showing:
- Model creation
- Forward pass results
- Loss computation
- Component testing

## Training Your First Model

### Step 1: Prepare Your Data

Create the following directory structure:

```
data/
├── videos/
│   ├── train/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── test/
│       ├── video1.mp4
│       └── ...
└── annotations.json
```

### Step 2: Create Annotations

Create an `annotations.json` file (see `data/README.md` for format details):

```json
{
  "video1.mp4": {
    "0": 0,
    "1": 0,
    "2": 1,
    "3": 1
  }
}
```

### Step 3: Train the Model

```bash
python src/train.py \
  --data_dir data/videos/train \
  --annotation_file data/annotations.json \
  --output_dir experiments/my_first_experiment \
  --config configs/default_config.yaml
```

### Step 4: Monitor Training

Training progress is logged to:
- Console output
- TensorBoard: `tensorboard --logdir experiments/my_first_experiment/logs`

### Step 5: Evaluate the Model

```bash
python src/evaluate.py \
  --checkpoint experiments/my_first_experiment/checkpoint_best.pth \
  --data_dir data/videos/test \
  --annotation_file data/test_annotations.json \
  --output_dir evaluation_results \
  --visualize
```

## Working with Pre-trained Models

If you have a pre-trained checkpoint:

```python
import torch
from src.models import CausalAnomalyDetector

# Load model
model = CausalAnomalyDetector(latent_dim=512)
checkpoint = torch.load('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
# ... (see main README for full example)
```

## Common Issues

### Out of Memory

If you encounter OOM errors:
1. Reduce `batch_size` in config
2. Reduce `sequence_length` in config
3. Reduce `frame_size` in config

Example config modification:
```yaml
training:
  batch_size: 4  # Reduced from 8

data:
  sequence_length: 8  # Reduced from 16
  frame_size: [128, 128]  # Reduced from [256, 256]
```

### Slow Training

To speed up training:
1. Use GPU instead of CPU
2. Increase `num_workers` in config
3. Use smaller frame sizes during development
4. Train on a subset of data first

### Module Not Found

If you get import errors:
```bash
# Make sure you're in the project root and add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Configuration Tips

### For Quick Experiments
```yaml
training:
  batch_size: 16
  num_epochs: 50
  
data:
  sequence_length: 8
  frame_size: [128, 128]
```

### For Best Performance
```yaml
training:
  batch_size: 8
  num_epochs: 200
  
data:
  sequence_length: 16
  frame_size: [256, 256]
```

### For Limited GPU Memory
```yaml
training:
  batch_size: 4
  
data:
  sequence_length: 8
  frame_size: [128, 128]
```

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Check [CONTRIBUTING.md](CONTRIBUTING.md) if you want to contribute
3. Explore the example configurations in `configs/`
4. Try different model architectures by modifying the config

## Getting Help

- Check the [Issues](https://github.com/pvvkishore/Causal-Learning-Based-Video-Anomaly-Detection_Paper_Code_Raw/issues) page
- Read the documentation in the code
- Contact: research@bjitgroup.com

Happy anomaly detecting! 🎯
