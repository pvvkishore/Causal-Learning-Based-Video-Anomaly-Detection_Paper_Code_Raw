# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-08

### Added
- Initial release of Causal Learning Based Video Anomaly Detection
- Causal Representation Learning framework
  - VAE-based encoder for disentangled representations
  - Decoder for frame reconstruction
  - Temporal causal module with LSTM and multi-head attention
- Complete anomaly detection model
  - End-to-end training pipeline
  - Frame-level anomaly score prediction
- Training infrastructure
  - Configurable training script
  - Support for multiple datasets (UCF-Crime, ShanghaiTech, etc.)
  - TensorBoard logging
  - Model checkpointing
- Evaluation framework
  - Comprehensive metrics (AUC-ROC, AUC-PR, F1, Precision, Recall)
  - Visualization tools for results
  - Frame-level and video-level evaluation
- Data loading utilities
  - Generic video dataset loader
  - Support for various video formats
  - Frame-level annotation support
- Utility modules
  - Metrics computation
  - Result visualization
  - Attention weight visualization
- Documentation
  - Comprehensive README
  - Quick start guide
  - Contributing guidelines
  - Code examples
- Configuration files
  - Default configuration
  - Lightweight configuration
  - High-performance configuration
- Example files
  - Demo script for testing
  - Example annotation format
  - Data format documentation

### Features
- Self-discovering temporal anomaly patterns
- Causal relationship modeling across time
- Variational autoencoder for robust representations
- Multi-head attention for temporal dependencies
- Frame-level anomaly detection
- Support for weakly supervised learning
- Configurable architecture and hyperparameters
- GPU acceleration support
- Batch processing for efficient training
- Comprehensive evaluation metrics

### Technical Details
- Python 3.8+ support
- PyTorch 2.0+ backend
- CUDA support for GPU acceleration
- Modular and extensible architecture
- Type hints throughout codebase
- Detailed docstrings

## [Unreleased]

### Planned Features
- Pre-trained model weights
- Additional dataset support (Avenue, UCSD Ped)
- Real-time inference API
- Web-based demo interface
- Model compression and optimization
- Multi-modal anomaly detection (audio + video)
- Explainable AI features
- Transfer learning support
- Docker containerization
- CI/CD pipeline

---

For more details, see the [README](README.md) and [documentation](docs/).
