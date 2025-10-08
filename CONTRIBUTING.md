# Contributing to Causal Video Anomaly Detection

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (OS, Python version, GPU/CPU, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:
- A clear and descriptive title
- A detailed description of the proposed functionality
- Any relevant examples or use cases

### Pull Requests

1. Fork the repository
2. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes:
   - Write clear, documented code
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation as needed

4. Commit your changes:
   ```bash
   git add .
   git commit -m "Add clear description of changes"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Open a Pull Request with:
   - Clear description of changes
   - Reference to related issues (if any)
   - Screenshots/videos for UI changes

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/pvvkishore/Causal-Learning-Based-Video-Anomaly-Detection_Paper_Code_Raw.git
   cd Causal-Learning-Based-Video-Anomaly-Detection_Paper_Code_Raw
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # Install in development mode with dev dependencies
   ```

## Code Style

- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Maximum line length: 100 characters

### Example:

```python
def compute_anomaly_score(
    features: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Compute anomaly scores from features.
    
    Args:
        features: Input features of shape (N, D)
        threshold: Classification threshold
        
    Returns:
        Anomaly scores of shape (N,)
    """
    # Implementation here
    pass
```

## Testing

Before submitting a pull request:

1. Run the demo script to ensure basic functionality:
   ```bash
   python src/demo.py
   ```

2. Test on sample data if possible

3. Ensure your changes don't break existing functionality

## Documentation

- Update README.md if you add new features
- Add inline comments for complex logic
- Update configuration examples if needed
- Add usage examples for new functionality

## Commit Messages

Write clear commit messages:
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests when relevant

Example:
```
Add temporal attention visualization

- Implement attention weight heatmap plotting
- Add save functionality for attention maps
- Update visualization utilities

Fixes #123
```

## Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, maintainers will merge your PR

## Community Guidelines

- Be respectful and constructive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Questions?

If you have questions, feel free to:
- Open an issue for discussion
- Contact the maintainers
- Check existing documentation

Thank you for contributing! 🎉
