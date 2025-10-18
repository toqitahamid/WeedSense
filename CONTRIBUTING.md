# Contributing to WeedSense

Thank you for your interest in contributing to WeedSense! This document provides guidelines for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in your interactions.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, PyTorch version, etc.)
- Any relevant error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- A clear description of the enhancement
- Why this enhancement would be useful
- Possible implementation approaches (if you have ideas)

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes**:
   - Write clear, commented code
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation as needed
3. **Test your changes**:
   - Ensure all existing tests pass
   - Add new tests for new features
   - Test on multiple environments if possible
4. **Commit your changes**:
   - Use clear, descriptive commit messages
   - Reference any related issues
5. **Submit a pull request**:
   - Provide a clear description of the changes
   - Link any related issues
   - Be responsive to feedback

## Development Setup

1. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/weedsense.git
cd weedsense
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks (optional):
```bash
pip install pre-commit
pre-commit install
```

## Code Style

- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Add type hints where appropriate

Example:
```python
def process_image(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Process and resize image.
    
    Args:
        image: Input image tensor (C, H, W)
        size: Target size (height, width)
    
    Returns:
        Processed image tensor
    """
    # Implementation
    pass
```

## Testing

Run tests with:
```bash
pytest tests/
```

## Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features

## Questions?

Feel free to open an issue or reach out to the maintainers at toqitahamid.sarker@siu.edu.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

