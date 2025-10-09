# Contributing to JAX NoProp Implementation

Thank you for your interest in contributing to the JAX NoProp implementation! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/jax-noprop.git
   cd jax-noprop
   ```
3. Install the package in development mode:
   ```bash
   pip install -e .
   pip install -e ".[dev]"  # Install development dependencies
   ```

## Development Setup

### Prerequisites
- Python 3.8+
- JAX and JAXLib
- Flax
- Other dependencies listed in `requirements.txt`

### Running Tests
```bash
python test_implementation.py
```

### Code Style
We use the following tools for code formatting and linting:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run these tools before submitting:
```bash
black src/ examples/ test_implementation.py
isort src/ examples/ test_implementation.py
flake8 src/ examples/ test_implementation.py
mypy src/
```

## Types of Contributions

### Bug Reports
When reporting bugs, please include:
- A clear description of the bug
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (Python version, JAX version, etc.)

### Feature Requests
For new features, please:
- Describe the feature clearly
- Explain why it would be useful
- Provide examples of how it would be used

### Code Contributions

#### Pull Request Process
1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

#### Code Style Guidelines
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and reasonably sized
- Use meaningful variable and function names

#### Testing
- Add tests for new functionality
- Ensure existing tests continue to pass
- Test edge cases and error conditions
- Include performance tests for critical paths

## Areas for Contribution

### High Priority
- **Performance optimizations**: Improve training speed and memory usage
- **Additional model architectures**: Support for more backbone networks
- **Advanced noise schedules**: Implement more sophisticated scheduling strategies
- **Better documentation**: Improve examples and tutorials

### Medium Priority
- **Additional datasets**: Support for more benchmark datasets
- **Visualization tools**: Plotting utilities for training progress
- **Checkpointing**: Better model saving and loading
- **Distributed training**: Multi-GPU support

### Low Priority
- **Web interface**: Simple web UI for training
- **Docker support**: Containerized environment
- **CI/CD**: Automated testing and deployment

## Research Contributions

We welcome research contributions that:
- Implement new NoProp variants
- Compare different approaches
- Provide theoretical analysis
- Include experimental results

For research contributions, please also include:
- Clear motivation and related work
- Experimental setup and results
- Discussion of limitations and future work

## Documentation

When contributing documentation:
- Use clear, concise language
- Include code examples
- Update the README if adding new features
- Add docstrings to new functions and classes

## Release Process

Releases are made by maintainers and follow semantic versioning:
- **Major version** (X.0.0): Breaking changes
- **Minor version** (X.Y.0): New features, backward compatible
- **Patch version** (X.Y.Z): Bug fixes, backward compatible

## Questions?

If you have questions about contributing:
- Open an issue for discussion
- Check existing issues and pull requests
- Review the codebase to understand the structure

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. Please be respectful and constructive in all interactions.

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.
