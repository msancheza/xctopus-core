# Contributing Guide to Xctopus

Thank you for your interest in contributing to Xctopus! This document provides guidelines for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Process](#development-process)
- [Code Standards](#code-standards)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please:

1. **Check that it hasn't been reported already**: Review existing [Issues](https://github.com/msancheza/xctopus-core/issues) before creating a new one.

2. **Create a detailed issue** with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected behavior vs. actual behavior
   - Python version and relevant dependencies
   - Logs or error messages (if applicable)
   - Operating system and version

### Suggesting Enhancements

Enhancement suggestions are welcome. Please:

1. **Open an issue** with the `enhancement` label
2. Clearly describe the proposed functionality
3. Explain why it would be useful
4. If possible, provide usage examples

### Contributing Code

1. **Fork the repository**
2. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/description-of-fix
   ```
3. **Make your changes** following the project standards
4. **Add tests** if necessary
5. **Ensure tests pass**:
   ```bash
   pytest
   ```
6. **Commit your changes** with descriptive messages
7. **Push to your fork** and create a Pull Request

## Development Process

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/msancheza/xctopus-core.git
   cd xctopus-core
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # For development with testing tools
   ```

### Project Structure

```
xctopus/
├── src/xctopus/          # Main source code
│   ├── core/             # Core components
│   ├── nodes/            # Nodes (Bayesian, Transformer)
│   └── modules/          # Additional modules
├── tests/                # Unit tests
├── scripts/              # Pipeline scripts
├── notebooks/            # Demonstration notebooks
└── docs/                 # Documentation
```

## Code Standards

### Code Style

- **Python 3.8+**: Ensure your code is compatible
- **PEP 8**: Follow Python style conventions
- **Type hints**: Use type hints when possible
- **Docstrings**: Document functions and classes using docstrings

### Formatting

The project uses automatic formatting tools. Before committing:

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/
```

### Tests

- Write tests for new features
- Ensure all existing tests pass
- Maintain high code coverage

```bash
# Run tests
pytest

# With coverage
pytest --cov=src/xctopus --cov-report=html
```

### Documentation

- Update documentation if you add new features
- Keep docstrings up to date
- Update the README if necessary

## Submitting Changes

### Pull Requests

1. **Descriptive title**: Use a clear title that describes the change
2. **Detailed description**: Explain what you changed and why
3. **Reference issues**: If it resolves an issue, mention `Closes #123`
4. **Tests**: Ensure all tests pass
5. **Documentation**: Update documentation if necessary

### Commit Messages

Use clear and descriptive commit messages:

```
feat: add support for new embedding models
fix: correct error in clustering with small datasets
docs: update installation guide
test: add tests for BayesianNode
refactor: simplify clustering logic
```

Common prefixes:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Add or modify tests
- `refactor`: Code refactoring
- `style`: Formatting changes
- `chore`: Maintenance tasks

## Reporting Issues

When reporting an issue, include:

- **Version**: Python version and project version
- **Operating system**: OS and version
- **Description**: What you expected vs. what happened
- **Reproduction**: Minimal steps to reproduce
- **Logs**: Relevant error messages
- **Context**: Any additional relevant information

## Feature Requests

To request new features:

1. **Open an issue** with the `feature-request` label
2. **Describe the use case**: What problem does it solve?
3. **Provide examples**: If possible, show how it would be used
4. **Consider alternatives**: Are there other ways to achieve the same?

## Contribution Areas

We are especially interested in contributions in:

- 🧠 **Continuous learning improvements**: Algorithms and techniques for Bayesian Continual Learning
- 🔧 **Performance optimization**: Efficiency and speed improvements
- 📊 **Visualizations**: Tools to visualize system state
- 🧪 **Tests**: Increase test coverage
- 📚 **Documentation**: Improve guides and examples
- 🌐 **Integrations**: Integrations with other tools and frameworks

## Questions

If you have questions about how to contribute:

- Open a [Discussion](https://github.com/msancheza/xctopus-core/discussions)
- Review the [documentation](https://xctopus.com)
- Contact the maintainers

## Recognition

All contributions are valuable and will be recognized. Thank you for helping make Xctopus better!

---

**Note**: This project is in an experimental phase. Guidelines may evolve as the project matures.
