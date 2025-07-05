# Contributing to Friday AI Trading System

## Welcome!

Thank you for considering contributing to the Friday AI Trading System! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## How Can I Contribute?

### Reporting Bugs

Bugs are tracked as GitHub issues. Create an issue and provide the following information:

- Use a clear and descriptive title
- Describe the exact steps to reproduce the bug
- Provide specific examples to demonstrate the steps
- Describe the behavior you observed and why it's a problem
- Include screenshots if applicable
- Include details about your environment (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are also tracked as GitHub issues. When creating an enhancement suggestion, include:

- A clear and descriptive title
- A detailed description of the proposed functionality
- Any potential implementation approaches you have in mind
- Why this enhancement would be useful to most users
- Examples of how this enhancement would be used

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Add or update tests as necessary
5. Update documentation as needed
6. Ensure all tests pass
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setting Up Development Environment

1. Clone your fork of the repository
   ```bash
   git clone https://github.com/YOUR-USERNAME/friday-ai-trading-system.git
   cd friday-ai-trading-system
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Set up pre-commit hooks
   ```bash
   pre-commit install
   ```

5. Create a `.env` file based on `.env.example`

### Running Tests

```bash
python -m pytest
```

To run tests with coverage:

```bash
python -m pytest --cov=src tests/
```

## Styleguides

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line
- Consider starting the commit message with an applicable emoji:
  - ✨ `:sparkles:` when adding a new feature
  - 🐛 `:bug:` when fixing a bug
  - 📚 `:books:` when adding or updating documentation
  - 🧹 `:broom:` when refactoring code
  - 🧪 `:test_tube:` when adding tests
  - ⚡️ `:zap:` when improving performance
  - 🔒 `:lock:` when dealing with security

### Python Styleguide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Use type hints for function parameters and return values
- Maximum line length is 88 characters (compatible with Black formatter)
- Use snake_case for variables and function names
- Use PascalCase for class names
- Use UPPER_CASE for constants

### Documentation Styleguide

- Use [Markdown](https://guides.github.com/features/mastering-markdown/) for documentation
- Follow the [documentation standards](docs/documentation_standards.md)

## Additional Notes

### Issue and Pull Request Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested
- `wontfix`: This will not be worked on

## Recognition

Contributors who make significant improvements will be recognized in the project's README and CONTRIBUTORS file.

Thank you for contributing to the Friday AI Trading System!