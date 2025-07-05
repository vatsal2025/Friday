# Friday AI Trading System - Developer Guide

## Overview

This guide is intended for developers who want to extend, customize, or contribute to the Friday AI Trading System. It provides detailed information about the system architecture, code organization, development workflow, and best practices.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Code Organization](#code-organization)
3. [Development Environment Setup](#development-environment-setup)
4. [Adding New Features](#adding-new-features)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Code Style and Best Practices](#code-style-and-best-practices)
8. [Contributing Guidelines](#contributing-guidelines)
9. [Advanced Topics](#advanced-topics)

## System Architecture

The Friday AI Trading System follows a modular, layered architecture designed for flexibility, extensibility, and maintainability.

### Architectural Layers

1. **Data Layer**
   - Responsible for data collection, storage, and preprocessing
   - Includes market data, alternative data, and feature engineering
   - Interfaces with databases (MongoDB, SQLite) and external data sources

2. **Model Layer**
   - Implements machine learning models and AI agents
   - Includes MCP servers for memory and sequential thinking
   - Handles model training, evaluation, and inference

3. **Strategy Layer**
   - Implements trading strategies and signal generation
   - Includes portfolio optimization and risk management
   - Handles strategy backtesting and evaluation

4. **Execution Layer**
   - Manages order execution and broker integration
   - Handles position management and trade lifecycle
   - Implements risk controls and compliance checks

5. **API Layer**
   - Provides RESTful API for system interaction
   - Implements WebSocket for real-time updates
   - Handles authentication and authorization

### Key Components

1. **MCP Servers**
   - Memory MCP Server: Maintains context and remembers important trading information
   - Sequential Thinking MCP Server: Enables step-by-step reasoning for complex decisions

2. **Database Systems**
   - MongoDB: Primary database for market data, trading records, and system state
   - Redis: High-performance cache and message broker
   - SQLite: Lightweight database for local storage and configuration

3. **API Server**
   - FastAPI-based RESTful API
   - WebSocket for real-time updates
   - Swagger UI for API documentation and testing

4. **CLI**
   - Command-line interface for system management
   - Supports various commands for trading, backtesting, and system administration

## Code Organization

The Friday AI Trading System codebase is organized into modules, each with a specific responsibility.

### Directory Structure

```
Friday/
├── data/                  # Data storage
├── logs/                  # System logs
├── models/                # Trained models
├── src/                   # Source code
│   ├── analytics/         # Analytics and reporting
│   │   ├── __init__.py
│   │   ├── performance.py # Performance metrics
│   │   ├── reporting.py   # Report generation
│   │   └── visualization.py # Data visualization
│   ├── application/       # Application layer
│   │   ├── __init__.py
│   │   ├── api/           # API server
│   │   │   ├── __init__.py
│   │   │   ├── main.py    # FastAPI application
│   │   │   ├── auth_router.py # Authentication routes
│   │   │   ├── broker_router.py # Broker routes
│   │   │   └── ...
│   │   └── cli/           # Command-line interface
│   │       ├── __init__.py
│   │       └── main.py    # CLI application
│   ├── backtesting/       # Backtesting framework
│   │   ├── __init__.py
│   │   ├── engine.py      # Backtesting engine
│   │   ├── metrics.py     # Performance metrics
│   │   └── ...
│   ├── data/              # Data collection and processing
│   │   ├── __init__.py
│   │   ├── collectors/    # Data collectors
│   │   ├── processors/    # Data processors
│   │   ├── storage/       # Data storage
│   │   └── ...
│   ├── infrastructure/    # Infrastructure components
│   │   ├── __init__.py
│   │   ├── database/      # Database integration
│   │   ├── logging/       # Logging configuration
│   │   ├── cache/         # Caching mechanisms
│   │   └── ...
│   ├── integration/       # External integrations
│   │   ├── __init__.py
│   │   ├── brokers/       # Broker integrations
│   │   ├── data_providers/ # Data provider integrations
│   │   └── ...
│   ├── orchestration/     # System orchestration
│   │   ├── __init__.py
│   │   ├── scheduler.py   # Task scheduler
│   │   ├── workflow.py    # Workflow management
│   │   └── ...
│   ├── portfolio/         # Portfolio management
│   │   ├── __init__.py
│   │   ├── allocation.py  # Asset allocation
│   │   ├── optimization.py # Portfolio optimization
│   │   └── ...
│   ├── risk/              # Risk management
│   │   ├── __init__.py
│   │   ├── position_sizing.py # Position sizing
│   │   ├── stop_loss.py   # Stop-loss mechanisms
│   │   └── ...
│   ├── services/          # Core services
│   │   ├── __init__.py
│   │   ├── mcp/           # MCP services
│   │   ├── trading/       # Trading services
│   │   └── ...
│   ├── mcp_client.py      # MCP client
│   └── mcp_servers.py     # MCP servers
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── ...
├── .env.example           # Example environment variables
├── requirements.txt       # Python dependencies
├── setup.py               # Package setup
└── README.md              # Project documentation
```

### Module Responsibilities

- **analytics**: Analytics and reporting functionality
- **application**: API server and CLI
- **backtesting**: Backtesting framework
- **data**: Data collection, processing, and storage
- **infrastructure**: Database, logging, and caching
- **integration**: External integrations (brokers, data providers)
- **orchestration**: System orchestration and workflow management
- **portfolio**: Portfolio management and optimization
- **risk**: Risk management and position sizing
- **services**: Core services, including MCP services

## Development Environment Setup

### Prerequisites

- Python 3.10 or later
- MongoDB 4.4 or later
- Redis 6.0 or later
- Git
- A code editor or IDE (e.g., Visual Studio Code, PyCharm)

### Setting Up the Development Environment

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/friday.git
cd friday
```

2. **Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
python install_dependencies.py --dev
```

This will install both the regular dependencies and development dependencies (testing tools, linters, etc.).

4. **Set Up Databases**

Ensure MongoDB and Redis are installed and running. Then initialize the databases:

```bash
python verify_databases.py
python src/infrastructure/database/setup_databases.py
```

5. **Configure the System**

Copy the example environment file and edit it with your settings:

```bash
cp .env.example .env
# Edit .env with your configuration
```

6. **Run Tests**

Verify that your development environment is set up correctly by running the tests:

```bash
python -m pytest tests/
```

## Adding New Features

When adding new features to the Friday AI Trading System, follow these steps:

### 1. Plan Your Feature

- Define the feature requirements and scope
- Identify which modules will be affected
- Consider how the feature fits into the existing architecture
- Plan the necessary changes to the codebase

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Implement the Feature

Follow these guidelines when implementing your feature:

- **Module Organization**: Place your code in the appropriate module based on its functionality
- **Class and Function Design**: Follow object-oriented design principles and keep functions focused on a single responsibility
- **Error Handling**: Implement proper error handling and logging
- **Configuration**: Make your feature configurable through the unified configuration system
- **Documentation**: Document your code with docstrings and comments

### 4. Write Tests

Write tests for your feature to ensure it works correctly and to prevent regressions:

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test how your feature interacts with other components
- **Functional Tests**: Test the feature from a user's perspective

```bash
python -m pytest tests/unit/your_module_test.py
```

### 5. Update Documentation

Update the relevant documentation to reflect your changes:

- **Code Documentation**: Update docstrings and comments
- **User Documentation**: Update user guides and README files
- **API Documentation**: Update API documentation if you've added or modified API endpoints

### 6. Submit a Pull Request

```bash
git add .
git commit -m "Add your feature description"
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with a detailed description of your changes.

## Testing

The Friday AI Trading System uses pytest for testing. The test suite is organized into unit tests, integration tests, and functional tests.

### Running Tests

```bash
# Run all tests
python -m pytest

# Run unit tests only
python -m pytest tests/unit/

# Run integration tests only
python -m pytest tests/integration/

# Run tests with coverage report
python -m pytest --cov=src tests/
```

### Writing Tests

When writing tests, follow these guidelines:

- **Test Organization**: Place tests in the appropriate directory based on their type (unit, integration, functional)
- **Test Naming**: Name test files with a `test_` prefix and test functions with a `test_` prefix
- **Test Coverage**: Aim for high test coverage, especially for critical components
- **Mocking**: Use mocking to isolate the code being tested from external dependencies
- **Fixtures**: Use pytest fixtures for test setup and teardown

### Example Test

```python
# tests/unit/risk/test_position_sizing.py
import pytest
from src.risk.position_sizing import calculate_position_size

def test_calculate_position_size():
    # Test with valid inputs
    capital = 10000
    risk_per_trade = 0.02  # 2%
    stop_loss_percent = 0.05  # 5%
    expected = 4000  # (10000 * 0.02) / 0.05
    
    result = calculate_position_size(capital, risk_per_trade, stop_loss_percent)
    assert result == expected
    
    # Test with zero inputs
    assert calculate_position_size(0, risk_per_trade, stop_loss_percent) == 0
    assert calculate_position_size(capital, 0, stop_loss_percent) == 0
    assert calculate_position_size(capital, risk_per_trade, 0) == 0
```

## Documentation

The Friday AI Trading System uses several types of documentation:

### Code Documentation

- **Docstrings**: Use Google-style docstrings for functions, classes, and modules
- **Comments**: Add comments for complex or non-obvious code
- **Type Hints**: Use Python type hints to document parameter and return types

### User Documentation

- **README.md**: Overview of the system and basic usage instructions
- **SETUP_INSTRUCTIONS.md**: Detailed setup instructions
- **CONFIGURATION_GUIDE.md**: Configuration options and best practices
- **SYSTEM_VERIFICATION.md**: System verification and troubleshooting
- **PRODUCTION_SETUP.md**: Production deployment guidelines
- **TROUBLESHOOTING.md**: Solutions for common issues

### API Documentation

- **Swagger UI**: Automatically generated API documentation from FastAPI
- **API Reference**: Detailed documentation for API endpoints

### Updating Documentation

When making changes to the codebase, always update the relevant documentation to reflect your changes.

## Code Style and Best Practices

The Friday AI Trading System follows PEP 8 style guidelines and other best practices for Python development.

### Code Style

- **PEP 8**: Follow PEP 8 style guidelines for Python code
- **Line Length**: Limit lines to 88 characters (compatible with Black formatter)
- **Imports**: Organize imports alphabetically and group them by standard library, third-party, and local imports
- **Naming Conventions**: Use descriptive names for variables, functions, and classes

### Best Practices

- **SOLID Principles**: Follow SOLID principles for object-oriented design
- **DRY (Don't Repeat Yourself)**: Avoid code duplication
- **KISS (Keep It Simple, Stupid)**: Keep code simple and easy to understand
- **Error Handling**: Implement proper error handling and logging
- **Configuration**: Make code configurable through the unified configuration system
- **Testing**: Write tests for all new code
- **Documentation**: Document all new code with docstrings and comments

### Code Linting and Formatting

Use the following tools for code linting and formatting:

- **Black**: Code formatter
- **Flake8**: Code linter
- **isort**: Import sorter
- **mypy**: Static type checker

```bash
# Format code with Black
black src/ tests/

# Check code with Flake8
flake8 src/ tests/

# Sort imports with isort
isort src/ tests/

# Check types with mypy
mypy src/
```

## Contributing Guidelines

When contributing to the Friday AI Trading System, follow these guidelines:

### 1. Fork the Repository

Fork the repository on GitHub and clone your fork:

```bash
git clone https://github.com/yourusername/friday.git
cd friday
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Implement Your Changes

Implement your changes following the code style and best practices described above.

### 4. Write Tests

Write tests for your changes to ensure they work correctly and to prevent regressions.

### 5. Update Documentation

Update the relevant documentation to reflect your changes.

### 6. Submit a Pull Request

```bash
git add .
git commit -m "Add your feature description"
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with a detailed description of your changes.

### 7. Code Review

Your pull request will be reviewed by the project maintainers. Be prepared to make changes based on their feedback.

### 8. Merge

Once your pull request is approved, it will be merged into the main branch.

## Advanced Topics

### Extending the MCP Servers

The MCP (Model Context Protocol) servers are a key component of the Friday AI Trading System. They provide memory and sequential thinking capabilities for AI-powered trading.

To extend the MCP servers, you can:

- Add new endpoints to the existing servers
- Implement new MCP servers for additional capabilities
- Enhance the existing MCP servers with new features

### Implementing New Trading Strategies

The Friday AI Trading System is designed to support a wide range of trading strategies. To implement a new trading strategy:

1. Create a new strategy class in the `src/services/trading/strategies` directory
2. Implement the required methods (e.g., `generate_signals`, `calculate_position_size`)
3. Register the strategy in the strategy registry
4. Add configuration options for the strategy in `unified_config.py`
5. Write tests for the strategy
6. Update documentation to include the new strategy

### Integrating New Data Sources

To integrate a new data source:

1. Create a new data collector in the `src/data/collectors` directory
2. Implement the required methods for collecting data from the source
3. Create a new data processor in the `src/data/processors` directory (if needed)
4. Add configuration options for the data source in `unified_config.py`
5. Write tests for the data collector and processor
6. Update documentation to include the new data source

### Implementing New Broker Integrations

To integrate a new broker:

1. Create a new broker integration in the `src/integration/brokers` directory
2. Implement the required methods for order placement, account information, etc.
3. Add configuration options for the broker in `unified_config.py`
4. Write tests for the broker integration
5. Update documentation to include the new broker

### Performance Optimization

To optimize the performance of the Friday AI Trading System:

1. Identify performance bottlenecks using profiling tools
2. Optimize database queries and indexes
3. Implement caching for frequently accessed data
4. Use asynchronous programming for I/O-bound operations
5. Optimize resource-intensive algorithms
6. Consider using compiled extensions for performance-critical code

---

For more information, refer to the `README.md` and other documentation files in the project root directory.