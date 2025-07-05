# Friday AI Trading System Documentation Standards

## Overview

This document outlines the documentation standards for the Friday AI Trading System project. Consistent documentation is crucial for maintainability, knowledge transfer, and collaboration.

## Docstring Style

We follow the **Google docstring style** for all Python code. This style is chosen for its readability and widespread adoption.

### Example Function Docstring

```python
def calculate_moving_average(prices, window_size=20):
    """Calculate the moving average for a series of prices.
    
    Args:
        prices (list[float]): A list of price values.
        window_size (int, optional): The size of the moving window. Defaults to 20.
        
    Returns:
        list[float]: The calculated moving averages.
        
    Raises:
        ValueError: If window_size is larger than the length of prices.
        TypeError: If prices contains non-numeric values.
        
    Examples:
        >>> calculate_moving_average([10, 11, 12, 13, 14], 3)
        [None, None, 11.0, 12.0, 13.0]
    """
```

### Example Class Docstring

```python
class TradingStrategy:
    """Base class for all trading strategies.
    
    This class defines the interface that all trading strategies must implement
    and provides common functionality for strategy execution.
    
    Attributes:
        name (str): The name of the strategy.
        timeframe (str): The timeframe this strategy operates on.
        parameters (dict): Strategy configuration parameters.
    """
```

## Required Documentation Sections

### For Functions and Methods

1. **Brief Description**: A concise explanation of what the function/method does.
2. **Args**: Document each parameter, its type, and purpose.
3. **Returns**: What the function returns, including type.
4. **Raises**: Any exceptions that might be raised.
5. **Examples**: Provide usage examples when helpful.

### For Classes

1. **Brief Description**: What the class represents or does.
2. **Attributes**: Class-level attributes.
3. **Methods**: Brief overview of important methods (detailed documentation in method docstrings).

### For Modules

1. **Brief Description**: Purpose of the module.
2. **Functions/Classes**: Overview of key components.
3. **Examples**: Usage examples if applicable.

## Optional Documentation Sections

1. **Notes**: Additional information that doesn't fit elsewhere.
2. **References**: Citations, links to papers, or other resources.
3. **Warnings**: Important cautions about using the code.

## Code Comments

- Use inline comments sparingly and only when the code's purpose isn't immediately obvious.
- Comment complex algorithms or business logic to explain "why" rather than "what".
- Keep comments up-to-date when code changes.

## Documentation Coverage

- All public classes, methods, and functions must be documented.
- Private methods (prefixed with `_`) should be documented when their purpose isn't obvious.
- Module-level variables should be documented if they're part of the public API.
- The project aims for at least 80% documentation coverage.

## Automated Documentation

- Documentation is generated using Sphinx.
- Run `make docs` to generate the latest documentation.
- Documentation is built in HTML and PDF formats.
- API documentation is automatically generated from docstrings.

## Code Review Guidelines

During code review, check that:

1. New code includes appropriate documentation.
2. Docstrings follow the Google style.
3. Required sections are present.
4. Examples are accurate and work as described.
5. Documentation is clear and helpful to other developers.

## Documentation Files

The project maintains the following documentation files:

- `README.md`: Project overview and getting started guide.
- `CONTRIBUTING.md`: Guidelines for contributing to the project.
- `CODE_OF_CONDUCT.md`: Community standards and expectations.
- `CHANGELOG.md`: Record of changes in each version.
- `LICENSE`: Project license information.
- `docs/`: Directory containing detailed documentation.