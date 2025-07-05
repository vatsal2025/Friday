# Contributing to the Portfolio Management System

Thank you for considering contributing to the Portfolio Management System! This document outlines the process for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our code of conduct. Please be respectful and considerate of others.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue tracker to see if the problem has already been reported. When you are creating a bug report, please include as many details as possible:

- Use a clear and descriptive title
- Describe the exact steps to reproduce the problem
- Provide specific examples to demonstrate the steps
- Describe the behavior you observed after following the steps
- Explain which behavior you expected to see instead and why
- Include screenshots if possible
- Include details about your configuration and environment

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- Use a clear and descriptive title
- Provide a step-by-step description of the suggested enhancement
- Provide specific examples to demonstrate the steps
- Describe the current behavior and explain which behavior you expected to see instead
- Explain why this enhancement would be useful to most users

### Pull Requests

- Fill in the required template
- Do not include issue numbers in the PR title
- Include screenshots and animated GIFs in your pull request whenever possible
- Follow the Python style guide
- Include tests for your changes
- Document new code
- End all files with a newline

## Development Process

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/friday.git`
3. Create a branch for your feature: `git checkout -b feature/your-feature-name`
4. Install development dependencies: `pip install -e ".[dev]"`

### Testing

We use pytest for testing. To run tests:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=portfolio
```

### Code Style

We use Black for code formatting and flake8 for linting. To format your code:

```bash
black portfolio tests
```

To check your code with flake8:

```bash
flake8 portfolio tests
```

Alternatively, you can use the Makefile commands:

```bash
make format
make lint
```

## Documentation

We use Google-style docstrings for code documentation. Please ensure all new code is properly documented.

## Visualization Guidelines

The Portfolio Management System supports various visualization capabilities for portfolio analysis and reporting. When contributing visualizations, please follow these guidelines:

### Visualization Libraries

- Use `matplotlib` for standard plots and charts
- Use `plotly` for interactive visualizations when appropriate
- Use `seaborn` for statistical visualizations and enhanced aesthetics
- Use `bokeh` for interactive web-based visualizations when needed
- Ensure visualizations are accessible and include proper labels, legends, and titles

### Standard Visualization Types

The system supports the following standard visualization types:

1. **Performance Visualizations**:
   - Equity curves with benchmark comparison
   - Drawdown charts with historical context
   - Returns distribution with statistical overlays
   - Rolling metrics (returns, volatility, Sharpe ratio)
   - Performance attribution by sector/asset
   - Underwater plots (drawdown duration)
   - Monthly/yearly returns heatmaps
   - Regime analysis charts

2. **Allocation Visualizations**:
   - Asset allocation pie charts and treemaps
   - Sector/category allocation bar charts
   - Allocation drift over time
   - Rebalancing impact analysis
   - Target vs. actual allocation comparison
   - Geographic exposure maps

3. **Risk Visualizations**:
   - Factor exposures with time series
   - Risk contribution pie charts and waterfall charts
   - Stress test results with scenario comparison
   - Correlation matrices and network graphs
   - Value at Risk (VaR) and Expected Shortfall visualizations
   - Risk decomposition by factor/sector
   - Monte Carlo simulation results

4. **Tax Visualizations**:
   - Realized gains/losses by time period
   - Tax lot distribution and aging
   - Wash sale impact analysis
   - Tax efficiency metrics
   - Tax-loss harvesting opportunities
   - After-tax returns comparison

### Interactive Visualization Features

For interactive visualizations, include the following features when appropriate:

- Tooltips with detailed information on hover
- Zoom and pan capabilities
- Filtering and sorting options
- Toggle visibility of data series
- Export and download options
- Responsive design for different screen sizes

### Adding New Visualizations

When adding new visualizations:

1. Follow the existing pattern in `backtesting/reporting.py` for consistency
2. Ensure visualizations are configurable (size, colors, labels, themes)
3. Support saving in multiple formats (PNG, PDF, SVG, HTML, JSON)
4. Make visualizations accessible (color-blind friendly palettes, text alternatives)
5. Implement both static and interactive versions when appropriate
6. Include examples in the documentation with sample code
7. Add appropriate tests for visualization functions
8. Optimize for performance with large datasets

### Visualization Configuration

All visualizations should support the following configuration options:

- Figure size and aspect ratio
- Color scheme/palette with defaults that match the system theme
- Font sizes and styles
- Title, axis labels, and legend positioning
- Grid lines and background styling
- Output format and resolution
- Data range selection

### Reporting Integration

Visualizations should be designed to work within the reporting framework:

1. Support inclusion in automated reports (PDF, HTML, dashboard)
2. Allow for batch generation of visualizations
3. Include metadata for report organization
4. Support templating for consistent styling across reports
5. Enable customization through the reporting configuration

## Submitting Changes

1. Push your changes to your fork: `git push origin feature/your-feature-name`
2. Submit a pull request to the main repository
3. The core team will review your pull request and provide feedback

## Release Process

The core team is responsible for releasing new versions. The process is as follows:

1. Update version number in `setup.py`
2. Update CHANGELOG.md
3. Create a new release on GitHub
4. Publish to PyPI

## Questions?

If you have any questions, please feel free to contact the core team.

Thank you for contributing to the Portfolio Management System!
