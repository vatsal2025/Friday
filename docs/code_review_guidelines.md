# Friday AI Trading System Code Review Guidelines

## Overview

This document outlines the code review process and standards for the Friday AI Trading System project. Code reviews are a critical part of our development process to ensure code quality, knowledge sharing, and consistent implementation.

## Code Review Process

### When to Request a Review

- All code changes, regardless of size, should be reviewed before merging to the main branch.
- Create a pull request (PR) with a clear description of the changes and the purpose.
- Assign at least one reviewer with knowledge of the affected subsystem.

### Reviewer Responsibilities

1. Review the code within 24 hours of assignment when possible.
2. Provide constructive feedback.
3. Verify that the code meets the standards outlined in this document.
4. Approve only when all issues have been addressed.

### Author Responsibilities

1. Respond to feedback in a timely manner.
2. Address all comments before requesting re-review.
3. Explain design decisions when necessary.
4. Keep PRs focused and reasonably sized (under 500 lines when possible).

## Code Quality Standards

### Functionality

- Code works as intended and meets requirements.
- Edge cases are handled appropriately.
- Error handling is comprehensive and user-friendly.
- Performance considerations are addressed.

### Readability and Maintainability

- Code is clear and easy to understand.
- Variable and function names are descriptive and follow naming conventions.
- Complex logic includes explanatory comments.
- Functions and methods are reasonably sized (generally under 50 lines).

### Architecture and Design

- Code follows the project's architectural patterns.
- Responsibilities are properly separated.
- Code reuses existing components when appropriate.
- New abstractions are justified and well-designed.

### Testing

- New code includes appropriate unit tests.
- Tests cover both normal operation and edge cases.
- Test coverage meets project standards (minimum 80%).
- Tests are clear and maintainable.

### Documentation

- Code includes proper docstrings following the Google style.
- Complex algorithms or business logic are explained.
- API changes are reflected in documentation.
- README and other documentation are updated as needed.

## Language-Specific Guidelines

### Python

- Follow PEP 8 style guidelines.
- Use type hints for function parameters and return values.
- Prefer explicit over implicit.
- Use context managers (`with` statements) for resource management.
- Follow the principle of least surprise.

### JavaScript/TypeScript (for Dashboard)

- Use ES6+ features appropriately.
- Prefer const over let, and avoid var.
- Use TypeScript interfaces for complex data structures.
- Follow functional programming principles when appropriate.

## Security Considerations

- No credentials or secrets in code.
- Input validation for all user inputs.
- SQL queries use parameterized statements.
- Authentication and authorization checks are comprehensive.
- API endpoints have appropriate rate limiting.

## Performance Considerations

- Database queries are optimized.
- Expensive operations are profiled.
- Memory usage is reasonable.
- Concurrency issues are addressed.
- Large data processing uses appropriate techniques (streaming, batching).

## Common Issues to Watch For

1. **Magic Numbers**: Unexplained numeric literals in code.
2. **Duplicated Code**: Similar code appearing in multiple places.
3. **Overly Complex Methods**: Methods that do too much or are difficult to understand.
4. **Inadequate Error Handling**: Catching exceptions without proper handling.
5. **Race Conditions**: Particularly in multi-threaded or asynchronous code.
6. **Insecure Practices**: Such as using `eval()` or storing sensitive data insecurely.
7. **Tight Coupling**: Components that are unnecessarily dependent on each other.

## Automated Checks

The following automated checks run on all PRs:

- Linting (flake8, eslint)
- Type checking (mypy)
- Unit tests
- Integration tests
- Security scanning
- Documentation generation

All automated checks must pass before a PR can be merged.

## Continuous Improvement

The code review process itself is subject to review and improvement. If you have suggestions for improving these guidelines or the review process, please open an issue or PR with your proposed changes.