"""
Common utilities and shared components for the Friday AI Trading System.
"""

from .mocks import install_optional_mocks

# Install optional package mocks when the common package is imported
install_optional_mocks()

__all__ = ['install_optional_mocks']
