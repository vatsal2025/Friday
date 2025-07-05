"""Multi-timeframe module for the Friday AI Trading System.

This module provides access to multi-timeframe processing functionality.
It re-exports the classes and functions from multi_timeframe_processor.py.
"""

# Re-export all functionality from multi_timeframe_processor.py
from src.data.processing.multi_timeframe_processor import (
    TimeframeConverter,
    TimeframeAlignment,
    DataTimeframe,
    MultiTimeframeProcessor,
)

# Add any additional functionality specific to this module here