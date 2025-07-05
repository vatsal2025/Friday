"""
Fallback mocks for optional packages that may not be available in CI environments.
These mocks provide minimal implementations to keep tests runnable.
"""

import sys
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock


class TALibMock:
    """Mock for TA-Lib technical analysis library."""
    
    @staticmethod
    def SMA(data, timeperiod=30):
        """Simple Moving Average mock."""
        if len(data) < timeperiod:
            return [0] * len(data)
        result = []
        for i in range(len(data)):
            if i < timeperiod - 1:
                result.append(0)
            else:
                result.append(sum(data[i-timeperiod+1:i+1]) / timeperiod)
        return result

    @staticmethod
    def EMA(data, timeperiod=30):
        """Exponential Moving Average mock."""
        if not data:
            return []
        result = [data[0]]
        multiplier = 2 / (timeperiod + 1)
        for i in range(1, len(data)):
            result.append((data[i] * multiplier) + (result[i-1] * (1 - multiplier)))
        return result

    @staticmethod
    def RSI(data, timeperiod=14):
        """Relative Strength Index mock."""
        if len(data) <= timeperiod:
            return [50] * len(data)
        return [50] * len(data)  # Simplified mock

    @staticmethod
    def MACD(data, fastperiod=12, slowperiod=26, signalperiod=9):
        """MACD mock."""
        size = len(data)
        return ([0] * size, [0] * size, [0] * size)

    @staticmethod
    def BBANDS(data, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
        """Bollinger Bands mock."""
        size = len(data)
        return ([max(data)] * size, [sum(data)/len(data)] * size, [min(data)] * size)


class OptionalPackageMock:
    """Generic mock for any missing optional package."""
    
    def __init__(self, package_name: str):
        self.package_name = package_name
    
    def __getattr__(self, name):
        return MagicMock()
    
    def __call__(self, *args, **kwargs):
        return MagicMock()


def install_optional_mocks():
    """Install mocks for optional packages that might not be available."""
    optional_packages = {
        'talib': TALibMock(),
        'TA_Lib': TALibMock(),
        'alpha_vantage': OptionalPackageMock('alpha_vantage'),
        'quandl': OptionalPackageMock('quandl'),
        'ccxt': OptionalPackageMock('ccxt'),
        'tweepy': OptionalPackageMock('tweepy'),
        'fredapi': OptionalPackageMock('fredapi'),
        'plotly': OptionalPackageMock('plotly'),
        'bokeh': OptionalPackageMock('bokeh'),
        'tensorflow': OptionalPackageMock('tensorflow'),
        'torch': OptionalPackageMock('torch'),
        'sklearn': OptionalPackageMock('sklearn'),
        'lightgbm': OptionalPackageMock('lightgbm'),
        'xgboost': OptionalPackageMock('xgboost'),
    }
    
    for package_name, mock_obj in optional_packages.items():
        if package_name not in sys.modules:
            try:
                __import__(package_name)
            except ImportError:
                sys.modules[package_name] = mock_obj


# Install mocks when this module is imported
install_optional_mocks()
