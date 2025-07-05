"""Pytest configuration file for the Friday AI Trading System.

This module contains fixtures and configuration for pytest.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Mock configuration for testing
MOCK_API_CONFIG = {
    'host': '127.0.0.1',
    'port': 5000,
    'debug': False,
    'threaded': True,
    'cors_enabled': True,
    'cors_origins': ['*'],
    'rate_limit_enabled': True,
    'rate_limit_window': 60,
    'rate_limit_max_requests': 100,
    'jwt_secret_key': 'test_secret_key',
    'jwt_expiry_minutes': 60,
    'max_content_length': 10 * 1024 * 1024,
    'api_docs_enabled': True,
    'api_docs_path': '/docs',
    'log_requests': True,
    'request_log_file': 'logs/api/requests.log'
}

@pytest.fixture(autouse=True)
def mock_config():
    """Mock the configuration for testing."""
    with patch('src.infrastructure.config.get_config') as mock_get_config:
        def side_effect(section=None, key=None):
            if section == 'API_CONFIG':
                if key is None:
                    return MOCK_API_CONFIG
                return MOCK_API_CONFIG.get(key)
            # Add other mock configurations as needed
            return MagicMock()
        
        mock_get_config.side_effect = side_effect
        yield mock_get_config