"""Tests for the analytics configuration module."""

import os
import json
import yaml
import tempfile
import unittest
from unittest.mock import patch, mock_open

from analytics.config import (
    ConfigFormat,
    AnalyticsConfig,
    VisualizationConfig,
    ReportingConfig,
    PerformanceConfig,
    RiskConfig,
    AttributionConfig,
    DashboardConfig
)


class TestAnalyticsConfig(unittest.TestCase):
    """Test cases for AnalyticsConfig class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AnalyticsConfig()
        self.test_config = {
            'visualization': {
                'default_style': 'test_style',
                'color_palette': 'test_palette'
            },
            'reporting': {
                'default_format': 'pdf'
            }
        }

    def test_default_config(self):
        """Test default configuration."""
        self.assertEqual(self.config.get('visualization', 'default_style'), 'seaborn')
        self.assertEqual(self.config.get('reporting', 'default_format'), 'html')
        self.assertEqual(self.config.get('performance', 'benchmark_id'), 'SPY')

    def test_get_config(self):
        """Test getting configuration values."""
        # Get section
        vis_section = self.config.get('visualization')
        self.assertIsInstance(vis_section, dict)
        self.assertEqual(vis_section['default_style'], 'seaborn')

        # Get key
        style = self.config.get('visualization', 'default_style')
        self.assertEqual(style, 'seaborn')

        # Get with default
        nonexistent = self.config.get('nonexistent', 'key', 'default_value')
        self.assertEqual(nonexistent, 'default_value')

    def test_set_config(self):
        """Test setting configuration values."""
        self.config.set('visualization', 'default_style', 'test_style')
        self.assertEqual(self.config.get('visualization', 'default_style'), 'test_style')

        # Set in nonexistent section
        self.config.set('new_section', 'key', 'value')
        self.assertEqual(self.config.get('new_section', 'key'), 'value')

    def test_update_config(self):
        """Test updating multiple configuration values."""
        updates = {
            'default_style': 'test_style',
            'color_palette': 'test_palette'
        }
        self.config.update('visualization', updates)
        self.assertEqual(self.config.get('visualization', 'default_style'), 'test_style')
        self.assertEqual(self.config.get('visualization', 'color_palette'), 'test_palette')

        # Update nonexistent section
        self.config.update('new_section', {'key': 'value'})
        self.assertEqual(self.config.get('new_section', 'key'), 'value')

    def test_reset_config(self):
        """Test resetting configuration."""
        # Modify config
        self.config.set('visualization', 'default_style', 'test_style')
        self.assertEqual(self.config.get('visualization', 'default_style'), 'test_style')

        # Reset section
        self.config.reset('visualization')
        self.assertEqual(self.config.get('visualization', 'default_style'), 'seaborn')

        # Reset all
        self.config.set('visualization', 'default_style', 'test_style')
        self.config.reset()
        self.assertEqual(self.config.get('visualization', 'default_style'), 'seaborn')

    def test_load_json_config(self):
        """Test loading JSON configuration."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
            temp_name = temp.name
            json.dump(self.test_config, temp)

        try:
            config = AnalyticsConfig(temp_name)
            self.assertEqual(config.get('visualization', 'default_style'), 'test_style')
            self.assertEqual(config.get('visualization', 'color_palette'), 'test_palette')
            self.assertEqual(config.get('reporting', 'default_format'), 'pdf')
            # Check that other defaults are preserved
            self.assertEqual(config.get('performance', 'benchmark_id'), 'SPY')
        finally:
            os.unlink(temp_name)

    def test_load_yaml_config(self):
        """Test loading YAML configuration."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp:
            temp_name = temp.name
            yaml.dump(self.test_config, temp)

        try:
            config = AnalyticsConfig(temp_name)
            self.assertEqual(config.get('visualization', 'default_style'), 'test_style')
            self.assertEqual(config.get('visualization', 'color_palette'), 'test_palette')
            self.assertEqual(config.get('reporting', 'default_format'), 'pdf')
        finally:
            os.unlink(temp_name)

    def test_save_json_config(self):
        """Test saving JSON configuration."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
            temp_name = temp.name

        try:
            self.config.set('visualization', 'default_style', 'test_style')
            self.config.save_config(temp_name, ConfigFormat.JSON)

            with open(temp_name, 'r') as f:
                loaded_config = json.load(f)

            self.assertEqual(loaded_config['visualization']['default_style'], 'test_style')
        finally:
            os.unlink(temp_name)

    def test_save_yaml_config(self):
        """Test saving YAML configuration."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp:
            temp_name = temp.name

        try:
            self.config.set('visualization', 'default_style', 'test_style')
            self.config.save_config(temp_name, ConfigFormat.YAML)

            with open(temp_name, 'r') as f:
                loaded_config = yaml.safe_load(f)

            self.assertEqual(loaded_config['visualization']['default_style'], 'test_style')
        finally:
            os.unlink(temp_name)

    @patch('logging.Logger.warning')
    def test_load_nonexistent_config(self, mock_warning):
        """Test loading nonexistent configuration file."""
        config = AnalyticsConfig('nonexistent_file.json')
        mock_warning.assert_called_once()
        self.assertEqual(config.get('visualization', 'default_style'), 'seaborn')

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_config_exception(self, mock_json_load, mock_file_open):
        """Test exception handling when loading configuration."""
        mock_json_load.side_effect = Exception('Test exception')

        with patch('logging.Logger.error') as mock_error:
            config = AnalyticsConfig('test.json')
            mock_error.assert_called_once()
            self.assertEqual(config.get('visualization', 'default_style'), 'seaborn')

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_config_exception(self, mock_json_dump, mock_file_open):
        """Test exception handling when saving configuration."""
        mock_json_dump.side_effect = Exception('Test exception')

        with patch('logging.Logger.error') as mock_error:
            self.config.save_config('test.json')
            mock_error.assert_called_once()


class TestSpecializedConfigs(unittest.TestCase):
    """Test cases for specialized configuration classes."""

    def setUp(self):
        """Set up test fixtures."""
        self.analytics_config = AnalyticsConfig()
        self.analytics_config.set('visualization', 'default_style', 'test_style')
        self.analytics_config.set('reporting', 'default_format', 'pdf')
        self.analytics_config.set('performance', 'benchmark_id', 'test_benchmark')
        self.analytics_config.set('risk', 'var_confidence_level', 0.99)
        self.analytics_config.set('attribution', 'factor_model', 'test_model')
        self.analytics_config.set('dashboard', 'refresh_interval', 30)

    def test_visualization_config(self):
        """Test VisualizationConfig class."""
        config = VisualizationConfig(self.analytics_config)
        self.assertEqual(config.style, 'test_style')
        self.assertEqual(config.color_palette, 'viridis')  # Default value

        # Test update
        config.update(color_palette='test_palette')
        self.assertEqual(config.color_palette, 'test_palette')

    def test_reporting_config(self):
        """Test ReportingConfig class."""
        config = ReportingConfig(self.analytics_config)
        self.assertEqual(config.default_format, 'pdf')
        self.assertEqual(config.include_code, False)  # Default value

        # Test update
        config.update(include_code=True)
        self.assertEqual(config.include_code, True)

    def test_performance_config(self):
        """Test PerformanceConfig class."""
        config = PerformanceConfig(self.analytics_config)
        self.assertEqual(config.benchmark_id, 'test_benchmark')
        self.assertEqual(config.risk_free_rate, 0.0)  # Default value

        # Test update
        config.update(risk_free_rate=0.02)
        self.assertEqual(config.risk_free_rate, 0.02)

    def test_risk_config(self):
        """Test RiskConfig class."""
        config = RiskConfig(self.analytics_config)
        self.assertEqual(config.var_confidence_level, 0.99)
        self.assertEqual(config.correlation_method, 'pearson')  # Default value

        # Test update
        config.update(correlation_method='spearman')
        self.assertEqual(config.correlation_method, 'spearman')

    def test_attribution_config(self):
        """Test AttributionConfig class."""
        config = AttributionConfig(self.analytics_config)
        self.assertEqual(config.factor_model, 'test_model')
        self.assertEqual(config.attribution_method, 'brinson')  # Default value

        # Test update
        config.update(attribution_method='test_method')
        self.assertEqual(config.attribution_method, 'test_method')

    def test_dashboard_config(self):
        """Test DashboardConfig class."""
        config = DashboardConfig(self.analytics_config)
        self.assertEqual(config.refresh_interval, 30)
        self.assertEqual(config.theme, 'light')  # Default value

        # Test update
        config.update(theme='dark')
        self.assertEqual(config.theme, 'dark')


if __name__ == '__main__':
    unittest.main()