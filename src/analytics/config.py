"""Configuration module for the analytics components.

This module provides configuration management for the analytics components,
including default settings, configuration loading/saving, and validation.
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import logging

# Set up logging
logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Enum for configuration file formats."""
    JSON = 'json'
    YAML = 'yaml'


class AnalyticsConfig:
    """Configuration manager for analytics components."""
    
    # Default configuration
    DEFAULT_CONFIG = {
        'visualization': {
            'default_style': 'seaborn',
            'color_palette': 'viridis',
            'figure_size': [10, 6],
            'dpi': 100,
            'interactive': True,
            'theme': 'plotly_white',
            'template_path': None
        },
        'reporting': {
            'default_format': 'html',
            'include_code': False,
            'include_metrics_table': True,
            'logo_path': None,
            'company_name': None,
            'disclaimer_text': None,
            'output_directory': 'reports'
        },
        'performance': {
            'benchmark_id': 'SPY',
            'risk_free_rate': 0.0,
            'annualization_factor': 252,
            'rolling_window': 252,
            'drawdown_threshold': -0.1
        },
        'allocation': {
            'drift_threshold': 0.05,
            'rebalance_frequency': 'quarterly',
            'sector_classification': 'gics',
            'include_cash': True
        },
        'risk': {
            'var_confidence_level': 0.95,
            'var_window': 252,
            'stress_test_scenarios': [
                {'name': 'Market Crash', 'equity': -0.3, 'bonds': 0.05, 'commodities': -0.1, 'real_estate': -0.25},
                {'name': 'Rising Rates', 'equity': -0.05, 'bonds': -0.1, 'commodities': 0.03, 'real_estate': -0.08},
                {'name': 'Inflation Shock', 'equity': -0.07, 'bonds': -0.08, 'commodities': 0.15, 'real_estate': 0.05}
            ],
            'correlation_method': 'pearson',
            'risk_model': 'historical'
        },
        'attribution': {
            'factor_model': 'fama_french_5',
            'attribution_method': 'brinson',
            'lookback_period': 252
        },
        'dashboard': {
            'refresh_interval': 60,  # seconds
            'default_port': 8050,
            'theme': 'light',
            'include_metrics_table': True,
            'include_drawdown_table': True
        },
        'export': {
            'default_format': 'png',
            'default_dpi': 300,
            'default_directory': 'figures'
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file not found: {config_path}")
            return
        
        # Determine format based on file extension
        _, ext = os.path.splitext(config_path)
        config_format = ConfigFormat.YAML if ext.lower() in ['.yml', '.yaml'] else ConfigFormat.JSON
        
        try:
            with open(config_path, 'r') as f:
                if config_format == ConfigFormat.JSON:
                    loaded_config = json.load(f)
                else:  # YAML
                    loaded_config = yaml.safe_load(f)
            
            # Update configuration with loaded values
            self._update_config_recursive(self.config, loaded_config)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    def save_config(self, config_path: str, config_format: ConfigFormat = ConfigFormat.JSON) -> None:
        """Save configuration to file.
        
        Args:
            config_path: Path to save configuration file
            config_format: Format to save configuration (JSON or YAML)
        """
        try:
            with open(config_path, 'w') as f:
                if config_format == ConfigFormat.JSON:
                    json.dump(self.config, f, indent=4)
                else:  # YAML
                    yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            section: Configuration section
            key: Optional key within section
            default: Default value if section/key not found
            
        Returns:
            Configuration value
        """
        if section not in self.config:
            return default
        
        if key is None:
            return self.config[section]
        
        return self.config[section].get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            section: Configuration section
            key: Key within section
            value: Value to set
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def update(self, section: str, values: Dict[str, Any]) -> None:
        """Update multiple configuration values in a section.
        
        Args:
            section: Configuration section
            values: Dictionary of values to update
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section].update(values)
    
    def reset(self, section: Optional[str] = None) -> None:
        """Reset configuration to defaults.
        
        Args:
            section: Optional section to reset (if None, reset all)
        """
        if section is None:
            self.config = self.DEFAULT_CONFIG.copy()
        elif section in self.DEFAULT_CONFIG:
            self.config[section] = self.DEFAULT_CONFIG[section].copy()
    
    def _update_config_recursive(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Recursively update configuration dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_config_recursive(target[key], value)
            else:
                target[key] = value


class VisualizationConfig:
    """Configuration manager for visualization settings."""
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize visualization configuration.
        
        Args:
            config: Optional analytics configuration instance
        """
        self.config = config or AnalyticsConfig()
    
    @property
    def style(self) -> str:
        """Get default visualization style."""
        return self.config.get('visualization', 'default_style', 'seaborn')
    
    @property
    def color_palette(self) -> str:
        """Get default color palette."""
        return self.config.get('visualization', 'color_palette', 'viridis')
    
    @property
    def figure_size(self) -> List[int]:
        """Get default figure size."""
        return self.config.get('visualization', 'figure_size', [10, 6])
    
    @property
    def dpi(self) -> int:
        """Get default DPI."""
        return self.config.get('visualization', 'dpi', 100)
    
    @property
    def interactive(self) -> bool:
        """Get whether to use interactive visualizations by default."""
        return self.config.get('visualization', 'interactive', True)
    
    @property
    def theme(self) -> str:
        """Get default theme for interactive visualizations."""
        return self.config.get('visualization', 'theme', 'plotly_white')
    
    @property
    def template_path(self) -> Optional[str]:
        """Get path to custom visualization template."""
        return self.config.get('visualization', 'template_path')
    
    def update(self, **kwargs) -> None:
        """Update visualization configuration.
        
        Args:
            **kwargs: Configuration values to update
        """
        self.config.update('visualization', kwargs)


class ReportingConfig:
    """Configuration manager for reporting settings."""
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize reporting configuration.
        
        Args:
            config: Optional analytics configuration instance
        """
        self.config = config or AnalyticsConfig()
    
    @property
    def default_format(self) -> str:
        """Get default report format."""
        return self.config.get('reporting', 'default_format', 'html')
    
    @property
    def include_code(self) -> bool:
        """Get whether to include code in reports."""
        return self.config.get('reporting', 'include_code', False)
    
    @property
    def include_metrics_table(self) -> bool:
        """Get whether to include metrics table in reports."""
        return self.config.get('reporting', 'include_metrics_table', True)
    
    @property
    def logo_path(self) -> Optional[str]:
        """Get path to logo for reports."""
        return self.config.get('reporting', 'logo_path')
    
    @property
    def company_name(self) -> Optional[str]:
        """Get company name for reports."""
        return self.config.get('reporting', 'company_name')
    
    @property
    def disclaimer_text(self) -> Optional[str]:
        """Get disclaimer text for reports."""
        return self.config.get('reporting', 'disclaimer_text')
    
    @property
    def output_directory(self) -> str:
        """Get default output directory for reports."""
        return self.config.get('reporting', 'output_directory', 'reports')
    
    def update(self, **kwargs) -> None:
        """Update reporting configuration.
        
        Args:
            **kwargs: Configuration values to update
        """
        self.config.update('reporting', kwargs)


class PerformanceConfig:
    """Configuration manager for performance analysis settings."""
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize performance configuration.
        
        Args:
            config: Optional analytics configuration instance
        """
        self.config = config or AnalyticsConfig()
    
    @property
    def benchmark_id(self) -> str:
        """Get default benchmark ID."""
        return self.config.get('performance', 'benchmark_id', 'SPY')
    
    @property
    def risk_free_rate(self) -> float:
        """Get default risk-free rate."""
        return self.config.get('performance', 'risk_free_rate', 0.0)
    
    @property
    def annualization_factor(self) -> int:
        """Get default annualization factor."""
        return self.config.get('performance', 'annualization_factor', 252)
    
    @property
    def rolling_window(self) -> int:
        """Get default rolling window size."""
        return self.config.get('performance', 'rolling_window', 252)
    
    @property
    def drawdown_threshold(self) -> float:
        """Get default drawdown threshold."""
        return self.config.get('performance', 'drawdown_threshold', -0.1)
    
    def update(self, **kwargs) -> None:
        """Update performance configuration.
        
        Args:
            **kwargs: Configuration values to update
        """
        self.config.update('performance', kwargs)


class RiskConfig:
    """Configuration manager for risk analysis settings."""
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize risk configuration.
        
        Args:
            config: Optional analytics configuration instance
        """
        self.config = config or AnalyticsConfig()
    
    @property
    def var_confidence_level(self) -> float:
        """Get default VaR confidence level."""
        return self.config.get('risk', 'var_confidence_level', 0.95)
    
    @property
    def var_window(self) -> int:
        """Get default VaR window size."""
        return self.config.get('risk', 'var_window', 252)
    
    @property
    def stress_test_scenarios(self) -> List[Dict[str, Any]]:
        """Get default stress test scenarios."""
        return self.config.get('risk', 'stress_test_scenarios', [])
    
    @property
    def correlation_method(self) -> str:
        """Get default correlation method."""
        return self.config.get('risk', 'correlation_method', 'pearson')
    
    @property
    def risk_model(self) -> str:
        """Get default risk model."""
        return self.config.get('risk', 'risk_model', 'historical')
    
    def update(self, **kwargs) -> None:
        """Update risk configuration.
        
        Args:
            **kwargs: Configuration values to update
        """
        self.config.update('risk', kwargs)


class AttributionConfig:
    """Configuration manager for attribution analysis settings."""
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize attribution configuration.
        
        Args:
            config: Optional analytics configuration instance
        """
        self.config = config or AnalyticsConfig()
    
    @property
    def factor_model(self) -> str:
        """Get default factor model."""
        return self.config.get('attribution', 'factor_model', 'fama_french_5')
    
    @property
    def attribution_method(self) -> str:
        """Get default attribution method."""
        return self.config.get('attribution', 'attribution_method', 'brinson')
    
    @property
    def lookback_period(self) -> int:
        """Get default lookback period."""
        return self.config.get('attribution', 'lookback_period', 252)
    
    def update(self, **kwargs) -> None:
        """Update attribution configuration.
        
        Args:
            **kwargs: Configuration values to update
        """
        self.config.update('attribution', kwargs)


class DashboardConfig:
    """Configuration manager for dashboard settings."""
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        """Initialize dashboard configuration.
        
        Args:
            config: Optional analytics configuration instance
        """
        self.config = config or AnalyticsConfig()
    
    @property
    def refresh_interval(self) -> int:
        """Get default dashboard refresh interval (seconds)."""
        return self.config.get('dashboard', 'refresh_interval', 60)
    
    @property
    def default_port(self) -> int:
        """Get default dashboard port."""
        return self.config.get('dashboard', 'default_port', 8050)
    
    @property
    def theme(self) -> str:
        """Get default dashboard theme."""
        return self.config.get('dashboard', 'theme', 'light')
    
    @property
    def include_metrics_table(self) -> bool:
        """Get whether to include metrics table in dashboards."""
        return self.config.get('dashboard', 'include_metrics_table', True)
    
    @property
    def include_drawdown_table(self) -> bool:
        """Get whether to include drawdown table in dashboards."""
        return self.config.get('dashboard', 'include_drawdown_table', True)
    
    def update(self, **kwargs) -> None:
        """Update dashboard configuration.
        
        Args:
            **kwargs: Configuration values to update
        """
        self.config.update('dashboard', kwargs)


# Export classes
__all__ = [
    'ConfigFormat',
    'AnalyticsConfig',
    'VisualizationConfig',
    'ReportingConfig',
    'PerformanceConfig',
    'RiskConfig',
    'AttributionConfig',
    'DashboardConfig'
]