# Enhanced Reporting and Analytics Module

"""
This module provides comprehensive performance analytics, interactive data visualizations,
a customizable reporting framework, advanced attribution analysis, and comparative/scenario
analysis capabilities for the Friday trading system.
"""

from .reporting import (
    EnhancedReport,
    ReportingEngine,
    ReportTemplate,
    ReportFormat,
    VisualizationType
)

from .visualization import (
    PerformanceVisualizer,
    AllocationVisualizer,
    RiskVisualizer,
    TaxVisualizer,
    InteractiveVisualizer
)

from .attribution import (
    AttributionAnalysis,
    FactorAttribution,
    SectorAttribution,
    StyleAttribution
)

from .comparative import (
    ScenarioAnalysis,
    BenchmarkComparison,
    StrategyComparison,
    HistoricalComparison
)

__all__ = [
    # Reporting
    'EnhancedReport',
    'ReportingEngine',
    'ReportTemplate',
    'ReportFormat',
    'VisualizationType',
    
    # Visualization
    'PerformanceVisualizer',
    'AllocationVisualizer',
    'RiskVisualizer',
    'TaxVisualizer',
    'InteractiveVisualizer',
    
    # Attribution
    'AttributionAnalysis',
    'FactorAttribution',
    'SectorAttribution',
    'StyleAttribution',
    
    # Comparative
    'ScenarioAnalysis',
    'BenchmarkComparison',
    'StrategyComparison',
    'HistoricalComparison'
]