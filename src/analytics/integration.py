"""Integration module for connecting analytics components with the rest of the system.

This module provides integration points between the analytics components and other
system components, such as portfolio management, risk management, and data sources.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

# Import local modules
from ..portfolio.portfolio_manager import PortfolioManager
from ..portfolio.performance_calculator import PerformanceCalculator
from ..portfolio.allocation_manager import AllocationManager
from ..risk.risk_manager import RiskManager
from ..risk.factor_model import FactorModel
from ..data.data_provider import DataProvider

# Import analytics modules
from .reporting import EnhancedReport, ReportingEngine, ReportFormat, VisualizationType
from .visualization import (
    PerformanceVisualizer, AllocationVisualizer, 
    RiskVisualizer, TaxVisualizer, InteractiveVisualizer
)
from .attribution import (
    BrinsionAttributionAnalyzer, FactorAttributionAnalyzer, RiskAttributionAnalyzer
)
from .comparative import StrategyComparator, ScenarioAnalyzer, MonteCarloSimulator
from .dashboard import (
    PerformanceDashboard, AllocationDashboard, RiskDashboard, 
    ComparativeDashboard, ScenarioDashboard
)


class AnalyticsIntegrator:
    """Main integration class for connecting analytics with other system components."""
    
    def __init__(
        self,
        portfolio_manager: Optional[PortfolioManager] = None,
        risk_manager: Optional[RiskManager] = None,
        data_provider: Optional[DataProvider] = None,
        config_path: Optional[str] = None
    ):
        """Initialize the analytics integrator.
        
        Args:
            portfolio_manager: Optional portfolio manager instance
            risk_manager: Optional risk manager instance
            data_provider: Optional data provider instance
            config_path: Optional path to configuration file
        """
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.data_provider = data_provider
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize analytics components
        self.reporting_engine = ReportingEngine()
        self.performance_visualizer = PerformanceVisualizer()
        self.allocation_visualizer = AllocationVisualizer()
        self.risk_visualizer = RiskVisualizer()
        self.tax_visualizer = TaxVisualizer()
        self.interactive_visualizer = InteractiveVisualizer()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            return {}
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def generate_portfolio_report(
        self,
        portfolio_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        report_format: ReportFormat = ReportFormat.HTML,
        output_path: Optional[str] = None,
        visualization_types: Optional[List[VisualizationType]] = None,
        include_benchmark: bool = True,
        benchmark_id: Optional[str] = None
    ) -> str:
        """Generate a comprehensive portfolio report.
        
        Args:
            portfolio_id: Portfolio identifier
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            report_format: Report format (HTML, PDF, etc.)
            output_path: Optional path to save the report
            visualization_types: Optional list of visualization types to include
            include_benchmark: Whether to include benchmark comparison
            benchmark_id: Optional benchmark identifier
            
        Returns:
            Path to the generated report
        """
        if self.portfolio_manager is None:
            raise ValueError("Portfolio manager is required for generating portfolio reports")
        
        # Get portfolio data
        portfolio = self.portfolio_manager.get_portfolio(portfolio_id)
        
        # Get performance data
        performance_calculator = portfolio.get_performance_calculator()
        returns = performance_calculator.get_returns(start_date, end_date)
        
        # Get benchmark data if requested
        benchmark_returns = None
        if include_benchmark and benchmark_id and self.data_provider:
            benchmark_returns = self.data_provider.get_returns(benchmark_id, start_date, end_date)
        
        # Get allocation data
        allocation_manager = portfolio.get_allocation_manager()
        current_weights = allocation_manager.get_current_weights()
        target_weights = allocation_manager.get_target_weights()
        historical_weights = allocation_manager.get_historical_weights(start_date, end_date)
        
        # Get risk data if risk manager is available
        risk_contributions = None
        factor_exposures = None
        if self.risk_manager:
            risk_contributions = self.risk_manager.get_risk_contributions(portfolio_id)
            factor_exposures = self.risk_manager.get_factor_exposures(portfolio_id)
        
        # Set default visualization types if not provided
        if visualization_types is None:
            visualization_types = [
                VisualizationType.PERFORMANCE_EQUITY_CURVE,
                VisualizationType.PERFORMANCE_DRAWDOWN,
                VisualizationType.PERFORMANCE_MONTHLY_RETURNS,
                VisualizationType.ALLOCATION_CURRENT,
                VisualizationType.ALLOCATION_DRIFT,
                VisualizationType.RISK_CONTRIBUTION,
                VisualizationType.RISK_FACTOR_EXPOSURE
            ]
        
        # Create report
        report = EnhancedReport(
            title=f"Portfolio Report: {portfolio_id}",
            subtitle=f"Period: {start_date.strftime('%Y-%m-%d') if start_date else 'Inception'} to {end_date.strftime('%Y-%m-%d') if end_date else 'Present'}",
            portfolio_id=portfolio_id,
            returns=returns,
            benchmark_returns=benchmark_returns,
            current_weights=current_weights,
            target_weights=target_weights,
            historical_weights=historical_weights,
            risk_contributions=risk_contributions,
            factor_exposures=factor_exposures
        )
        
        # Generate report using reporting engine
        report_path = self.reporting_engine.generate_report(
            report,
            visualization_types=visualization_types,
            format=report_format,
            output_path=output_path
        )
        
        return report_path
    
    def generate_attribution_report(
        self,
        portfolio_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        attribution_type: str = "factor",
        report_format: ReportFormat = ReportFormat.HTML,
        output_path: Optional[str] = None
    ) -> str:
        """Generate an attribution analysis report.
        
        Args:
            portfolio_id: Portfolio identifier
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            attribution_type: Type of attribution analysis ("factor", "brinson", "risk")
            report_format: Report format (HTML, PDF, etc.)
            output_path: Optional path to save the report
            
        Returns:
            Path to the generated report
        """
        if self.portfolio_manager is None:
            raise ValueError("Portfolio manager is required for generating attribution reports")
        
        # Get portfolio data
        portfolio = self.portfolio_manager.get_portfolio(portfolio_id)
        
        # Get performance data
        performance_calculator = portfolio.get_performance_calculator()
        returns = performance_calculator.get_returns(start_date, end_date)
        
        # Get attribution data based on type
        if attribution_type == "factor" and self.risk_manager:
            # Factor attribution
            factor_model = self.risk_manager.get_factor_model(portfolio_id)
            factor_returns = factor_model.get_factor_returns(start_date, end_date)
            factor_exposures = factor_model.get_factor_exposures(portfolio_id)
            
            analyzer = FactorAttributionAnalyzer()
            attribution_data = analyzer.analyze(
                returns=returns,
                factor_returns=factor_returns,
                factor_exposures=factor_exposures
            )
            
            # Create report with factor attribution visualizations
            report = EnhancedReport(
                title=f"Factor Attribution Analysis: {portfolio_id}",
                subtitle=f"Period: {start_date.strftime('%Y-%m-%d') if start_date else 'Inception'} to {end_date.strftime('%Y-%m-%d') if end_date else 'Present'}",
                portfolio_id=portfolio_id,
                returns=returns,
                attribution_data=attribution_data
            )
            
            visualization_types = [
                VisualizationType.ATTRIBUTION_FACTOR_CONTRIBUTION,
                VisualizationType.ATTRIBUTION_FACTOR_RETURN,
                VisualizationType.ATTRIBUTION_FACTOR_EXPOSURE
            ]
        
        elif attribution_type == "brinson":
            # Brinson attribution
            allocation_manager = portfolio.get_allocation_manager()
            sector_weights = allocation_manager.get_sector_weights()
            benchmark_sector_weights = allocation_manager.get_benchmark_sector_weights()
            sector_returns = performance_calculator.get_sector_returns(start_date, end_date)
            benchmark_sector_returns = performance_calculator.get_benchmark_sector_returns(start_date, end_date)
            
            analyzer = BrinsionAttributionAnalyzer()
            attribution_data = analyzer.analyze(
                portfolio_weights=sector_weights,
                benchmark_weights=benchmark_sector_weights,
                portfolio_returns=sector_returns,
                benchmark_returns=benchmark_sector_returns
            )
            
            # Create report with Brinson attribution visualizations
            report = EnhancedReport(
                title=f"Brinson Attribution Analysis: {portfolio_id}",
                subtitle=f"Period: {start_date.strftime('%Y-%m-%d') if start_date else 'Inception'} to {end_date.strftime('%Y-%m-%d') if end_date else 'Present'}",
                portfolio_id=portfolio_id,
                returns=returns,
                attribution_data=attribution_data
            )
            
            visualization_types = [
                VisualizationType.ATTRIBUTION_ALLOCATION_EFFECT,
                VisualizationType.ATTRIBUTION_SELECTION_EFFECT,
                VisualizationType.ATTRIBUTION_INTERACTION_EFFECT,
                VisualizationType.ATTRIBUTION_TOTAL_EFFECT
            ]
        
        elif attribution_type == "risk" and self.risk_manager:
            # Risk attribution
            risk_contributions = self.risk_manager.get_risk_contributions(portfolio_id)
            factor_risk_contributions = self.risk_manager.get_factor_risk_contributions(portfolio_id)
            
            analyzer = RiskAttributionAnalyzer()
            attribution_data = {
                'asset_risk_contributions': risk_contributions,
                'factor_risk_contributions': factor_risk_contributions
            }
            
            # Create report with risk attribution visualizations
            report = EnhancedReport(
                title=f"Risk Attribution Analysis: {portfolio_id}",
                subtitle=f"Period: {start_date.strftime('%Y-%m-%d') if start_date else 'Inception'} to {end_date.strftime('%Y-%m-%d') if end_date else 'Present'}",
                portfolio_id=portfolio_id,
                returns=returns,
                attribution_data=attribution_data
            )
            
            visualization_types = [
                VisualizationType.RISK_CONTRIBUTION,
                VisualizationType.RISK_FACTOR_CONTRIBUTION
            ]
        
        else:
            raise ValueError(f"Unsupported attribution type: {attribution_type}")
        
        # Generate report using reporting engine
        report_path = self.reporting_engine.generate_report(
            report,
            visualization_types=visualization_types,
            format=report_format,
            output_path=output_path
        )
        
        return report_path
    
    def generate_comparative_report(
        self,
        portfolio_ids: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        benchmark_id: Optional[str] = None,
        report_format: ReportFormat = ReportFormat.HTML,
        output_path: Optional[str] = None
    ) -> str:
        """Generate a comparative analysis report for multiple portfolios.
        
        Args:
            portfolio_ids: List of portfolio identifiers
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            benchmark_id: Optional benchmark identifier
            report_format: Report format (HTML, PDF, etc.)
            output_path: Optional path to save the report
            
        Returns:
            Path to the generated report
        """
        if self.portfolio_manager is None:
            raise ValueError("Portfolio manager is required for generating comparative reports")
        
        # Get returns for each portfolio
        returns_dict = {}
        for portfolio_id in portfolio_ids:
            portfolio = self.portfolio_manager.get_portfolio(portfolio_id)
            performance_calculator = portfolio.get_performance_calculator()
            returns_dict[portfolio_id] = performance_calculator.get_returns(start_date, end_date)
        
        # Get benchmark returns if requested
        benchmark_returns = None
        if benchmark_id and self.data_provider:
            benchmark_returns = self.data_provider.get_returns(benchmark_id, start_date, end_date)
        
        # Create strategy comparator
        comparator = StrategyComparator()
        comparison_data = comparator.compare(returns_dict, benchmark_returns=benchmark_returns)
        
        # Create report
        report = EnhancedReport(
            title="Portfolio Comparison Report",
            subtitle=f"Period: {start_date.strftime('%Y-%m-%d') if start_date else 'Inception'} to {end_date.strftime('%Y-%m-%d') if end_date else 'Present'}",
            portfolio_ids=portfolio_ids,
            returns_dict=returns_dict,
            benchmark_returns=benchmark_returns,
            comparison_data=comparison_data
        )
        
        # Set visualization types for comparative report
        visualization_types = [
            VisualizationType.COMPARATIVE_EQUITY_CURVE,
            VisualizationType.COMPARATIVE_DRAWDOWN,
            VisualizationType.COMPARATIVE_METRICS,
            VisualizationType.COMPARATIVE_ROLLING_RETURNS,
            VisualizationType.COMPARATIVE_ROLLING_VOLATILITY,
            VisualizationType.COMPARATIVE_ROLLING_SHARPE
        ]
        
        # Generate report using reporting engine
        report_path = self.reporting_engine.generate_report(
            report,
            visualization_types=visualization_types,
            format=report_format,
            output_path=output_path
        )
        
        return report_path
    
    def generate_scenario_report(
        self,
        portfolio_ids: List[str],
        scenarios: List[Dict[str, Any]],
        report_format: ReportFormat = ReportFormat.HTML,
        output_path: Optional[str] = None
    ) -> str:
        """Generate a scenario analysis report.
        
        Args:
            portfolio_ids: List of portfolio identifiers
            scenarios: List of scenario definitions
            report_format: Report format (HTML, PDF, etc.)
            output_path: Optional path to save the report
            
        Returns:
            Path to the generated report
        """
        if self.portfolio_manager is None:
            raise ValueError("Portfolio manager is required for generating scenario reports")
        
        # Get portfolio weights for each portfolio
        portfolios = {}
        for portfolio_id in portfolio_ids:
            portfolio = self.portfolio_manager.get_portfolio(portfolio_id)
            allocation_manager = portfolio.get_allocation_manager()
            portfolios[portfolio_id] = allocation_manager.get_current_weights()
        
        # Create scenario analyzer
        analyzer = ScenarioAnalyzer()
        scenario_results = analyzer.analyze(portfolios, scenarios)
        
        # Create report
        report = EnhancedReport(
            title="Scenario Analysis Report",
            subtitle=f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}",
            portfolio_ids=portfolio_ids,
            scenario_results=scenario_results,
            scenarios=scenarios
        )
        
        # Set visualization types for scenario report
        visualization_types = [
            VisualizationType.SCENARIO_HEATMAP,
            VisualizationType.SCENARIO_BAR,
            VisualizationType.SCENARIO_COMPARISON
        ]
        
        # Generate report using reporting engine
        report_path = self.reporting_engine.generate_report(
            report,
            visualization_types=visualization_types,
            format=report_format,
            output_path=output_path
        )
        
        return report_path
    
    def launch_performance_dashboard(
        self,
        portfolio_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        benchmark_id: Optional[str] = None,
        port: int = 8050,
        debug: bool = False
    ) -> None:
        """Launch an interactive performance dashboard.
        
        Args:
            portfolio_id: Portfolio identifier
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            benchmark_id: Optional benchmark identifier
            port: Port to run the dashboard on
            debug: Whether to run in debug mode
        """
        if self.portfolio_manager is None:
            raise ValueError("Portfolio manager is required for launching dashboards")
        
        # Get portfolio data
        portfolio = self.portfolio_manager.get_portfolio(portfolio_id)
        performance_calculator = portfolio.get_performance_calculator()
        returns = performance_calculator.get_returns(start_date, end_date)
        
        # Get benchmark returns if requested
        benchmark_returns = None
        if benchmark_id and self.data_provider:
            benchmark_returns = self.data_provider.get_returns(benchmark_id, start_date, end_date)
        
        # Create and launch dashboard
        dashboard = PerformanceDashboard(
            returns=returns,
            benchmark_returns=benchmark_returns,
            title=f"Performance Dashboard: {portfolio_id}"
        )
        
        dashboard.run_server(debug=debug, port=port)
    
    def launch_allocation_dashboard(
        self,
        portfolio_id: str,
        port: int = 8051,
        debug: bool = False
    ) -> None:
        """Launch an interactive allocation dashboard.
        
        Args:
            portfolio_id: Portfolio identifier
            port: Port to run the dashboard on
            debug: Whether to run in debug mode
        """
        if self.portfolio_manager is None:
            raise ValueError("Portfolio manager is required for launching dashboards")
        
        # Get portfolio data
        portfolio = self.portfolio_manager.get_portfolio(portfolio_id)
        allocation_manager = portfolio.get_allocation_manager()
        current_weights = allocation_manager.get_current_weights()
        target_weights = allocation_manager.get_target_weights()
        historical_weights = allocation_manager.get_historical_weights()
        
        # Create and launch dashboard
        dashboard = AllocationDashboard(
            portfolio_weights=current_weights,
            target_weights=target_weights,
            historical_weights=historical_weights,
            title=f"Allocation Dashboard: {portfolio_id}"
        )
        
        dashboard.run_server(debug=debug, port=port)
    
    def launch_risk_dashboard(
        self,
        portfolio_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        port: int = 8052,
        debug: bool = False
    ) -> None:
        """Launch an interactive risk dashboard.
        
        Args:
            portfolio_id: Portfolio identifier
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            port: Port to run the dashboard on
            debug: Whether to run in debug mode
        """
        if self.portfolio_manager is None or self.risk_manager is None:
            raise ValueError("Portfolio manager and risk manager are required for launching risk dashboards")
        
        # Get portfolio data
        portfolio = self.portfolio_manager.get_portfolio(portfolio_id)
        performance_calculator = portfolio.get_performance_calculator()
        returns = performance_calculator.get_returns(start_date, end_date)
        
        # Get asset returns
        asset_returns = performance_calculator.get_asset_returns(start_date, end_date)
        
        # Get risk data
        risk_contributions = self.risk_manager.get_risk_contributions(portfolio_id)
        factor_exposures = self.risk_manager.get_factor_exposures(portfolio_id)
        factor_returns = self.risk_manager.get_factor_returns(start_date, end_date)
        
        # Create and launch dashboard
        dashboard = RiskDashboard(
            returns=returns,
            asset_returns=asset_returns,
            factor_exposures=factor_exposures,
            factor_returns=factor_returns,
            risk_contributions=risk_contributions,
            title=f"Risk Dashboard: {portfolio_id}"
        )
        
        dashboard.run_server(debug=debug, port=port)
    
    def launch_comparative_dashboard(
        self,
        portfolio_ids: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        benchmark_id: Optional[str] = None,
        port: int = 8053,
        debug: bool = False
    ) -> None:
        """Launch an interactive comparative dashboard.
        
        Args:
            portfolio_ids: List of portfolio identifiers
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            benchmark_id: Optional benchmark identifier
            port: Port to run the dashboard on
            debug: Whether to run in debug mode
        """
        if self.portfolio_manager is None:
            raise ValueError("Portfolio manager is required for launching dashboards")
        
        # Get returns for each portfolio
        returns_dict = {}
        for portfolio_id in portfolio_ids:
            portfolio = self.portfolio_manager.get_portfolio(portfolio_id)
            performance_calculator = portfolio.get_performance_calculator()
            returns_dict[portfolio_id] = performance_calculator.get_returns(start_date, end_date)
        
        # Get benchmark returns if requested
        benchmark_returns = None
        if benchmark_id and self.data_provider:
            benchmark_returns = self.data_provider.get_returns(benchmark_id, start_date, end_date)
        
        # Create and launch dashboard
        dashboard = ComparativeDashboard(
            returns_dict=returns_dict,
            benchmark_returns=benchmark_returns,
            title="Portfolio Comparison Dashboard"
        )
        
        dashboard.run_server(debug=debug, port=port)
    
    def launch_scenario_dashboard(
        self,
        portfolio_ids: List[str],
        scenarios: List[Dict[str, Any]],
        port: int = 8054,
        debug: bool = False
    ) -> None:
        """Launch an interactive scenario dashboard.
        
        Args:
            portfolio_ids: List of portfolio identifiers
            scenarios: List of scenario definitions
            port: Port to run the dashboard on
            debug: Whether to run in debug mode
        """
        if self.portfolio_manager is None:
            raise ValueError("Portfolio manager is required for launching dashboards")
        
        # Get portfolio weights for each portfolio
        portfolios = {}
        for portfolio_id in portfolio_ids:
            portfolio = self.portfolio_manager.get_portfolio(portfolio_id)
            allocation_manager = portfolio.get_allocation_manager()
            portfolios[portfolio_id] = allocation_manager.get_current_weights()
        
        # Create and launch dashboard
        dashboard = ScenarioDashboard(
            portfolios=portfolios,
            scenarios=scenarios,
            title="Scenario Analysis Dashboard"
        )
        
        dashboard.run_server(debug=debug, port=port)


# Export classes
__all__ = ['AnalyticsIntegrator']