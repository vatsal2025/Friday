# Enhanced Reporting and Analytics - Reporting Module

"""
This module provides a comprehensive reporting framework for generating
detailed performance reports with customizable templates and formats.
"""

import os
import json
import pandas as pd
import numpy as np
from enum import Enum, auto
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

from ..backtesting.reporting import ReportFormat as BaseReportFormat, ChartType
from ..portfolio.performance import PerformanceCalculator


class ReportFormat(BaseReportFormat):
    """Extended report format enum with additional formats."""
    DASHBOARD = auto()  # Interactive web dashboard
    API = auto()         # API response format
    INTERACTIVE = auto()  # Interactive HTML with JavaScript


class VisualizationType(Enum):
    """Types of visualizations available in the enhanced reporting system."""
    # Performance visualizations
    EQUITY_CURVE = auto()
    DRAWDOWN = auto()
    RETURNS_DISTRIBUTION = auto()
    ROLLING_RETURNS = auto()
    ROLLING_VOLATILITY = auto()
    ROLLING_SHARPE = auto()
    UNDERWATER = auto()
    MONTHLY_RETURNS_HEATMAP = auto()
    YEARLY_RETURNS_HEATMAP = auto()
    REGIME_ANALYSIS = auto()
    PERFORMANCE_ATTRIBUTION = auto()
    
    # Allocation visualizations
    ASSET_ALLOCATION = auto()
    SECTOR_ALLOCATION = auto()
    ALLOCATION_DRIFT = auto()
    REBALANCING_IMPACT = auto()
    TARGET_VS_ACTUAL = auto()
    GEOGRAPHIC_EXPOSURE = auto()
    
    # Risk visualizations
    FACTOR_EXPOSURE = auto()
    RISK_CONTRIBUTION = auto()
    STRESS_TEST = auto()
    CORRELATION_MATRIX = auto()
    VALUE_AT_RISK = auto()
    RISK_DECOMPOSITION = auto()
    MONTE_CARLO = auto()
    
    # Tax visualizations
    REALIZED_GAINS = auto()
    TAX_LOT_DISTRIBUTION = auto()
    WASH_SALE_IMPACT = auto()
    TAX_EFFICIENCY = auto()
    TAX_LOSS_HARVESTING = auto()
    AFTER_TAX_RETURNS = auto()
    
    # Comparative visualizations
    BENCHMARK_COMPARISON = auto()
    STRATEGY_COMPARISON = auto()
    SCENARIO_COMPARISON = auto()
    HISTORICAL_COMPARISON = auto()


class ReportTemplate:
    """Template for customizing report structure and appearance."""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize a report template.
        
        Args:
            name: Template name
            description: Template description
        """
        self.name = name
        self.description = description
        self.sections = []
        self.style = {
            "colors": {
                "primary": "#1f77b4",
                "secondary": "#ff7f0e",
                "positive": "#2ca02c",
                "negative": "#d62728",
                "neutral": "#7f7f7f"
            },
            "fonts": {
                "title": "Arial, 16pt, bold",
                "subtitle": "Arial, 14pt",
                "body": "Arial, 12pt",
                "caption": "Arial, 10pt"
            },
            "layout": {
                "page_size": "letter",
                "orientation": "portrait",
                "margins": {"top": 1, "right": 1, "bottom": 1, "left": 1}
            }
        }
        self.metadata = {}
    
    def add_section(self, title: str, visualizations: List[VisualizationType], 
                   description: str = "", order: int = None) -> None:
        """Add a section to the report template.
        
        Args:
            title: Section title
            visualizations: List of visualizations to include
            description: Section description
            order: Order of the section (lower numbers appear first)
        """
        section = {
            "title": title,
            "description": description,
            "visualizations": visualizations
        }
        
        if order is not None:
            section["order"] = order
            self.sections.insert(order, section)
        else:
            section["order"] = len(self.sections)
            self.sections.append(section)
    
    def set_style(self, style_dict: Dict[str, Any]) -> None:
        """Update the template style.
        
        Args:
            style_dict: Dictionary with style settings
        """
        for key, value in style_dict.items():
            if key in self.style:
                if isinstance(self.style[key], dict) and isinstance(value, dict):
                    self.style[key].update(value)
                else:
                    self.style[key] = value
    
    def save(self, filepath: str) -> None:
        """Save the template to a JSON file.
        
        Args:
            filepath: Path to save the template
        """
        template_data = {
            "name": self.name,
            "description": self.description,
            "sections": self.sections,
            "style": self.style,
            "metadata": self.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(template_data, f, indent=4)
    
    @classmethod
    def load(cls, filepath: str) -> 'ReportTemplate':
        """Load a template from a JSON file.
        
        Args:
            filepath: Path to the template file
            
        Returns:
            ReportTemplate: Loaded template
        """
        with open(filepath, 'r') as f:
            template_data = json.load(f)
        
        template = cls(template_data["name"], template_data["description"])
        template.sections = template_data["sections"]
        template.style = template_data["style"]
        template.metadata = template_data.get("metadata", {})
        
        return template


class EnhancedReport:
    """Enhanced report class for generating comprehensive performance reports."""
    
    def __init__(self, 
                 equity_curve: pd.Series,
                 returns: pd.Series,
                 positions: Optional[pd.DataFrame] = None,
                 trades: Optional[pd.DataFrame] = None,
                 benchmark_returns: Optional[pd.Series] = None,
                 strategy_name: str = "Strategy",
                 portfolio_name: str = "Portfolio",
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 initial_capital: float = 10000.0,
                 output_dir: str = "./reports",
                 template: Optional[ReportTemplate] = None):
        """Initialize the enhanced report.
        
        Args:
            equity_curve: Series of portfolio equity values over time
            returns: Series of portfolio returns
            positions: DataFrame of position data
            trades: DataFrame of trade data
            benchmark_returns: Series of benchmark returns
            strategy_name: Name of the strategy
            portfolio_name: Name of the portfolio
            start_date: Start date of the analysis period
            end_date: End date of the analysis period
            initial_capital: Initial capital amount
            output_dir: Directory to save reports
            template: Report template to use
        """
        self.equity_curve = equity_curve
        self.returns = returns
        self.positions = positions
        self.trades = trades
        self.benchmark_returns = benchmark_returns
        self.strategy_name = strategy_name
        self.portfolio_name = portfolio_name
        self.start_date = start_date or equity_curve.index[0]
        self.end_date = end_date or equity_curve.index[-1]
        self.initial_capital = initial_capital
        self.output_dir = output_dir
        self.template = template or self._create_default_template()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize performance calculator
        self.performance_calculator = PerformanceCalculator()
        
        # Calculate performance metrics
        self.metrics = self._calculate_metrics()
        
        # Initialize visualizers (will be imported from visualization module)
        self.visualizers = {}
        
    def _create_default_template(self) -> ReportTemplate:
        """Create a default report template.
        
        Returns:
            ReportTemplate: Default template
        """
        template = ReportTemplate("Default Template", "Standard performance report template")
        
        # Performance section
        template.add_section(
            "Performance Overview",
            [VisualizationType.EQUITY_CURVE, VisualizationType.DRAWDOWN, 
             VisualizationType.RETURNS_DISTRIBUTION],
            "Overview of strategy performance",
            0
        )
        
        # Risk section
        template.add_section(
            "Risk Analysis",
            [VisualizationType.ROLLING_VOLATILITY, VisualizationType.VALUE_AT_RISK, 
             VisualizationType.CORRELATION_MATRIX],
            "Analysis of strategy risk metrics",
            1
        )
        
        # Allocation section (if positions data available)
        template.add_section(
            "Allocation Analysis",
            [VisualizationType.ASSET_ALLOCATION, VisualizationType.SECTOR_ALLOCATION],
            "Analysis of portfolio allocations",
            2
        )
        
        # Comparative section (if benchmark data available)
        template.add_section(
            "Comparative Analysis",
            [VisualizationType.BENCHMARK_COMPARISON, VisualizationType.HISTORICAL_COMPARISON],
            "Comparison with benchmarks and historical performance",
            3
        )
        
        return template
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics.
        
        Returns:
            Dict: Dictionary of performance metrics
        """
        metrics = {}
        
        # Calculate basic metrics
        total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
        metrics["total_return"] = total_return
        
        # Calculate annualized return
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        years = days / 365.25
        metrics["annualized_return"] = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate volatility
        metrics["volatility"] = self.returns.std() * np.sqrt(252)  # Annualized
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        metrics["sharpe_ratio"] = metrics["annualized_return"] / metrics["volatility"] if metrics["volatility"] > 0 else 0
        
        # Calculate max drawdown
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        metrics["max_drawdown"] = drawdown.min()
        
        # Calculate Calmar ratio
        metrics["calmar_ratio"] = abs(metrics["annualized_return"] / metrics["max_drawdown"]) if metrics["max_drawdown"] < 0 else 0
        
        # Calculate Sortino ratio (downside deviation)
        negative_returns = self.returns[self.returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        metrics["sortino_ratio"] = metrics["annualized_return"] / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate win rate if trades data is available
        if self.trades is not None and not self.trades.empty:
            winning_trades = self.trades[self.trades["pnl"] > 0]
            metrics["win_rate"] = len(winning_trades) / len(self.trades) if len(self.trades) > 0 else 0
            metrics["profit_factor"] = abs(winning_trades["pnl"].sum() / self.trades[self.trades["pnl"] < 0]["pnl"].sum()) \
                if self.trades[self.trades["pnl"] < 0]["pnl"].sum() != 0 else 0
        
        # Calculate benchmark metrics if benchmark data is available
        if self.benchmark_returns is not None and not self.benchmark_returns.empty:
            benchmark_total_return = (1 + self.benchmark_returns).prod() - 1
            metrics["benchmark_total_return"] = benchmark_total_return
            metrics["benchmark_annualized_return"] = (1 + benchmark_total_return) ** (1 / years) - 1 if years > 0 else 0
            metrics["benchmark_volatility"] = self.benchmark_returns.std() * np.sqrt(252)  # Annualized
            metrics["benchmark_sharpe_ratio"] = metrics["benchmark_annualized_return"] / metrics["benchmark_volatility"] \
                if metrics["benchmark_volatility"] > 0 else 0
            
            # Calculate alpha and beta
            covariance = np.cov(self.returns.fillna(0), self.benchmark_returns.fillna(0))[0, 1]
            benchmark_variance = np.var(self.benchmark_returns.fillna(0))
            metrics["beta"] = covariance / benchmark_variance if benchmark_variance > 0 else 0
            metrics["alpha"] = metrics["annualized_return"] - metrics["beta"] * metrics["benchmark_annualized_return"]
            
            # Calculate information ratio
            tracking_error = (self.returns - self.benchmark_returns).std() * np.sqrt(252)
            metrics["information_ratio"] = (metrics["annualized_return"] - metrics["benchmark_annualized_return"]) / tracking_error \
                if tracking_error > 0 else 0
        
        return metrics
    
    def generate_report(self, 
                       report_format: ReportFormat = ReportFormat.HTML,
                       output_file: Optional[str] = None,
                       visualizations: Optional[List[VisualizationType]] = None,
                       show_plots: bool = False) -> str:
        """Generate a report in the specified format.
        
        Args:
            report_format: Format of the report
            output_file: Output file path (if None, a default name will be generated)
            visualizations: List of visualizations to include (if None, use template)
            show_plots: Whether to display plots (only applicable for interactive formats)
            
        Returns:
            str: Path to the generated report
        """
        # This is a placeholder for the actual implementation
        # The actual implementation will use the visualization module to generate the visualizations
        # and then combine them into a report based on the template and format
        
        # Generate default output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.strategy_name.replace(' ', '_')}_{timestamp}"
            
            if report_format == ReportFormat.HTML:
                output_file = os.path.join(self.output_dir, f"{filename}.html")
            elif report_format == ReportFormat.PDF:
                output_file = os.path.join(self.output_dir, f"{filename}.pdf")
            elif report_format == ReportFormat.PNG:
                output_file = os.path.join(self.output_dir, f"{filename}.png")
            elif report_format == ReportFormat.JPG:
                output_file = os.path.join(self.output_dir, f"{filename}.jpg")
            elif report_format == ReportFormat.SVG:
                output_file = os.path.join(self.output_dir, f"{filename}.svg")
            elif report_format == ReportFormat.JSON:
                output_file = os.path.join(self.output_dir, f"{filename}.json")
            elif report_format == ReportFormat.CSV:
                output_file = os.path.join(self.output_dir, f"{filename}.csv")
            elif report_format == ReportFormat.EXCEL:
                output_file = os.path.join(self.output_dir, f"{filename}.xlsx")
            elif report_format == ReportFormat.DASHBOARD:
                output_file = os.path.join(self.output_dir, f"{filename}_dashboard")
            elif report_format == ReportFormat.API:
                output_file = os.path.join(self.output_dir, f"{filename}_api.json")
            elif report_format == ReportFormat.INTERACTIVE:
                output_file = os.path.join(self.output_dir, f"{filename}_interactive.html")
        
        # Placeholder for actual report generation
        print(f"Generating {report_format.name} report to {output_file}")
        
        # Return the path to the generated report
        return output_file
    
    def export_metrics(self, output_file: Optional[str] = None) -> str:
        """Export performance metrics to a JSON file.
        
        Args:
            output_file: Output file path (if None, a default name will be generated)
            
        Returns:
            str: Path to the exported metrics file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.strategy_name.replace(' ', '_')}_metrics_{timestamp}.json"
            output_file = os.path.join(self.output_dir, filename)
        
        with open(output_file, 'w') as f:
            json.dump(self.metrics, f, indent=4, default=str)
        
        return output_file


class ReportingEngine:
    """Engine for generating and managing reports."""
    
    def __init__(self, output_dir: str = "./reports"):
        """Initialize the reporting engine.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        self.templates = {}
        self.reports = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load default templates
        self._load_default_templates()
    
    def _load_default_templates(self) -> None:
        """Load default report templates."""
        # Performance template
        performance_template = ReportTemplate("Performance", "Comprehensive performance analysis template")
        performance_template.add_section(
            "Performance Overview",
            [VisualizationType.EQUITY_CURVE, VisualizationType.DRAWDOWN, 
             VisualizationType.RETURNS_DISTRIBUTION, VisualizationType.MONTHLY_RETURNS_HEATMAP],
            "Overview of strategy performance",
            0
        )
        performance_template.add_section(
            "Detailed Performance Metrics",
            [VisualizationType.ROLLING_RETURNS, VisualizationType.ROLLING_VOLATILITY, 
             VisualizationType.ROLLING_SHARPE, VisualizationType.UNDERWATER],
            "Detailed performance metrics over time",
            1
        )
        self.templates["performance"] = performance_template
        
        # Risk template
        risk_template = ReportTemplate("Risk", "Comprehensive risk analysis template")
        risk_template.add_section(
            "Risk Overview",
            [VisualizationType.VALUE_AT_RISK, VisualizationType.STRESS_TEST, 
             VisualizationType.RISK_CONTRIBUTION],
            "Overview of strategy risk",
            0
        )
        risk_template.add_section(
            "Detailed Risk Analysis",
            [VisualizationType.FACTOR_EXPOSURE, VisualizationType.CORRELATION_MATRIX, 
             VisualizationType.RISK_DECOMPOSITION, VisualizationType.MONTE_CARLO],
            "Detailed risk analysis",
            1
        )
        self.templates["risk"] = risk_template
        
        # Allocation template
        allocation_template = ReportTemplate("Allocation", "Portfolio allocation analysis template")
        allocation_template.add_section(
            "Allocation Overview",
            [VisualizationType.ASSET_ALLOCATION, VisualizationType.SECTOR_ALLOCATION, 
             VisualizationType.GEOGRAPHIC_EXPOSURE],
            "Overview of portfolio allocations",
            0
        )
        allocation_template.add_section(
            "Allocation Analysis",
            [VisualizationType.ALLOCATION_DRIFT, VisualizationType.TARGET_VS_ACTUAL, 
             VisualizationType.REBALANCING_IMPACT],
            "Analysis of allocation changes and impact",
            1
        )
        self.templates["allocation"] = allocation_template
        
        # Tax template
        tax_template = ReportTemplate("Tax", "Tax analysis template")
        tax_template.add_section(
            "Tax Overview",
            [VisualizationType.REALIZED_GAINS, VisualizationType.TAX_LOT_DISTRIBUTION],
            "Overview of tax implications",
            0
        )
        tax_template.add_section(
            "Detailed Tax Analysis",
            [VisualizationType.WASH_SALE_IMPACT, VisualizationType.TAX_EFFICIENCY, 
             VisualizationType.TAX_LOSS_HARVESTING, VisualizationType.AFTER_TAX_RETURNS],
            "Detailed tax analysis and optimization",
            1
        )
        self.templates["tax"] = tax_template
        
        # Comparative template
        comparative_template = ReportTemplate("Comparative", "Comparative analysis template")
        comparative_template.add_section(
            "Benchmark Comparison",
            [VisualizationType.BENCHMARK_COMPARISON, VisualizationType.HISTORICAL_COMPARISON],
            "Comparison with benchmarks and historical performance",
            0
        )
        comparative_template.add_section(
            "Strategy and Scenario Analysis",
            [VisualizationType.STRATEGY_COMPARISON, VisualizationType.SCENARIO_COMPARISON],
            "Comparison with other strategies and scenarios",
            1
        )
        self.templates["comparative"] = comparative_template
    
    def create_report(self, 
                     equity_curve: pd.Series,
                     returns: pd.Series,
                     positions: Optional[pd.DataFrame] = None,
                     trades: Optional[pd.DataFrame] = None,
                     benchmark_returns: Optional[pd.Series] = None,
                     strategy_name: str = "Strategy",
                     portfolio_name: str = "Portfolio",
                     template_name: Optional[str] = None) -> EnhancedReport:
        """Create a new report.
        
        Args:
            equity_curve: Series of portfolio equity values over time
            returns: Series of portfolio returns
            positions: DataFrame of position data
            trades: DataFrame of trade data
            benchmark_returns: Series of benchmark returns
            strategy_name: Name of the strategy
            portfolio_name: Name of the portfolio
            template_name: Name of the template to use (if None, use default)
            
        Returns:
            EnhancedReport: Created report
        """
        template = None
        if template_name is not None and template_name in self.templates:
            template = self.templates[template_name]
        
        report = EnhancedReport(
            equity_curve=equity_curve,
            returns=returns,
            positions=positions,
            trades=trades,
            benchmark_returns=benchmark_returns,
            strategy_name=strategy_name,
            portfolio_name=portfolio_name,
            output_dir=self.output_dir,
            template=template
        )
        
        self.reports.append(report)
        return report
    
    def add_template(self, template: ReportTemplate) -> None:
        """Add a template to the engine.
        
        Args:
            template: Template to add
        """
        self.templates[template.name] = template
    
    def save_template(self, template_name: str, filepath: Optional[str] = None) -> str:
        """Save a template to a file.
        
        Args:
            template_name: Name of the template to save
            filepath: Path to save the template (if None, use default)
            
        Returns:
            str: Path to the saved template
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        if filepath is None:
            os.makedirs(os.path.join(self.output_dir, "templates"), exist_ok=True)
            filepath = os.path.join(self.output_dir, "templates", f"{template_name.lower()}.json")
        
        self.templates[template_name].save(filepath)
        return filepath
    
    def load_template(self, filepath: str) -> str:
        """Load a template from a file.
        
        Args:
            filepath: Path to the template file
            
        Returns:
            str: Name of the loaded template
        """
        template = ReportTemplate.load(filepath)
        self.templates[template.name] = template
        return template.name
    
    def batch_generate_reports(self, 
                              report_format: ReportFormat = ReportFormat.HTML,
                              template_name: Optional[str] = None) -> List[str]:
        """Generate reports for all strategies in batch.
        
        Args:
            report_format: Format of the reports
            template_name: Name of the template to use (if None, use default)
            
        Returns:
            List[str]: Paths to the generated reports
        """
        report_paths = []
        
        for report in self.reports:
            if template_name is not None and template_name in self.templates:
                report.template = self.templates[template_name]
            
            report_path = report.generate_report(report_format=report_format)
            report_paths.append(report_path)
        
        return report_paths