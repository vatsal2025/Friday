"""Command-line interface for the analytics module.

This module provides command-line access to the analytics functionality,
including report generation, dashboard launching, and analysis tools.
"""

import argparse
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import local modules
from ..portfolio.portfolio_factory import PortfolioFactory
from ..risk.risk_management_factory import RiskManagementFactory
from ..data.data_provider_factory import DataProviderFactory

# Import analytics modules
from .integration import AnalyticsIntegrator
from .reporting import ReportFormat, VisualizationType


def parse_date(date_str: str) -> datetime:
    """Parse date string into datetime object.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        
    Returns:
        Datetime object
    """
    return datetime.strptime(date_str, "%Y-%m-%d")


def parse_visualization_types(types_str: str) -> List[VisualizationType]:
    """Parse visualization types string into list of VisualizationType enums.
    
    Args:
        types_str: Comma-separated string of visualization types
        
    Returns:
        List of VisualizationType enums
    """
    if not types_str:
        return None
    
    type_map = {
        # Performance visualizations
        "equity_curve": VisualizationType.PERFORMANCE_EQUITY_CURVE,
        "drawdown": VisualizationType.PERFORMANCE_DRAWDOWN,
        "returns_dist": VisualizationType.PERFORMANCE_RETURNS_DISTRIBUTION,
        "monthly_returns": VisualizationType.PERFORMANCE_MONTHLY_RETURNS,
        "rolling_returns": VisualizationType.PERFORMANCE_ROLLING_RETURNS,
        "rolling_vol": VisualizationType.PERFORMANCE_ROLLING_VOLATILITY,
        "rolling_sharpe": VisualizationType.PERFORMANCE_ROLLING_SHARPE,
        "underwater": VisualizationType.PERFORMANCE_UNDERWATER,
        
        # Allocation visualizations
        "allocation": VisualizationType.ALLOCATION_CURRENT,
        "allocation_drift": VisualizationType.ALLOCATION_DRIFT,
        "allocation_history": VisualizationType.ALLOCATION_HISTORY,
        "sector_allocation": VisualizationType.ALLOCATION_SECTOR,
        
        # Risk visualizations
        "risk_contrib": VisualizationType.RISK_CONTRIBUTION,
        "corr_matrix": VisualizationType.RISK_CORRELATION_MATRIX,
        "factor_exposure": VisualizationType.RISK_FACTOR_EXPOSURE,
        "var": VisualizationType.RISK_VALUE_AT_RISK,
        "stress_test": VisualizationType.RISK_STRESS_TEST,
        
        # Tax visualizations
        "realized_gains": VisualizationType.TAX_REALIZED_GAINS,
        "unrealized_gains": VisualizationType.TAX_UNREALIZED_GAINS,
        "tax_impact": VisualizationType.TAX_IMPACT,
        "wash_sales": VisualizationType.TAX_WASH_SALES,
        
        # Attribution visualizations
        "factor_contrib": VisualizationType.ATTRIBUTION_FACTOR_CONTRIBUTION,
        "factor_return": VisualizationType.ATTRIBUTION_FACTOR_RETURN,
        "factor_exposure_attr": VisualizationType.ATTRIBUTION_FACTOR_EXPOSURE,
        "allocation_effect": VisualizationType.ATTRIBUTION_ALLOCATION_EFFECT,
        "selection_effect": VisualizationType.ATTRIBUTION_SELECTION_EFFECT,
        "interaction_effect": VisualizationType.ATTRIBUTION_INTERACTION_EFFECT,
        "total_effect": VisualizationType.ATTRIBUTION_TOTAL_EFFECT,
        
        # Comparative visualizations
        "comp_equity": VisualizationType.COMPARATIVE_EQUITY_CURVE,
        "comp_drawdown": VisualizationType.COMPARATIVE_DRAWDOWN,
        "comp_metrics": VisualizationType.COMPARATIVE_METRICS,
        "comp_rolling_returns": VisualizationType.COMPARATIVE_ROLLING_RETURNS,
        "comp_rolling_vol": VisualizationType.COMPARATIVE_ROLLING_VOLATILITY,
        "comp_rolling_sharpe": VisualizationType.COMPARATIVE_ROLLING_SHARPE,
        
        # Scenario visualizations
        "scenario_heatmap": VisualizationType.SCENARIO_HEATMAP,
        "scenario_bar": VisualizationType.SCENARIO_BAR,
        "scenario_comparison": VisualizationType.SCENARIO_COMPARISON
    }
    
    types = []
    for type_str in types_str.split(","):
        type_str = type_str.strip().lower()
        if type_str in type_map:
            types.append(type_map[type_str])
        else:
            print(f"Warning: Unknown visualization type '{type_str}'")
    
    return types


def parse_report_format(format_str: str) -> ReportFormat:
    """Parse report format string into ReportFormat enum.
    
    Args:
        format_str: Report format string
        
    Returns:
        ReportFormat enum
    """
    format_map = {
        "html": ReportFormat.HTML,
        "pdf": ReportFormat.PDF,
        "jupyter": ReportFormat.JUPYTER,
        "excel": ReportFormat.EXCEL,
        "json": ReportFormat.JSON,
        "csv": ReportFormat.CSV,
        "markdown": ReportFormat.MARKDOWN,
        "interactive": ReportFormat.INTERACTIVE
    }
    
    format_str = format_str.lower()
    if format_str in format_map:
        return format_map[format_str]
    else:
        print(f"Warning: Unknown report format '{format_str}', defaulting to HTML")
        return ReportFormat.HTML


def load_scenarios_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load scenario definitions from a JSON file.
    
    Args:
        file_path: Path to JSON file containing scenario definitions
        
    Returns:
        List of scenario definitions
    """
    if not os.path.exists(file_path):
        print(f"Error: Scenario file '{file_path}' not found")
        sys.exit(1)
    
    try:
        with open(file_path, 'r') as f:
            scenarios = json.load(f)
        
        if not isinstance(scenarios, list):
            print(f"Error: Scenario file must contain a list of scenario definitions")
            sys.exit(1)
        
        return scenarios
    except json.JSONDecodeError:
        print(f"Error: Failed to parse scenario file '{file_path}' as JSON")
        sys.exit(1)


def setup_integrator(config_path: Optional[str] = None) -> AnalyticsIntegrator:
    """Set up the analytics integrator with required components.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured AnalyticsIntegrator instance
    """
    # Load configuration if provided
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Create factories
    portfolio_factory = PortfolioFactory(config.get('portfolio', {}))
    risk_factory = RiskManagementFactory(config.get('risk', {}))
    data_factory = DataProviderFactory(config.get('data', {}))
    
    # Create components
    portfolio_manager = portfolio_factory.create_portfolio_manager()
    risk_manager = risk_factory.create_risk_manager()
    data_provider = data_factory.create_data_provider()
    
    # Create and return integrator
    return AnalyticsIntegrator(
        portfolio_manager=portfolio_manager,
        risk_manager=risk_manager,
        data_provider=data_provider,
        config_path=config_path
    )


def portfolio_report_command(args: argparse.Namespace) -> None:
    """Handle portfolio report command.
    
    Args:
        args: Command-line arguments
    """
    # Parse dates if provided
    start_date = parse_date(args.start_date) if args.start_date else None
    end_date = parse_date(args.end_date) if args.end_date else None
    
    # Parse visualization types if provided
    viz_types = parse_visualization_types(args.visualization_types)
    
    # Parse report format
    report_format = parse_report_format(args.format)
    
    # Set up integrator
    integrator = setup_integrator(args.config)
    
    # Generate report
    report_path = integrator.generate_portfolio_report(
        portfolio_id=args.portfolio_id,
        start_date=start_date,
        end_date=end_date,
        report_format=report_format,
        output_path=args.output,
        visualization_types=viz_types,
        include_benchmark=not args.no_benchmark,
        benchmark_id=args.benchmark_id
    )
    
    print(f"Portfolio report generated: {report_path}")


def attribution_report_command(args: argparse.Namespace) -> None:
    """Handle attribution report command.
    
    Args:
        args: Command-line arguments
    """
    # Parse dates if provided
    start_date = parse_date(args.start_date) if args.start_date else None
    end_date = parse_date(args.end_date) if args.end_date else None
    
    # Parse report format
    report_format = parse_report_format(args.format)
    
    # Set up integrator
    integrator = setup_integrator(args.config)
    
    # Generate report
    report_path = integrator.generate_attribution_report(
        portfolio_id=args.portfolio_id,
        start_date=start_date,
        end_date=end_date,
        attribution_type=args.attribution_type,
        report_format=report_format,
        output_path=args.output
    )
    
    print(f"Attribution report generated: {report_path}")


def comparative_report_command(args: argparse.Namespace) -> None:
    """Handle comparative report command.
    
    Args:
        args: Command-line arguments
    """
    # Parse dates if provided
    start_date = parse_date(args.start_date) if args.start_date else None
    end_date = parse_date(args.end_date) if args.end_date else None
    
    # Parse report format
    report_format = parse_report_format(args.format)
    
    # Parse portfolio IDs
    portfolio_ids = args.portfolio_ids.split(',')
    
    # Set up integrator
    integrator = setup_integrator(args.config)
    
    # Generate report
    report_path = integrator.generate_comparative_report(
        portfolio_ids=portfolio_ids,
        start_date=start_date,
        end_date=end_date,
        benchmark_id=args.benchmark_id,
        report_format=report_format,
        output_path=args.output
    )
    
    print(f"Comparative report generated: {report_path}")


def scenario_report_command(args: argparse.Namespace) -> None:
    """Handle scenario report command.
    
    Args:
        args: Command-line arguments
    """
    # Parse report format
    report_format = parse_report_format(args.format)
    
    # Parse portfolio IDs
    portfolio_ids = args.portfolio_ids.split(',')
    
    # Load scenarios from file
    scenarios = load_scenarios_from_file(args.scenarios_file)
    
    # Set up integrator
    integrator = setup_integrator(args.config)
    
    # Generate report
    report_path = integrator.generate_scenario_report(
        portfolio_ids=portfolio_ids,
        scenarios=scenarios,
        report_format=report_format,
        output_path=args.output
    )
    
    print(f"Scenario report generated: {report_path}")


def performance_dashboard_command(args: argparse.Namespace) -> None:
    """Handle performance dashboard command.
    
    Args:
        args: Command-line arguments
    """
    # Parse dates if provided
    start_date = parse_date(args.start_date) if args.start_date else None
    end_date = parse_date(args.end_date) if args.end_date else None
    
    # Set up integrator
    integrator = setup_integrator(args.config)
    
    # Launch dashboard
    print(f"Launching performance dashboard for portfolio {args.portfolio_id} on port {args.port}...")
    integrator.launch_performance_dashboard(
        portfolio_id=args.portfolio_id,
        start_date=start_date,
        end_date=end_date,
        benchmark_id=args.benchmark_id,
        port=args.port,
        debug=args.debug
    )


def allocation_dashboard_command(args: argparse.Namespace) -> None:
    """Handle allocation dashboard command.
    
    Args:
        args: Command-line arguments
    """
    # Set up integrator
    integrator = setup_integrator(args.config)
    
    # Launch dashboard
    print(f"Launching allocation dashboard for portfolio {args.portfolio_id} on port {args.port}...")
    integrator.launch_allocation_dashboard(
        portfolio_id=args.portfolio_id,
        port=args.port,
        debug=args.debug
    )


def risk_dashboard_command(args: argparse.Namespace) -> None:
    """Handle risk dashboard command.
    
    Args:
        args: Command-line arguments
    """
    # Parse dates if provided
    start_date = parse_date(args.start_date) if args.start_date else None
    end_date = parse_date(args.end_date) if args.end_date else None
    
    # Set up integrator
    integrator = setup_integrator(args.config)
    
    # Launch dashboard
    print(f"Launching risk dashboard for portfolio {args.portfolio_id} on port {args.port}...")
    integrator.launch_risk_dashboard(
        portfolio_id=args.portfolio_id,
        start_date=start_date,
        end_date=end_date,
        port=args.port,
        debug=args.debug
    )


def comparative_dashboard_command(args: argparse.Namespace) -> None:
    """Handle comparative dashboard command.
    
    Args:
        args: Command-line arguments
    """
    # Parse dates if provided
    start_date = parse_date(args.start_date) if args.start_date else None
    end_date = parse_date(args.end_date) if args.end_date else None
    
    # Parse portfolio IDs
    portfolio_ids = args.portfolio_ids.split(',')
    
    # Set up integrator
    integrator = setup_integrator(args.config)
    
    # Launch dashboard
    print(f"Launching comparative dashboard for portfolios {args.portfolio_ids} on port {args.port}...")
    integrator.launch_comparative_dashboard(
        portfolio_ids=portfolio_ids,
        start_date=start_date,
        end_date=end_date,
        benchmark_id=args.benchmark_id,
        port=args.port,
        debug=args.debug
    )


def scenario_dashboard_command(args: argparse.Namespace) -> None:
    """Handle scenario dashboard command.
    
    Args:
        args: Command-line arguments
    """
    # Parse portfolio IDs
    portfolio_ids = args.portfolio_ids.split(',')
    
    # Load scenarios from file
    scenarios = load_scenarios_from_file(args.scenarios_file)
    
    # Set up integrator
    integrator = setup_integrator(args.config)
    
    # Launch dashboard
    print(f"Launching scenario dashboard for portfolios {args.portfolio_ids} on port {args.port}...")
    integrator.launch_scenario_dashboard(
        portfolio_ids=portfolio_ids,
        scenarios=scenarios,
        port=args.port,
        debug=args.debug
    )


def main():
    """Main entry point for the analytics CLI."""
    parser = argparse.ArgumentParser(description="Portfolio Analytics CLI")
    parser.add_argument("--config", help="Path to configuration file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Portfolio report command
    portfolio_parser = subparsers.add_parser("portfolio-report", help="Generate portfolio report")
    portfolio_parser.add_argument("portfolio_id", help="Portfolio ID")
    portfolio_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    portfolio_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    portfolio_parser.add_argument("--format", default="html", help="Report format (html, pdf, jupyter, excel, json, csv, markdown, interactive)")
    portfolio_parser.add_argument("--output", help="Output path")
    portfolio_parser.add_argument("--visualization-types", help="Comma-separated list of visualization types")
    portfolio_parser.add_argument("--no-benchmark", action="store_true", help="Exclude benchmark comparison")
    portfolio_parser.add_argument("--benchmark-id", help="Benchmark ID")
    portfolio_parser.set_defaults(func=portfolio_report_command)
    
    # Attribution report command
    attribution_parser = subparsers.add_parser("attribution-report", help="Generate attribution report")
    attribution_parser.add_argument("portfolio_id", help="Portfolio ID")
    attribution_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    attribution_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    attribution_parser.add_argument("--attribution-type", default="factor", choices=["factor", "brinson", "risk"], help="Attribution type")
    attribution_parser.add_argument("--format", default="html", help="Report format (html, pdf, jupyter, excel, json, csv, markdown, interactive)")
    attribution_parser.add_argument("--output", help="Output path")
    attribution_parser.set_defaults(func=attribution_report_command)
    
    # Comparative report command
    comparative_parser = subparsers.add_parser("comparative-report", help="Generate comparative report")
    comparative_parser.add_argument("portfolio_ids", help="Comma-separated list of portfolio IDs")
    comparative_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    comparative_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    comparative_parser.add_argument("--benchmark-id", help="Benchmark ID")
    comparative_parser.add_argument("--format", default="html", help="Report format (html, pdf, jupyter, excel, json, csv, markdown, interactive)")
    comparative_parser.add_argument("--output", help="Output path")
    comparative_parser.set_defaults(func=comparative_report_command)
    
    # Scenario report command
    scenario_parser = subparsers.add_parser("scenario-report", help="Generate scenario report")
    scenario_parser.add_argument("portfolio_ids", help="Comma-separated list of portfolio IDs")
    scenario_parser.add_argument("scenarios_file", help="Path to JSON file containing scenario definitions")
    scenario_parser.add_argument("--format", default="html", help="Report format (html, pdf, jupyter, excel, json, csv, markdown, interactive)")
    scenario_parser.add_argument("--output", help="Output path")
    scenario_parser.set_defaults(func=scenario_report_command)
    
    # Performance dashboard command
    perf_dashboard_parser = subparsers.add_parser("performance-dashboard", help="Launch performance dashboard")
    perf_dashboard_parser.add_argument("portfolio_id", help="Portfolio ID")
    perf_dashboard_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    perf_dashboard_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    perf_dashboard_parser.add_argument("--benchmark-id", help="Benchmark ID")
    perf_dashboard_parser.add_argument("--port", type=int, default=8050, help="Port to run the dashboard on")
    perf_dashboard_parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    perf_dashboard_parser.set_defaults(func=performance_dashboard_command)
    
    # Allocation dashboard command
    alloc_dashboard_parser = subparsers.add_parser("allocation-dashboard", help="Launch allocation dashboard")
    alloc_dashboard_parser.add_argument("portfolio_id", help="Portfolio ID")
    alloc_dashboard_parser.add_argument("--port", type=int, default=8051, help="Port to run the dashboard on")
    alloc_dashboard_parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    alloc_dashboard_parser.set_defaults(func=allocation_dashboard_command)
    
    # Risk dashboard command
    risk_dashboard_parser = subparsers.add_parser("risk-dashboard", help="Launch risk dashboard")
    risk_dashboard_parser.add_argument("portfolio_id", help="Portfolio ID")
    risk_dashboard_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    risk_dashboard_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    risk_dashboard_parser.add_argument("--port", type=int, default=8052, help="Port to run the dashboard on")
    risk_dashboard_parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    risk_dashboard_parser.set_defaults(func=risk_dashboard_command)
    
    # Comparative dashboard command
    comp_dashboard_parser = subparsers.add_parser("comparative-dashboard", help="Launch comparative dashboard")
    comp_dashboard_parser.add_argument("portfolio_ids", help="Comma-separated list of portfolio IDs")
    comp_dashboard_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    comp_dashboard_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    comp_dashboard_parser.add_argument("--benchmark-id", help="Benchmark ID")
    comp_dashboard_parser.add_argument("--port", type=int, default=8053, help="Port to run the dashboard on")
    comp_dashboard_parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    comp_dashboard_parser.set_defaults(func=comparative_dashboard_command)
    
    # Scenario dashboard command
    scen_dashboard_parser = subparsers.add_parser("scenario-dashboard", help="Launch scenario dashboard")
    scen_dashboard_parser.add_argument("portfolio_ids", help="Comma-separated list of portfolio IDs")
    scen_dashboard_parser.add_argument("scenarios_file", help="Path to JSON file containing scenario definitions")
    scen_dashboard_parser.add_argument("--port", type=int, default=8054, help="Port to run the dashboard on")
    scen_dashboard_parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    scen_dashboard_parser.set_defaults(func=scenario_dashboard_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()