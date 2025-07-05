"""Integration module for backtesting framework.

This module provides integration between the backtesting framework and other
components of the Friday system, such as the data providers, strategy registry,
and model registry.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.backtesting.engine import BacktestEngine, Event, EventType
from src.backtesting.performance import PerformanceAnalytics
from src.backtesting.costs import TransactionCostModel, CompositeCostModel
from src.backtesting.reporting import BacktestReport, ReportFormat
from src.data.providers.base import DataProvider
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class BacktestRunner:
    """Backtest runner for integrating with the Friday system.
    
    This class provides a high-level interface for running backtests using
    the Friday system's components.
    """
    
    def __init__(
        self,
        data_provider: DataProvider,
        strategy_name: str,
        initial_capital: float = 100000.0,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        transaction_cost_model: Optional[TransactionCostModel] = None,
        benchmark_symbol: Optional[str] = None,
        risk_free_rate: float = 0.0,
        output_dir: str = "reports",
    ):
        """Initialize the backtest runner.
        
        Args:
            data_provider: Data provider for fetching market data
            strategy_name: Name of the strategy to backtest
            initial_capital: Initial capital for the backtest (default: 100000.0)
            start_date: Start date for the backtest (default: None, use all available data)
            end_date: End date for the backtest (default: None, use all available data)
            transaction_cost_model: Transaction cost model (default: None)
            benchmark_symbol: Symbol for benchmark comparison (default: None)
            risk_free_rate: Annual risk-free rate (default: 0.0)
            output_dir: Output directory for reports (default: "reports")
        """
        self.data_provider = data_provider
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.transaction_cost_model = transaction_cost_model
        self.benchmark_symbol = benchmark_symbol
        self.risk_free_rate = risk_free_rate
        self.output_dir = output_dir
        
        # Initialize backtest engine
        self.engine = BacktestEngine(initial_capital=initial_capital)
        
        # Initialize results storage
        self.results = None
        self.report = None
        
        logger.info(f"Initialized backtest runner for strategy: {strategy_name}")
    
    def load_strategy(self, strategy_class: Any, **strategy_params) -> None:
        """Load a strategy for backtesting.
        
        Args:
            strategy_class: Strategy class to instantiate
            **strategy_params: Parameters to pass to the strategy constructor
        """
        # Instantiate strategy
        self.strategy = strategy_class(**strategy_params)
        
        # Register strategy with backtest engine
        self.engine.register_strategy(self.strategy)
        
        logger.info(f"Loaded strategy: {self.strategy.__class__.__name__}")
    
    def load_data(self, symbols: List[str], timeframe: str = "1d") -> None:
        """Load data for backtesting.
        
        Args:
            symbols: List of symbols to load data for
            timeframe: Timeframe for the data (default: "1d")
        """
        # Load data for each symbol
        for symbol in symbols:
            # Fetch data from data provider
            data = self.data_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=self.start_date,
                end_date=self.end_date,
            )
            
            # Add data to backtest engine
            self.engine.add_data(symbol=symbol, data=data)
        
        # Load benchmark data if specified
        if self.benchmark_symbol is not None and self.benchmark_symbol not in symbols:
            benchmark_data = self.data_provider.get_historical_data(
                symbol=self.benchmark_symbol,
                timeframe=timeframe,
                start_date=self.start_date,
                end_date=self.end_date,
            )
            
            # Add benchmark data to backtest engine
            self.engine.add_data(symbol=self.benchmark_symbol, data=benchmark_data)
        
        logger.info(f"Loaded data for symbols: {symbols}")
    
    def run(self, verbose: bool = False) -> Dict[str, Any]:
        """Run the backtest.
        
        Args:
            verbose: Whether to print progress (default: False)
            
        Returns:
            Dictionary with backtest results
        """
        # Set transaction cost model if provided
        if self.transaction_cost_model is not None:
            self.engine.set_transaction_cost_model(self.transaction_cost_model)
        
        # Run backtest
        self.engine.run(verbose=verbose)
        
        # Get results
        self.results = self.engine.get_results()
        
        # Create benchmark returns if benchmark symbol is provided
        benchmark_returns = None
        if self.benchmark_symbol is not None:
            benchmark_data = self.engine.get_data(self.benchmark_symbol)
            if benchmark_data is not None and "close" in benchmark_data.columns:
                benchmark_returns = pd.DataFrame(
                    benchmark_data["close"].pct_change().dropna()
                )
        
        # Create backtest report
        self.report = BacktestReport(
            equity_curve=self.results["equity_curve"],
            returns=self.results["returns"],
            trades=self.results["trades"],
            positions=self.results.get("positions"),
            benchmark_returns=benchmark_returns,
            strategy_name=self.strategy_name,
            initial_capital=self.initial_capital,
            risk_free_rate=self.risk_free_rate,
            output_dir=self.output_dir,
        )
        
        # Get summary
        summary = self.report.get_summary()
        
        logger.info(f"Backtest completed for strategy: {self.strategy_name}")
        logger.info(f"Total return: {summary['total_return_pct']:.2f}%")
        logger.info(f"Sharpe ratio: {summary['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {summary['max_drawdown']:.2f}%")
        
        return summary
    
    def generate_report(
        self,
        report_format: Union[str, ReportFormat] = ReportFormat.HTML,
        include_charts: Optional[List[str]] = None,
        filename: Optional[str] = None,
        show_plots: bool = False,
    ) -> str:
        """Generate a backtest report.
        
        Args:
            report_format: Report format (default: HTML)
            include_charts: List of charts to include (default: all)
            filename: Output filename (default: auto-generated)
            show_plots: Whether to show plots (default: False)
            
        Returns:
            Path to the generated report
        """
        if self.report is None:
            raise ValueError("Backtest must be run before generating a report")
        
        # Generate report
        report_path = self.report.generate_report(
            report_format=report_format,
            include_charts=include_charts,
            filename=filename,
            show_plots=show_plots,
        )
        
        logger.info(f"Generated backtest report: {report_path}")
        
        return report_path
    
    def get_results(self) -> Dict[str, Any]:
        """Get the backtest results.
        
        Returns:
            Dictionary with backtest results
        """
        if self.results is None:
            raise ValueError("Backtest must be run before getting results")
        
        return self.results
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get the performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if self.report is None:
            raise ValueError("Backtest must be run before getting performance metrics")
        
        return self.report.performance.get_metrics()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the backtest results.
        
        Returns:
            Dictionary with backtest summary
        """
        if self.report is None:
            raise ValueError("Backtest must be run before getting summary")
        
        return self.report.get_summary()


class WalkForwardAnalyzer:
    """Walk-forward analyzer for strategy optimization and validation.
    
    This class implements walk-forward analysis, a technique for strategy
    optimization and validation that combines in-sample optimization with
    out-of-sample testing.
    """
    
    def __init__(
        self,
        data_provider: DataProvider,
        strategy_class: Any,
        initial_capital: float = 100000.0,
        train_size: float = 0.7,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        transaction_cost_model: Optional[TransactionCostModel] = None,
        benchmark_symbol: Optional[str] = None,
        risk_free_rate: float = 0.0,
        output_dir: str = "reports",
    ):
        """Initialize the walk-forward analyzer.
        
        Args:
            data_provider: Data provider for fetching market data
            strategy_class: Strategy class to instantiate
            initial_capital: Initial capital for the backtest (default: 100000.0)
            train_size: Proportion of data to use for training (default: 0.7)
            window_size: Size of each window in days (default: None, use all data)
            step_size: Size of each step in days (default: None, use window_size)
            transaction_cost_model: Transaction cost model (default: None)
            benchmark_symbol: Symbol for benchmark comparison (default: None)
            risk_free_rate: Annual risk-free rate (default: 0.0)
            output_dir: Output directory for reports (default: "reports")
        """
        self.data_provider = data_provider
        self.strategy_class = strategy_class
        self.initial_capital = initial_capital
        self.train_size = train_size
        self.window_size = window_size
        self.step_size = step_size if step_size is not None else window_size
        self.transaction_cost_model = transaction_cost_model
        self.benchmark_symbol = benchmark_symbol
        self.risk_free_rate = risk_free_rate
        self.output_dir = output_dir
        
        # Initialize results storage
        self.results = []
        self.optimized_params = []
        self.out_of_sample_results = []
        
        logger.info(f"Initialized walk-forward analyzer for strategy: {strategy_class.__name__}")
    
    def load_data(self, symbols: List[str], timeframe: str = "1d", start_date: Optional[Union[str, datetime]] = None, end_date: Optional[Union[str, datetime]] = None) -> None:
        """Load data for walk-forward analysis.
        
        Args:
            symbols: List of symbols to load data for
            timeframe: Timeframe for the data (default: "1d")
            start_date: Start date for the data (default: None, use all available data)
            end_date: End date for the data (default: None, use all available data)
        """
        self.symbols = symbols
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        
        # Load data for each symbol
        self.data = {}
        for symbol in symbols:
            # Fetch data from data provider
            data = self.data_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
            
            # Store data
            self.data[symbol] = data
        
        # Load benchmark data if specified
        if self.benchmark_symbol is not None and self.benchmark_symbol not in symbols:
            self.data[self.benchmark_symbol] = self.data_provider.get_historical_data(
                symbol=self.benchmark_symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
        
        logger.info(f"Loaded data for symbols: {symbols}")
    
    def optimize(self, param_grid: Dict[str, List[Any]], metric: str = "sharpe_ratio", maximize: bool = True) -> List[Dict[str, Any]]:
        """Perform walk-forward optimization.
        
        Args:
            param_grid: Dictionary with parameter names as keys and lists of parameter values as values
            metric: Metric to optimize (default: "sharpe_ratio")
            maximize: Whether to maximize or minimize the metric (default: True)
            
        Returns:
            List of dictionaries with optimization results for each window
        """
        if not self.data:
            raise ValueError("Data must be loaded before optimization")
        
        # Get the date range from the first symbol's data
        first_symbol = next(iter(self.data))
        dates = self.data[first_symbol].index
        
        # Calculate window and step sizes if not provided
        if self.window_size is None:
            self.window_size = len(dates)
        
        if self.step_size is None:
            self.step_size = self.window_size
        
        # Create windows
        windows = []
        for i in range(0, len(dates) - self.window_size + 1, self.step_size):
            window_start = dates[i]
            window_end = dates[min(i + self.window_size - 1, len(dates) - 1)]
            
            # Calculate train/test split
            split_idx = i + int(self.window_size * self.train_size)
            train_end = dates[min(split_idx, len(dates) - 1)]
            
            windows.append({
                "window_start": window_start,
                "window_end": window_end,
                "train_start": window_start,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": window_end,
            })
        
        # Perform optimization for each window
        for window in windows:
            logger.info(f"Optimizing window: {window['window_start']} to {window['window_end']}")
            
            # Create parameter combinations
            from itertools import product
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            param_combinations = list(product(*param_values))
            
            # Initialize best parameters and best metric value
            best_params = None
            best_metric_value = float("-inf") if maximize else float("inf")
            
            # Test each parameter combination
            for params in param_combinations:
                # Create parameter dictionary
                param_dict = dict(zip(param_names, params))
                
                # Create backtest runner for in-sample period
                runner = BacktestRunner(
                    data_provider=self.data_provider,
                    strategy_name=f"{self.strategy_class.__name__}_optimization",
                    initial_capital=self.initial_capital,
                    start_date=window["train_start"],
                    end_date=window["train_end"],
                    transaction_cost_model=self.transaction_cost_model,
                    benchmark_symbol=self.benchmark_symbol,
                    risk_free_rate=self.risk_free_rate,
                    output_dir=self.output_dir,
                )
                
                # Load strategy with current parameters
                runner.load_strategy(self.strategy_class, **param_dict)
                
                # Load data
                runner.load_data(symbols=self.symbols, timeframe=self.timeframe)
                
                # Run backtest
                runner.run(verbose=False)
                
                # Get performance metrics
                metrics = runner.get_performance_metrics()
                
                # Check if this is the best parameter set so far
                metric_value = metrics.get(metric, 0)
                if (maximize and metric_value > best_metric_value) or (not maximize and metric_value < best_metric_value):
                    best_metric_value = metric_value
                    best_params = param_dict
            
            # Store best parameters for this window
            self.optimized_params.append({
                "window": window,
                "params": best_params,
                "metric": metric,
                "metric_value": best_metric_value,
            })
            
            # Run out-of-sample test with best parameters
            runner = BacktestRunner(
                data_provider=self.data_provider,
                strategy_name=f"{self.strategy_class.__name__}_validation",
                initial_capital=self.initial_capital,
                start_date=window["test_start"],
                end_date=window["test_end"],
                transaction_cost_model=self.transaction_cost_model,
                benchmark_symbol=self.benchmark_symbol,
                risk_free_rate=self.risk_free_rate,
                output_dir=self.output_dir,
            )
            
            # Load strategy with best parameters
            runner.load_strategy(self.strategy_class, **best_params)
            
            # Load data
            runner.load_data(symbols=self.symbols, timeframe=self.timeframe)
            
            # Run backtest
            results = runner.run(verbose=False)
            
            # Store out-of-sample results
            self.out_of_sample_results.append({
                "window": window,
                "params": best_params,
                "results": results,
            })
            
            logger.info(f"Best parameters for window: {best_params}")
            logger.info(f"Out-of-sample performance: {results['total_return_pct']:.2f}% return, {results['sharpe_ratio']:.2f} Sharpe ratio")
        
        return self.optimized_params
    
    def generate_report(self, report_format: Union[str, ReportFormat] = ReportFormat.HTML, filename: Optional[str] = None) -> str:
        """Generate a walk-forward analysis report.
        
        Args:
            report_format: Report format (default: HTML)
            filename: Output filename (default: auto-generated)
            
        Returns:
            Path to the generated report
        """
        if not self.out_of_sample_results:
            raise ValueError("Walk-forward analysis must be run before generating a report")
        
        # Create filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"walkforward_{self.strategy_class.__name__}_{timestamp}"
        
        # Create report based on format
        if isinstance(report_format, str):
            report_format = ReportFormat(report_format.lower())
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate report based on format
        if report_format == ReportFormat.HTML:
            report_path = self._generate_html_report(filename)
        elif report_format == ReportFormat.JSON:
            report_path = self._generate_json_report(filename)
        elif report_format == ReportFormat.EXCEL:
            report_path = self._generate_excel_report(filename)
        else:
            raise ValueError(f"Unsupported report format for walk-forward analysis: {report_format}")
        
        logger.info(f"Generated walk-forward analysis report: {report_path}")
        
        return report_path
    
    def _generate_html_report(self, filename: str) -> str:
        """Generate HTML report for walk-forward analysis.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the generated report
        """
        import base64
        from io import BytesIO
        import matplotlib.pyplot as plt
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.strategy_class.__name__} - Walk-Forward Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                .metrics {{ display: flex; flex-wrap: wrap; margin-bottom: 20px; }}
                .metric {{ background-color: #f5f5f5; border-radius: 5px; padding: 15px; margin: 10px; flex: 1; min-width: 200px; }}
                .metric h3 {{ margin-top: 0; color: #333; }}
                .metric p {{ font-size: 24px; font-weight: bold; margin: 5px 0; }}
                .chart {{ margin: 20px 0; text-align: center; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>{self.strategy_class.__name__} - Walk-Forward Analysis</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Summary</h2>
            <div class="metrics">
        """
        
        # Calculate summary metrics
        total_returns = [result["results"]["total_return_pct"] for result in self.out_of_sample_results]
        sharpe_ratios = [result["results"]["sharpe_ratio"] for result in self.out_of_sample_results]
        max_drawdowns = [result["results"]["max_drawdown"] for result in self.out_of_sample_results]
        
        # Add summary metrics
        metrics = [
            ("Average Return", f"{sum(total_returns) / len(total_returns):.2f}%"),
            ("Average Sharpe", f"{sum(sharpe_ratios) / len(sharpe_ratios):.2f}"),
            ("Average Max DD", f"{sum(max_drawdowns) / len(max_drawdowns):.2f}%"),
            ("Win Rate", f"{len([r for r in total_returns if r > 0]) / len(total_returns) * 100:.2f}%"),
            ("# Windows", f"{len(self.out_of_sample_results)}"),
        ]
        
        for name, value in metrics:
            html_content += f"""
                <div class="metric">
                    <h3>{name}</h3>
                    <p>{value}</p>
                </div>
            """
        
        html_content += "</div>\n"
        
        # Create performance chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot out-of-sample returns
        ax.bar(range(len(total_returns)), total_returns, color=["green" if r > 0 else "red" for r in total_returns])
        
        # Add labels and title
        ax.set_title("Out-of-Sample Returns by Window")
        ax.set_xlabel("Window")
        ax.set_ylabel("Return (%)")
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        
        # Convert figure to base64 image
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        
        # Add chart to HTML
        html_content += f"""
        <h2>Performance by Window</h2>
        <div class="chart">
            <img src="data:image/png;base64,{img_str}" alt="Performance by Window" width="800">
        </div>
        """
        
        # Add window details table
        html_content += """
        <h2>Window Details</h2>
        <table>
            <tr>
                <th>Window</th>
                <th>Train Period</th>
                <th>Test Period</th>
                <th>Best Parameters</th>
                <th>Return</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown</th>
            </tr>
        """
        
        for i, result in enumerate(self.out_of_sample_results):
            window = result["window"]
            params = result["params"]
            results = result["results"]
            
            # Format parameters as string
            params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            
            html_content += f"""
            <tr>
                <td>{i + 1}</td>
                <td>{window["train_start"].strftime("%Y-%m-%d")} to {window["train_end"].strftime("%Y-%m-%d")}</td>
                <td>{window["test_start"].strftime("%Y-%m-%d")} to {window["test_end"].strftime("%Y-%m-%d")}</td>
                <td>{params_str}</td>
                <td>{results["total_return_pct"]:.2f}%</td>
                <td>{results["sharpe_ratio"]:.2f}</td>
                <td>{results["max_drawdown"]:.2f}%</td>
            </tr>
            """
        
        html_content += """
        </table>
        </body>
        </html>
        """
        
        # Save HTML to file
        if not filename.endswith(".html"):
            filename += ".html"
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w") as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_json_report(self, filename: str) -> str:
        """Generate JSON report for walk-forward analysis.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the generated report
        """
        import json
        
        # Create report data
        report_data = {
            "strategy": self.strategy_class.__name__,
            "generated_at": datetime.now().isoformat(),
            "windows": [],
        }
        
        # Add window details
        for i, result in enumerate(self.out_of_sample_results):
            window = result["window"]
            params = result["params"]
            results = result["results"]
            
            report_data["windows"].append({
                "window_number": i + 1,
                "train_start": window["train_start"].isoformat(),
                "train_end": window["train_end"].isoformat(),
                "test_start": window["test_start"].isoformat(),
                "test_end": window["test_end"].isoformat(),
                "best_parameters": params,
                "results": results,
            })
        
        # Calculate summary metrics
        total_returns = [result["results"]["total_return_pct"] for result in self.out_of_sample_results]
        sharpe_ratios = [result["results"]["sharpe_ratio"] for result in self.out_of_sample_results]
        max_drawdowns = [result["results"]["max_drawdown"] for result in self.out_of_sample_results]
        
        report_data["summary"] = {
            "average_return": sum(total_returns) / len(total_returns),
            "average_sharpe_ratio": sum(sharpe_ratios) / len(sharpe_ratios),
            "average_max_drawdown": sum(max_drawdowns) / len(max_drawdowns),
            "win_rate": len([r for r in total_returns if r > 0]) / len(total_returns),
            "number_of_windows": len(self.out_of_sample_results),
        }
        
        # Save to file
        if not filename.endswith(".json"):
            filename += ".json"
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=4)
        
        return output_path
    
    def _generate_excel_report(self, filename: str) -> str:
        """Generate Excel report for walk-forward analysis.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the generated report
        """
        try:
            import xlsxwriter
        except ImportError:
            logger.error("XlsxWriter is required for Excel generation. Install with 'pip install xlsxwriter'")
            return ""
        
        # Create Excel filename
        if not filename.endswith(".xlsx"):
            filename += ".xlsx"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Create Excel writer
        writer = pd.ExcelWriter(output_path, engine="xlsxwriter")
        
        # Create workbook and add worksheets
        workbook = writer.book
        
        # Create summary worksheet
        summary_data = {
            "Metric": [
                "Strategy",
                "Number of Windows",
                "Average Return (%)",
                "Average Sharpe Ratio",
                "Average Max Drawdown (%)",
                "Win Rate (%)",
            ],
            "Value": [
                self.strategy_class.__name__,
                len(self.out_of_sample_results),
                sum([result["results"]["total_return_pct"] for result in self.out_of_sample_results]) / len(self.out_of_sample_results),
                sum([result["results"]["sharpe_ratio"] for result in self.out_of_sample_results]) / len(self.out_of_sample_results),
                sum([result["results"]["max_drawdown"] for result in self.out_of_sample_results]) / len(self.out_of_sample_results),
                len([r for r in [result["results"]["total_return_pct"] for result in self.out_of_sample_results] if r > 0]) / len(self.out_of_sample_results) * 100,
            ],
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        
        # Format summary worksheet
        summary_sheet = writer.sheets["Summary"]
        summary_sheet.set_column("A:A", 30)
        summary_sheet.set_column("B:B", 15)
        
        # Create windows worksheet
        windows_data = []
        for i, result in enumerate(self.out_of_sample_results):
            window = result["window"]
            params = result["params"]
            results = result["results"]
            
            # Add window data
            window_data = {
                "Window": i + 1,
                "Train Start": window["train_start"],
                "Train End": window["train_end"],
                "Test Start": window["test_start"],
                "Test End": window["test_end"],
                "Return (%)": results["total_return_pct"],
                "Sharpe Ratio": results["sharpe_ratio"],
                "Max Drawdown (%)": results["max_drawdown"],
            }
            
            # Add parameters
            for param_name, param_value in params.items():
                window_data[f"Param: {param_name}"] = param_value
            
            windows_data.append(window_data)
        
        windows_df = pd.DataFrame(windows_data)
        windows_df.to_excel(writer, sheet_name="Windows", index=False)
        
        # Format windows worksheet
        windows_sheet = writer.sheets["Windows"]
        windows_sheet.set_column("A:Z", 15)
        
        # Save Excel file
        writer.close()
        
        return output_path


class MonteCarloSimulator:
    """Monte Carlo simulator for strategy robustness testing.
    
    This class implements Monte Carlo simulation for testing the robustness
    of trading strategies under different market conditions and parameter
    variations.
    """
    
    def __init__(
        self,
        backtest_runner: BacktestRunner,
        num_simulations: int = 100,
        output_dir: str = "reports",
    ):
        """Initialize the Monte Carlo simulator.
        
        Args:
            backtest_runner: Backtest runner instance
            num_simulations: Number of simulations to run (default: 100)
            output_dir: Output directory for reports (default: "reports")
        """
        self.backtest_runner = backtest_runner
        self.num_simulations = num_simulations
        self.output_dir = output_dir
        
        # Initialize results storage
        self.results = []
        
        logger.info(f"Initialized Monte Carlo simulator with {num_simulations} simulations")
    
    def run_price_simulations(self, volatility_factor: float = 0.1) -> List[Dict[str, Any]]:
        """Run Monte Carlo simulations with price variations.
        
        Args:
            volatility_factor: Factor to scale price volatility (default: 0.1)
            
        Returns:
            List of dictionaries with simulation results
        """
        import numpy as np
        
        # Get original data
        original_data = {}
        for symbol in self.backtest_runner.engine.data:
            original_data[symbol] = self.backtest_runner.engine.data[symbol].copy()
        
        # Run simulations
        self.results = []
        for i in range(self.num_simulations):
            logger.info(f"Running simulation {i + 1} of {self.num_simulations}")
            
            # Create modified data with price variations
            for symbol in original_data:
                # Get original prices
                data = original_data[symbol].copy()
                
                # Generate random walk
                returns = np.random.normal(0, volatility_factor, len(data))
                price_factors = np.cumprod(1 + returns)
                
                # Apply price factors to OHLC data
                if "open" in data.columns:
                    data["open"] = data["open"] * price_factors
                if "high" in data.columns:
                    data["high"] = data["high"] * price_factors
                if "low" in data.columns:
                    data["low"] = data["low"] * price_factors
                if "close" in data.columns:
                    data["close"] = data["close"] * price_factors
                
                # Update data in backtest engine
                self.backtest_runner.engine.data[symbol] = data
            
            # Run backtest with modified data
            self.backtest_runner.engine.reset()
            self.backtest_runner.run(verbose=False)
            
            # Store results
            self.results.append(self.backtest_runner.get_summary())
        
        # Restore original data
        for symbol in original_data:
            self.backtest_runner.engine.data[symbol] = original_data[symbol]
        
        return self.results
    
    def run_parameter_simulations(self, param_ranges: Dict[str, Tuple[float, float]]) -> List[Dict[str, Any]]:
        """Run Monte Carlo simulations with parameter variations.
        
        Args:
            param_ranges: Dictionary with parameter names as keys and (min, max) tuples as values
            
        Returns:
            List of dictionaries with simulation results
        """
        import numpy as np
        
        # Get original strategy parameters
        original_params = {}
        for param_name in param_ranges:
            if hasattr(self.backtest_runner.strategy, param_name):
                original_params[param_name] = getattr(self.backtest_runner.strategy, param_name)
        
        # Run simulations
        self.results = []
        for i in range(self.num_simulations):
            logger.info(f"Running simulation {i + 1} of {self.num_simulations}")
            
            # Generate random parameters
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                params[param_name] = min_val + np.random.random() * (max_val - min_val)
            
            # Update strategy parameters
            for param_name, param_value in params.items():
                setattr(self.backtest_runner.strategy, param_name, param_value)
            
            # Run backtest with modified parameters
            self.backtest_runner.engine.reset()
            self.backtest_runner.run(verbose=False)
            
            # Store results with parameters
            results = self.backtest_runner.get_summary()
            results["parameters"] = params
            self.results.append(results)
        
        # Restore original parameters
        for param_name, param_value in original_params.items():
            setattr(self.backtest_runner.strategy, param_name, param_value)
        
        return self.results
    
    def generate_report(self, report_format: Union[str, ReportFormat] = ReportFormat.HTML, filename: Optional[str] = None) -> str:
        """Generate a Monte Carlo simulation report.
        
        Args:
            report_format: Report format (default: HTML)
            filename: Output filename (default: auto-generated)
            
        Returns:
            Path to the generated report
        """
        if not self.results:
            raise ValueError("Monte Carlo simulation must be run before generating a report")
        
        # Create filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"montecarlo_{self.backtest_runner.strategy_name}_{timestamp}"
        
        # Create report based on format
        if isinstance(report_format, str):
            report_format = ReportFormat(report_format.lower())
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate report based on format
        if report_format == ReportFormat.HTML:
            report_path = self._generate_html_report(filename)
        elif report_format == ReportFormat.JSON:
            report_path = self._generate_json_report(filename)
        elif report_format == ReportFormat.EXCEL:
            report_path = self._generate_excel_report(filename)
        else:
            raise ValueError(f"Unsupported report format for Monte Carlo simulation: {report_format}")
        
        logger.info(f"Generated Monte Carlo simulation report: {report_path}")
        
        return report_path
    
    def _generate_html_report(self, filename: str) -> str:
        """Generate HTML report for Monte Carlo simulation.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the generated report
        """
        import base64
        from io import BytesIO
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.backtest_runner.strategy_name} - Monte Carlo Simulation</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                .metrics {{ display: flex; flex-wrap: wrap; margin-bottom: 20px; }}
                .metric {{ background-color: #f5f5f5; border-radius: 5px; padding: 15px; margin: 10px; flex: 1; min-width: 200px; }}
                .metric h3 {{ margin-top: 0; color: #333; }}
                .metric p {{ font-size: 24px; font-weight: bold; margin: 5px 0; }}
                .chart {{ margin: 20px 0; text-align: center; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>{self.backtest_runner.strategy_name} - Monte Carlo Simulation</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Number of simulations: {self.num_simulations}</p>
            
            <h2>Summary</h2>
            <div class="metrics">
        """
        
        # Calculate summary metrics
        total_returns = [result["total_return_pct"] for result in self.results]
        sharpe_ratios = [result["sharpe_ratio"] for result in self.results]
        max_drawdowns = [result["max_drawdown"] for result in self.results]
        
        # Add summary metrics
        metrics = [
            ("Mean Return", f"{np.mean(total_returns):.2f}%"),
            ("Median Return", f"{np.median(total_returns):.2f}%"),
            ("Std Dev Return", f"{np.std(total_returns):.2f}%"),
            ("Min Return", f"{np.min(total_returns):.2f}%"),
            ("Max Return", f"{np.max(total_returns):.2f}%"),
            ("Mean Sharpe", f"{np.mean(sharpe_ratios):.2f}"),
            ("Mean Max DD", f"{np.mean(max_drawdowns):.2f}%"),
            ("Win Rate", f"{len([r for r in total_returns if r > 0]) / len(total_returns) * 100:.2f}%"),
        ]
        
        for name, value in metrics:
            html_content += f"""
                <div class="metric">
                    <h3>{name}</h3>
                    <p>{value}</p>
                </div>
            """
        
        html_content += "</div>\n"
        
        # Create return distribution chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot return distribution
        ax.hist(total_returns, bins=20, alpha=0.7, color="blue")
        
        # Add labels and title
        ax.set_title("Return Distribution")
        ax.set_xlabel("Return (%)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at mean
        ax.axvline(x=np.mean(total_returns), color="red", linestyle="--", alpha=0.7, label=f"Mean: {np.mean(total_returns):.2f}%")
        
        # Add vertical line at 0
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        
        ax.legend()
        
        # Convert figure to base64 image
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        
        # Add chart to HTML
        html_content += f"""
        <h2>Return Distribution</h2>
        <div class="chart">
            <img src="data:image/png;base64,{img_str}" alt="Return Distribution" width="800">
        </div>
        """
        
        # Create Sharpe ratio distribution chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Sharpe ratio distribution
        ax.hist(sharpe_ratios, bins=20, alpha=0.7, color="green")
        
        # Add labels and title
        ax.set_title("Sharpe Ratio Distribution")
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at mean
        ax.axvline(x=np.mean(sharpe_ratios), color="red", linestyle="--", alpha=0.7, label=f"Mean: {np.mean(sharpe_ratios):.2f}")
        
        # Add vertical line at 0
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        
        ax.legend()
        
        # Convert figure to base64 image
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        
        # Add chart to HTML
        html_content += f"""
        <h2>Sharpe Ratio Distribution</h2>
        <div class="chart">
            <img src="data:image/png;base64,{img_str}" alt="Sharpe Ratio Distribution" width="800">
        </div>
        """
        
        # Create max drawdown distribution chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot max drawdown distribution
        ax.hist(max_drawdowns, bins=20, alpha=0.7, color="red")
        
        # Add labels and title
        ax.set_title("Max Drawdown Distribution")
        ax.set_xlabel("Max Drawdown (%)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at mean
        ax.axvline(x=np.mean(max_drawdowns), color="blue", linestyle="--", alpha=0.7, label=f"Mean: {np.mean(max_drawdowns):.2f}%")
        
        ax.legend()
        
        # Convert figure to base64 image
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        
        # Add chart to HTML
        html_content += f"""
        <h2>Max Drawdown Distribution</h2>
        <div class="chart">
            <img src="data:image/png;base64,{img_str}" alt="Max Drawdown Distribution" width="800">
        </div>
        """
        
        # Add simulation details table
        html_content += """
        <h2>Simulation Details</h2>
        <table>
            <tr>
                <th>Simulation</th>
                <th>Return (%)</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown (%)</th>
        """
        
        # Add parameter columns if available
        if "parameters" in self.results[0]:
            param_names = list(self.results[0]["parameters"].keys())
            for param_name in param_names:
                html_content += f"<th>{param_name}</th>\n"
        
        html_content += "</tr>\n"
        
        # Add simulation rows
        for i, result in enumerate(self.results):
            html_content += f"""
            <tr>
                <td>{i + 1}</td>
                <td>{result["total_return_pct"]:.2f}%</td>
                <td>{result["sharpe_ratio"]:.2f}</td>
                <td>{result["max_drawdown"]:.2f}%</td>
            """
            
            # Add parameter values if available
            if "parameters" in result:
                for param_name in param_names:
                    param_value = result["parameters"].get(param_name, "")
                    html_content += f"<td>{param_value:.4f}</td>\n"
            
            html_content += "</tr>\n"
        
        html_content += """
        </table>
        </body>
        </html>
        """
        
        # Save HTML to file
        if not filename.endswith(".html"):
            filename += ".html"
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w") as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_json_report(self, filename: str) -> str:
        """Generate JSON report for Monte Carlo simulation.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the generated report
        """
        import json
        import numpy as np
        
        # Create report data
        report_data = {
            "strategy": self.backtest_runner.strategy_name,
            "generated_at": datetime.now().isoformat(),
            "num_simulations": self.num_simulations,
            "simulations": self.results,
        }
        
        # Calculate summary metrics
        total_returns = [result["total_return_pct"] for result in self.results]
        sharpe_ratios = [result["sharpe_ratio"] for result in self.results]
        max_drawdowns = [result["max_drawdown"] for result in self.results]
        
        report_data["summary"] = {
            "mean_return": float(np.mean(total_returns)),
            "median_return": float(np.median(total_returns)),
            "std_dev_return": float(np.std(total_returns)),
            "min_return": float(np.min(total_returns)),
            "max_return": float(np.max(total_returns)),
            "mean_sharpe_ratio": float(np.mean(sharpe_ratios)),
            "mean_max_drawdown": float(np.mean(max_drawdowns)),
            "win_rate": float(len([r for r in total_returns if r > 0]) / len(total_returns)),
        }
        
        # Save to file
        if not filename.endswith(".json"):
            filename += ".json"
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=4, default=str)
        
        return output_path
    
    def _generate_excel_report(self, filename: str) -> str:
        """Generate Excel report for Monte Carlo simulation.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the generated report
        """
        try:
            import xlsxwriter
        except ImportError:
            logger.error("XlsxWriter is required for Excel generation. Install with 'pip install xlsxwriter'")
            return ""
        
        import numpy as np
        
        # Create Excel filename
        if not filename.endswith(".xlsx"):
            filename += ".xlsx"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Create Excel writer
        writer = pd.ExcelWriter(output_path, engine="xlsxwriter")
        
        # Create workbook and add worksheets
        workbook = writer.book
        
        # Calculate summary metrics
        total_returns = [result["total_return_pct"] for result in self.results]
        sharpe_ratios = [result["sharpe_ratio"] for result in self.results]
        max_drawdowns = [result["max_drawdown"] for result in self.results]
        
        # Create summary worksheet
        summary_data = {
            "Metric": [
                "Strategy",
                "Number of Simulations",
                "Mean Return (%)",
                "Median Return (%)",
                "Std Dev Return (%)",
                "Min Return (%)",
                "Max Return (%)",
                "Mean Sharpe Ratio",
                "Mean Max Drawdown (%)",
                "Win Rate (%)",
            ],
            "Value": [
                self.backtest_runner.strategy_name,
                self.num_simulations,
                np.mean(total_returns),
                np.median(total_returns),
                np.std(total_returns),
                np.min(total_returns),
                np.max(total_returns),
                np.mean(sharpe_ratios),
                np.mean(max_drawdowns),
                len([r for r in total_returns if r > 0]) / len(total_returns) * 100,
            ],
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        
        # Format summary worksheet
        summary_sheet = writer.sheets["Summary"]
        summary_sheet.set_column("A:A", 30)
        summary_sheet.set_column("B:B", 15)
        
        # Create simulations worksheet
        simulations_data = []
        for i, result in enumerate(self.results):
            # Create simulation data
            simulation_data = {
                "Simulation": i + 1,
                "Return (%)": result["total_return_pct"],
                "Sharpe Ratio": result["sharpe_ratio"],
                "Max Drawdown (%)": result["max_drawdown"],
            }
            
            # Add parameters if available
            if "parameters" in result:
                for param_name, param_value in result["parameters"].items():
                    simulation_data[f"Param: {param_name}"] = param_value
            
            simulations_data.append(simulation_data)
        
        simulations_df = pd.DataFrame(simulations_data)
        simulations_df.to_excel(writer, sheet_name="Simulations", index=False)
        
        # Format simulations worksheet
        simulations_sheet = writer.sheets["Simulations"]
        simulations_sheet.set_column("A:Z", 15)
        
        # Save Excel file
        writer.close()
        
        return output_path