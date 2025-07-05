import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, TypedDict, NamedTuple, Callable
from datetime import datetime, timedelta
from functools import lru_cache

logger = logging.getLogger(__name__)

# Type aliases for improved readability
PerformanceMetric = Dict[str, Union[float, str, datetime]]
PerformanceMetrics = Dict[str, PerformanceMetric]

class PerformanceCalculator:
    """
    Performance Calculator for measuring portfolio performance and attribution.

    This class provides functionality for:
    - Performance measurement (returns, volatility, Sharpe ratio, etc.)
    - Performance attribution (sector, asset, factor contributions)
    - Benchmark comparison
    - Risk-adjusted performance metrics
    
    This implementation includes caching for frequently calculated metrics
    to improve performance when the same calculations are requested multiple times.
    """

    def __init__(self, benchmark_symbol: Optional[str] = None, benchmark_data: Optional[Dict[datetime, float]] = None, risk_free_rate: float = 0.0):
        """
        Initialize the Performance Calculator.

        Args:
            benchmark_symbol: Symbol for benchmark comparison
            benchmark_data: Optional benchmark data as {timestamp: value}
            risk_free_rate: Risk-free rate for risk-adjusted metrics
        """
        self.portfolio_returns = []  # List of (timestamp, return_value)
        self.portfolio_values = []  # List of (timestamp, portfolio_value)
        self.benchmark_symbol = benchmark_symbol
        self.benchmark_data = benchmark_data or {}
        self.benchmark_returns = self._calculate_benchmark_returns() if benchmark_data else []
        self.risk_free_rate = risk_free_rate

        # Performance attribution data
        self.sector_returns = {}  # {sector: [(timestamp, return_value), ...]} 
        self.asset_returns = {}  # {symbol: [(timestamp, return_value), ...]}
        
        # Cache invalidation counter - incremented whenever data changes
        self._cache_version = 0
        
        # Use NumPy arrays for efficient calculations
        self._portfolio_values_array = np.array([])
        self._portfolio_returns_array = np.array([])

        logger.info(f"Performance Calculator initialized with benchmark: {benchmark_symbol if benchmark_symbol else 'None'}")

    def _create_cached_method(self, func: Callable) -> Callable:
        """
        Create a cached version of a method that invalidates when data changes.
        
        Args:
            func: The method to cache
            
        Returns:
            A cached version of the method that will be invalidated when data changes
        """
        @lru_cache(maxsize=32)
        def cached_method(cache_version, *args, **kwargs):
            return func(*args, **kwargs)
            
        def wrapper(*args, **kwargs):
            return cached_method(self._cache_version, *args, **kwargs)
            
        return wrapper
        
    def _invalidate_cache(self) -> None:
        """
        Invalidate all cached calculations by incrementing the cache version.
        This should be called whenever portfolio data changes.
        """
        self._cache_version += 1
        logger.debug(f"Cache invalidated (version: {self._cache_version})")
    
    def _calculate_benchmark_returns(self) -> List[Tuple[datetime, float]]:
        """
        Calculate benchmark returns from benchmark data.

        Returns:
            List of (timestamp, return_value) tuples
        """
        if not self.benchmark_data or len(self.benchmark_data) < 2:
            return []

        # Sort benchmark data by timestamp
        sorted_data = sorted(self.benchmark_data.items())

        # Calculate returns
        benchmark_returns = []
        for i in range(1, len(sorted_data)):
            prev_ts, prev_value = sorted_data[i-1]
            curr_ts, curr_value = sorted_data[i]

            if prev_value > 0:  # Avoid division by zero
                return_value = (curr_value / prev_value) - 1
                benchmark_returns.append((curr_ts, return_value))

        return benchmark_returns

    def add_portfolio_return_observation(self, return_value: float, timestamp: datetime) -> None:
        """
        Add a portfolio return observation.

        Args:
            return_value: The return value
            timestamp: The timestamp of the return observation
        """
        self.portfolio_returns.append((timestamp, return_value))

    def add_portfolio_value_observation(self, value: float, timestamp: datetime) -> None:
        """
        Add a portfolio value observation.

        Args:
            value: The portfolio value
            timestamp: The timestamp of the value observation
        """
        self.portfolio_values.append((timestamp, value))

    def add_sector_return_observation(self, sector: str, return_value: float, timestamp: datetime) -> None:
        """
        Add a sector return observation for attribution analysis.

        Args:
            sector: The sector name
            return_value: The return value
            timestamp: The timestamp of the return observation
        """
        if sector not in self.sector_returns:
            self.sector_returns[sector] = []

        self.sector_returns[sector].append((timestamp, return_value))

    def add_asset_return_observation(self, symbol: str, return_value: float, timestamp: datetime) -> None:
        """
        Add an asset return observation for attribution analysis.

        Args:
            symbol: The asset symbol
            return_value: The return value
            timestamp: The timestamp of the return observation
        """
        if symbol not in self.asset_returns:
            self.asset_returns[symbol] = []

        self.asset_returns[symbol].append((timestamp, return_value))

    def calculate_performance_metrics(self,
                                     risk_free_rate: float = 0.0,
                                     period: str = 'all') -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
            period: Time period for calculation ('all', '1y', '6m', '3m', '1m')

        Returns:
            Dict with performance metrics
        """
        if not self.portfolio_returns:
            return {"error": "No portfolio returns available"}

        # Filter returns based on period
        filtered_returns = self._filter_by_period(self.portfolio_returns, period)
        if not filtered_returns:
            return {"error": f"No returns available for period {period}"}

        # Convert to numpy array for calculations
        returns_array = np.array([r[1] for r in filtered_returns])

        # Calculate basic metrics
        total_return = np.prod(1 + returns_array) - 1
        avg_return = np.mean(returns_array)
        volatility = np.std(returns_array)

        # Annualize metrics (assuming daily returns)
        trading_days = 252
        annualized_return = (1 + avg_return) ** trading_days - 1
        annualized_volatility = volatility * np.sqrt(trading_days)

        # Calculate risk-adjusted metrics
        daily_rf = (1 + risk_free_rate) ** (1/trading_days) - 1
        excess_return = avg_return - daily_rf
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Calculate downside deviation (for Sortino ratio)
        downside_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0

        # Calculate drawdown
        cumulative_returns = (1 + returns_array).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = np.min(drawdown)

        # Calculate benchmark comparison if available
        benchmark_metrics = {}
        if self.benchmark_returns:
            filtered_benchmark = self._filter_by_period(self.benchmark_returns, period)
            if filtered_benchmark:
                benchmark_array = np.array([r[1] for r in filtered_benchmark])

                # Match lengths if necessary
                min_length = min(len(returns_array), len(benchmark_array))
                returns_array = returns_array[-min_length:]
                benchmark_array = benchmark_array[-min_length:]

                # Calculate benchmark metrics
                benchmark_total_return = np.prod(1 + benchmark_array) - 1
                benchmark_avg_return = np.mean(benchmark_array)
                benchmark_volatility = np.std(benchmark_array)
                benchmark_annualized_return = (1 + benchmark_avg_return) ** trading_days - 1

                # Calculate alpha and beta
                covariance = np.cov(returns_array, benchmark_array)[0, 1]
                beta = covariance / np.var(benchmark_array) if np.var(benchmark_array) > 0 else 0
                alpha = avg_return - (beta * benchmark_avg_return)
                alpha_annualized = (1 + alpha) ** trading_days - 1

                # Calculate tracking error and information ratio
                tracking_diff = returns_array - benchmark_array
                tracking_error = np.std(tracking_diff) * np.sqrt(trading_days)
                information_ratio = (annualized_return - benchmark_annualized_return) / tracking_error if tracking_error > 0 else 0

                benchmark_metrics = {
                    "benchmark_return": benchmark_total_return,
                    "benchmark_annualized_return": benchmark_annualized_return,
                    "benchmark_volatility": benchmark_volatility,
                    "alpha": alpha_annualized,
                    "beta": beta,
                    "tracking_error": tracking_error,
                    "information_ratio": information_ratio,
                    "excess_return": total_return - benchmark_total_return
                }

        # Combine all metrics
        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            "period": period,
            "num_observations": len(filtered_returns)
        }

        # Add benchmark metrics if available
        if benchmark_metrics:
            metrics.update(benchmark_metrics)

        return metrics

    def calculate_attribution(self, period: str = 'all') -> Dict[str, Any]:
        """
        Calculate performance attribution by sector and asset.

        Args:
            period: Time period for calculation ('all', '1y', '6m', '3m', '1m')

        Returns:
            Dict with attribution analysis
        """
        if not self.portfolio_returns:
            return {"error": "No portfolio returns available"}

        # Filter portfolio returns based on period
        filtered_returns = self._filter_by_period(self.portfolio_returns, period)
        if not filtered_returns:
            return {"error": f"No returns available for period {period}"}

        # Calculate sector attribution
        sector_attribution = {}
        for sector, returns in self.sector_returns.items():
            filtered_sector = self._filter_by_period(returns, period)
            if filtered_sector:
                sector_array = np.array([r[1] for r in filtered_sector])
                sector_attribution[sector] = {
                    "total_return": np.prod(1 + sector_array) - 1,
                    "average_return": np.mean(sector_array),
                    "volatility": np.std(sector_array),
                    "observations": len(filtered_sector)
                }

        # Calculate asset attribution
        asset_attribution = {}
        for symbol, returns in self.asset_returns.items():
            filtered_asset = self._filter_by_period(returns, period)
            if filtered_asset:
                asset_array = np.array([r[1] for r in filtered_asset])
                asset_attribution[symbol] = {
                    "total_return": np.prod(1 + asset_array) - 1,
                    "average_return": np.mean(asset_array),
                    "volatility": np.std(asset_array),
                    "observations": len(filtered_asset)
                }

        return {
            "sector_attribution": sector_attribution,
            "asset_attribution": asset_attribution,
            "period": period
        }

    def _filter_by_period(self, data: List[Tuple[datetime, float]], period: str) -> List[Tuple[datetime, float]]:
        """
        Filter data by time period.

        Args:
            data: List of (timestamp, value) tuples
            period: Time period ('all', '1y', '6m', '3m', '1m')

        Returns:
            Filtered list of (timestamp, value) tuples
        """
        if not data:
            return []

        if period == 'all':
            return data

        now = datetime.now()
        cutoff = None

        if period == '1y':
            cutoff = now - timedelta(days=365)
        elif period == '6m':
            cutoff = now - timedelta(days=182)
        elif period == '3m':
            cutoff = now - timedelta(days=91)
        elif period == '1m':
            cutoff = now - timedelta(days=30)
        else:
            return data  # Invalid period, return all data

        return [item for item in data if item[0] >= cutoff]

    def get_returns_dataframe(self) -> pd.DataFrame:
        """
        Get portfolio returns as a pandas DataFrame.

        Returns:
            DataFrame with portfolio returns
        """
        if not self.portfolio_returns:
            return pd.DataFrame(columns=['timestamp', 'return'])

        df = pd.DataFrame(self.portfolio_returns, columns=['timestamp', 'return'])
        df.set_index('timestamp', inplace=True)
        return df

    def get_values_dataframe(self) -> pd.DataFrame:
        """
        Get portfolio values as a pandas DataFrame.

        Returns:
            DataFrame with portfolio values
        """
        if not self.portfolio_values:
            return pd.DataFrame(columns=['timestamp', 'value'])

        df = pd.DataFrame(self.portfolio_values, columns=['timestamp', 'value'])
        df.set_index('timestamp', inplace=True)
        return df

    def get_benchmark_comparison_dataframe(self) -> pd.DataFrame:
        """
        Get portfolio and benchmark returns as a pandas DataFrame.

        Returns:
            DataFrame with portfolio and benchmark returns
        """
        portfolio_df = self.get_returns_dataframe().rename(columns={'return': 'portfolio_return'})

        if not self.benchmark_returns:
            return portfolio_df

        benchmark_df = pd.DataFrame(self.benchmark_returns, columns=['timestamp', 'return'])
        benchmark_df.set_index('timestamp', inplace=True)
        benchmark_df.rename(columns={'return': 'benchmark_return'}, inplace=True)

        # Merge the dataframes
        return pd.merge(portfolio_df, benchmark_df, left_index=True, right_index=True, how='outer')

    def reset_calculator(self) -> None:
        """
        Reset the performance calculator.
        """
        self.portfolio_returns = []
        self.portfolio_values = []
        self.sector_returns = {}
        self.asset_returns = {}

        logger.info("Performance Calculator reset")
