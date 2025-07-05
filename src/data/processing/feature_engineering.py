"""Feature engineering module for the Friday AI Trading System.

This module provides the FeatureEngineer class and related components for
creating technical indicators and other features from market data.
"""

from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager
from src.data.processing.data_processor import DataProcessor, ProcessingStep, DataProcessingError

# Create logger
logger = get_logger(__name__)


class FeatureCategory(Enum):
    """Enum for feature categories."""

    PRICE = auto()  # Price-based features
    VOLUME = auto()  # Volume-based features
    VOLATILITY = auto()  # Volatility-based features
    TREND = auto()  # Trend-based features
    MOMENTUM = auto()  # Momentum-based features
    OSCILLATOR = auto()  # Oscillator-based features
    PATTERN = auto()  # Pattern-based features
    CUSTOM = auto()  # Custom features


class FeatureSet:
    """Class representing a set of features.

    Attributes:
        name: Name of the feature set.
        category: Category of the feature set.
        features: List of feature names in the set.
        dependencies: List of column names required to compute the features.
        description: Description of the feature set.
    """

    def __init__(
        self,
        name: str,
        category: FeatureCategory,
        features: List[str],
        dependencies: List[str],
        description: str = "",
    ):
        """Initialize a feature set.

        Args:
            name: Name of the feature set.
            category: Category of the feature set.
            features: List of feature names in the set.
            dependencies: List of column names required to compute the features.
            description: Description of the feature set.
        """
        self.name = name
        self.category = category
        self.features = features
        self.dependencies = dependencies
        self.description = description

    def __repr__(self) -> str:
        """Return a string representation of the feature set."""
        return f"FeatureSet(name={self.name}, category={self.category.name}, features={self.features})"


class FeatureEngineer(DataProcessor):
    """Class for creating technical indicators and other features from market data.

    This class provides methods for generating various technical indicators and
    other features commonly used in financial analysis and trading strategies.

    Attributes:
        config: Configuration manager.
        feature_sets: Dictionary of feature sets.
        enabled_features: Set of enabled feature names.
    """

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        enable_all_features: bool = False,
    ):
        """Initialize a feature engineer.

        Args:
            config: Configuration manager. If None, a new one will be created.
            enable_all_features: Whether to enable all features by default.
        """
        super().__init__(config)
        self.feature_sets: Dict[str, FeatureSet] = {}
        self.enabled_features: Set[str] = set()

        # Register default processing step
        self.add_processing_step(ProcessingStep.FEATURE_ENGINEERING, self.generate_features)

        # Register standard feature sets
        self._register_standard_feature_sets()

        # Load configuration for default enabled feature sets
        self._load_default_configuration()

        # Enable all features if requested (overrides config)
        if enable_all_features:
            self.enable_all_features()
        
        # Log initialization
        enabled_sets = self.get_enabled_feature_sets()
        logger.info(f"FeatureEngineer initialized with {len(enabled_sets)} enabled feature sets: {enabled_sets}")

    def _register_standard_feature_sets(self) -> None:
        """Register standard feature sets."""
        # Price-based features
        self.register_feature_set(
            FeatureSet(
                name="price_derived",
                category=FeatureCategory.PRICE,
                features=[
                    "typical_price",
                    "price_avg",
                    "price_log_return",
                    "price_pct_change",
                ],
                dependencies=["open", "high", "low", "close"],
                description="Basic price-derived features",
            )
        )

        # Moving averages
        self.register_feature_set(
            FeatureSet(
                name="moving_averages",
                category=FeatureCategory.TREND,
                features=[
                    "sma_5",
                    "sma_10",
                    "sma_20",
                    "sma_50",
                    "sma_200",
                    "ema_5",
                    "ema_10",
                    "ema_20",
                    "ema_50",
                    "ema_200",
                ],
                dependencies=["close"],
                description="Simple and exponential moving averages",
            )
        )

        # Volatility indicators
        self.register_feature_set(
            FeatureSet(
                name="volatility",
                category=FeatureCategory.VOLATILITY,
                features=[
                    "atr_14",
                    "bollinger_upper",
                    "bollinger_middle",
                    "bollinger_lower",
                    "bollinger_width",
                    "keltner_upper",
                    "keltner_middle",
                    "keltner_lower",
                ],
                dependencies=["open", "high", "low", "close"],
                description="Volatility-based indicators",
            )
        )

        # Momentum indicators
        self.register_feature_set(
            FeatureSet(
                name="momentum",
                category=FeatureCategory.MOMENTUM,
                features=[
                    "rsi_14",
                    "stoch_k_14",
                    "stoch_d_14",
                    "macd_line",
                    "macd_signal",
                    "macd_histogram",
                    "roc_10",
                ],
                dependencies=["close", "high", "low"],
                description="Momentum-based indicators",
            )
        )

        # Volume indicators
        self.register_feature_set(
            FeatureSet(
                name="volume",
                category=FeatureCategory.VOLUME,
                features=[
                    "volume_sma_5",
                    "volume_sma_10",
                    "volume_sma_20",
                    "volume_ratio",
                    "obv",
                    "vwap",
                ],
                dependencies=["close", "volume"],
                description="Volume-based indicators",
            )
        )

        # Trend indicators
        self.register_feature_set(
            FeatureSet(
                name="trend",
                category=FeatureCategory.TREND,
                features=[
                    "adx_14",
                    "di_plus_14",
                    "di_minus_14",
                    "aroon_up_14",
                    "aroon_down_14",
                    "aroon_oscillator_14",
                    "cci_20",
                ],
                dependencies=["high", "low", "close"],
                description="Trend-based indicators",
            )
        )

    def register_feature_set(self, feature_set: FeatureSet) -> None:
        """Register a feature set.

        Args:
            feature_set: The feature set to register.
        """
        self.feature_sets[feature_set.name] = feature_set
        logger.info(f"Registered feature set: {feature_set.name}")

    def enable_feature_set(self, feature_set_name: str) -> None:
        """Enable a feature set.

        Args:
            feature_set_name: Name of the feature set to enable.

        Raises:
            ValueError: If the feature set is not registered.
        """
        if feature_set_name not in self.feature_sets:
            raise ValueError(f"Feature set '{feature_set_name}' is not registered")

        feature_set = self.feature_sets[feature_set_name]
        self.enabled_features.update(feature_set.features)
        logger.info(f"Enabled feature set: {feature_set_name}")

    def disable_feature_set(self, feature_set_name: str) -> None:
        """Disable a feature set.

        Args:
            feature_set_name: Name of the feature set to disable.

        Raises:
            ValueError: If the feature set is not registered.
        """
        if feature_set_name not in self.feature_sets:
            raise ValueError(f"Feature set '{feature_set_name}' is not registered")

        feature_set = self.feature_sets[feature_set_name]
        self.enabled_features.difference_update(feature_set.features)
        logger.info(f"Disabled feature set: {feature_set_name}")

    def enable_all_features(self) -> None:
        """Enable all registered feature sets."""
        for feature_set_name in self.feature_sets:
            self.enable_feature_set(feature_set_name)

    def disable_all_features(self) -> None:
        """Disable all feature sets."""
        self.enabled_features.clear()
        logger.info("Disabled all features")

    def get_enabled_feature_sets(self) -> List[str]:
        """Get the names of all enabled feature sets.

        Returns:
            List[str]: List of enabled feature set names.
        """
        enabled_sets = []
        for name, feature_set in self.feature_sets.items():
            if all(feature in self.enabled_features for feature in feature_set.features):
                enabled_sets.append(name)
        return enabled_sets
    
    def save_configuration(self) -> None:
        """Save current feature configuration to config manager."""
        enabled_sets = self.get_enabled_feature_sets()
        self.config.set("features.enabled_feature_sets", enabled_sets)
        logger.info(f"Saved feature configuration: enabled sets = {enabled_sets}")
    
    def get_feature_set_info(self, feature_set_name: str) -> Dict[str, Any]:
        """Get detailed information about a feature set.
        
        Args:
            feature_set_name: Name of the feature set.
            
        Returns:
            Dict[str, Any]: Information about the feature set.
            
        Raises:
            ValueError: If the feature set is not registered.
        """
        if feature_set_name not in self.feature_sets:
            raise ValueError(f"Feature set '{feature_set_name}' is not registered")
        
        feature_set = self.feature_sets[feature_set_name]
        enabled = all(feature in self.enabled_features for feature in feature_set.features)
        
        return {
            "name": feature_set.name,
            "category": feature_set.category.name,
            "description": feature_set.description,
            "features": feature_set.features,
            "dependencies": feature_set.dependencies,
            "enabled": enabled,
            "feature_count": len(feature_set.features),
            "dependency_count": len(feature_set.dependencies)
        }
    
    def get_all_feature_sets_info(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all feature sets.
        
        Returns:
            Dict[str, Dict[str, Any]]: Information about all feature sets.
        """
        return {name: self.get_feature_set_info(name) for name in self.feature_sets}
    
    def validate_data_columns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate that data contains required columns for enabled features.
        
        Args:
            data: The input data to validate.
            
        Returns:
            Dict[str, Any]: Validation results.
        """
        required_columns = self.get_required_columns()
        available_columns = set(data.columns)
        
        missing_columns = required_columns - available_columns
        extra_columns = available_columns - required_columns
        
        validation_result = {
            "valid": len(missing_columns) == 0,
            "required_columns": sorted(required_columns),
            "available_columns": sorted(available_columns),
            "missing_columns": sorted(missing_columns),
            "extra_columns": sorted(extra_columns),
            "enabled_feature_sets": self.get_enabled_feature_sets(),
            "total_features": len(self.enabled_features)
        }
        
        return validation_result

    def get_required_columns(self) -> Set[str]:
        """Get the set of columns required for enabled features.

        Returns:
            Set[str]: Set of required column names.
        """
        required_columns = set()
        for name, feature_set in self.feature_sets.items():
            if any(feature in self.enabled_features for feature in feature_set.features):
                required_columns.update(feature_set.dependencies)
        return required_columns

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate all enabled features.

        Args:
            data: The input data.

        Returns:
            pd.DataFrame: The data with added features.

        Raises:
            DataProcessingError: If an error occurs during feature generation.
        """
        try:
            # Make a copy of the input data
            result = data.copy()

            # Check if any features are enabled
            if not self.enabled_features:
                logger.warning("No features are enabled")
                return result

            # Check if required columns are present
            required_columns = self.get_required_columns()
            missing_columns = required_columns - set(result.columns)
            if missing_columns:
                raise DataProcessingError(
                    f"Missing required columns for feature generation: {', '.join(missing_columns)}"
                )

            # Generate price-derived features
            if any(f in self.enabled_features for f in self.feature_sets.get("price_derived", FeatureSet("", FeatureCategory.CUSTOM, [], [])).features):
                result = self._generate_price_derived_features(result)

            # Generate moving averages
            if any(f in self.enabled_features for f in self.feature_sets.get("moving_averages", FeatureSet("", FeatureCategory.CUSTOM, [], [])).features):
                result = self._generate_moving_averages(result)

            # Generate volatility indicators
            if any(f in self.enabled_features for f in self.feature_sets.get("volatility", FeatureSet("", FeatureCategory.CUSTOM, [], [])).features):
                result = self._generate_volatility_indicators(result)

            # Generate momentum indicators
            if any(f in self.enabled_features for f in self.feature_sets.get("momentum", FeatureSet("", FeatureCategory.CUSTOM, [], [])).features):
                result = self._generate_momentum_indicators(result)

            # Generate volume indicators
            if any(f in self.enabled_features for f in self.feature_sets.get("volume", FeatureSet("", FeatureCategory.CUSTOM, [], [])).features):
                result = self._generate_volume_indicators(result)

            # Generate trend indicators
            if any(f in self.enabled_features for f in self.feature_sets.get("trend", FeatureSet("", FeatureCategory.CUSTOM, [], [])).features):
                result = self._generate_trend_indicators(result)

            # Record feature generation in metadata
            self.metadata["generated_features"] = list(self.enabled_features)
            self.metadata["feature_count"] = len(self.enabled_features)

            return result

        except Exception as e:
            error_msg = f"Error generating features: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)

    def _generate_price_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate price-derived features.

        Args:
            data: The input data.

        Returns:
            pd.DataFrame: The data with added features.
        """
        result = data.copy()

        # Typical price: (high + low + close) / 3
        if "typical_price" in self.enabled_features:
            result["typical_price"] = (result["high"] + result["low"] + result["close"]) / 3

        # Average price: (open + high + low + close) / 4
        if "price_avg" in self.enabled_features:
            result["price_avg"] = (result["open"] + result["high"] + result["low"] + result["close"]) / 4

        # Log returns: ln(close_t / close_t-1)
        if "price_log_return" in self.enabled_features:
            result["price_log_return"] = np.log(result["close"] / result["close"].shift(1))

        # Percentage change: (close_t - close_t-1) / close_t-1
        if "price_pct_change" in self.enabled_features:
            result["price_pct_change"] = result["close"].pct_change()

        return result

    def _generate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate moving average features.

        Args:
            data: The input data.

        Returns:
            pd.DataFrame: The data with added features.
        """
        result = data.copy()

        # Simple Moving Averages (SMA)
        for period in [5, 10, 20, 50, 200]:
            feature_name = f"sma_{period}"
            if feature_name in self.enabled_features:
                result[feature_name] = result["close"].rolling(window=period).mean()

        # Exponential Moving Averages (EMA)
        for period in [5, 10, 20, 50, 200]:
            feature_name = f"ema_{period}"
            if feature_name in self.enabled_features:
                result[feature_name] = result["close"].ewm(span=period, adjust=False).mean()

        return result

    def _generate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility indicator features.

        Args:
            data: The input data.

        Returns:
            pd.DataFrame: The data with added features.
        """
        result = data.copy()

        # Average True Range (ATR)
        if "atr_14" in self.enabled_features:
            high_low = result["high"] - result["low"]
            high_close = np.abs(result["high"] - result["close"].shift(1))
            low_close = np.abs(result["low"] - result["close"].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            result["atr_14"] = true_range.rolling(window=14).mean()

        # Bollinger Bands
        if any(f in self.enabled_features for f in ["bollinger_upper", "bollinger_middle", "bollinger_lower", "bollinger_width"]):
            # Middle band is 20-day SMA
            middle = result["close"].rolling(window=20).mean()
            # Standard deviation of price
            std = result["close"].rolling(window=20).std()
            # Upper band is middle + 2*std
            upper = middle + 2 * std
            # Lower band is middle - 2*std
            lower = middle - 2 * std

            if "bollinger_middle" in self.enabled_features:
                result["bollinger_middle"] = middle
            if "bollinger_upper" in self.enabled_features:
                result["bollinger_upper"] = upper
            if "bollinger_lower" in self.enabled_features:
                result["bollinger_lower"] = lower
            if "bollinger_width" in self.enabled_features:
                result["bollinger_width"] = (upper - lower) / middle

        # Keltner Channels
        if any(f in self.enabled_features for f in ["keltner_upper", "keltner_middle", "keltner_lower"]):
            # Middle line is 20-day EMA
            middle = result["close"].ewm(span=20, adjust=False).mean()

            # ATR calculation for Keltner
            high_low = result["high"] - result["low"]
            high_close = np.abs(result["high"] - result["close"].shift(1))
            low_close = np.abs(result["low"] - result["close"].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()

            # Upper line is middle + 2*ATR
            upper = middle + 2 * atr
            # Lower line is middle - 2*ATR
            lower = middle - 2 * atr

            if "keltner_middle" in self.enabled_features:
                result["keltner_middle"] = middle
            if "keltner_upper" in self.enabled_features:
                result["keltner_upper"] = upper
            if "keltner_lower" in self.enabled_features:
                result["keltner_lower"] = lower

        return result

    def _generate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum indicator features.

        Args:
            data: The input data.

        Returns:
            pd.DataFrame: The data with added features.
        """
        result = data.copy()

        # Relative Strength Index (RSI)
        if "rsi_14" in self.enabled_features:
            delta = result["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            result["rsi_14"] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        if "stoch_k_14" in self.enabled_features or "stoch_d_14" in self.enabled_features:
            low_14 = result["low"].rolling(window=14).min()
            high_14 = result["high"].rolling(window=14).max()
            k = 100 * ((result["close"] - low_14) / (high_14 - low_14))

            if "stoch_k_14" in self.enabled_features:
                result["stoch_k_14"] = k
            if "stoch_d_14" in self.enabled_features:
                result["stoch_d_14"] = k.rolling(window=3).mean()

        # MACD (Moving Average Convergence Divergence)
        if any(f in self.enabled_features for f in ["macd_line", "macd_signal", "macd_histogram"]):
            ema_12 = result["close"].ewm(span=12, adjust=False).mean()
            ema_26 = result["close"].ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line

            if "macd_line" in self.enabled_features:
                result["macd_line"] = macd_line
            if "macd_signal" in self.enabled_features:
                result["macd_signal"] = signal_line
            if "macd_histogram" in self.enabled_features:
                result["macd_histogram"] = histogram

        # Rate of Change (ROC)
        if "roc_10" in self.enabled_features:
            result["roc_10"] = (result["close"] - result["close"].shift(10)) / result["close"].shift(10) * 100

        return result

    def _generate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume indicator features.

        Args:
            data: The input data.

        Returns:
            pd.DataFrame: The data with added features.
        """
        result = data.copy()

        # Volume Moving Averages
        for period in [5, 10, 20]:
            feature_name = f"volume_sma_{period}"
            if feature_name in self.enabled_features:
                result[feature_name] = result["volume"].rolling(window=period).mean()

        # Volume Ratio (current volume / average volume)
        if "volume_ratio" in self.enabled_features:
            avg_volume = result["volume"].rolling(window=20).mean()
            result["volume_ratio"] = result["volume"] / avg_volume

        # On-Balance Volume (OBV)
        if "obv" in self.enabled_features:
            obv = pd.Series(index=result.index, dtype=float)
            obv.iloc[0] = 0

            for i in range(1, len(result)):
                if result["close"].iloc[i] > result["close"].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + result["volume"].iloc[i]
                elif result["close"].iloc[i] < result["close"].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - result["volume"].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]

            result["obv"] = obv

        # Volume-Weighted Average Price (VWAP)
        if "vwap" in self.enabled_features:
            # Check if we have a datetime index to reset each day
            if isinstance(result.index, pd.DatetimeIndex):
                # Group by date
                result["date"] = result.index.date
                groups = result.groupby("date")

                # Calculate VWAP for each day
                vwap = pd.Series(index=result.index)
                for date, group in groups:
                    typical_price = (group["high"] + group["low"] + group["close"]) / 3
                    vwap_group = (typical_price * group["volume"]).cumsum() / group["volume"].cumsum()
                    vwap.loc[group.index] = vwap_group

                result["vwap"] = vwap
                result.drop("date", axis=1, inplace=True)
            else:
                # If no datetime index, calculate VWAP for the entire dataset
                typical_price = (result["high"] + result["low"] + result["close"]) / 3
                result["vwap"] = (typical_price * result["volume"]).cumsum() / result["volume"].cumsum()

        return result

    def _generate_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trend indicator features.

        Args:
            data: The input data.

        Returns:
            pd.DataFrame: The data with added features.
        """
        result = data.copy()

        # Average Directional Index (ADX)
        if any(f in self.enabled_features for f in ["adx_14", "di_plus_14", "di_minus_14"]):
            # True Range
            high_low = result["high"] - result["low"]
            high_close = np.abs(result["high"] - result["close"].shift(1))
            low_close = np.abs(result["low"] - result["close"].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = true_range.rolling(window=14).mean()

            # Plus Directional Movement (+DM)
            plus_dm = result["high"].diff()
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > -result["low"].diff()), 0)

            # Minus Directional Movement (-DM)
            minus_dm = -result["low"].diff()
            minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > result["high"].diff()), 0)

            # Smoothed +DM and -DM
            plus_dm_14 = plus_dm.rolling(window=14).sum()
            minus_dm_14 = minus_dm.rolling(window=14).sum()

            # Directional Indicators
            di_plus_14 = 100 * plus_dm_14 / atr_14
            di_minus_14 = 100 * minus_dm_14 / atr_14

            # Directional Index
            dx = 100 * np.abs(di_plus_14 - di_minus_14) / (di_plus_14 + di_minus_14)

            # Average Directional Index
            adx_14 = dx.rolling(window=14).mean()

            if "di_plus_14" in self.enabled_features:
                result["di_plus_14"] = di_plus_14
            if "di_minus_14" in self.enabled_features:
                result["di_minus_14"] = di_minus_14
            if "adx_14" in self.enabled_features:
                result["adx_14"] = adx_14

        # Aroon Indicator
        if any(f in self.enabled_features for f in ["aroon_up_14", "aroon_down_14", "aroon_oscillator_14"]):
            # Aroon Up: ((14 - periods since 14-period high) / 14) * 100
            high_idx = result["high"].rolling(window=14).apply(lambda x: x.argmax(), raw=True)
            aroon_up = ((14 - high_idx) / 14) * 100

            # Aroon Down: ((14 - periods since 14-period low) / 14) * 100
            low_idx = result["low"].rolling(window=14).apply(lambda x: x.argmin(), raw=True)
            aroon_down = ((14 - low_idx) / 14) * 100

            # Aroon Oscillator: Aroon Up - Aroon Down
            aroon_osc = aroon_up - aroon_down

            if "aroon_up_14" in self.enabled_features:
                result["aroon_up_14"] = aroon_up
            if "aroon_down_14" in self.enabled_features:
                result["aroon_down_14"] = aroon_down
            if "aroon_oscillator_14" in self.enabled_features:
                result["aroon_oscillator_14"] = aroon_osc

        # Commodity Channel Index (CCI)
        if "cci_20" in self.enabled_features:
            typical_price = (result["high"] + result["low"] + result["close"]) / 3
            mean_dev = pd.Series(index=result.index)

            # Calculate mean deviation
            for i in range(20, len(typical_price)):
                mean_dev.iloc[i] = np.mean(np.abs(typical_price.iloc[i-20:i] - typical_price.iloc[i-20:i].mean()))

            # Calculate CCI
            sma_tp = typical_price.rolling(window=20).mean()
            result["cci_20"] = (typical_price - sma_tp) / (0.015 * mean_dev)

        return result
    
    def _load_default_configuration(self) -> None:
        """Load default feature configuration from config manager."""
        try:
            # Get default enabled feature sets from configuration
            default_feature_sets = self.config.get("features.default_enabled", [
                "price_derived", "moving_averages", "volatility", "momentum", "volume", "trend"
            ])
            
            # Enable default feature sets
            for feature_set_name in default_feature_sets:
                try:
                    self.enable_feature_set(feature_set_name)
                    logger.debug(f"Enabled default feature set: {feature_set_name}")
                except ValueError as e:
                    logger.warning(f"Failed to enable default feature set {feature_set_name}: {e}")
            
            # Check for user-configured enabled sets
            user_enabled_sets = self.config.get("features.enabled_feature_sets", [])
            if user_enabled_sets:
                # Disable all and enable only user-specified sets
                self.disable_all_features()
                for feature_set_name in user_enabled_sets:
                    try:
                        self.enable_feature_set(feature_set_name)
                        logger.debug(f"Enabled user-configured feature set: {feature_set_name}")
                    except ValueError as e:
                        logger.warning(f"Failed to enable user-configured feature set {feature_set_name}: {e}")
            
            logger.info(f"Loaded feature configuration from config manager")
            
        except Exception as e:
            logger.warning(f"Failed to load feature configuration: {e}. Using default settings.")
            # Enable default feature sets as fallback
            default_sets = ["price_derived", "moving_averages", "volatility", "momentum", "volume", "trend"]
            for feature_set_name in default_sets:
                try:
                    self.enable_feature_set(feature_set_name)
                except ValueError:
                    pass
