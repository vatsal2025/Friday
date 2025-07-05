"""Multi-timeframe processor module for the Friday AI Trading System.

This module provides classes for handling data across different timeframes,
including conversion between timeframes and alignment of data from multiple timeframes.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager
from src.data.processing.data_processor import DataProcessor, ProcessingStep, DataProcessingError
from src.data.acquisition.data_fetcher import DataTimeframe

# Create logger
logger = get_logger(__name__)


class TimeframeConverter:
    """Class for converting data between different timeframes.

    This class provides methods for resampling data to higher timeframes (aggregation)
    and for expanding data to lower timeframes (expansion).
    """

    @staticmethod
    def resample_to_higher_timeframe(
        data: pd.DataFrame,
        source_timeframe: Union[DataTimeframe, str],
        target_timeframe: Union[DataTimeframe, str],
    ) -> pd.DataFrame:
        """Resample data to a higher timeframe.

        Args:
            data: The input data with a datetime index.
            source_timeframe: The source timeframe.
            target_timeframe: The target timeframe (must be higher than source).

        Returns:
            pd.DataFrame: The resampled data.

        Raises:
            ValueError: If the target timeframe is not higher than the source timeframe.
        """
        # Convert enum to string if needed
        if isinstance(source_timeframe, DataTimeframe):
            source_timeframe = source_timeframe.value
        if isinstance(target_timeframe, DataTimeframe):
            target_timeframe = target_timeframe.value

        # Validate timeframes
        if not TimeframeConverter._is_higher_timeframe(target_timeframe, source_timeframe):
            raise ValueError(
                f"Target timeframe '{target_timeframe}' must be higher than source timeframe '{source_timeframe}'"
            )

        # Ensure data has a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index for resampling")

        # Convert pandas resample rule
        resample_rule = TimeframeConverter._convert_to_pandas_rule(target_timeframe)

        # Define aggregation functions for different column types
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        # Add default aggregation for other columns
        for col in data.columns:
            if col not in agg_dict:
                agg_dict[col] = "last"

        # Resample the data
        resampled = data.resample(resample_rule).agg(agg_dict)

        return resampled

    @staticmethod
    def expand_to_lower_timeframe(
        data: pd.DataFrame,
        source_timeframe: Union[DataTimeframe, str],
        target_timeframe: Union[DataTimeframe, str],
        method: str = "ffill",
    ) -> pd.DataFrame:
        """Expand data to a lower timeframe.

        Args:
            data: The input data with a datetime index.
            source_timeframe: The source timeframe.
            target_timeframe: The target timeframe (must be lower than source).
            method: The method to use for filling values ("ffill", "bfill", or "nearest").

        Returns:
            pd.DataFrame: The expanded data.

        Raises:
            ValueError: If the target timeframe is not lower than the source timeframe.
        """
        # Convert enum to string if needed
        if isinstance(source_timeframe, DataTimeframe):
            source_timeframe = source_timeframe.value
        if isinstance(target_timeframe, DataTimeframe):
            target_timeframe = target_timeframe.value

        # Validate timeframes
        if not TimeframeConverter._is_lower_timeframe(target_timeframe, source_timeframe):
            raise ValueError(
                f"Target timeframe '{target_timeframe}' must be lower than source timeframe '{source_timeframe}'"
            )

        # Ensure data has a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index for expansion")

        # Get start and end times
        start_time = data.index.min()
        end_time = data.index.max()

        # Create a new index with the target timeframe
        target_rule = TimeframeConverter._convert_to_pandas_rule(target_timeframe)
        new_index = pd.date_range(start=start_time, end=end_time, freq=target_rule)

        # Reindex the data
        expanded = data.reindex(new_index, method=method)

        # Special handling for OHLC data
        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            # For OHLC data, we need to adjust the values
            # Each bar in the source timeframe becomes multiple bars in the target timeframe
            # The first bar gets the open, high, low, close values
            # Subsequent bars get the close value for open, high, low, close

            # Get the source timeframe bars
            source_bars = data.index.tolist()

            # For each source bar, find the corresponding target bars
            for i in range(len(source_bars)):
                current_bar = source_bars[i]
                next_bar = source_bars[i + 1] if i < len(source_bars) - 1 else end_time + pd.Timedelta(seconds=1)

                # Get the target bars within this source bar
                mask = (expanded.index >= current_bar) & (expanded.index < next_bar)
                target_bars = expanded.index[mask]

                if len(target_bars) > 1:
                    # First bar gets the OHLC values
                    # Subsequent bars get the close value for OHLC
                    for j in range(1, len(target_bars)):
                        expanded.loc[target_bars[j], "open"] = data.loc[current_bar, "close"]
                        expanded.loc[target_bars[j], "high"] = data.loc[current_bar, "close"]
                        expanded.loc[target_bars[j], "low"] = data.loc[current_bar, "close"]

        return expanded

    @staticmethod
    def _convert_to_pandas_rule(timeframe: str) -> str:
        """Convert a timeframe string to a pandas resample rule.

        Args:
            timeframe: The timeframe string (e.g., "1m", "1h", "1d").

        Returns:
            str: The pandas resample rule.

        Raises:
            ValueError: If the timeframe is not supported.
        """
        # Extract the number and unit
        if len(timeframe) < 2:
            raise ValueError(f"Invalid timeframe format: {timeframe}")

        # Find the position where the numeric part ends
        for i in range(len(timeframe)):
            if not timeframe[i].isdigit():
                break
        else:
            raise ValueError(f"Invalid timeframe format: {timeframe}")

        number = int(timeframe[:i])
        unit = timeframe[i:]

        # Convert to pandas rule
        if unit == "m" or unit == "min":
            return f"{number}T"  # T is the pandas code for minutes
        elif unit == "h":
            return f"{number}H"
        elif unit == "d":
            return f"{number}D"
        elif unit == "w":
            return f"{number}W"
        elif unit == "mo":
            return f"{number}M"
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

    @staticmethod
    def _is_higher_timeframe(timeframe1: str, timeframe2: str) -> bool:
        """Check if timeframe1 is higher than timeframe2.

        Args:
            timeframe1: The first timeframe.
            timeframe2: The second timeframe.

        Returns:
            bool: True if timeframe1 is higher than timeframe2, False otherwise.
        """
        # Convert timeframes to minutes for comparison
        minutes1 = TimeframeConverter._convert_to_minutes(timeframe1)
        minutes2 = TimeframeConverter._convert_to_minutes(timeframe2)

        return minutes1 > minutes2

    @staticmethod
    def _is_lower_timeframe(timeframe1: str, timeframe2: str) -> bool:
        """Check if timeframe1 is lower than timeframe2.

        Args:
            timeframe1: The first timeframe.
            timeframe2: The second timeframe.

        Returns:
            bool: True if timeframe1 is lower than timeframe2, False otherwise.
        """
        # Convert timeframes to minutes for comparison
        minutes1 = TimeframeConverter._convert_to_minutes(timeframe1)
        minutes2 = TimeframeConverter._convert_to_minutes(timeframe2)

        return minutes1 < minutes2

    @staticmethod
    def _convert_to_minutes(timeframe: str) -> int:
        """Convert a timeframe string to minutes.

        Args:
            timeframe: The timeframe string (e.g., "1m", "1h", "1d").

        Returns:
            int: The number of minutes.

        Raises:
            ValueError: If the timeframe is not supported.
        """
        # Extract the number and unit
        if len(timeframe) < 2:
            raise ValueError(f"Invalid timeframe format: {timeframe}")

        # Find the position where the numeric part ends
        for i in range(len(timeframe)):
            if not timeframe[i].isdigit():
                break
        else:
            raise ValueError(f"Invalid timeframe format: {timeframe}")

        number = int(timeframe[:i])
        unit = timeframe[i:]

        # Convert to minutes
        if unit == "m" or unit == "min":
            return number
        elif unit == "h":
            return number * 60
        elif unit == "d":
            return number * 60 * 24
        elif unit == "w":
            return number * 60 * 24 * 7
        elif unit == "mo":
            # Approximate a month as 30 days
            return number * 60 * 24 * 30
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")


class TimeframeAlignment:
    """Class for aligning data from multiple timeframes.

    This class provides methods for aligning data from multiple timeframes
    to a common timeframe or for creating multi-timeframe features.
    """

    @staticmethod
    def align_to_common_timeframe(
        data_dict: Dict[str, pd.DataFrame],
        target_timeframe: Union[DataTimeframe, str],
    ) -> Dict[str, pd.DataFrame]:
        """Align data from multiple timeframes to a common timeframe.

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames.
            target_timeframe: The target timeframe to align to.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of aligned DataFrames.
        """
        # Convert enum to string if needed
        if isinstance(target_timeframe, DataTimeframe):
            target_timeframe = target_timeframe.value

        result = {}

        for tf, df in data_dict.items():
            # Skip if already at target timeframe
            if tf == target_timeframe:
                result[tf] = df
                continue

            # Convert to target timeframe
            if TimeframeConverter._is_higher_timeframe(target_timeframe, tf):
                # Resample to higher timeframe
                result[tf] = TimeframeConverter.resample_to_higher_timeframe(
                    df, tf, target_timeframe
                )
            else:
                # Expand to lower timeframe
                result[tf] = TimeframeConverter.expand_to_lower_timeframe(
                    df, tf, target_timeframe
                )

        return result

    @staticmethod
    def create_multi_timeframe_features(
        base_data: pd.DataFrame,
        base_timeframe: Union[DataTimeframe, str],
        higher_timeframe_data: Dict[str, pd.DataFrame],
        feature_columns: List[str],
        suffix_format: str = "_{tf}",
    ) -> pd.DataFrame:
        """Create multi-timeframe features by adding higher timeframe data to base timeframe.

        Args:
            base_data: The base timeframe data.
            base_timeframe: The base timeframe.
            higher_timeframe_data: Dictionary mapping timeframe strings to DataFrames.
            feature_columns: List of column names to include from higher timeframes.
            suffix_format: Format string for the suffix to add to column names.

        Returns:
            pd.DataFrame: DataFrame with multi-timeframe features.

        Raises:
            ValueError: If any of the higher timeframes are not actually higher than the base timeframe.
        """
        # Convert enum to string if needed
        if isinstance(base_timeframe, DataTimeframe):
            base_timeframe = base_timeframe.value

        # Make a copy of the base data
        result = base_data.copy()

        # Process each higher timeframe
        for tf, df in higher_timeframe_data.items():
            # Verify that this is a higher timeframe
            if not TimeframeConverter._is_higher_timeframe(tf, base_timeframe):
                raise ValueError(
                    f"Timeframe '{tf}' is not higher than base timeframe '{base_timeframe}'"
                )

            # Expand the higher timeframe data to the base timeframe
            expanded = TimeframeConverter.expand_to_lower_timeframe(
                df, tf, base_timeframe, method="ffill"
            )

            # Add the selected columns with appropriate suffix
            suffix = suffix_format.format(tf=tf)
            for col in feature_columns:
                if col in expanded.columns:
                    result[f"{col}{suffix}"] = expanded[col]

        return result


class MultiTimeframeProcessor(DataProcessor):
    """Class for processing data across multiple timeframes.

    This class provides methods for handling data from multiple timeframes,
    including conversion, alignment, and feature creation.

    Attributes:
        config: Configuration manager.
        base_timeframe: The base timeframe for processing.
        higher_timeframes: List of higher timeframes to include.
        feature_columns: List of columns to include from higher timeframes.
    """

    def __init__(
        self,
        base_timeframe: Union[DataTimeframe, str],
        higher_timeframes: List[Union[DataTimeframe, str]],
        feature_columns: List[str],
        config: Optional[ConfigManager] = None,
    ):
        """Initialize a multi-timeframe processor.

        Args:
            base_timeframe: The base timeframe for processing.
            higher_timeframes: List of higher timeframes to include.
            feature_columns: List of columns to include from higher timeframes.
            config: Configuration manager. If None, a new one will be created.
        """
        super().__init__(config)

        # Convert enums to strings if needed
        self.base_timeframe = base_timeframe.value if isinstance(base_timeframe, DataTimeframe) else base_timeframe
        self.higher_timeframes = [
            tf.value if isinstance(tf, DataTimeframe) else tf for tf in higher_timeframes
        ]
        self.feature_columns = feature_columns

        # Validate that all higher timeframes are actually higher
        for tf in self.higher_timeframes:
            if not TimeframeConverter._is_higher_timeframe(tf, self.base_timeframe):
                raise ValueError(
                    f"Timeframe '{tf}' is not higher than base timeframe '{self.base_timeframe}'"
                )

        # Register default processing step
        self.add_processing_step(
            ProcessingStep.MULTI_TIMEFRAME_PROCESSING, self.process_multi_timeframe
        )

    def process_multi_timeframe(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Process data from multiple timeframes.

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames.

        Returns:
            pd.DataFrame: Processed data with multi-timeframe features.

        Raises:
            DataProcessingError: If an error occurs during processing.
            ValueError: If the base timeframe data is not provided.
        """
        try:
            # Check if base timeframe data is provided
            if self.base_timeframe not in data_dict:
                raise ValueError(
                    f"Base timeframe '{self.base_timeframe}' data not provided"
                )

            # Get the base data
            base_data = data_dict[self.base_timeframe]

            # Filter higher timeframes that are available in the data
            available_higher_tfs = {
                tf: data_dict[tf]
                for tf in self.higher_timeframes
                if tf in data_dict
            }

            # If no higher timeframe data is available, return the base data
            if not available_higher_tfs:
                logger.warning("No higher timeframe data available for multi-timeframe processing")
                return base_data

            # Create multi-timeframe features
            result = TimeframeAlignment.create_multi_timeframe_features(
                base_data,
                self.base_timeframe,
                available_higher_tfs,
                self.feature_columns,
            )

            # Record multi-timeframe processing in metadata
            self.metadata["base_timeframe"] = self.base_timeframe
            self.metadata["higher_timeframes"] = list(available_higher_tfs.keys())
            self.metadata["feature_columns"] = self.feature_columns

            return result

        except Exception as e:
            error_msg = f"Error processing multi-timeframe data: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise DataProcessingError(error_msg)