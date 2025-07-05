"""Pattern Recognition System for Friday AI Trading System.

This module provides functionality for detecting and analyzing patterns in financial
market data, including candlestick patterns, chart patterns, and technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from enum import Enum, auto

from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class PatternType(Enum):
    """Enum for different types of patterns in financial data."""
    CANDLESTICK = auto()  # Single or multiple candlestick patterns
    CHART = auto()        # Chart patterns like head and shoulders, triangles, etc.
    INDICATOR = auto()    # Technical indicator patterns like MACD crossover, RSI overbought, etc.
    HARMONIC = auto()     # Harmonic patterns like Gartley, Butterfly, etc.
    ELLIOTT_WAVE = auto() # Elliott Wave patterns
    CUSTOM = auto()       # Custom user-defined patterns


class PatternStrength(Enum):
    """Enum for the strength or reliability of a detected pattern."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


class PatternDirection(Enum):
    """Enum for the expected direction after a pattern."""
    BULLISH = auto()  # Upward movement expected
    BEARISH = auto()  # Downward movement expected
    NEUTRAL = auto()  # No clear direction expected
    CONTINUATION = auto()  # Continuation of current trend expected
    REVERSAL = auto()  # Reversal of current trend expected


class PatternResult:
    """Class to store the result of a pattern detection."""
    
    def __init__(self, 
                 pattern_name: str,
                 pattern_type: PatternType,
                 direction: PatternDirection,
                 strength: PatternStrength,
                 start_index: int,
                 end_index: int,
                 confidence: float,
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize a pattern result.
        
        Args:
            pattern_name: Name of the detected pattern.
            pattern_type: Type of the pattern.
            direction: Expected direction after the pattern.
            strength: Strength or reliability of the pattern.
            start_index: Start index of the pattern in the data.
            end_index: End index of the pattern in the data.
            confidence: Confidence score of the pattern detection (0.0 to 1.0).
            metadata: Additional metadata about the pattern.
        """
        self.pattern_name = pattern_name
        self.pattern_type = pattern_type
        self.direction = direction
        self.strength = strength
        self.start_index = start_index
        self.end_index = end_index
        self.confidence = confidence
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        """String representation of the pattern result."""
        return (f"{self.pattern_name} ({self.pattern_type.name}): {self.direction.name} pattern "
                f"with {self.strength.name} strength at indices {self.start_index}-{self.end_index} "
                f"(confidence: {self.confidence:.2f})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the pattern result to a dictionary."""
        return {
            "pattern_name": self.pattern_name,
            "pattern_type": self.pattern_type.name,
            "direction": self.direction.name,
            "strength": self.strength.name,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class PatternDetector:
    """Base class for pattern detectors."""
    
    def __init__(self, name: str, pattern_type: PatternType):
        """Initialize a pattern detector.
        
        Args:
            name: Name of the pattern detector.
            pattern_type: Type of patterns this detector handles.
        """
        self.name = name
        self.pattern_type = pattern_type
        logger.info(f"Initialized {name} pattern detector for {pattern_type.name} patterns")
    
    def detect(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect patterns in the data.
        
        Args:
            data: DataFrame containing the market data.
            
        Returns:
            List[PatternResult]: List of detected patterns.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the detect method")


class CandlestickPatternDetector(PatternDetector):
    """Detector for candlestick patterns."""
    
    def __init__(self):
        """Initialize a candlestick pattern detector."""
        super().__init__("Candlestick", PatternType.CANDLESTICK)
        self.patterns = self._register_patterns()
    
    def _register_patterns(self) -> Dict[str, Callable]:
        """Register candlestick pattern detection functions.
        
        Returns:
            Dict[str, Callable]: Dictionary of pattern names and detection functions.
        """
        return {
            "Doji": self._detect_doji,
            "Hammer": self._detect_hammer,
            "Shooting Star": self._detect_shooting_star,
            "Engulfing": self._detect_engulfing,
            "Morning Star": self._detect_morning_star,
            "Evening Star": self._detect_evening_star,
            "Harami": self._detect_harami,
            "Three White Soldiers": self._detect_three_white_soldiers,
            "Three Black Crows": self._detect_three_black_crows,
            "Piercing Line": self._detect_piercing_line,
            "Dark Cloud Cover": self._detect_dark_cloud_cover
        }
    
    def detect(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect candlestick patterns in the data.
        
        Args:
            data: DataFrame containing OHLCV data.
            
        Returns:
            List[PatternResult]: List of detected candlestick patterns.
            
        Raises:
            ValueError: If the data does not contain required columns.
        """
        # Check if data contains required columns
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col.lower() in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Standardize column names to lowercase
        data_std = data.copy()
        for col in required_columns:
            if col in data.columns:
                data_std[col.lower()] = data[col]
            elif col.upper() in data.columns:
                data_std[col.lower()] = data[col.upper()]
        
        # Detect patterns
        results = []
        for pattern_name, detect_func in self.patterns.items():
            try:
                pattern_results = detect_func(data_std)
                results.extend(pattern_results)
            except Exception as e:
                logger.error(f"Error detecting {pattern_name} pattern: {str(e)}")
        
        return results
    
    def _detect_doji(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Doji candlestick patterns.
        
        A Doji occurs when the opening and closing prices are very close or equal.
        
        Args:
            data: DataFrame containing OHLCV data.
            
        Returns:
            List[PatternResult]: List of detected Doji patterns.
        """
        results = []
        
        # Calculate body size as percentage of the high-low range
        body_size = abs(data['close'] - data['open'])
        candle_range = data['high'] - data['low']
        body_percentage = body_size / candle_range
        
        # Doji threshold: body is less than 10% of the candle range
        doji_threshold = 0.1
        
        for i in range(len(data)):
            if candle_range[i] > 0 and body_percentage[i] < doji_threshold:
                # Determine pattern direction based on trend
                if i > 0:
                    prev_trend = data['close'][i-1] > data['open'][i-1]
                    direction = PatternDirection.REVERSAL
                else:
                    direction = PatternDirection.NEUTRAL
                
                # Determine strength based on volume if available
                if 'volume' in data.columns and i > 0:
                    avg_volume = data['volume'].iloc[max(0, i-5):i].mean()
                    if data['volume'][i] > 1.5 * avg_volume:
                        strength = PatternStrength.STRONG
                    elif data['volume'][i] > avg_volume:
                        strength = PatternStrength.MODERATE
                    else:
                        strength = PatternStrength.WEAK
                else:
                    strength = PatternStrength.MODERATE
                
                # Calculate confidence based on how close to a perfect doji
                confidence = 1.0 - (body_percentage[i] / doji_threshold)
                
                results.append(PatternResult(
                    pattern_name="Doji",
                    pattern_type=PatternType.CANDLESTICK,
                    direction=direction,
                    strength=strength,
                    start_index=i,
                    end_index=i,
                    confidence=confidence,
                    metadata={
                        "body_percentage": body_percentage[i],
                        "candle_range": candle_range[i]
                    }
                ))
        
        return results
    
    def _detect_hammer(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Hammer candlestick patterns.
        
        A Hammer has a small body at the top with a long lower shadow.
        
        Args:
            data: DataFrame containing OHLCV data.
            
        Returns:
            List[PatternResult]: List of detected Hammer patterns.
        """
        results = []
        
        # Calculate body and shadow sizes
        body_size = abs(data['close'] - data['open'])
        upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)
        lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
        candle_range = data['high'] - data['low']
        
        for i in range(len(data)):
            if candle_range[i] > 0:
                # Hammer criteria:
                # 1. Lower shadow is at least twice the body size
                # 2. Upper shadow is small (less than 10% of candle range)
                # 3. Body is in the upper third of the candle range
                if (lower_shadow[i] >= 2 * body_size[i] and 
                    upper_shadow[i] <= 0.1 * candle_range[i] and 
                    data[['open', 'close']].min(axis=1)[i] >= data['low'][i] + 0.6 * candle_range[i]):
                    
                    # Determine if it's a bullish pattern (appears in a downtrend)
                    is_downtrend = False
                    if i >= 3:
                        # Simple downtrend detection: lower lows and lower highs
                        prev_lows = data['low'].iloc[i-3:i]
                        prev_highs = data['high'].iloc[i-3:i]
                        is_downtrend = (prev_lows.is_monotonic_decreasing and 
                                        prev_highs.is_monotonic_decreasing)
                    
                    direction = PatternDirection.BULLISH if is_downtrend else PatternDirection.NEUTRAL
                    
                    # Determine strength based on the lower shadow length
                    lower_shadow_ratio = lower_shadow[i] / body_size[i]
                    if lower_shadow_ratio > 4:
                        strength = PatternStrength.VERY_STRONG
                    elif lower_shadow_ratio > 3:
                        strength = PatternStrength.STRONG
                    elif lower_shadow_ratio > 2:
                        strength = PatternStrength.MODERATE
                    else:
                        strength = PatternStrength.WEAK
                    
                    # Calculate confidence
                    confidence = min(1.0, lower_shadow_ratio / 5.0)
                    
                    results.append(PatternResult(
                        pattern_name="Hammer",
                        pattern_type=PatternType.CANDLESTICK,
                        direction=direction,
                        strength=strength,
                        start_index=i,
                        end_index=i,
                        confidence=confidence,
                        metadata={
                            "lower_shadow_ratio": lower_shadow_ratio,
                            "is_downtrend": is_downtrend
                        }
                    ))
        
        return results
    
    def _detect_shooting_star(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Shooting Star candlestick patterns.
        
        A Shooting Star has a small body at the bottom with a long upper shadow.
        
        Args:
            data: DataFrame containing OHLCV data.
            
        Returns:
            List[PatternResult]: List of detected Shooting Star patterns.
        """
        results = []
        
        # Calculate body and shadow sizes
        body_size = abs(data['close'] - data['open'])
        upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)
        lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
        candle_range = data['high'] - data['low']
        
        for i in range(len(data)):
            if candle_range[i] > 0:
                # Shooting Star criteria:
                # 1. Upper shadow is at least twice the body size
                # 2. Lower shadow is small (less than 10% of candle range)
                # 3. Body is in the lower third of the candle range
                if (upper_shadow[i] >= 2 * body_size[i] and 
                    lower_shadow[i] <= 0.1 * candle_range[i] and 
                    data[['open', 'close']].max(axis=1)[i] <= data['low'][i] + 0.4 * candle_range[i]):
                    
                    # Determine if it's a bearish pattern (appears in an uptrend)
                    is_uptrend = False
                    if i >= 3:
                        # Simple uptrend detection: higher highs and higher lows
                        prev_lows = data['low'].iloc[i-3:i]
                        prev_highs = data['high'].iloc[i-3:i]
                        is_uptrend = (prev_lows.is_monotonic_increasing and 
                                      prev_highs.is_monotonic_increasing)
                    
                    direction = PatternDirection.BEARISH if is_uptrend else PatternDirection.NEUTRAL
                    
                    # Determine strength based on the upper shadow length
                    upper_shadow_ratio = upper_shadow[i] / body_size[i]
                    if upper_shadow_ratio > 4:
                        strength = PatternStrength.VERY_STRONG
                    elif upper_shadow_ratio > 3:
                        strength = PatternStrength.STRONG
                    elif upper_shadow_ratio > 2:
                        strength = PatternStrength.MODERATE
                    else:
                        strength = PatternStrength.WEAK
                    
                    # Calculate confidence
                    confidence = min(1.0, upper_shadow_ratio / 5.0)
                    
                    results.append(PatternResult(
                        pattern_name="Shooting Star",
                        pattern_type=PatternType.CANDLESTICK,
                        direction=direction,
                        strength=strength,
                        start_index=i,
                        end_index=i,
                        confidence=confidence,
                        metadata={
                            "upper_shadow_ratio": upper_shadow_ratio,
                            "is_uptrend": is_uptrend
                        }
                    ))
        
        return results
    
    def _detect_engulfing(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Bullish and Bearish Engulfing patterns.
        
        An Engulfing pattern occurs when a candle's body completely engulfs the previous candle's body.
        
        Args:
            data: DataFrame containing OHLCV data.
            
        Returns:
            List[PatternResult]: List of detected Engulfing patterns.
        """
        results = []
        
        for i in range(1, len(data)):
            # Get current and previous candle data
            curr_open = data['open'][i]
            curr_close = data['close'][i]
            prev_open = data['open'][i-1]
            prev_close = data['close'][i-1]
            
            # Calculate body sizes
            curr_body_size = abs(curr_close - curr_open)
            prev_body_size = abs(prev_close - prev_open)
            
            # Skip if either candle has a very small body
            if curr_body_size < 0.001 or prev_body_size < 0.001:
                continue
            
            # Bullish Engulfing: current candle is bullish and engulfs previous bearish candle
            is_bullish_engulfing = (
                curr_close > curr_open and  # Current candle is bullish
                prev_close < prev_open and  # Previous candle is bearish
                curr_open <= prev_close and  # Current open is below or equal to previous close
                curr_close >= prev_open      # Current close is above or equal to previous open
            )
            
            # Bearish Engulfing: current candle is bearish and engulfs previous bullish candle
            is_bearish_engulfing = (
                curr_close < curr_open and  # Current candle is bearish
                prev_close > prev_open and  # Previous candle is bullish
                curr_open >= prev_close and  # Current open is above or equal to previous close
                curr_close <= prev_open      # Current close is below or equal to previous open
            )
            
            if is_bullish_engulfing or is_bearish_engulfing:
                # Determine pattern name and direction
                if is_bullish_engulfing:
                    pattern_name = "Bullish Engulfing"
                    direction = PatternDirection.BULLISH
                else:
                    pattern_name = "Bearish Engulfing"
                    direction = PatternDirection.BEARISH
                
                # Determine strength based on size ratio and volume
                size_ratio = curr_body_size / prev_body_size
                
                if size_ratio > 2.0:
                    strength = PatternStrength.VERY_STRONG
                elif size_ratio > 1.5:
                    strength = PatternStrength.STRONG
                elif size_ratio > 1.2:
                    strength = PatternStrength.MODERATE
                else:
                    strength = PatternStrength.WEAK
                
                # Adjust strength based on volume if available
                if 'volume' in data.columns:
                    vol_ratio = data['volume'][i] / data['volume'][i-1]
                    if vol_ratio > 2.0:
                        strength = PatternStrength(min(4, strength.value + 1))
                
                # Calculate confidence
                confidence = min(1.0, size_ratio / 3.0)
                
                results.append(PatternResult(
                    pattern_name=pattern_name,
                    pattern_type=PatternType.CANDLESTICK,
                    direction=direction,
                    strength=strength,
                    start_index=i-1,
                    end_index=i,
                    confidence=confidence,
                    metadata={
                        "size_ratio": size_ratio,
                        "vol_ratio": vol_ratio if 'volume' in data.columns else None
                    }
                ))
        
        return results
    
    # Placeholder methods for other candlestick patterns
    # These would be implemented similarly to the patterns above
    
    def _detect_morning_star(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Morning Star patterns."""
        # Implementation would go here
        return []
    
    def _detect_evening_star(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Evening Star patterns."""
        # Implementation would go here
        return []
    
    def _detect_harami(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Harami patterns."""
        # Implementation would go here
        return []
    
    def _detect_three_white_soldiers(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Three White Soldiers patterns."""
        # Implementation would go here
        return []
    
    def _detect_three_black_crows(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Three Black Crows patterns."""
        # Implementation would go here
        return []
    
    def _detect_piercing_line(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Piercing Line patterns."""
        # Implementation would go here
        return []
    
    def _detect_dark_cloud_cover(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Dark Cloud Cover patterns."""
        # Implementation would go here
        return []


class ChartPatternDetector(PatternDetector):
    """Detector for chart patterns."""
    
    def __init__(self):
        """Initialize a chart pattern detector."""
        super().__init__("Chart", PatternType.CHART)
        self.patterns = self._register_patterns()
    
    def _register_patterns(self) -> Dict[str, Callable]:
        """Register chart pattern detection functions.
        
        Returns:
            Dict[str, Callable]: Dictionary of pattern names and detection functions.
        """
        return {
            "Head and Shoulders": self._detect_head_and_shoulders,
            "Inverse Head and Shoulders": self._detect_inverse_head_and_shoulders,
            "Double Top": self._detect_double_top,
            "Double Bottom": self._detect_double_bottom,
            "Triple Top": self._detect_triple_top,
            "Triple Bottom": self._detect_triple_bottom,
            "Ascending Triangle": self._detect_ascending_triangle,
            "Descending Triangle": self._detect_descending_triangle,
            "Symmetrical Triangle": self._detect_symmetrical_triangle,
            "Rectangle": self._detect_rectangle,
            "Cup and Handle": self._detect_cup_and_handle,
            "Wedge": self._detect_wedge
        }
    
    def detect(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect chart patterns in the data.
        
        Args:
            data: DataFrame containing OHLCV data.
            
        Returns:
            List[PatternResult]: List of detected chart patterns.
            
        Raises:
            ValueError: If the data does not contain required columns.
        """
        # Check if data contains required columns
        required_columns = ['high', 'low', 'close']
        if not all(col.lower() in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Standardize column names to lowercase
        data_std = data.copy()
        for col in required_columns:
            if col in data.columns:
                data_std[col.lower()] = data[col]
            elif col.upper() in data.columns:
                data_std[col.lower()] = data[col.upper()]
        
        # Detect patterns
        results = []
        for pattern_name, detect_func in self.patterns.items():
            try:
                pattern_results = detect_func(data_std)
                results.extend(pattern_results)
            except Exception as e:
                logger.error(f"Error detecting {pattern_name} pattern: {str(e)}")
        
        return results
    
    # Placeholder methods for chart patterns
    # These would be implemented with more complex algorithms
    
    def _detect_head_and_shoulders(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Head and Shoulders patterns."""
        # Implementation would go here
        return []
    
    def _detect_inverse_head_and_shoulders(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Inverse Head and Shoulders patterns."""
        # Implementation would go here
        return []
    
    def _detect_double_top(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Double Top patterns."""
        # Implementation would go here
        return []
    
    def _detect_double_bottom(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Double Bottom patterns."""
        # Implementation would go here
        return []
    
    def _detect_triple_top(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Triple Top patterns."""
        # Implementation would go here
        return []
    
    def _detect_triple_bottom(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Triple Bottom patterns."""
        # Implementation would go here
        return []
    
    def _detect_ascending_triangle(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Ascending Triangle patterns."""
        # Implementation would go here
        return []
    
    def _detect_descending_triangle(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Descending Triangle patterns."""
        # Implementation would go here
        return []
    
    def _detect_symmetrical_triangle(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Symmetrical Triangle patterns."""
        # Implementation would go here
        return []
    
    def _detect_rectangle(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Rectangle patterns."""
        # Implementation would go here
        return []
    
    def _detect_cup_and_handle(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Cup and Handle patterns."""
        # Implementation would go here
        return []
    
    def _detect_wedge(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Wedge patterns."""
        # Implementation would go here
        return []


class IndicatorPatternDetector(PatternDetector):
    """Detector for technical indicator patterns."""
    
    def __init__(self):
        """Initialize a technical indicator pattern detector."""
        super().__init__("Indicator", PatternType.INDICATOR)
        self.patterns = self._register_patterns()
    
    def _register_patterns(self) -> Dict[str, Callable]:
        """Register technical indicator pattern detection functions.
        
        Returns:
            Dict[str, Callable]: Dictionary of pattern names and detection functions.
        """
        return {
            "MACD Crossover": self._detect_macd_crossover,
            "RSI Overbought/Oversold": self._detect_rsi_extreme,
            "Bollinger Band Squeeze": self._detect_bollinger_squeeze,
            "Moving Average Crossover": self._detect_ma_crossover,
            "Stochastic Crossover": self._detect_stochastic_crossover
        }
    
    def detect(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect technical indicator patterns in the data.
        
        Args:
            data: DataFrame containing OHLCV data.
            
        Returns:
            List[PatternResult]: List of detected indicator patterns.
            
        Raises:
            ValueError: If the data does not contain required columns.
        """
        # Check if data contains required columns
        required_columns = ['close']
        if not all(col.lower() in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Standardize column names to lowercase
        data_std = data.copy()
        for col in required_columns:
            if col in data.columns:
                data_std[col.lower()] = data[col]
            elif col.upper() in data.columns:
                data_std[col.lower()] = data[col.upper()]
        
        # Calculate common indicators
        data_std = self._calculate_indicators(data_std)
        
        # Detect patterns
        results = []
        for pattern_name, detect_func in self.patterns.items():
            try:
                pattern_results = detect_func(data_std)
                results.extend(pattern_results)
            except Exception as e:
                logger.error(f"Error detecting {pattern_name} pattern: {str(e)}")
        
        return results
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate common technical indicators.
        
        Args:
            data: DataFrame containing OHLCV data.
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators.
        """
        df = data.copy()
        
        # Calculate MACD
        try:
            # MACD Line = 12-period EMA - 26-period EMA
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema12'] - df['ema26']
            # Signal Line = 9-period EMA of MACD Line
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            # MACD Histogram = MACD Line - Signal Line
            df['macd_hist'] = df['macd'] - df['macd_signal']
        except Exception as e:
            logger.warning(f"Error calculating MACD: {str(e)}")
        
        # Calculate RSI
        try:
            # RSI = 100 - (100 / (1 + RS))
            # RS = Average Gain / Average Loss
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
        except Exception as e:
            logger.warning(f"Error calculating RSI: {str(e)}")
        
        # Calculate Bollinger Bands
        try:
            # Middle Band = 20-period SMA
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            # Standard Deviation of price over 20 periods
            df['bb_std'] = df['close'].rolling(window=20).std()
            # Upper Band = Middle Band + (2 * Standard Deviation)
            df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
            # Lower Band = Middle Band - (2 * Standard Deviation)
            df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
            # Bandwidth = (Upper Band - Lower Band) / Middle Band
            df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {str(e)}")
        
        # Calculate Moving Averages
        try:
            df['sma50'] = df['close'].rolling(window=50).mean()
            df['sma200'] = df['close'].rolling(window=200).mean()
        except Exception as e:
            logger.warning(f"Error calculating Moving Averages: {str(e)}")
        
        # Calculate Stochastic Oscillator
        try:
            # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = ((df['close'] - low_14) / (high_14 - low_14)) * 100
            # %D = 3-period SMA of %K
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        except Exception as e:
            logger.warning(f"Error calculating Stochastic Oscillator: {str(e)}")
        
        return df
    
    def _detect_macd_crossover(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect MACD crossover patterns.
        
        Args:
            data: DataFrame containing MACD indicators.
            
        Returns:
            List[PatternResult]: List of detected MACD crossover patterns.
        """
        results = []
        
        # Check if MACD indicators are available
        if not all(col in data.columns for col in ['macd', 'macd_signal']):
            return results
        
        # Look for crossovers
        for i in range(1, len(data)):
            # Bullish crossover: MACD crosses above Signal Line
            if (data['macd'][i-1] < data['macd_signal'][i-1] and 
                data['macd'][i] > data['macd_signal'][i]):
                
                # Determine strength based on histogram size and trend
                hist_size = data['macd_hist'][i]
                is_above_zero = data['macd'][i] > 0
                
                if is_above_zero and hist_size > 0.5:
                    strength = PatternStrength.VERY_STRONG
                elif is_above_zero:
                    strength = PatternStrength.STRONG
                elif hist_size > 0.3:
                    strength = PatternStrength.MODERATE
                else:
                    strength = PatternStrength.WEAK
                
                # Calculate confidence
                confidence = min(1.0, abs(hist_size) / 0.5)
                
                results.append(PatternResult(
                    pattern_name="MACD Bullish Crossover",
                    pattern_type=PatternType.INDICATOR,
                    direction=PatternDirection.BULLISH,
                    strength=strength,
                    start_index=i-1,
                    end_index=i,
                    confidence=confidence,
                    metadata={
                        "macd": data['macd'][i],
                        "signal": data['macd_signal'][i],
                        "histogram": hist_size,
                        "is_above_zero": is_above_zero
                    }
                ))
            
            # Bearish crossover: MACD crosses below Signal Line
            elif (data['macd'][i-1] > data['macd_signal'][i-1] and 
                  data['macd'][i] < data['macd_signal'][i]):
                
                # Determine strength based on histogram size and trend
                hist_size = abs(data['macd_hist'][i])
                is_below_zero = data['macd'][i] < 0
                
                if is_below_zero and hist_size > 0.5:
                    strength = PatternStrength.VERY_STRONG
                elif is_below_zero:
                    strength = PatternStrength.STRONG
                elif hist_size > 0.3:
                    strength = PatternStrength.MODERATE
                else:
                    strength = PatternStrength.WEAK
                
                # Calculate confidence
                confidence = min(1.0, hist_size / 0.5)
                
                results.append(PatternResult(
                    pattern_name="MACD Bearish Crossover",
                    pattern_type=PatternType.INDICATOR,
                    direction=PatternDirection.BEARISH,
                    strength=strength,
                    start_index=i-1,
                    end_index=i,
                    confidence=confidence,
                    metadata={
                        "macd": data['macd'][i],
                        "signal": data['macd_signal'][i],
                        "histogram": -hist_size,
                        "is_below_zero": is_below_zero
                    }
                ))
        
        return results
    
    # Placeholder methods for other indicator patterns
    # These would be implemented similarly to the MACD crossover pattern
    
    def _detect_rsi_extreme(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect RSI overbought/oversold patterns."""
        # Implementation would go here
        return []
    
    def _detect_bollinger_squeeze(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Bollinger Band squeeze patterns."""
        # Implementation would go here
        return []
    
    def _detect_ma_crossover(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Moving Average crossover patterns."""
        # Implementation would go here
        return []
    
    def _detect_stochastic_crossover(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect Stochastic Oscillator crossover patterns."""
        # Implementation would go here
        return []


class PatternRecognitionSystem:
    """System for detecting and analyzing patterns in financial market data."""
    
    def __init__(self, detectors: Optional[List[PatternDetector]] = None):
        """Initialize the pattern recognition system.
        
        Args:
            detectors: List of pattern detectors to use. If None, all available detectors are used.
        """
        if detectors is None:
            self.detectors = [
                CandlestickPatternDetector(),
                ChartPatternDetector(),
                IndicatorPatternDetector()
            ]
        else:
            self.detectors = detectors
        
        logger.info(f"Initialized PatternRecognitionSystem with {len(self.detectors)} detectors")
    
    def detect_patterns(self, data: pd.DataFrame) -> List[PatternResult]:
        """Detect patterns in the data using all registered detectors.
        
        Args:
            data: DataFrame containing market data.
            
        Returns:
            List[PatternResult]: List of all detected patterns.
        """
        all_results = []
        
        for detector in self.detectors:
            try:
                results = detector.detect(data)
                all_results.extend(results)
                logger.info(f"Detected {len(results)} patterns with {detector.name} detector")
            except Exception as e:
                logger.error(f"Error with {detector.name} detector: {str(e)}")
        
        # Sort results by end_index (most recent first) and then by confidence (highest first)
        all_results.sort(key=lambda x: (-x.end_index, -x.confidence))
        
        return all_results
    
    def get_patterns_by_type(self, results: List[PatternResult], pattern_type: PatternType) -> List[PatternResult]:
        """Filter pattern results by pattern type.
        
        Args:
            results: List of pattern results.
            pattern_type: Type of patterns to filter for.
            
        Returns:
            List[PatternResult]: Filtered list of pattern results.
        """
        return [r for r in results if r.pattern_type == pattern_type]
    
    def get_patterns_by_direction(self, results: List[PatternResult], direction: PatternDirection) -> List[PatternResult]:
        """Filter pattern results by expected direction.
        
        Args:
            results: List of pattern results.
            direction: Expected direction to filter for.
            
        Returns:
            List[PatternResult]: Filtered list of pattern results.
        """
        return [r for r in results if r.direction == direction]
    
    def get_patterns_by_strength(self, results: List[PatternResult], min_strength: PatternStrength) -> List[PatternResult]:
        """Filter pattern results by minimum strength.
        
        Args:
            results: List of pattern results.
            min_strength: Minimum strength to filter for.
            
        Returns:
            List[PatternResult]: Filtered list of pattern results.
        """
        return [r for r in results if r.strength.value >= min_strength.value]
    
    def get_patterns_by_confidence(self, results: List[PatternResult], min_confidence: float) -> List[PatternResult]:
        """Filter pattern results by minimum confidence.
        
        Args:
            results: List of pattern results.
            min_confidence: Minimum confidence score to filter for (0.0 to 1.0).
            
        Returns:
            List[PatternResult]: Filtered list of pattern results.
        """
        return [r for r in results if r.confidence >= min_confidence]
    
    def get_recent_patterns(self, results: List[PatternResult], lookback: int) -> List[PatternResult]:
        """Get patterns that occurred within the recent lookback period.
        
        Args:
            results: List of pattern results.
            lookback: Number of periods to look back.
            
        Returns:
            List[PatternResult]: List of recent pattern results.
        """
        if not results:
            return []
        
        # Find the most recent index
        most_recent_idx = max(r.end_index for r in results)
        
        # Filter for patterns within the lookback period
        return [r for r in results if r.end_index >= most_recent_idx - lookback]