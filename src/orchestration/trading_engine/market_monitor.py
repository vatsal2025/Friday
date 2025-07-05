"""Market Condition Monitoring for Production Trading Engine.

This module provides tools for monitoring market conditions in real-time,
detecting unusual market behavior, and triggering appropriate responses
to protect trading capital during volatile or abnormal market conditions.
"""

from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Set
from enum import Enum
import time
from datetime import datetime, timedelta
import threading
import numpy as np
import pytz
from collections import deque

from src.infrastructure.logging import get_logger
from src.infrastructure.event import EventSystem, EventHandler
from src.orchestration.trading_engine.emergency import EmergencyTrigger, EmergencyLevel, EmergencyAction

# Configure logger
logger = get_logger(__name__)


class MarketCondition(Enum):
    """Market condition classifications."""
    NORMAL = "normal"  # Normal market conditions
    VOLATILE = "volatile"  # Higher than normal volatility
    HIGHLY_VOLATILE = "highly_volatile"  # Extremely high volatility
    ILLIQUID = "illiquid"  # Low liquidity conditions
    GAPPING = "gapping"  # Price gaps occurring
    FAST_MARKET = "fast_market"  # Rapid price movements
    SLOW_MARKET = "slow_market"  # Delayed price updates
    HALTED = "halted"  # Trading halted
    CLOSED = "closed"  # Market closed
    OPENING = "opening"  # Market opening period
    CLOSING = "closing"  # Market closing period
    NEWS_IMPACT = "news_impact"  # Market affected by news
    ABNORMAL = "abnormal"  # Abnormal conditions not otherwise classified


class VolatilityLevel(Enum):
    """Volatility level classifications."""
    VERY_LOW = 0
    LOW = 1
    NORMAL = 2
    ELEVATED = 3
    HIGH = 4
    VERY_HIGH = 5
    EXTREME = 6


class LiquidityLevel(Enum):
    """Liquidity level classifications."""
    VERY_LOW = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    VERY_HIGH = 4


class MarketMetrics:
    """Container for market metrics."""
    def __init__(self, symbol: str, timeframe: str = "1m"):
        """Initialize market metrics.
        
        Args:
            symbol: The market symbol
            timeframe: The timeframe for metrics calculation
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.timestamp = datetime.now(pytz.UTC)
        
        # Price metrics
        self.last_price: Optional[float] = None
        self.bid_price: Optional[float] = None
        self.ask_price: Optional[float] = None
        self.mid_price: Optional[float] = None
        self.open_price: Optional[float] = None
        self.high_price: Optional[float] = None
        self.low_price: Optional[float] = None
        self.close_price: Optional[float] = None
        self.vwap: Optional[float] = None
        
        # Volume metrics
        self.volume: Optional[float] = None
        self.avg_volume: Optional[float] = None
        self.relative_volume: Optional[float] = None
        
        # Volatility metrics
        self.volatility: Optional[float] = None  # Realized volatility
        self.implied_volatility: Optional[float] = None  # Option implied volatility
        self.historical_volatility: Optional[float] = None  # Historical volatility
        self.volatility_level: VolatilityLevel = VolatilityLevel.NORMAL
        
        # Liquidity metrics
        self.spread: Optional[float] = None
        self.relative_spread: Optional[float] = None  # Spread as percentage of price
        self.depth: Optional[float] = None  # Order book depth
        self.liquidity_level: LiquidityLevel = LiquidityLevel.NORMAL
        
        # Market condition
        self.condition: MarketCondition = MarketCondition.NORMAL
        
        # Abnormal indicators
        self.price_gaps: List[float] = []  # Recent price gaps
        self.price_spikes: List[float] = []  # Recent price spikes
        self.unusual_volume_spikes: List[float] = []  # Recent volume spikes
        self.unusual_spread_widening: List[float] = []  # Recent spread widening events
        
        # Market status
        self.is_halted: bool = False
        self.is_open: bool = True
        
        # News impact
        self.recent_news_impact: bool = False
        self.news_sentiment: Optional[float] = None  # -1.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "last_price": self.last_price,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "mid_price": self.mid_price,
            "open_price": self.open_price,
            "high_price": self.high_price,
            "low_price": self.low_price,
            "close_price": self.close_price,
            "vwap": self.vwap,
            "volume": self.volume,
            "avg_volume": self.avg_volume,
            "relative_volume": self.relative_volume,
            "volatility": self.volatility,
            "implied_volatility": self.implied_volatility,
            "historical_volatility": self.historical_volatility,
            "volatility_level": self.volatility_level.name,
            "spread": self.spread,
            "relative_spread": self.relative_spread,
            "depth": self.depth,
            "liquidity_level": self.liquidity_level.name,
            "condition": self.condition.value,
            "price_gaps": self.price_gaps,
            "price_spikes": self.price_spikes,
            "unusual_volume_spikes": self.unusual_volume_spikes,
            "unusual_spread_widening": self.unusual_spread_widening,
            "is_halted": self.is_halted,
            "is_open": self.is_open,
            "recent_news_impact": self.recent_news_impact,
            "news_sentiment": self.news_sentiment
        }


class MarketMonitor:
    """Monitor market conditions and detect abnormal behavior."""
    def __init__(
        self,
        event_system: EventSystem,
        config: Dict[str, Any] = None,
        emergency_callback: Optional[Callable[[EmergencyTrigger, EmergencyLevel, str, EmergencyAction, List[str], List[str], Dict[str, Any]], None]] = None
    ):
        """Initialize the market monitor.
        
        Args:
            event_system: Event system for publishing and subscribing to events
            config: Configuration for the market monitor
            emergency_callback: Callback function for emergency situations
        """
        self.event_system = event_system
        self.config = config or {}
        self.emergency_callback = emergency_callback
        
        # Initialize metrics storage
        self.metrics: Dict[str, MarketMetrics] = {}
        self.historical_metrics: Dict[str, Dict[str, deque]] = {}
        
        # Initialize monitoring state
        self.monitored_symbols: Set[str] = set()
        self.running = False
        self.lock = threading.Lock()
        
        # Configure monitoring parameters
        self._configure_monitoring_parameters()
        
        # Register event handlers
        self._register_event_handlers()
    
    def _configure_monitoring_parameters(self) -> None:
        """Configure monitoring parameters from config."""
        # Historical data window sizes
        self.price_history_size = self.config.get("price_history_size", 100)
        self.volume_history_size = self.config.get("volume_history_size", 20)
        self.volatility_history_size = self.config.get("volatility_history_size", 20)
        
        # Volatility thresholds
        self.volatility_thresholds = self.config.get("volatility_thresholds", {
            VolatilityLevel.VERY_LOW.name: 0.5,  # 0.5x normal volatility
            VolatilityLevel.LOW.name: 0.75,  # 0.75x normal volatility
            VolatilityLevel.NORMAL.name: 1.0,  # Baseline
            VolatilityLevel.ELEVATED.name: 1.5,  # 1.5x normal volatility
            VolatilityLevel.HIGH.name: 2.0,  # 2x normal volatility
            VolatilityLevel.VERY_HIGH.name: 3.0,  # 3x normal volatility
            VolatilityLevel.EXTREME.name: 5.0  # 5x normal volatility
        })
        
        # Price gap thresholds
        self.price_gap_threshold = self.config.get("price_gap_threshold", 0.01)  # 1% gap
        self.price_spike_threshold = self.config.get("price_spike_threshold", 0.02)  # 2% spike
        
        # Volume thresholds
        self.volume_spike_threshold = self.config.get("volume_spike_threshold", 3.0)  # 3x normal volume
        
        # Spread thresholds
        self.spread_widening_threshold = self.config.get("spread_widening_threshold", 3.0)  # 3x normal spread
        
        # Emergency thresholds
        self.emergency_thresholds = self.config.get("emergency_thresholds", {
            "volatility": {
                EmergencyLevel.LOW.name: self.volatility_thresholds[VolatilityLevel.ELEVATED.name],
                EmergencyLevel.MEDIUM.name: self.volatility_thresholds[VolatilityLevel.HIGH.name],
                EmergencyLevel.HIGH.name: self.volatility_thresholds[VolatilityLevel.VERY_HIGH.name],
                EmergencyLevel.CRITICAL.name: self.volatility_thresholds[VolatilityLevel.EXTREME.name]
            },
            "price_gap": {
                EmergencyLevel.LOW.name: 0.02,  # 2% gap
                EmergencyLevel.MEDIUM.name: 0.05,  # 5% gap
                EmergencyLevel.HIGH.name: 0.1,  # 10% gap
                EmergencyLevel.CRITICAL.name: 0.2  # 20% gap
            },
            "volume_spike": {
                EmergencyLevel.LOW.name: 3.0,  # 3x normal volume
                EmergencyLevel.MEDIUM.name: 5.0,  # 5x normal volume
                EmergencyLevel.HIGH.name: 10.0,  # 10x normal volume
                EmergencyLevel.CRITICAL.name: 20.0  # 20x normal volume
            },
            "spread_widening": {
                EmergencyLevel.LOW.name: 3.0,  # 3x normal spread
                EmergencyLevel.MEDIUM.name: 5.0,  # 5x normal spread
                EmergencyLevel.HIGH.name: 10.0,  # 10x normal spread
                EmergencyLevel.CRITICAL.name: 20.0  # 20x normal spread
            }
        })
        
        # Monitoring intervals
        self.update_interval_seconds = self.config.get("update_interval_seconds", 1.0)
        self.analysis_interval_seconds = self.config.get("analysis_interval_seconds", 5.0)
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant events."""
        # Register for market data events
        self.event_system.subscribe("market_data", self._handle_market_data)
        self.event_system.subscribe("order_book", self._handle_order_book)
        self.event_system.subscribe("trade", self._handle_trade)
        self.event_system.subscribe("bar", self._handle_bar)
        
        # Register for market status events
        self.event_system.subscribe("market_status", self._handle_market_status)
        
        # Register for news events
        self.event_system.subscribe("news", self._handle_news)
    
    def start(self) -> None:
        """Start the market monitor."""
        with self.lock:
            if not self.running:
                self.running = True
                
                # Start analysis thread
                self.analysis_thread = threading.Thread(
                    target=self._analysis_loop,
                    daemon=True
                )
                self.analysis_thread.start()
                
                logger.info("Market monitor started")
    
    def stop(self) -> None:
        """Stop the market monitor."""
        with self.lock:
            self.running = False
            if hasattr(self, "analysis_thread") and self.analysis_thread.is_alive():
                self.analysis_thread.join(timeout=5.0)
            logger.info("Market monitor stopped")
    
    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to monitor.
        
        Args:
            symbol: Symbol to monitor
        """
        with self.lock:
            if symbol not in self.monitored_symbols:
                self.monitored_symbols.add(symbol)
                self.metrics[symbol] = MarketMetrics(symbol)
                self.historical_metrics[symbol] = {
                    "prices": deque(maxlen=self.price_history_size),
                    "volumes": deque(maxlen=self.volume_history_size),
                    "spreads": deque(maxlen=self.volume_history_size),
                    "volatilities": deque(maxlen=self.volatility_history_size)
                }
                logger.info(f"Added symbol {symbol} to market monitor")
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from monitoring.
        
        Args:
            symbol: Symbol to remove
        """
        with self.lock:
            if symbol in self.monitored_symbols:
                self.monitored_symbols.remove(symbol)
                if symbol in self.metrics:
                    del self.metrics[symbol]
                if symbol in self.historical_metrics:
                    del self.historical_metrics[symbol]
                logger.info(f"Removed symbol {symbol} from market monitor")
    
    def get_market_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market metrics for a symbol.
        
        Args:
            symbol: Symbol to get metrics for
            
        Returns:
            Dict[str, Any]: Market metrics as dictionary or None if not found
        """
        with self.lock:
            if symbol in self.metrics:
                return self.metrics[symbol].to_dict()
            return None
    
    def get_all_market_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get current market metrics for all monitored symbols.
        
        Returns:
            Dict[str, Dict[str, Any]]: Market metrics for all symbols
        """
        with self.lock:
            return {symbol: metrics.to_dict() for symbol, metrics in self.metrics.items()}
    
    def get_market_condition(self, symbol: str) -> Optional[MarketCondition]:
        """Get current market condition for a symbol.
        
        Args:
            symbol: Symbol to get condition for
            
        Returns:
            MarketCondition: Current market condition or None if not found
        """
        with self.lock:
            if symbol in self.metrics:
                return self.metrics[symbol].condition
            return None
    
    def _analysis_loop(self) -> None:
        """Background thread for periodic market analysis."""
        last_analysis_time = time.time()
        
        while self.running:
            time.sleep(0.1)  # Short sleep to prevent CPU hogging
            
            current_time = time.time()
            if current_time - last_analysis_time >= self.analysis_interval_seconds:
                self._analyze_market_conditions()
                last_analysis_time = current_time
    
    def _analyze_market_conditions(self) -> None:
        """Analyze current market conditions for all monitored symbols."""
        with self.lock:
            for symbol in list(self.monitored_symbols):
                if symbol not in self.metrics or symbol not in self.historical_metrics:
                    continue
                
                metrics = self.metrics[symbol]
                history = self.historical_metrics[symbol]
                
                # Skip if we don't have enough data yet
                if (len(history["prices"]) < 5 or 
                    len(history["volumes"]) < 3 or 
                    len(history["spreads"]) < 3):
                    continue
                
                # Analyze volatility
                self._analyze_volatility(symbol, metrics, history)
                
                # Analyze price movements
                self._analyze_price_movements(symbol, metrics, history)
                
                # Analyze volume
                self._analyze_volume(symbol, metrics, history)
                
                # Analyze liquidity
                self._analyze_liquidity(symbol, metrics, history)
                
                # Determine overall market condition
                self._determine_market_condition(symbol, metrics)
                
                # Check for emergency conditions
                self._check_emergency_conditions(symbol, metrics, history)
                
                # Publish updated metrics
                self.event_system.publish("market_metrics_updated", {
                    "symbol": symbol,
                    "metrics": metrics.to_dict()
                })
    
    def _analyze_volatility(self, symbol: str, metrics: MarketMetrics, history: Dict[str, deque]) -> None:
        """Analyze volatility for a symbol.
        
        Args:
            symbol: Symbol to analyze
            metrics: Current metrics for the symbol
            history: Historical metrics for the symbol
        """
        # Calculate current volatility (standard deviation of returns)
        prices = list(history["prices"])
        if len(prices) >= 2:
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:  # Avoid division by zero
                    returns.append((prices[i] - prices[i-1]) / prices[i-1])
            
            if returns:
                current_volatility = np.std(returns) * np.sqrt(252 * 390)  # Annualized volatility
                metrics.volatility = current_volatility
                
                # Store in history
                history["volatilities"].append(current_volatility)
                
                # Calculate average volatility
                avg_volatility = np.mean(list(history["volatilities"]))
                
                # Determine volatility level
                if avg_volatility > 0:  # Avoid division by zero
                    relative_volatility = current_volatility / avg_volatility
                    
                    if relative_volatility >= self.volatility_thresholds[VolatilityLevel.EXTREME.name]:
                        metrics.volatility_level = VolatilityLevel.EXTREME
                    elif relative_volatility >= self.volatility_thresholds[VolatilityLevel.VERY_HIGH.name]:
                        metrics.volatility_level = VolatilityLevel.VERY_HIGH
                    elif relative_volatility >= self.volatility_thresholds[VolatilityLevel.HIGH.name]:
                        metrics.volatility_level = VolatilityLevel.HIGH
                    elif relative_volatility >= self.volatility_thresholds[VolatilityLevel.ELEVATED.name]:
                        metrics.volatility_level = VolatilityLevel.ELEVATED
                    elif relative_volatility <= self.volatility_thresholds[VolatilityLevel.VERY_LOW.name]:
                        metrics.volatility_level = VolatilityLevel.VERY_LOW
                    elif relative_volatility <= self.volatility_thresholds[VolatilityLevel.LOW.name]:
                        metrics.volatility_level = VolatilityLevel.LOW
                    else:
                        metrics.volatility_level = VolatilityLevel.NORMAL
    
    def _analyze_price_movements(self, symbol: str, metrics: MarketMetrics, history: Dict[str, deque]) -> None:
        """Analyze price movements for a symbol.
        
        Args:
            symbol: Symbol to analyze
            metrics: Current metrics for the symbol
            history: Historical metrics for the symbol
        """
        prices = list(history["prices"])
        if len(prices) >= 2:
            # Check for price gaps
            metrics.price_gaps = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:  # Avoid division by zero
                    price_change = abs(prices[i] - prices[i-1]) / prices[i-1]
                    if price_change >= self.price_gap_threshold:
                        metrics.price_gaps.append(price_change)
            
            # Check for price spikes (rapid movements followed by reversals)
            metrics.price_spikes = []
            if len(prices) >= 3:
                for i in range(1, len(prices) - 1):
                    if prices[i-1] > 0 and prices[i] > 0:  # Avoid division by zero
                        change1 = (prices[i] - prices[i-1]) / prices[i-1]
                        change2 = (prices[i+1] - prices[i]) / prices[i]
                        
                        # If changes are in opposite directions and both significant
                        if (change1 * change2 < 0 and 
                            abs(change1) >= self.price_spike_threshold and 
                            abs(change2) >= self.price_spike_threshold):
                            metrics.price_spikes.append(max(abs(change1), abs(change2)))
    
    def _analyze_volume(self, symbol: str, metrics: MarketMetrics, history: Dict[str, deque]) -> None:
        """Analyze volume for a symbol.
        
        Args:
            symbol: Symbol to analyze
            metrics: Current metrics for the symbol
            history: Historical metrics for the symbol
        """
        volumes = list(history["volumes"])
        if len(volumes) >= 2 and metrics.volume is not None:
            # Calculate average volume
            avg_volume = np.mean(volumes[:-1])  # Exclude current volume
            metrics.avg_volume = avg_volume
            
            # Calculate relative volume
            if avg_volume > 0:  # Avoid division by zero
                relative_volume = metrics.volume / avg_volume
                metrics.relative_volume = relative_volume
                
                # Check for volume spikes
                if relative_volume >= self.volume_spike_threshold:
                    metrics.unusual_volume_spikes.append(relative_volume)
    
    def _analyze_liquidity(self, symbol: str, metrics: MarketMetrics, history: Dict[str, deque]) -> None:
        """Analyze liquidity for a symbol.
        
        Args:
            symbol: Symbol to analyze
            metrics: Current metrics for the symbol
            history: Historical metrics for the symbol
        """
        spreads = list(history["spreads"])
        if len(spreads) >= 2 and metrics.spread is not None:
            # Calculate average spread
            avg_spread = np.mean(spreads[:-1])  # Exclude current spread
            
            # Calculate relative spread
            if avg_spread > 0:  # Avoid division by zero
                relative_spread = metrics.spread / avg_spread
                metrics.relative_spread = relative_spread
                
                # Check for spread widening
                if relative_spread >= self.spread_widening_threshold:
                    metrics.unusual_spread_widening.append(relative_spread)
                
                # Determine liquidity level based on spread
                if relative_spread >= 3.0:
                    metrics.liquidity_level = LiquidityLevel.VERY_LOW
                elif relative_spread >= 2.0:
                    metrics.liquidity_level = LiquidityLevel.LOW
                elif relative_spread <= 0.5:
                    metrics.liquidity_level = LiquidityLevel.VERY_HIGH
                elif relative_spread <= 0.75:
                    metrics.liquidity_level = LiquidityLevel.HIGH
                else:
                    metrics.liquidity_level = LiquidityLevel.NORMAL
    
    def _determine_market_condition(self, symbol: str, metrics: MarketMetrics) -> None:
        """Determine overall market condition for a symbol.
        
        Args:
            symbol: Symbol to analyze
            metrics: Current metrics for the symbol
        """
        # Check if market is halted or closed
        if metrics.is_halted:
            metrics.condition = MarketCondition.HALTED
            return
        
        if not metrics.is_open:
            metrics.condition = MarketCondition.CLOSED
            return
        
        # Check for news impact
        if metrics.recent_news_impact:
            metrics.condition = MarketCondition.NEWS_IMPACT
            return
        
        # Check for extreme volatility
        if metrics.volatility_level == VolatilityLevel.EXTREME:
            metrics.condition = MarketCondition.HIGHLY_VOLATILE
            return
        
        # Check for high volatility
        if metrics.volatility_level in [VolatilityLevel.VERY_HIGH, VolatilityLevel.HIGH]:
            metrics.condition = MarketCondition.VOLATILE
            return
        
        # Check for price gaps
        if metrics.price_gaps and max(metrics.price_gaps) >= 0.05:  # 5% gap
            metrics.condition = MarketCondition.GAPPING
            return
        
        # Check for fast market (high volatility and high volume)
        if (metrics.volatility_level in [VolatilityLevel.ELEVATED, VolatilityLevel.HIGH] and
            metrics.relative_volume is not None and metrics.relative_volume >= 2.0):
            metrics.condition = MarketCondition.FAST_MARKET
            return
        
        # Check for illiquid market
        if metrics.liquidity_level in [LiquidityLevel.VERY_LOW, LiquidityLevel.LOW]:
            metrics.condition = MarketCondition.ILLIQUID
            return
        
        # Check for slow market (low volatility and low volume)
        if (metrics.volatility_level in [VolatilityLevel.VERY_LOW, VolatilityLevel.LOW] and
            metrics.relative_volume is not None and metrics.relative_volume <= 0.5):
            metrics.condition = MarketCondition.SLOW_MARKET
            return
        
        # Default to normal
        metrics.condition = MarketCondition.NORMAL
    
    def _check_emergency_conditions(self, symbol: str, metrics: MarketMetrics, history: Dict[str, deque]) -> None:
        """Check for emergency conditions that require immediate action.
        
        Args:
            symbol: Symbol to analyze
            metrics: Current metrics for the symbol
            history: Historical metrics for the symbol
        """
        if not self.emergency_callback:
            return
        
        # Check for extreme volatility
        if metrics.volatility is not None and metrics.volatility_level in [VolatilityLevel.VERY_HIGH, VolatilityLevel.EXTREME]:
            # Determine emergency level based on volatility
            emergency_level = None
            for level_name, threshold in self.emergency_thresholds["volatility"].items():
                if metrics.volatility >= threshold:
                    emergency_level = EmergencyLevel[level_name]
            
            if emergency_level:
                self.emergency_callback(
                    EmergencyTrigger.MARKET_VOLATILITY,
                    emergency_level,
                    f"Extreme volatility detected for {symbol}",
                    EmergencyAction.THROTTLE if emergency_level.value < EmergencyLevel.HIGH.value else EmergencyAction.PAUSE_NEW_ORDERS,
                    [],  # affected_markets
                    [symbol],  # affected_symbols
                    {"volatility": metrics.volatility, "volatility_level": metrics.volatility_level.name}
                )
        
        # Check for large price gaps
        if metrics.price_gaps:
            max_gap = max(metrics.price_gaps)
            emergency_level = None
            for level_name, threshold in self.emergency_thresholds["price_gap"].items():
                if max_gap >= threshold:
                    emergency_level = EmergencyLevel[level_name]
            
            if emergency_level:
                self.emergency_callback(
                    EmergencyTrigger.MARKET_VOLATILITY,
                    emergency_level,
                    f"Large price gap detected for {symbol} ({max_gap:.2%})",
                    EmergencyAction.THROTTLE if emergency_level.value < EmergencyLevel.HIGH.value else EmergencyAction.PAUSE_NEW_ORDERS,
                    [],  # affected_markets
                    [symbol],  # affected_symbols
                    {"price_gap": max_gap}
                )
        
        # Check for unusual volume spikes
        if metrics.unusual_volume_spikes and metrics.relative_volume is not None:
            max_volume_spike = metrics.relative_volume
            emergency_level = None
            for level_name, threshold in self.emergency_thresholds["volume_spike"].items():
                if max_volume_spike >= threshold:
                    emergency_level = EmergencyLevel[level_name]
            
            if emergency_level:
                self.emergency_callback(
                    EmergencyTrigger.MARKET_VOLATILITY,
                    emergency_level,
                    f"Unusual volume spike detected for {symbol} ({max_volume_spike:.2f}x normal)",
                    EmergencyAction.MONITOR if emergency_level.value < EmergencyLevel.MEDIUM.value else EmergencyAction.THROTTLE,
                    [],  # affected_markets
                    [symbol],  # affected_symbols
                    {"volume_spike": max_volume_spike}
                )
        
        # Check for unusual spread widening
        if metrics.unusual_spread_widening and metrics.relative_spread is not None:
            max_spread_widening = metrics.relative_spread
            emergency_level = None
            for level_name, threshold in self.emergency_thresholds["spread_widening"].items():
                if max_spread_widening >= threshold:
                    emergency_level = EmergencyLevel[level_name]
            
            if emergency_level:
                self.emergency_callback(
                    EmergencyTrigger.LIQUIDITY_ISSUE,
                    emergency_level,
                    f"Unusual spread widening detected for {symbol} ({max_spread_widening:.2f}x normal)",
                    EmergencyAction.MONITOR if emergency_level.value < EmergencyLevel.MEDIUM.value else EmergencyAction.THROTTLE,
                    [],  # affected_markets
                    [symbol],  # affected_symbols
                    {"spread_widening": max_spread_widening}
                )
    
    # Event handlers
    
    def _handle_market_data(self, event_data: Dict[str, Any]) -> None:
        """Handle market data events.
        
        Args:
            event_data: Event data
        """
        symbol = event_data.get("symbol")
        if not symbol or symbol not in self.monitored_symbols:
            return
        
        with self.lock:
            if symbol not in self.metrics:
                self.metrics[symbol] = MarketMetrics(symbol)
            
            metrics = self.metrics[symbol]
            
            # Update price metrics
            if "last" in event_data:
                metrics.last_price = event_data["last"]
                
                # Add to price history
                if symbol in self.historical_metrics:
                    self.historical_metrics[symbol]["prices"].append(metrics.last_price)
            
            if "bid" in event_data:
                metrics.bid_price = event_data["bid"]
            
            if "ask" in event_data:
                metrics.ask_price = event_data["ask"]
            
            # Calculate mid price and spread
            if metrics.bid_price is not None and metrics.ask_price is not None:
                metrics.mid_price = (metrics.bid_price + metrics.ask_price) / 2.0
                metrics.spread = metrics.ask_price - metrics.bid_price
                
                # Add to spread history
                if symbol in self.historical_metrics:
                    self.historical_metrics[symbol]["spreads"].append(metrics.spread)
            
            # Update timestamp
            metrics.timestamp = datetime.now(pytz.UTC)
    
    def _handle_order_book(self, event_data: Dict[str, Any]) -> None:
        """Handle order book events.
        
        Args:
            event_data: Event data
        """
        symbol = event_data.get("symbol")
        if not symbol or symbol not in self.monitored_symbols:
            return
        
        with self.lock:
            if symbol not in self.metrics:
                self.metrics[symbol] = MarketMetrics(symbol)
            
            metrics = self.metrics[symbol]
            
            # Calculate order book depth
            bids = event_data.get("bids", [])
            asks = event_data.get("asks", [])
            
            bid_depth = sum(bid[1] for bid in bids) if bids else 0
            ask_depth = sum(ask[1] for ask in asks) if asks else 0
            
            metrics.depth = bid_depth + ask_depth
            
            # Update timestamp
            metrics.timestamp = datetime.now(pytz.UTC)
    
    def _handle_trade(self, event_data: Dict[str, Any]) -> None:
        """Handle trade events.
        
        Args:
            event_data: Event data
        """
        symbol = event_data.get("symbol")
        if not symbol or symbol not in self.monitored_symbols:
            return
        
        with self.lock:
            if symbol not in self.metrics:
                self.metrics[symbol] = MarketMetrics(symbol)
            
            metrics = self.metrics[symbol]
            
            # Update price and volume
            if "price" in event_data:
                metrics.last_price = event_data["price"]
                
                # Add to price history
                if symbol in self.historical_metrics:
                    self.historical_metrics[symbol]["prices"].append(metrics.last_price)
            
            if "volume" in event_data:
                # Accumulate volume
                if metrics.volume is None:
                    metrics.volume = event_data["volume"]
                else:
                    metrics.volume += event_data["volume"]
            
            # Update timestamp
            metrics.timestamp = datetime.now(pytz.UTC)
    
    def _handle_bar(self, event_data: Dict[str, Any]) -> None:
        """Handle bar (OHLCV) events.
        
        Args:
            event_data: Event data
        """
        symbol = event_data.get("symbol")
        if not symbol or symbol not in self.monitored_symbols:
            return
        
        with self.lock:
            if symbol not in self.metrics:
                self.metrics[symbol] = MarketMetrics(symbol)
            
            metrics = self.metrics[symbol]
            
            # Update OHLCV data
            if "open" in event_data:
                metrics.open_price = event_data["open"]
            
            if "high" in event_data:
                metrics.high_price = event_data["high"]
            
            if "low" in event_data:
                metrics.low_price = event_data["low"]
            
            if "close" in event_data:
                metrics.close_price = event_data["close"]
                metrics.last_price = event_data["close"]
                
                # Add to price history
                if symbol in self.historical_metrics:
                    self.historical_metrics[symbol]["prices"].append(metrics.close_price)
            
            if "volume" in event_data:
                metrics.volume = event_data["volume"]
                
                # Add to volume history
                if symbol in self.historical_metrics:
                    self.historical_metrics[symbol]["volumes"].append(metrics.volume)
            
            if "vwap" in event_data:
                metrics.vwap = event_data["vwap"]
            
            # Reset accumulated values for the new bar
            if event_data.get("is_new_bar", False):
                metrics.price_gaps = []
                metrics.price_spikes = []
                metrics.unusual_volume_spikes = []
                metrics.unusual_spread_widening = []
            
            # Update timestamp
            metrics.timestamp = datetime.now(pytz.UTC)
    
    def _handle_market_status(self, event_data: Dict[str, Any]) -> None:
        """Handle market status events.
        
        Args:
            event_data: Event data
        """
        symbol = event_data.get("symbol")
        if not symbol:
            return
        
        status = event_data.get("status")
        if status is None:
            return
        
        with self.lock:
            # Add symbol if not already monitored
            if symbol not in self.monitored_symbols and event_data.get("auto_add", False):
                self.add_symbol(symbol)
            
            if symbol in self.metrics:
                metrics = self.metrics[symbol]
                
                # Update market status
                if status == "halted":
                    metrics.is_halted = True
                    metrics.is_open = False
                    metrics.condition = MarketCondition.HALTED
                
                elif status == "closed":
                    metrics.is_halted = False
                    metrics.is_open = False
                    metrics.condition = MarketCondition.CLOSED
                
                elif status == "pre_open" or status == "opening":
                    metrics.is_halted = False
                    metrics.is_open = True
                    metrics.condition = MarketCondition.OPENING
                
                elif status == "closing":
                    metrics.is_halted = False
                    metrics.is_open = True
                    metrics.condition = MarketCondition.CLOSING
                
                elif status == "open":
                    metrics.is_halted = False
                    metrics.is_open = True
                    # Don't set condition here, let the analysis determine it
                
                # Update timestamp
                metrics.timestamp = datetime.now(pytz.UTC)
    
    def _handle_news(self, event_data: Dict[str, Any]) -> None:
        """Handle news events.
        
        Args:
            event_data: Event data
        """
        affected_symbols = event_data.get("affected_symbols", [])
        if not affected_symbols:
            return
        
        sentiment = event_data.get("sentiment")
        importance = event_data.get("importance", 0.0)
        
        with self.lock:
            for symbol in affected_symbols:
                if symbol in self.metrics:
                    metrics = self.metrics[symbol]
                    
                    # Update news impact
                    if importance >= 0.7:  # High importance news
                        metrics.recent_news_impact = True
                        metrics.news_sentiment = sentiment
                        
                        # Set condition to news impact
                        metrics.condition = MarketCondition.NEWS_IMPACT
                    
                    # Update timestamp
                    metrics.timestamp = datetime.now(pytz.UTC)


# Factory function to create market monitor
def create_market_monitor(
    event_system: EventSystem,
    config: Dict[str, Any] = None,
    emergency_handler = None
) -> MarketMonitor:
    """Create a market monitor with the specified configuration.
    
    Args:
        event_system: Event system for publishing and subscribing to events
        config: Configuration for the market monitor
        emergency_handler: Emergency handler for market emergencies
        
    Returns:
        MarketMonitor: Configured market monitor
    """
    # Create emergency callback if emergency handler is provided
    emergency_callback = None
    if emergency_handler is not None:
        emergency_callback = emergency_handler.declare_emergency
    
    # Create market monitor
    monitor = MarketMonitor(event_system, config, emergency_callback)
    
    # Start the monitor
    monitor.start()
    
    return monitor