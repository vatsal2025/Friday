import logging
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Base Risk Manager class that provides signal adjustment functionality.
    
    This class serves as a simple implementation of a risk manager that can
    adjust trading signals based on risk parameters. It can be used directly
    or extended by more complex risk management implementations.
    """
    
    def __init__(self, 
                 max_signal_strength: float = 1.0,
                 min_signal_strength: float = 0.1,
                 risk_adjustment_factor: float = 1.0,
                 enable_signal_filtering: bool = True):
        """
        Initialize the RiskManager.
        
        Args:
            max_signal_strength: Maximum allowed signal strength (default: 1.0)
            min_signal_strength: Minimum signal strength to consider valid (default: 0.1)
            risk_adjustment_factor: Factor to adjust signal strength (default: 1.0)
            enable_signal_filtering: Whether to enable signal filtering (default: True)
        """
        self.max_signal_strength = max_signal_strength
        self.min_signal_strength = min_signal_strength
        self.risk_adjustment_factor = risk_adjustment_factor
        self.enable_signal_filtering = enable_signal_filtering
        
        # Track adjusted signals for monitoring
        self.adjusted_signals = []
        
        logger.info(f"Initialized RiskManager with adjustment factor: {risk_adjustment_factor}")
    
    def adjust_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust a trading signal based on risk parameters.
        
        This method applies risk-based adjustments to trading signals before they
        are processed by the trading engine. It can modify signal strength,
        filter out weak signals, or apply other risk-based transformations.
        
        Args:
            signal: A dictionary containing signal information including:
                - symbol: The asset symbol
                - direction: Trade direction ("long", "short", or "neutral")
                - confidence: Signal confidence/strength (0.0 to 1.0)
                - timestamp: Signal generation time
                - metadata: Additional signal information
        
        Returns:
            The adjusted signal dictionary
        """
        if not signal or not isinstance(signal, dict):
            logger.warning("Invalid signal format provided to risk manager")
            return signal
        
        # Create a copy to avoid modifying the original
        adjusted_signal = signal.copy()
        
        # Extract key signal components
        symbol = adjusted_signal.get('symbol')
        confidence = adjusted_signal.get('confidence', 0.0)
        direction = adjusted_signal.get('direction', 'neutral')
        
        # Apply risk adjustment factor to confidence
        if confidence is not None and isinstance(confidence, (int, float)):
            original_confidence = confidence
            adjusted_confidence = confidence * self.risk_adjustment_factor
            
            # Ensure confidence stays within bounds
            adjusted_confidence = min(adjusted_confidence, self.max_signal_strength)
            adjusted_confidence = max(adjusted_confidence, 0.0)
            
            adjusted_signal['confidence'] = adjusted_confidence
            
            # Log the adjustment
            logger.debug(f"Adjusted signal confidence for {symbol} from {original_confidence:.4f} to {adjusted_confidence:.4f}")
            
            # Filter out weak signals if enabled
            if self.enable_signal_filtering and adjusted_confidence < self.min_signal_strength:
                logger.info(f"Filtered out weak signal for {symbol} (confidence: {adjusted_confidence:.4f})")
                adjusted_signal['direction'] = 'neutral'
                adjusted_signal['filtered_reason'] = 'low_confidence'
        
        # Track this adjustment for monitoring
        self._record_adjustment(symbol, direction, confidence, adjusted_signal.get('confidence', 0.0))
        
        return adjusted_signal
    
    def _record_adjustment(self, symbol: str, direction: str, 
                          original_confidence: float, adjusted_confidence: float) -> None:
        """
        Record signal adjustment for monitoring and analysis.
        
        Args:
            symbol: Asset symbol
            direction: Trade direction
            original_confidence: Original signal confidence
            adjusted_confidence: Adjusted signal confidence
        """
        self.adjusted_signals.append({
            'symbol': symbol,
            'direction': direction,
            'original_confidence': original_confidence,
            'adjusted_confidence': adjusted_confidence,
            'adjustment_factor': self.risk_adjustment_factor,
            'timestamp': datetime.now()
        })
        
        # Keep only the last 1000 adjustments
        if len(self.adjusted_signals) > 1000:
            self.adjusted_signals = self.adjusted_signals[-1000:]
    
    def set_risk_adjustment_factor(self, factor: float) -> None:
        """
        Update the risk adjustment factor.
        
        Args:
            factor: New risk adjustment factor
        """
        self.risk_adjustment_factor = factor
        logger.info(f"Updated risk adjustment factor to {factor}")
    
    def get_adjustment_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get history of signal adjustments.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of adjustment records
        """
        return self.adjusted_signals[-limit:]
    
    def reset(self) -> None:
        """
        Reset the risk manager state.
        """
        self.adjusted_signals = []
        logger.info("Reset RiskManager state")