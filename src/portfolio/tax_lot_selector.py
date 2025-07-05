import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from enum import Enum

from portfolio.tax_manager import TaxManager, TaxLot, TaxLotMethod

logger = logging.getLogger(__name__)

class TaxLotSelectionStrategy(Enum):
    """Strategies for tax lot selection."""
    FIFO = "First-In-First-Out"  # Oldest lots sold first
    LIFO = "Last-In-First-Out"   # Newest lots sold first
    HIFO = "Highest-In-First-Out"  # Highest cost lots sold first
    LOFO = "Lowest-In-First-Out"   # Lowest cost lots sold first
    SPECIFIC = "Specific Identification"  # Manually specified lots
    MIN_TAX = "Minimize Tax Impact"  # Minimize taxes
    MAX_TAX = "Maximize Tax Impact"  # Maximize tax losses
    TAX_EFFICIENT = "Tax Efficient"  # Balance between tax minimization and other factors

class TaxLotSelector:
    """
    Tax Lot Selector for sophisticated tax-lot selection strategies.
    
    This class provides functionality for:
    - Selecting tax lots using various strategies
    - Calculating tax impact of different selection methods
    - Previewing tax lot selection results
    - Implementing custom selection rules
    """
    
    def __init__(self, 
                 tax_manager: TaxManager,
                 default_strategy: TaxLotSelectionStrategy = TaxLotSelectionStrategy.FIFO,
                 short_term_tax_rate: float = 0.35,
                 long_term_tax_rate: float = 0.15,
                 tax_loss_preference_factor: float = 0.5):
        """
        Initialize the Tax Lot Selector.
        
        Args:
            tax_manager: The tax manager instance
            default_strategy: Default tax lot selection strategy
            short_term_tax_rate: Tax rate for short-term gains
            long_term_tax_rate: Tax rate for long-term gains
            tax_loss_preference_factor: Factor to prefer tax losses (0-1)
        """
        self.tax_manager = tax_manager
        self.default_strategy = default_strategy
        self.short_term_tax_rate = short_term_tax_rate
        self.long_term_tax_rate = long_term_tax_rate
        self.tax_loss_preference_factor = tax_loss_preference_factor
        
        # Symbol-specific strategies
        self.symbol_strategies = {}
        
        # Selection history
        self.selection_history = []
        
        logger.info(f"Tax Lot Selector initialized with {default_strategy.value} strategy")
    
    def set_strategy(self, symbol: str, strategy: TaxLotSelectionStrategy) -> None:
        """
        Set the tax lot selection strategy for a specific symbol.
        
        Args:
            symbol: Security symbol
            strategy: Tax lot selection strategy
        """
        self.symbol_strategies[symbol] = strategy
        logger.info(f"Set tax lot selection strategy for {symbol} to {strategy.value}")
    
    def select_lots(self,
                   symbol: str,
                   quantity: float,
                   current_price: float,
                   strategy: Optional[TaxLotSelectionStrategy] = None,
                   specific_lots: Optional[List[str]] = None,
                   timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Select tax lots for a sale using the specified strategy.
        
        Args:
            symbol: Security symbol
            quantity: Number of shares/units to sell
            current_price: Current price per share/unit
            strategy: Tax lot selection strategy (default: use symbol's strategy or default)
            specific_lots: List of lot IDs for specific identification strategy
            timestamp: Timestamp for the selection (default: now)
            
        Returns:
            Dict with selected lots and tax impact details
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Determine the strategy to use
        if strategy is None:
            strategy = self.symbol_strategies.get(symbol, self.default_strategy)
            
        # Get tax lots for this symbol
        tax_lots = self.tax_manager.get_tax_lots(symbol=symbol)
        
        if not tax_lots:
            return {
                "error": f"No tax lots found for {symbol}",
                "symbol": symbol,
                "quantity": quantity,
                "current_price": current_price,
                "strategy": strategy.value,
                "timestamp": timestamp
            }
            
        # Convert to list of TaxLot objects for easier manipulation
        lots = []
        for lot_dict in tax_lots:
            lot = TaxLot(
                quantity=lot_dict["quantity"],
                purchase_price=lot_dict["purchase_price"],
                purchase_date=lot_dict["purchase_date"],
                lot_id=lot_dict["lot_id"]
            )
            lots.append(lot)
            
        # Select lots based on the strategy
        selected_lots = []
        remaining_quantity = quantity
        
        if strategy == TaxLotSelectionStrategy.SPECIFIC and specific_lots:
            # Specific lot selection
            for lot_id in specific_lots:
                if remaining_quantity <= 0:
                    break
                    
                # Find the lot with this ID
                matching_lots = [lot for lot in lots if lot.lot_id == lot_id]
                
                if not matching_lots:
                    continue
                    
                lot = matching_lots[0]
                
                # Determine how much to sell from this lot
                sell_quantity = min(lot.quantity, remaining_quantity)
                remaining_quantity -= sell_quantity
                
                # Add to selected lots
                selected_lots.append({
                    "lot_id": lot.lot_id,
                    "quantity": sell_quantity,
                    "purchase_price": lot.purchase_price,
                    "purchase_date": lot.purchase_date,
                    "holding_period_days": (timestamp - lot.purchase_date).days,
                    "long_term": (timestamp - lot.purchase_date).days > 365,
                    "cost_basis": sell_quantity * lot.purchase_price,
                    "market_value": sell_quantity * current_price,
                    "gain_loss": sell_quantity * (current_price - lot.purchase_price)
                })
        else:
            # Sort lots based on the strategy
            sorted_lots = self._sort_lots_by_strategy(lots, strategy, current_price, timestamp)
            
            # Select lots in order
            for lot in sorted_lots:
                if remaining_quantity <= 0:
                    break
                    
                # Determine how much to sell from this lot
                sell_quantity = min(lot.quantity, remaining_quantity)
                remaining_quantity -= sell_quantity
                
                # Add to selected lots
                selected_lots.append({
                    "lot_id": lot.lot_id,
                    "quantity": sell_quantity,
                    "purchase_price": lot.purchase_price,
                    "purchase_date": lot.purchase_date,
                    "holding_period_days": (timestamp - lot.purchase_date).days,
                    "long_term": (timestamp - lot.purchase_date).days > 365,
                    "cost_basis": sell_quantity * lot.purchase_price,
                    "market_value": sell_quantity * current_price,
                    "gain_loss": sell_quantity * (current_price - lot.purchase_price)
                })
        
        # Calculate tax impact
        tax_impact = self._calculate_tax_impact(selected_lots)
        
        # Record the selection
        selection_record = {
            "symbol": symbol,
            "quantity": quantity - remaining_quantity,
            "current_price": current_price,
            "strategy": strategy.value,
            "timestamp": timestamp,
            "selected_lots": selected_lots,
            "tax_impact": tax_impact,
            "remaining_quantity": remaining_quantity
        }
        
        self.selection_history.append(selection_record)
        
        return selection_record
    
    def compare_strategies(self,
                         symbol: str,
                         quantity: float,
                         current_price: float,
                         timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Compare different tax lot selection strategies for a potential sale.
        
        Args:
            symbol: Security symbol
            quantity: Number of shares/units to sell
            current_price: Current price per share/unit
            timestamp: Timestamp for the comparison (default: now)
            
        Returns:
            Dict with comparison results for different strategies
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Get results for each strategy
        results = {}
        
        for strategy in TaxLotSelectionStrategy:
            # Skip SPECIFIC strategy as it requires lot IDs
            if strategy == TaxLotSelectionStrategy.SPECIFIC:
                continue
                
            result = self.select_lots(
                symbol=symbol,
                quantity=quantity,
                current_price=current_price,
                strategy=strategy,
                timestamp=timestamp
            )
            
            # Remove selected_lots detail to keep the comparison concise
            if "selected_lots" in result:
                del result["selected_lots"]
                
            results[strategy.value] = result
        
        # Find the optimal strategy based on after-tax proceeds
        optimal_strategy = None
        max_after_tax = float('-inf')
        
        for strategy_name, result in results.items():
            if "error" in result:
                continue
                
            after_tax = result.get("tax_impact", {}).get("after_tax_proceeds", 0)
            
            if after_tax > max_after_tax:
                max_after_tax = after_tax
                optimal_strategy = strategy_name
        
        return {
            "symbol": symbol,
            "quantity": quantity,
            "current_price": current_price,
            "timestamp": timestamp,
            "results": results,
            "optimal_strategy": optimal_strategy
        }
    
    def get_selection_history(self,
                            symbol: Optional[str] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get tax lot selection history with optional filtering.
        
        Args:
            symbol: Filter by symbol (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)
            
        Returns:
            List of selection history entries
        """
        filtered_history = self.selection_history
        
        if symbol:
            filtered_history = [h for h in filtered_history if h["symbol"] == symbol]
            
        if start_date:
            filtered_history = [h for h in filtered_history if h["timestamp"] >= start_date]
            
        if end_date:
            filtered_history = [h for h in filtered_history if h["timestamp"] <= end_date]
            
        return filtered_history
    
    def _sort_lots_by_strategy(self, 
                             lots: List[TaxLot], 
                             strategy: TaxLotSelectionStrategy,
                             current_price: float,
                             timestamp: datetime) -> List[TaxLot]:
        """
        Sort tax lots based on the selected strategy.
        
        Args:
            lots: List of tax lots
            strategy: Tax lot selection strategy
            current_price: Current price per share/unit
            timestamp: Timestamp for the sorting
            
        Returns:
            Sorted list of tax lots
        """
        if strategy == TaxLotSelectionStrategy.FIFO:
            # Oldest first
            return sorted(lots, key=lambda lot: lot.purchase_date)
            
        elif strategy == TaxLotSelectionStrategy.LIFO:
            # Newest first
            return sorted(lots, key=lambda lot: lot.purchase_date, reverse=True)
            
        elif strategy == TaxLotSelectionStrategy.HIFO:
            # Highest cost first
            return sorted(lots, key=lambda lot: lot.purchase_price, reverse=True)
            
        elif strategy == TaxLotSelectionStrategy.LOFO:
            # Lowest cost first
            return sorted(lots, key=lambda lot: lot.purchase_price)
            
        elif strategy == TaxLotSelectionStrategy.MIN_TAX:
            # Minimize tax impact
            # Calculate tax impact for each lot
            lots_with_tax = []
            
            for lot in lots:
                gain_loss = (current_price - lot.purchase_price) * lot.quantity
                holding_period = (timestamp - lot.purchase_date).days
                long_term = holding_period > 365
                
                if gain_loss <= 0:  # Loss or break-even
                    tax_impact = gain_loss  # Tax benefit from loss
                else:  # Gain
                    tax_rate = self.long_term_tax_rate if long_term else self.short_term_tax_rate
                    tax_impact = gain_loss * tax_rate
                    
                lots_with_tax.append((lot, tax_impact))
            
            # Sort by tax impact (lowest first)
            sorted_lots_with_tax = sorted(lots_with_tax, key=lambda x: x[1])
            return [lot for lot, _ in sorted_lots_with_tax]
            
        elif strategy == TaxLotSelectionStrategy.MAX_TAX:
            # Maximize tax losses
            # Calculate tax impact for each lot
            lots_with_tax = []
            
            for lot in lots:
                gain_loss = (current_price - lot.purchase_price) * lot.quantity
                holding_period = (timestamp - lot.purchase_date).days
                long_term = holding_period > 365
                
                if gain_loss <= 0:  # Loss or break-even
                    tax_impact = gain_loss  # Tax benefit from loss
                else:  # Gain
                    tax_rate = self.long_term_tax_rate if long_term else self.short_term_tax_rate
                    tax_impact = gain_loss * tax_rate
                    
                lots_with_tax.append((lot, tax_impact))
            
            # Sort by tax impact (lowest first, which means largest losses first)
            sorted_lots_with_tax = sorted(lots_with_tax, key=lambda x: x[1])
            return [lot for lot, _ in sorted_lots_with_tax]
            
        elif strategy == TaxLotSelectionStrategy.TAX_EFFICIENT:
            # Balance between tax minimization and other factors
            # Calculate a score for each lot that considers:
            # 1. Tax impact
            # 2. Holding period (prefer long-term over short-term)
            # 3. Size of gain/loss
            lots_with_score = []
            
            for lot in lots:
                gain_loss = (current_price - lot.purchase_price) * lot.quantity
                holding_period = (timestamp - lot.purchase_date).days
                long_term = holding_period > 365
                
                # Calculate tax impact
                if gain_loss <= 0:  # Loss or break-even
                    tax_impact = gain_loss  # Tax benefit from loss
                else:  # Gain
                    tax_rate = self.long_term_tax_rate if long_term else self.short_term_tax_rate
                    tax_impact = gain_loss * tax_rate
                
                # Calculate holding period factor (0-1)
                holding_factor = min(1.0, holding_period / 365)
                
                # Calculate score (lower is better)
                # For losses, we want to prioritize larger losses
                # For gains, we want to prioritize smaller gains and longer holding periods
                if gain_loss <= 0:
                    # For losses, prioritize larger losses (more negative tax_impact)
                    score = tax_impact
                else:
                    # For gains, balance tax impact with holding period
                    # Longer holding periods reduce the score (making them more favorable)
                    score = tax_impact * (1 - (holding_factor * self.tax_loss_preference_factor))
                
                lots_with_score.append((lot, score))
            
            # Sort by score (lowest first)
            sorted_lots_with_score = sorted(lots_with_score, key=lambda x: x[1])
            return [lot for lot, _ in sorted_lots_with_score]
        
        # Default to FIFO if strategy not implemented
        return sorted(lots, key=lambda lot: lot.purchase_date)
    
    def _calculate_tax_impact(self, selected_lots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate the tax impact of selected lots.
        
        Args:
            selected_lots: List of selected tax lots
            
        Returns:
            Dict with tax impact details
        """
        if not selected_lots:
            return {
                "total_proceeds": 0.0,
                "total_cost_basis": 0.0,
                "total_gain_loss": 0.0,
                "short_term_gain_loss": 0.0,
                "long_term_gain_loss": 0.0,
                "tax_due": 0.0,
                "after_tax_proceeds": 0.0
            }
        
        total_proceeds = sum(lot["market_value"] for lot in selected_lots)
        total_cost_basis = sum(lot["cost_basis"] for lot in selected_lots)
        total_gain_loss = sum(lot["gain_loss"] for lot in selected_lots)
        
        # Separate short-term and long-term gains/losses
        short_term_gain_loss = sum(lot["gain_loss"] for lot in selected_lots if not lot["long_term"])
        long_term_gain_loss = sum(lot["gain_loss"] for lot in selected_lots if lot["long_term"])
        
        # Calculate tax due
        short_term_tax = max(0, short_term_gain_loss) * self.short_term_tax_rate
        long_term_tax = max(0, long_term_gain_loss) * self.long_term_tax_rate
        tax_due = short_term_tax + long_term_tax
        
        # Calculate after-tax proceeds
        after_tax_proceeds = total_proceeds - tax_due
        
        return {
            "total_proceeds": total_proceeds,
            "total_cost_basis": total_cost_basis,
            "total_gain_loss": total_gain_loss,
            "short_term_gain_loss": short_term_gain_loss,
            "long_term_gain_loss": long_term_gain_loss,
            "short_term_tax": short_term_tax,
            "long_term_tax": long_term_tax,
            "tax_due": tax_due,
            "after_tax_proceeds": after_tax_proceeds
        }
    
    def reset(self) -> None:
        """
        Reset the tax lot selector.
        """
        self.symbol_strategies = {}
        self.selection_history = []
        
        logger.info("Tax Lot Selector reset")