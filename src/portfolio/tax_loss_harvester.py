import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum

from portfolio.tax_manager import TaxManager, TaxLot, TaxLotMethod

logger = logging.getLogger(__name__)

class HarvestingStrategy(Enum):
    """Strategies for tax-loss harvesting."""
    THRESHOLD_BASED = "Threshold Based"  # Harvest losses exceeding thresholds
    OPTIMAL_TIMING = "Optimal Timing"    # Time harvesting for maximum benefit
    YEAR_END = "Year End"               # Focus on year-end harvesting
    CONTINUOUS = "Continuous"           # Continuously monitor for opportunities

class ReplacementStrategy(Enum):
    """Strategies for selecting replacement securities."""
    SIMILAR_ETF = "Similar ETF"          # Replace with similar ETF
    SIMILAR_SECTOR = "Similar Sector"    # Replace with similar sector fund
    CORRELATED_ASSET = "Correlated Asset" # Replace with correlated asset
    TEMPORARY_HOLD = "Temporary Hold"    # Hold cash temporarily

class TaxLossHarvester:
    """
    Tax Loss Harvester for automated tax-loss harvesting.
    
    This class provides functionality for:
    - Identifying tax-loss harvesting opportunities
    - Executing tax-loss harvesting strategies
    - Preventing wash sales
    - Selecting replacement securities
    - Analyzing harvesting impact
    """
    
    def __init__(self, 
                 tax_manager: TaxManager,
                 threshold_absolute: float = 500.0,
                 threshold_relative: float = 0.05,
                 wash_sale_window_days: int = 30,
                 harvesting_strategy: HarvestingStrategy = HarvestingStrategy.THRESHOLD_BASED,
                 replacement_strategy: ReplacementStrategy = ReplacementStrategy.SIMILAR_ETF,
                 max_harvest_per_year: float = 50000.0,
                 harvest_frequency: str = "weekly",
                 similar_securities_map: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the Tax Loss Harvester.
        
        Args:
            tax_manager: The tax manager instance
            threshold_absolute: Minimum loss in dollars to consider harvesting
            threshold_relative: Minimum loss as percentage to consider harvesting
            wash_sale_window_days: Number of days to look for wash sales
            harvesting_strategy: Strategy for harvesting losses
            replacement_strategy: Strategy for selecting replacement securities
            max_harvest_per_year: Maximum losses to harvest per year
            harvest_frequency: How often to check for harvesting opportunities
            similar_securities_map: Dict mapping securities to similar alternatives
        """
        self.tax_manager = tax_manager
        self.threshold_absolute = threshold_absolute
        self.threshold_relative = threshold_relative
        self.wash_sale_window_days = wash_sale_window_days
        self.harvesting_strategy = harvesting_strategy
        self.replacement_strategy = replacement_strategy
        self.max_harvest_per_year = max_harvest_per_year
        self.harvest_frequency = harvest_frequency
        self.similar_securities_map = similar_securities_map or {}
        
        # Track harvesting history
        self.harvest_history = []
        # Track current year harvested amount
        self.current_year_harvested = 0.0
        # Track last harvest date by symbol
        self.last_harvest_date = {}
        # Track wash sale prevention list (symbols to avoid buying)
        self.wash_sale_prevention_list = set()
        
        logger.info(f"Tax Loss Harvester initialized with {harvesting_strategy.value} strategy")
    
    def find_harvesting_opportunities(self, 
                                     portfolio_values: Dict[str, float],
                                     current_prices: Dict[str, float],
                                     timestamp: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Find tax-loss harvesting opportunities in the portfolio.
        
        Args:
            portfolio_values: Dict of {symbol: current_value}
            current_prices: Dict of {symbol: current_price}
            timestamp: Timestamp for the analysis (default: now)
            
        Returns:
            List of harvesting opportunities with details
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        opportunities = []
        
        # Get all tax lots from the tax manager
        all_tax_lots = self.tax_manager.get_tax_lots()
        
        # Check each symbol for harvesting opportunities
        for symbol, lots in all_tax_lots.items():
            # Skip if we don't have current price or value
            if symbol not in current_prices or symbol not in portfolio_values:
                continue
                
            current_price = current_prices[symbol]
            current_value = portfolio_values[symbol]
            
            # Calculate total quantity and cost basis
            total_quantity = sum(lot["quantity"] for lot in lots)
            total_cost_basis = sum(lot["cost_basis"] for lot in lots)
            
            # Skip if no quantity
            if total_quantity <= 0:
                continue
                
            # Calculate unrealized loss
            market_value = total_quantity * current_price
            unrealized_loss = market_value - total_cost_basis
            
            # Only consider losses
            if unrealized_loss >= 0:
                continue
                
            # Check if loss meets thresholds
            absolute_loss = abs(unrealized_loss)
            relative_loss = absolute_loss / total_cost_basis if total_cost_basis > 0 else 0
            
            meets_absolute_threshold = absolute_loss >= self.threshold_absolute
            meets_relative_threshold = relative_loss >= self.threshold_relative
            
            if meets_absolute_threshold and meets_relative_threshold:
                # Check if we've harvested this symbol recently
                last_harvest = self.last_harvest_date.get(symbol)
                min_days_between_harvests = self._get_min_days_between_harvests()
                
                if last_harvest and (timestamp - last_harvest).days < min_days_between_harvests:
                    # Too soon to harvest again
                    continue
                    
                # Check if harvesting would exceed yearly limit
                if self.current_year_harvested + absolute_loss > self.max_harvest_per_year:
                    # Would exceed yearly limit
                    continue
                    
                # Find potential replacement securities
                replacements = self._find_replacement_securities(symbol)
                
                # Create opportunity record
                opportunity = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "current_value": current_value,
                    "total_quantity": total_quantity,
                    "cost_basis": total_cost_basis,
                    "unrealized_loss": unrealized_loss,
                    "absolute_loss": absolute_loss,
                    "relative_loss": relative_loss,
                    "timestamp": timestamp,
                    "potential_replacements": replacements,
                    "lots": lots
                }
                
                opportunities.append(opportunity)
        
        # Sort opportunities by absolute loss (largest first)
        opportunities.sort(key=lambda x: x["absolute_loss"], reverse=True)
        
        return opportunities
    
    def create_harvest_plan(self, 
                          opportunities: List[Dict[str, Any]],
                          constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a tax-loss harvesting plan from identified opportunities.
        
        Args:
            opportunities: List of harvesting opportunities
            constraints: Dict of constraints for the plan (optional)
            
        Returns:
            Dict with harvesting plan details
        """
        if not opportunities:
            return {
                "harvest_trades": [],
                "replacement_trades": [],
                "total_harvest_amount": 0.0,
                "timestamp": datetime.now(),
                "message": "No harvesting opportunities found"
            }
            
        constraints = constraints or {}
        max_trades = constraints.get("max_trades", len(opportunities))
        max_harvest_amount = constraints.get("max_harvest_amount", self.max_harvest_per_year - self.current_year_harvested)
        
        # Select opportunities to include in the plan
        selected_opportunities = []
        total_harvest_amount = 0.0
        
        for opportunity in opportunities:
            if len(selected_opportunities) >= max_trades:
                break
                
            if total_harvest_amount + opportunity["absolute_loss"] > max_harvest_amount:
                continue
                
            selected_opportunities.append(opportunity)
            total_harvest_amount += opportunity["absolute_loss"]
        
        # Create harvest and replacement trades
        harvest_trades = []
        replacement_trades = []
        
        for opportunity in selected_opportunities:
            symbol = opportunity["symbol"]
            quantity = opportunity["total_quantity"]
            current_price = opportunity["current_price"]
            current_value = opportunity["current_value"]
            
            # Create harvest trade
            harvest_trade = {
                "symbol": symbol,
                "action": "SELL",
                "quantity": quantity,
                "price": current_price,
                "amount": current_value,
                "reason": "Tax-Loss Harvest"
            }
            
            harvest_trades.append(harvest_trade)
            
            # Select replacement security
            replacements = opportunity["potential_replacements"]
            if replacements:
                replacement = replacements[0]  # Use first replacement
                
                # Create replacement trade
                replacement_trade = {
                    "symbol": replacement,
                    "action": "BUY",
                    "amount": current_value,  # Reinvest full amount
                    "reason": f"Replacement for {symbol}"
                }
                
                replacement_trades.append(replacement_trade)
        
        return {
            "harvest_trades": harvest_trades,
            "replacement_trades": replacement_trades,
            "total_harvest_amount": total_harvest_amount,
            "opportunities": selected_opportunities,
            "timestamp": datetime.now()
        }
    
    def execute_harvest_plan(self, 
                           harvest_plan: Dict[str, Any],
                           current_prices: Dict[str, float],
                           timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Execute a tax-loss harvesting plan.
        
        Args:
            harvest_plan: The harvesting plan to execute
            current_prices: Dict of {symbol: current_price}
            timestamp: Timestamp for the execution (default: now)
            
        Returns:
            Dict with execution results
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        harvest_trades = harvest_plan.get("harvest_trades", [])
        replacement_trades = harvest_plan.get("replacement_trades", [])
        
        if not harvest_trades:
            return {
                "success": True,
                "message": "No trades to execute",
                "timestamp": timestamp,
                "executed_harvest_trades": [],
                "executed_replacement_trades": [],
                "total_harvested": 0.0
            }
        
        # Execute harvest trades
        executed_harvest_trades = []
        total_harvested = 0.0
        
        for trade in harvest_trades:
            symbol = trade["symbol"]
            quantity = trade["quantity"]
            price = current_prices.get(symbol, trade["price"])  # Use latest price if available
            
            # Execute the sale using the tax manager
            sale_result = self.tax_manager.sell_tax_lots(
                symbol=symbol,
                quantity=quantity,
                sale_price=price,
                sale_date=timestamp
            )
            
            # Record the harvest
            if "error" not in sale_result:
                executed_trade = {
                    "symbol": symbol,
                    "quantity": sale_result["quantity_sold"],
                    "price": price,
                    "amount": sale_result["quantity_sold"] * price,
                    "realized_loss": abs(sale_result["realized_gain"]) if sale_result["realized_gain"] < 0 else 0,
                    "timestamp": timestamp
                }
                
                executed_harvest_trades.append(executed_trade)
                total_harvested += executed_trade["realized_loss"]
                
                # Update tracking
                self.last_harvest_date[symbol] = timestamp
                self.wash_sale_prevention_list.add(symbol)
                
                # Add to harvest history
                self.harvest_history.append({
                    "symbol": symbol,
                    "quantity": executed_trade["quantity"],
                    "price": price,
                    "amount": executed_trade["amount"],
                    "realized_loss": executed_trade["realized_loss"],
                    "timestamp": timestamp
                })
        
        # Update current year harvested amount
        current_year = timestamp.year
        self.current_year_harvested = sum(
            h["realized_loss"] for h in self.harvest_history 
            if h["timestamp"].year == current_year
        )
        
        # Execute replacement trades (in a real system, this would interface with a trading system)
        executed_replacement_trades = []
        
        for trade in replacement_trades:
            symbol = trade["symbol"]
            amount = trade["amount"]
            
            # Check if this would cause a wash sale
            if symbol in self.wash_sale_prevention_list:
                logger.warning(f"Skipping replacement trade for {symbol} to prevent wash sale")
                continue
                
            # In a real system, this would execute the trade
            # For now, we just record it as if it was executed
            executed_trade = {
                "symbol": symbol,
                "amount": amount,
                "timestamp": timestamp,
                "reason": trade.get("reason", "Replacement")
            }
            
            executed_replacement_trades.append(executed_trade)
        
        return {
            "success": True,
            "timestamp": timestamp,
            "executed_harvest_trades": executed_harvest_trades,
            "executed_replacement_trades": executed_replacement_trades,
            "total_harvested": total_harvested,
            "current_year_harvested": self.current_year_harvested
        }
    
    def get_harvest_history(self,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get tax-loss harvesting history with optional date filtering.
        
        Args:
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)
            
        Returns:
            List of harvest history entries
        """
        filtered_history = self.harvest_history
        
        if start_date:
            filtered_history = [h for h in filtered_history if h["timestamp"] >= start_date]
            
        if end_date:
            filtered_history = [h for h in filtered_history if h["timestamp"] <= end_date]
            
        return filtered_history
    
    def get_yearly_harvest_summary(self, year: Optional[int] = None) -> Dict[str, Any]:
        """
        Get a summary of harvesting for a specific year.
        
        Args:
            year: The year to summarize (default: current year)
            
        Returns:
            Dict with yearly harvest summary
        """
        if year is None:
            year = datetime.now().year
            
        # Filter history for the specified year
        year_history = [h for h in self.harvest_history if h["timestamp"].year == year]
        
        if not year_history:
            return {
                "year": year,
                "total_harvested": 0.0,
                "trade_count": 0,
                "symbols_harvested": [],
                "largest_harvest": None,
                "remaining_harvest_capacity": self.max_harvest_per_year
            }
        
        # Calculate summary statistics
        total_harvested = sum(h["realized_loss"] for h in year_history)
        symbols_harvested = list(set(h["symbol"] for h in year_history))
        largest_harvest = max(year_history, key=lambda h: h["realized_loss"])
        remaining_capacity = max(0, self.max_harvest_per_year - total_harvested)
        
        return {
            "year": year,
            "total_harvested": total_harvested,
            "trade_count": len(year_history),
            "symbols_harvested": symbols_harvested,
            "largest_harvest": {
                "symbol": largest_harvest["symbol"],
                "amount": largest_harvest["realized_loss"],
                "date": largest_harvest["timestamp"]
            },
            "remaining_harvest_capacity": remaining_capacity
        }
    
    def update_wash_sale_prevention_list(self) -> None:
        """
        Update the wash sale prevention list by removing symbols
        that are past the wash sale window.
        """
        current_time = datetime.now()
        symbols_to_remove = set()
        
        for symbol, last_date in self.last_harvest_date.items():
            days_since_harvest = (current_time - last_date).days
            
            if days_since_harvest > self.wash_sale_window_days:
                symbols_to_remove.add(symbol)
        
        # Remove symbols from prevention list
        self.wash_sale_prevention_list -= symbols_to_remove
        
        # Remove from last harvest date tracking
        for symbol in symbols_to_remove:
            if symbol in self.last_harvest_date:
                del self.last_harvest_date[symbol]
    
    def _find_replacement_securities(self, symbol: str) -> List[str]:
        """
        Find suitable replacement securities for a harvested position.
        
        Args:
            symbol: The symbol being harvested
            
        Returns:
            List of potential replacement securities
        """
        # Check if we have predefined replacements for this symbol
        if symbol in self.similar_securities_map:
            return self.similar_securities_map[symbol]
        
        # In a real implementation, this would use more sophisticated logic
        # to find similar securities based on correlation, sector, etc.
        # For now, return an empty list
        return []
    
    def _get_min_days_between_harvests(self) -> int:
        """
        Get the minimum number of days between harvests based on frequency setting.
        
        Returns:
            Minimum days between harvests
        """
        if self.harvest_frequency == "daily":
            return 1
        elif self.harvest_frequency == "weekly":
            return 7
        elif self.harvest_frequency == "monthly":
            return 30
        elif self.harvest_frequency == "quarterly":
            return 90
        else:  # Default to monthly
            return 30
    
    def reset(self) -> None:
        """
        Reset the tax loss harvester.
        """
        self.harvest_history = []
        self.current_year_harvested = 0.0
        self.last_harvest_date = {}
        self.wash_sale_prevention_list = set()
        
        logger.info("Tax Loss Harvester reset")