import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from enum import Enum

from portfolio.allocation_manager import AllocationManager, RebalanceMethod
from portfolio.tax_manager import TaxManager, TaxLot

logger = logging.getLogger(__name__)

class TaxAwareRebalanceMethod(Enum):
    """Methods for tax-aware portfolio rebalancing."""
    MINIMIZE_TAX_IMPACT = "Minimize Tax Impact"  # Prioritize minimizing taxes
    CAPITAL_GAINS_BUDGET = "Capital Gains Budget"  # Stay within a capital gains budget
    ASSET_LOCATION = "Asset Location"  # Consider tax efficiency of different account types
    MULTI_PERIOD = "Multi-Period Optimization"  # Optimize across multiple time periods

class TaxAwareRebalancer:
    """
    Tax-Aware Rebalancer for portfolio rebalancing with tax considerations.
    
    This class extends the functionality of the AllocationManager by adding
    tax-aware rebalancing strategies that minimize tax impact while
    maintaining target allocations.
    """
    
    def __init__(self,
                 allocation_manager: AllocationManager,
                 tax_manager: TaxManager,
                 tax_aware_method: TaxAwareRebalanceMethod = TaxAwareRebalanceMethod.MINIMIZE_TAX_IMPACT,
                 capital_gains_budget: float = 0.0,
                 long_term_preference_factor: float = 0.5,
                 harvest_losses_first: bool = True):
        """
        Initialize the Tax-Aware Rebalancer.
        
        Args:
            allocation_manager: The allocation manager instance
            tax_manager: The tax manager instance
            tax_aware_method: Method for tax-aware rebalancing
            capital_gains_budget: Maximum capital gains to realize (for CAPITAL_GAINS_BUDGET method)
            long_term_preference_factor: Factor to prefer long-term over short-term gains (0-1)
            harvest_losses_first: Whether to prioritize harvesting losses before rebalancing
        """
        self.allocation_manager = allocation_manager
        self.tax_manager = tax_manager
        self.tax_aware_method = tax_aware_method
        self.capital_gains_budget = capital_gains_budget
        self.long_term_preference_factor = long_term_preference_factor
        self.harvest_losses_first = harvest_losses_first
        
        # Dictionary to store symbol-specific tax lot methods
        self.symbol_methods = {}
        
        # History of tax-aware rebalance operations
        self.rebalance_history = []
        
        logger.info(f"Tax-Aware Rebalancer initialized with {tax_aware_method.value} method")
    
    def generate_tax_aware_rebalance_plan(self, 
                                      timestamp: datetime,
                                      portfolio_values: Dict[str, float],
                                      categories: Optional[Dict[str, str]] = None,
                                      tax_loss_harvester: Optional[Any] = None) -> Dict[str, Any]:
        """
        Generate a tax-aware rebalance plan based on the selected method.
        
        Args:
            timestamp: The timestamp for the rebalance
            portfolio_values: Dict of {symbol: value}
            categories: Optional dict of {symbol: category}
            tax_loss_harvester: Optional TaxLossHarvester instance for integrated harvesting
            
        Returns:
            A rebalance plan that considers tax implications
        """
        # First, get the standard rebalance plan from the allocation manager
        standard_plan = self.allocation_manager.generate_rebalance_plan(
            timestamp=timestamp,
            portfolio_values=portfolio_values,
            categories=categories
        )
        
        # If tax loss harvester is provided, check for harvesting opportunities
        harvest_plan = None
        if tax_loss_harvester is not None:
            # Get harvesting opportunities
            harvest_opportunities = tax_loss_harvester.identify_harvest_opportunities(
                timestamp=timestamp,
                min_loss_threshold=0.0  # Use the harvester's default threshold
            )
            
            if harvest_opportunities:
                # Create a harvest plan
                harvest_plan = tax_loss_harvester.create_harvest_plan(
                    opportunities=harvest_opportunities,
                    timestamp=timestamp
                )
                
                # Add harvesting information to the rebalance plan
                standard_plan["tax_loss_harvesting"] = {
                    "opportunities": len(harvest_opportunities),
                    "harvest_plan": harvest_plan
                }
                
                # Integrate harvesting trades with rebalance trades
                if "trades" not in standard_plan:
                    standard_plan["trades"] = []
                    
                # Add sell trades from harvest plan
                for harvest_item in harvest_plan.get("harvests", []):
                    symbol = harvest_item.get("symbol")
                    amount = harvest_item.get("amount")
                    estimated_loss = harvest_item.get("estimated_loss")
                    
                    # Check if this symbol is already in the rebalance plan
                    existing_trade = next((t for t in standard_plan["trades"] 
                                         if t.get("name") == symbol and t.get("action") == "SELL"), None)
                    
                    if existing_trade:
                        # Update existing sell trade
                        existing_trade["amount"] = max(existing_trade.get("amount", 0), amount)
                        existing_trade["tax_loss_harvest"] = True
                        existing_trade["estimated_loss"] = estimated_loss
                    else:
                        # Add new sell trade
                        standard_plan["trades"].append({
                            "name": symbol,
                            "action": "SELL",
                            "amount": amount,
                            "tax_loss_harvest": True,
                            "estimated_loss": estimated_loss
                        })
                    
                # Add buy trades for replacements
                for replacement in harvest_plan.get("replacements", []):
                    symbol = replacement.get("symbol")
                    amount = replacement.get("amount")
                    original_symbol = replacement.get("original_symbol")
                    
                    # Check if this symbol is already in the rebalance plan
                    existing_trade = next((t for t in standard_plan["trades"] 
                                         if t.get("name") == symbol and t.get("action") == "BUY"), None)
                    
                    if existing_trade:
                        # Update existing buy trade
                        existing_trade["amount"] = max(existing_trade.get("amount", 0), amount)
                        existing_trade["tax_loss_harvest_replacement"] = True
                        existing_trade["original_symbol"] = original_symbol
                    else:
                        # Add new buy trade
                        standard_plan["trades"].append({
                            "name": symbol,
                            "action": "BUY",
                            "amount": amount,
                            "tax_loss_harvest_replacement": True,
                            "original_symbol": original_symbol
                        })
        
        # If no rebalancing needed and no harvesting opportunities, return the standard plan
        if not standard_plan.get("trades", []):
            logger.info("No rebalancing or harvesting needed, returning standard plan")
            return standard_plan
        
        # Apply the selected tax-aware optimization method
        if self.tax_aware_method == TaxAwareRebalanceMethod.MINIMIZE_TAX_IMPACT:
            return self._minimize_tax_impact_plan(standard_plan, portfolio_values, categories, timestamp)
        elif self.tax_aware_method == TaxAwareRebalanceMethod.CAPITAL_GAINS_BUDGET:
            return self._capital_gains_budget_plan(standard_plan, portfolio_values, categories, timestamp)
        elif self.tax_aware_method == TaxAwareRebalanceMethod.ASSET_LOCATION:
            return self._asset_location_plan(standard_plan, portfolio_values, categories, timestamp)
        elif self.tax_aware_method == TaxAwareRebalanceMethod.MULTI_PERIOD:
            return self._multi_period_plan(standard_plan, portfolio_values, categories, timestamp)
        else:
            # Default to standard plan if method not recognized
            logger.warning(f"Unrecognized tax-aware method: {self.tax_aware_method}, using standard plan")
            return standard_plan
    
    def _minimize_tax_impact_plan(self, 
                                 standard_plan: Dict[str, Any], 
                                 portfolio_values: Dict[str, float],
                                 categories: Optional[Dict[str, str]],
                                 timestamp: datetime) -> Dict[str, Any]:
        """
        Generate a rebalance plan that minimizes tax impact.
        
        This method modifies the standard rebalance plan to:
        1. Prioritize selling positions with losses
        2. Avoid selling positions with short-term gains
        3. Minimize selling positions with large long-term gains
        4. Use new cash inflows for buying instead of selling existing positions
        5. Consider tax-loss harvesting opportunities
        
        Args:
            standard_plan: The standard rebalance plan from allocation manager
            portfolio_values: Dict of {name: value}
            categories: Dict of {name: category}
            timestamp: Timestamp for the plan
            
        Returns:
            Modified rebalance plan with tax considerations
        """
        # Start with the standard plan
        tax_aware_plan = standard_plan.copy()
        trades = standard_plan.get("trades", [])
        
        # Separate buy and sell trades
        buy_trades = [t for t in trades if t["action"] == "BUY"]
        sell_trades = [t for t in trades if t["action"] == "SELL"]
        
        # Get tax lot information for all symbols that might be sold
        symbols_to_sell = [t["name"] for t in sell_trades]
        tax_lots_by_symbol = {}
        potential_tax_impact = {}
        
        for symbol in symbols_to_sell:
            # Get all tax lots for this symbol
            tax_lots = self.tax_manager.get_tax_lots(symbol=symbol)
            tax_lots_by_symbol[symbol] = tax_lots
            
            # Calculate potential tax impact
            current_price = portfolio_values.get(symbol, 0) / sum(lot.quantity for lot in tax_lots) if tax_lots else 0
            
            # Calculate unrealized gains/losses and tax impact
            short_term_impact = 0
            long_term_impact = 0
            loss_harvest_potential = 0
            
            for lot in tax_lots:
                unrealized_gain = (current_price - lot.purchase_price) * lot.quantity
                holding_period = (timestamp - lot.purchase_date).days
                
                if unrealized_gain < 0:  # Loss
                    loss_harvest_potential += abs(unrealized_gain)
                elif holding_period <= 365:  # Short-term gain
                    short_term_impact += unrealized_gain
                else:  # Long-term gain
                    long_term_impact += unrealized_gain
            
            potential_tax_impact[symbol] = {
                "short_term_impact": short_term_impact,
                "long_term_impact": long_term_impact,
                "loss_harvest_potential": loss_harvest_potential,
                "total_impact": short_term_impact + (long_term_impact * (1 - self.long_term_preference_factor))
            }
        
        # Sort sell trades by tax impact (lowest first)
        sell_trades_with_impact = []
        for trade in sell_trades:
            symbol = trade["name"]
            impact = potential_tax_impact.get(symbol, {"total_impact": 0}).get("total_impact", 0)
            sell_trades_with_impact.append((trade, impact))
        
        # Sort by tax impact (prioritize harvesting losses, then minimize gains)
        sell_trades_with_impact.sort(key=lambda x: x[1])
        
        # Rebuild the trades list with the optimized sell order
        optimized_trades = buy_trades + [t[0] for t in sell_trades_with_impact]
        
        # Update the plan with the optimized trades
        tax_aware_plan["trades"] = optimized_trades
        tax_aware_plan["tax_impact_analysis"] = {
            "potential_tax_impact": potential_tax_impact,
            "method": self.tax_aware_method.value,
            "long_term_preference_factor": self.long_term_preference_factor
        }
        
        # Record this plan in history
        self.rebalance_history.append({
            "timestamp": timestamp,
            "method": self.tax_aware_method.value,
            "standard_plan": standard_plan,
            "tax_aware_plan": tax_aware_plan
        })
        
        return tax_aware_plan
    
    def _capital_gains_budget_plan(self, 
                                 standard_plan: Dict[str, Any], 
                                 portfolio_values: Dict[str, float],
                                 categories: Optional[Dict[str, str]],
                                 timestamp: datetime) -> Dict[str, Any]:
        """
        Generate a rebalance plan that stays within a capital gains budget.
        
        This method modifies the standard rebalance plan to:
        1. Estimate the capital gains that would be realized by the standard plan
        2. If the estimated gains exceed the budget, reduce sells to stay within budget
        3. Prioritize the most important allocation adjustments
        
        Args:
            standard_plan: The standard rebalance plan from allocation manager
            portfolio_values: Dict of {name: value}
            categories: Dict of {name: category}
            timestamp: Timestamp for the plan
            
        Returns:
            Modified rebalance plan with capital gains budget constraint
        """
        # Start with the standard plan
        tax_aware_plan = standard_plan.copy()
        trades = standard_plan.get("trades", [])
        
        # Separate buy and sell trades
        buy_trades = [t for t in trades if t["action"] == "BUY"]
        sell_trades = [t for t in trades if t["action"] == "SELL"]
        
        # Calculate potential capital gains from sell trades
        total_estimated_gains = 0
        sell_trades_with_gains = []
        
        for trade in sell_trades:
            symbol = trade["name"]
            amount_to_sell = trade["amount"]
            
            # Simulate the sale to estimate gains
            estimated_gain = self._estimate_gain_from_sale(symbol, amount_to_sell, timestamp)
            
            sell_trades_with_gains.append({
                "trade": trade,
                "estimated_gain": estimated_gain,
                "drift_percentage": self._get_drift_percentage(standard_plan, symbol)
            })
            
            total_estimated_gains += estimated_gain
        
        # Check if we're within budget
        if total_estimated_gains <= self.capital_gains_budget:
            # We're within budget, use the standard plan but add tax impact info
            tax_aware_plan["estimated_capital_gains"] = total_estimated_gains
            tax_aware_plan["capital_gains_budget"] = self.capital_gains_budget
            tax_aware_plan["within_budget"] = True
        else:
            # We need to reduce sells to stay within budget
            # Sort by drift percentage (highest first) to prioritize most important adjustments
            sell_trades_with_gains.sort(key=lambda x: x["drift_percentage"], reverse=True)
            
            # Keep adding trades until we hit the budget
            approved_sell_trades = []
            running_gain_total = 0
            
            for trade_info in sell_trades_with_gains:
                if running_gain_total + trade_info["estimated_gain"] <= self.capital_gains_budget:
                    approved_sell_trades.append(trade_info["trade"])
                    running_gain_total += trade_info["estimated_gain"]
                else:
                    # We can't add the full trade, see if we can add a partial trade
                    remaining_budget = self.capital_gains_budget - running_gain_total
                    if remaining_budget > 0:
                        # Calculate what portion of the trade we can do
                        portion = remaining_budget / trade_info["estimated_gain"]
                        partial_trade = trade_info["trade"].copy()
                        partial_trade["amount"] = trade_info["trade"]["amount"] * portion
                        partial_trade["partial"] = True
                        approved_sell_trades.append(partial_trade)
                        running_gain_total += remaining_budget
                    break
            
            # Recalculate buys based on available cash from approved sells
            available_cash = sum(t["amount"] for t in approved_sell_trades)
            
            # Distribute available cash proportionally among buy trades
            total_buy_amount = sum(t["amount"] for t in buy_trades)
            adjusted_buy_trades = []
            
            if total_buy_amount > 0:
                for trade in buy_trades:
                    adjusted_trade = trade.copy()
                    adjusted_trade["amount"] = (trade["amount"] / total_buy_amount) * available_cash
                    adjusted_buy_trades.append(adjusted_trade)
            
            # Update the plan with the budget-constrained trades
            tax_aware_plan["trades"] = adjusted_buy_trades + approved_sell_trades
            tax_aware_plan["estimated_capital_gains"] = running_gain_total
            tax_aware_plan["capital_gains_budget"] = self.capital_gains_budget
            tax_aware_plan["within_budget"] = False
            tax_aware_plan["original_estimated_gains"] = total_estimated_gains
        
        # Record this plan in history
        self.rebalance_history.append({
            "timestamp": timestamp,
            "method": self.tax_aware_method.value,
            "standard_plan": standard_plan,
            "tax_aware_plan": tax_aware_plan
        })
        
        return tax_aware_plan
    
    def _asset_location_plan(self, 
                           standard_plan: Dict[str, Any], 
                           portfolio_values: Dict[str, float],
                           categories: Optional[Dict[str, str]],
                           timestamp: datetime) -> Dict[str, Any]:
        """
        Generate a rebalance plan that considers asset location tax efficiency.
        
        This method optimizes asset location by placing tax-inefficient assets in
        tax-advantaged accounts and tax-efficient assets in taxable accounts.
        
        Args:
            standard_plan: The standard rebalance plan from allocation manager
            portfolio_values: Dict of {name: value}
            categories: Dict of {name: category}
            timestamp: Timestamp for the plan
            
        Returns:
            Modified rebalance plan with asset location considerations
        """
        # Start with the standard plan
        tax_aware_plan = standard_plan.copy()
        trades = standard_plan.get("trades", [])
        
        # If no categories provided, we can't do asset location optimization
        if not categories:
            logger.warning("Asset location optimization requires category information")
            
            # Record this plan in history
            self.rebalance_history.append({
                "timestamp": timestamp,
                "method": self.tax_aware_method.value,
                "standard_plan": standard_plan,
                "tax_aware_plan": standard_plan
            })
            
            return standard_plan
        
        # Define tax efficiency ratings for different asset classes (1-10 scale, higher is more tax-efficient)
        tax_efficiency_ratings = {
            "us_stock": 7,  # Qualified dividends, long-term capital gains
            "international_stock": 6,  # Some foreign dividends may not be qualified
            "emerging_markets": 5,  # Higher turnover, less tax-efficient
            "us_bond": 3,  # Interest taxed as ordinary income
            "municipal_bond": 10,  # Tax-exempt interest
            "corporate_bond": 2,  # Interest taxed as ordinary income, higher yield
            "high_yield_bond": 1,  # High interest taxed as ordinary income
            "reit": 2,  # Dividends mostly non-qualified
            "commodity": 4,  # Complex tax treatment
            "cash": 3,  # Interest taxed as ordinary income
            # Default for unknown categories
            "default": 5
        }
        
        # Define account types and their tax treatment (1-10 scale, higher is more tax-advantaged)
        account_types = {
            "taxable": 1,  # Taxable account
            "ira": 8,  # Traditional IRA
            "roth": 10,  # Roth IRA
            "401k": 8,  # Traditional 401(k)
            "roth_401k": 10,  # Roth 401(k)
            "hsa": 10,  # Health Savings Account
            # Default for unknown account types
            "default": 5
        }
        
        # Extract account information from categories
        # Assuming categories contains account type information in format "asset_class:account_type"
        asset_accounts = {}
        asset_classes = {}
        
        for symbol, category in categories.items():
            parts = category.split(":")
            
            if len(parts) >= 2:
                asset_class = parts[0].lower()
                account_type = parts[1].lower()
            else:
                asset_class = category.lower()
                account_type = "default"
            
            asset_accounts[symbol] = account_type
            asset_classes[symbol] = asset_class
        
        # Calculate asset location score for each symbol
        # Higher score means the asset should preferably be in a taxable account
        # Lower score means the asset should preferably be in a tax-advantaged account
        asset_location_scores = {}
        
        for symbol in portfolio_values.keys():
            asset_class = asset_classes.get(symbol, "default")
            account_type = asset_accounts.get(symbol, "default")
            
            tax_efficiency = tax_efficiency_ratings.get(asset_class, tax_efficiency_ratings["default"])
            account_advantage = account_types.get(account_type, account_types["default"])
            
            # Calculate location score: tax efficiency * account advantage
            # Higher score means better location
            asset_location_scores[symbol] = tax_efficiency * account_advantage
        
        # Identify mislocated assets
        mislocated_assets = []
        
        for symbol, score in asset_location_scores.items():
            asset_class = asset_classes.get(symbol, "default")
            account_type = asset_accounts.get(symbol, "default")
            tax_efficiency = tax_efficiency_ratings.get(asset_class, tax_efficiency_ratings["default"])
            
            # Check if asset is mislocated
            # Tax-inefficient assets (low rating) should be in tax-advantaged accounts (high rating)
            # Tax-efficient assets (high rating) should be in taxable accounts (low rating)
            if (tax_efficiency <= 4 and account_types.get(account_type, 5) <= 5) or \
               (tax_efficiency >= 8 and account_types.get(account_type, 5) >= 8):
                mislocated_assets.append({
                    "symbol": symbol,
                    "asset_class": asset_class,
                    "account_type": account_type,
                    "tax_efficiency": tax_efficiency,
                    "location_score": score,
                    "value": portfolio_values.get(symbol, 0)
                })
        
        # If no mislocated assets, return the standard plan
        if not mislocated_assets:
            tax_aware_plan["asset_location_analysis"] = {
                "asset_location_scores": asset_location_scores,
                "mislocated_assets": [],
                "message": "No asset location improvements identified"
            }
            
            # Record this plan in history
            self.rebalance_history.append({
                "timestamp": timestamp,
                "method": self.tax_aware_method.value,
                "standard_plan": standard_plan,
                "tax_aware_plan": tax_aware_plan
            })
            
            return tax_aware_plan
        
        # Modify the standard plan to improve asset location
        # This is a simplified approach that prioritizes fixing the most severely mislocated assets
        
        # Sort mislocated assets by severity (difference between ideal and actual location)
        for asset in mislocated_assets:
            asset_class = asset["asset_class"]
            account_type = asset["account_type"]
            tax_efficiency = asset["tax_efficiency"]
            account_advantage = account_types.get(account_type, account_types["default"])
            
            # Calculate severity: how far from ideal location
            # For tax-inefficient assets (low rating), severity is higher when in less tax-advantaged accounts
            # For tax-efficient assets (high rating), severity is higher when in more tax-advantaged accounts
            if tax_efficiency <= 5:  # Tax-inefficient asset
                asset["severity"] = (10 - account_advantage) * (5 - tax_efficiency) / 5
            else:  # Tax-efficient asset
                asset["severity"] = account_advantage * (tax_efficiency - 5) / 5
        
        # Sort by severity (highest first)
        mislocated_assets.sort(key=lambda x: x["severity"], reverse=True)
        
        # Add asset location recommendations to the plan
        tax_aware_plan["asset_location_analysis"] = {
            "asset_location_scores": asset_location_scores,
            "mislocated_assets": mislocated_assets,
            "recommendations": []
        }
        
        # Generate recommendations for improving asset location
        for asset in mislocated_assets:
            symbol = asset["symbol"]
            asset_class = asset["asset_class"]
            account_type = asset["account_type"]
            tax_efficiency = asset["tax_efficiency"]
            
            if tax_efficiency <= 5:  # Tax-inefficient asset
                # Recommend moving to more tax-advantaged account
                recommended_accounts = [acct for acct, rating in account_types.items() 
                                      if rating > account_types.get(account_type, 5) and acct != "default"]
                
                if recommended_accounts:
                    tax_aware_plan["asset_location_analysis"]["recommendations"].append({
                        "symbol": symbol,
                        "current_location": account_type,
                        "recommended_locations": recommended_accounts,
                        "reason": f"Tax-inefficient asset ({asset_class}) should be in a more tax-advantaged account"
                    })
            else:  # Tax-efficient asset
                # Recommend moving to less tax-advantaged account
                recommended_accounts = [acct for acct, rating in account_types.items() 
                                      if rating < account_types.get(account_type, 5) and acct != "default"]
                
                if recommended_accounts:
                    tax_aware_plan["asset_location_analysis"]["recommendations"].append({
                        "symbol": symbol,
                        "current_location": account_type,
                        "recommended_locations": recommended_accounts,
                        "reason": f"Tax-efficient asset ({asset_class}) should be in a less tax-advantaged account"
                    })
        
        # Record this plan in history
        self.rebalance_history.append({
            "timestamp": timestamp,
            "method": self.tax_aware_method.value,
            "standard_plan": standard_plan,
            "tax_aware_plan": tax_aware_plan
        })
        
        return tax_aware_plan
    
    def _multi_period_plan(self, 
                         standard_plan: Dict[str, Any], 
                         portfolio_values: Dict[str, float],
                         categories: Optional[Dict[str, str]],
                         timestamp: datetime) -> Dict[str, Any]:
        """
        Generate a rebalance plan that optimizes across multiple time periods.
        
        This method optimizes rebalancing over multiple time periods to minimize
        tax impact by spreading large trades across multiple periods and timing
        trades to maximize tax efficiency.
        
        Args:
            standard_plan: The standard rebalance plan from allocation manager
            portfolio_values: Dict of {name: value}
            categories: Dict of {name: category}
            timestamp: Timestamp for the plan
            
        Returns:
            Modified rebalance plan with multi-period considerations
        """
        # Start with the standard plan
        tax_aware_plan = standard_plan.copy()
        trades = standard_plan.get("trades", [])
        
        # If no trades, return the standard plan
        if not trades:
            # Record this plan in history
            self.rebalance_history.append({
                "timestamp": timestamp,
                "method": self.tax_aware_method.value,
                "standard_plan": standard_plan,
                "tax_aware_plan": standard_plan
            })
            
            return standard_plan
        
        # Define parameters for multi-period optimization
        num_periods = 3  # Number of periods to spread trades over
        tax_year_end = datetime(timestamp.year, 12, 31)  # End of current tax year
        days_to_year_end = (tax_year_end - timestamp).days
        
        # Check if we're close to tax year end (less than 60 days)
        near_year_end = days_to_year_end < 60
        
        # Get historical rebalance data to analyze frequency and patterns
        rebalance_history = self.get_rebalance_history()
        
        # Calculate average time between rebalances (if history exists)
        avg_rebalance_interval = 90  # Default to 90 days if no history
        
        if len(rebalance_history) >= 2:
            timestamps = [entry["timestamp"] for entry in rebalance_history]
            timestamps.sort()
            
            intervals = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]
            if intervals:
                avg_rebalance_interval = sum(intervals) / len(intervals)
        
        # Analyze trades to determine which should be executed now vs. deferred
        immediate_trades = []
        deferred_trades = []
        
        for trade in trades:
            symbol = trade.get("name")
            action = trade.get("action")
            amount = trade.get("amount", 0)
            
            # Skip trades with no value
            if amount == 0:
                continue
            
            # Estimate tax impact of the trade
            tax_impact = 0
            
            if action == "SELL":
                # Estimate potential gain/loss from this sale
                gain_loss = self._estimate_gain_from_sale(symbol, amount, timestamp)
                
                # If it's a loss, execute immediately (tax-loss harvesting)
                if gain_loss < 0:
                    trade["priority"] = "immediate"
                    trade["reason"] = "Tax loss harvesting opportunity"
                    immediate_trades.append(trade)
                    continue
                
                # If it's a gain, calculate tax impact
                tax_impact = gain_loss * 0.20  # Simplified tax rate assumption
                
                # If near year end and we have losses to offset, prioritize gains
                if near_year_end:
                    # Get realized gains/losses for the year
                    # This is a placeholder - would need actual tax manager implementation
                    total_realized_gain_loss = -5000  # Placeholder value
                    
                    # If we have net losses for the year, prioritize realizing some gains
                    if total_realized_gain_loss < 0 and gain_loss > 0:
                        # Only realize gains up to the amount of losses (to offset)
                        if gain_loss <= abs(total_realized_gain_loss):
                            trade["priority"] = "immediate"
                            trade["reason"] = "Offsetting existing losses before year-end"
                            immediate_trades.append(trade)
                            continue
            
            # For buys, prioritize those that align with tax-loss harvesting or rebalancing needs
            if action == "BUY":
                # Check if this is a replacement for a tax-loss harvesting sale
                # (simplified check - in a real implementation, would track specific replacements)
                is_replacement = False
                for sell_trade in trades:
                    if sell_trade.get("action") == "SELL" and sell_trade.get("priority") == "immediate":
                        # This is a simplification - would need more logic to properly identify replacements
                        is_replacement = True
                        break
                
                if is_replacement:
                    trade["priority"] = "immediate"
                    trade["reason"] = "Replacement for tax-loss harvesting sale"
                    immediate_trades.append(trade)
                    continue
            
            # Calculate drift percentage to determine urgency
            drift_percentage = self._get_drift_percentage(standard_plan, symbol)
            
            # High drift trades should be prioritized
            if abs(drift_percentage) > 0.05:  # 5% drift threshold
                trade["priority"] = "immediate"
                trade["reason"] = f"High drift ({drift_percentage:.2%}) requires immediate attention"
                immediate_trades.append(trade)
            else:
                # For trades with lower drift, consider deferring
                trade["priority"] = "deferred"
                trade["reason"] = f"Low drift ({drift_percentage:.2%}) allows for deferral"
                trade["tax_impact"] = tax_impact
                deferred_trades.append(trade)
        
        # Sort deferred trades by tax impact (highest first) and drift (highest first)
        deferred_trades.sort(key=lambda x: (x.get("tax_impact", 0), abs(self._get_drift_percentage(standard_plan, x.get("name", "")))), reverse=True)
        
        # Distribute deferred trades across periods
        periods = []
        for i in range(num_periods):
            period_trades = []
            
            # Calculate expected date for this period
            period_days = min(avg_rebalance_interval * (i + 1), 365)  # Cap at 1 year
            period_date = timestamp + timedelta(days=period_days)
            
            # Assign trades to this period
            start_idx = i * (len(deferred_trades) // num_periods)
            end_idx = (i + 1) * (len(deferred_trades) // num_periods) if i < num_periods - 1 else len(deferred_trades)
            
            for j in range(start_idx, end_idx):
                if j < len(deferred_trades):
                    trade = deferred_trades[j].copy()
                    trade["scheduled_date"] = period_date.strftime("%Y-%m-%d")
                    period_trades.append(trade)
            
            periods.append({
                "period": i + 1,
                "estimated_date": period_date.strftime("%Y-%m-%d"),
                "trades": period_trades
            })
        
        # Update the plan with immediate and deferred trades
        tax_aware_plan["trades"] = immediate_trades
        tax_aware_plan["multi_period_plan"] = {
            "immediate_trades": len(immediate_trades),
            "deferred_trades": len(deferred_trades),
            "num_periods": num_periods,
            "avg_rebalance_interval_days": avg_rebalance_interval,
            "periods": periods
        }
        
        # Calculate estimated tax savings
        total_standard_tax_impact = 0
        total_multi_period_tax_impact = 0
        
        for trade in trades:
            if trade.get("action") == "SELL":
                symbol = trade.get("name")
                amount = trade.get("amount", 0)
                gain_loss = self._estimate_gain_from_sale(symbol, amount, timestamp)
                
                if gain_loss > 0:  # Only consider gains for tax impact
                    total_standard_tax_impact += gain_loss * 0.20  # Simplified tax rate
        
        for trade in immediate_trades:
            if trade.get("action") == "SELL":
                symbol = trade.get("name")
                amount = trade.get("amount", 0)
                gain_loss = self._estimate_gain_from_sale(symbol, amount, timestamp)
                
                if gain_loss > 0:  # Only consider gains for tax impact
                    total_multi_period_tax_impact += gain_loss * 0.20  # Simplified tax rate
        
        # Estimate deferred tax impact (with time value of money discount)
        for period in periods:
            period_date = datetime.strptime(period["estimated_date"], "%Y-%m-%d")
            years_deferred = (period_date - timestamp).days / 365.0
            discount_factor = 1 / (1 + 0.05) ** years_deferred  # 5% discount rate
            
            for trade in period["trades"]:
                if trade.get("action") == "SELL":
                    tax_impact = trade.get("tax_impact", 0)
                    total_multi_period_tax_impact += tax_impact * discount_factor
        
        tax_aware_plan["multi_period_plan"]["tax_impact"] = {
            "standard_plan": total_standard_tax_impact,
            "multi_period_plan": total_multi_period_tax_impact,
            "estimated_savings": total_standard_tax_impact - total_multi_period_tax_impact
        }
        
        # Record this plan in history
        self.rebalance_history.append({
            "timestamp": timestamp,
            "method": self.tax_aware_method.value,
            "standard_plan": standard_plan,
            "tax_aware_plan": tax_aware_plan
        })
        
        return tax_aware_plan
    
    def _estimate_gain_from_sale(self, symbol: str, amount: float, timestamp: datetime) -> float:
        """
        Estimate the capital gain or loss from selling a position.
        
        Args:
            symbol: The symbol to sell
            amount: The amount to sell
            timestamp: The timestamp of the sale
            
        Returns:
            Estimated capital gain or loss (positive for gain, negative for loss)
        """
        # Get tax lots for the symbol
        tax_lots = self.tax_manager.get_tax_lots(symbol=symbol)
        
        # If no tax lots, return 0 (no gain/loss)
        if not tax_lots:
            return 0.0
        
        # Calculate total quantity
        total_quantity = sum(lot.quantity for lot in tax_lots)
        
        # If total quantity is 0, return 0 (no gain/loss)
        if total_quantity == 0:
            return 0
            
        # Get current price from tax manager
        current_price = self.tax_manager.get_current_price(symbol, timestamp)
        
        # Determine the tax lot method for this symbol
        tax_lot_method = self.tax_manager.get_tax_lot_method(symbol)
        
        # Estimate quantity to sell based on amount and current price
        quantity_to_sell = amount / current_price if current_price > 0 else 0
        
        # Simulate the sale using the tax manager's logic
        # This is a simplified version that doesn't actually modify the tax lots
        total_proceeds = 0.0
        total_cost_basis = 0.0
        remaining_quantity = quantity_to_sell
        
        # Sort tax lots according to the tax lot method
        if tax_lot_method == "FIFO":
            # First In, First Out - sell oldest lots first
            sorted_lots = sorted(tax_lots, key=lambda lot: lot.purchase_date)
        elif tax_lot_method == "LIFO":
            # Last In, First Out - sell newest lots first
            sorted_lots = sorted(tax_lots, key=lambda lot: lot.purchase_date, reverse=True)
        elif tax_lot_method == "HIFO":
            # Highest In, First Out - sell highest cost basis first (minimizes gains)
            sorted_lots = sorted(tax_lots, key=lambda lot: lot.purchase_price, reverse=True)
        elif tax_lot_method == "LOFO":
            # Lowest In, First Out - sell lowest cost basis first (maximizes gains)
            sorted_lots = sorted(tax_lots, key=lambda lot: lot.purchase_price)
        elif tax_lot_method == "MINTAX":
            # Minimize Tax Impact - prioritize lots based on tax impact
            # First use lots with losses, then lots with smallest gains
            lots_with_metrics = []
            for lot in tax_lots:
                lot_gain = (current_price - lot.purchase_price) * lot.quantity
                holding_period = (timestamp - lot.purchase_date).days
                lots_with_metrics.append((lot, lot_gain, holding_period >= 365))
            
            # Sort: losses first (oldest first), then long-term gains, then short-term gains
            sorted_lots = [lot for lot, _, _ in sorted(lots_with_metrics, key=lambda x: (
                0 if x[1] < 0 else 1,  # Losses first
                -x[0].purchase_date.timestamp() if x[1] < 0 else 0,  # Oldest losses first
                0 if x[2] else 1,  # Long-term gains before short-term
                x[1] if x[1] >= 0 else 0  # Smallest gains first
            ))]
        elif tax_lot_method == "MAXTAX":
            # Maximize Tax Impact - useful for tax-gain harvesting or loss offsetting
            # First use lots with largest gains that are long-term, then short-term gains, then losses
            lots_with_metrics = []
            for lot in tax_lots:
                lot_gain = (current_price - lot.purchase_price) * lot.quantity
                holding_period = (timestamp - lot.purchase_date).days
                lots_with_metrics.append((lot, lot_gain, holding_period >= 365))
            
            # Sort: long-term gains first (largest first), then short-term gains, then losses
            sorted_lots = [lot for lot, _, _ in sorted(lots_with_metrics, key=lambda x: (
                0 if x[1] >= 0 and x[2] else (1 if x[1] >= 0 else 2),  # Long-term gains, short-term gains, then losses
                -x[1] if x[1] >= 0 else 0  # Largest gains first
            ))]
        else:  # Default to FIFO
            sorted_lots = sorted(tax_lots, key=lambda lot: lot.purchase_date)
        
        # Calculate gain from each lot until we've covered the quantity to sell
        for lot in sorted_lots:
            if remaining_quantity <= 0:
                break
                
            lot_sell_quantity = min(lot.quantity, remaining_quantity)
            lot_proceeds = lot_sell_quantity * current_price
            lot_cost = lot_sell_quantity * lot.purchase_price
            
            total_proceeds += lot_proceeds
            total_cost_basis += lot_cost
            remaining_quantity -= lot_sell_quantity
        
        # Calculate gain or loss
        gain_loss = total_proceeds - total_cost_basis
        
        return gain_loss
    
    def _get_drift_percentage(self, plan: Dict[str, Any], symbol: str) -> float:
        """
        Get the drift percentage for a symbol from the rebalance plan.
        
        Args:
            plan: The rebalance plan
            symbol: The symbol to get drift for
            
        Returns:
            Absolute drift percentage or 0 if not found
        """
        for component in plan.get("drift_components", []):
            if component.get("name") == symbol:
                return abs(component.get("drift_percentage", 0))
        return 0
    
    def get_rebalance_history(self,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get tax-aware rebalance history with optional date filtering.
        
        Args:
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)
            
        Returns:
            List of rebalance history entries
        """
        filtered_history = self.rebalance_history
        
        if start_date:
            filtered_history = [h for h in filtered_history if h["timestamp"] >= start_date]
            
        if end_date:
            filtered_history = [h for h in filtered_history if h["timestamp"] <= end_date]
            
        return filtered_history
    
    def get_tax_impact_summary(self) -> Dict[str, Any]:
        """
        Get a summary of tax impact from rebalancing operations.
        
        Returns:
            Dict with tax impact summary
        """
        if not self.rebalance_history:
            return {
                "total_rebalances": 0,
                "total_tax_savings": 0,
                "average_tax_savings": 0
            }
        
        total_standard_impact = 0
        total_tax_aware_impact = 0
        
        for entry in self.rebalance_history:
            # This is a simplified calculation that would need to be enhanced
            # with actual tax impact calculations in a real implementation
            standard_plan = entry.get("standard_plan", {})
            tax_aware_plan = entry.get("tax_aware_plan", {})
            
            standard_impact = standard_plan.get("estimated_capital_gains", 0)
            tax_aware_impact = tax_aware_plan.get("estimated_capital_gains", 0)
            
            total_standard_impact += standard_impact
            total_tax_aware_impact += tax_aware_impact
        
        tax_savings = total_standard_impact - total_tax_aware_impact
        
        return {
            "total_rebalances": len(self.rebalance_history),
            "total_tax_savings": tax_savings,
            "average_tax_savings": tax_savings / len(self.rebalance_history) if self.rebalance_history else 0,
            "total_standard_impact": total_standard_impact,
            "total_tax_aware_impact": total_tax_aware_impact
        }
    
    def set_symbol_tax_lot_method(self, symbol: str, method: str) -> None:
        """
        Set a specific tax lot selection method for a symbol.
        
        This allows different tax lot selection strategies for different assets.
        
        Args:
            symbol: The symbol to set the method for
            method: The tax lot method (FIFO, LIFO, HIFO, LOFO, MINTAX, MAXTAX)
        """
        valid_methods = ["FIFO", "LIFO", "HIFO", "LOFO", "MINTAX", "MAXTAX"]
        
        if method not in valid_methods:
            logger.warning(f"Invalid tax lot method '{method}' for symbol '{symbol}'. Using default method.")
            return
            
        self.symbol_methods[symbol] = method
        logger.info(f"Set tax lot method for {symbol} to {method}")
    
    def reset(self) -> None:
        """
        Reset the tax-aware rebalancer.
        """
        self.rebalance_history = []
        self.symbol_methods = {}
        logger.info("Tax-Aware Rebalancer reset")