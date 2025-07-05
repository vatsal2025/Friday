import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, date, timedelta
from enum import Enum

from portfolio.tax_manager import TaxManager
from portfolio.tax_loss_harvester import TaxLossHarvester
from portfolio.tax_lot_selector import TaxLotSelector

logger = logging.getLogger(__name__)

class TaxPlanningHorizon(Enum):
    """Tax planning time horizons."""
    YEAR_END = "Year End"  # Planning for current tax year
    ONE_YEAR = "One Year"  # One year forecast
    THREE_YEAR = "Three Year"  # Three year forecast
    FIVE_YEAR = "Five Year"  # Five year forecast
    CUSTOM = "Custom"  # Custom time horizon

class TaxPlanner:
    """
    Tax Planner for tax planning and forecasting.
    
    This class provides functionality for:
    - Forecasting tax liabilities
    - Year-end tax planning
    - Tax-efficient withdrawal strategies
    - Long-term tax optimization
    - What-if scenario analysis
    """
    
    def __init__(self, 
                 tax_manager: TaxManager,
                 tax_loss_harvester: Optional[TaxLossHarvester] = None,
                 tax_lot_selector: Optional[TaxLotSelector] = None,
                 short_term_tax_rate: float = 0.35,
                 long_term_tax_rate: float = 0.15,
                 income_tax_brackets: Optional[Dict[str, List[Tuple[float, float]]]] = None,
                 default_planning_horizon: TaxPlanningHorizon = TaxPlanningHorizon.YEAR_END):
        """
        Initialize the Tax Planner.
        
        Args:
            tax_manager: The tax manager instance
            tax_loss_harvester: The tax loss harvester instance (optional)
            tax_lot_selector: The tax lot selector instance (optional)
            short_term_tax_rate: Default tax rate for short-term gains
            long_term_tax_rate: Default tax rate for long-term gains
            income_tax_brackets: Dict of income tax brackets by filing status
                e.g., {"single": [(0, 0.10), (9950, 0.12), ...], "married": [...], ...}
            default_planning_horizon: Default tax planning horizon
        """
        self.tax_manager = tax_manager
        self.tax_loss_harvester = tax_loss_harvester
        self.tax_lot_selector = tax_lot_selector
        self.short_term_tax_rate = short_term_tax_rate
        self.long_term_tax_rate = long_term_tax_rate
        self.default_planning_horizon = default_planning_horizon
        
        # Set default income tax brackets if not provided
        if income_tax_brackets is None:
            self.income_tax_brackets = {
                "single": [
                    (0, 0.10),
                    (10275, 0.12),
                    (41775, 0.22),
                    (89075, 0.24),
                    (170050, 0.32),
                    (215950, 0.35),
                    (539900, 0.37)
                ],
                "married": [
                    (0, 0.10),
                    (20550, 0.12),
                    (83550, 0.22),
                    (178150, 0.24),
                    (340100, 0.32),
                    (431900, 0.35),
                    (647850, 0.37)
                ],
                "head_of_household": [
                    (0, 0.10),
                    (14650, 0.12),
                    (55900, 0.22),
                    (89050, 0.24),
                    (170050, 0.32),
                    (215950, 0.35),
                    (539900, 0.37)
                ]
            }
        else:
            self.income_tax_brackets = income_tax_brackets
            
        # Planning scenarios
        self.scenarios = {}
        
        # Planning history
        self.planning_history = []
        
        logger.info(f"Tax Planner initialized with {default_planning_horizon.value} planning horizon")
    
    def forecast_tax_liability(self,
                             year: int,
                             expected_income: float,
                             expected_deductions: float = 0.0,
                             filing_status: str = "single",
                             include_unrealized: bool = False,
                             price_appreciation_rate: float = 0.05,
                             expected_additional_gains: float = 0.0,
                             expected_additional_losses: float = 0.0) -> Dict[str, Any]:
        """
        Forecast tax liability for a specific year.
        
        Args:
            year: Tax year to forecast
            expected_income: Expected ordinary income
            expected_deductions: Expected deductions
            filing_status: Tax filing status (single, married, head_of_household)
            include_unrealized: Whether to include unrealized gains in the forecast
            price_appreciation_rate: Expected annual price appreciation rate for unrealized positions
            expected_additional_gains: Expected additional capital gains not in tax manager
            expected_additional_losses: Expected additional capital losses not in tax manager
            
        Returns:
            Dict with tax liability forecast details
        """
        # Get current date
        current_date = datetime.now().date()
        current_year = current_date.year
        
        # Calculate days remaining in current year
        if year == current_year:
            year_end = date(current_year, 12, 31)
            days_remaining = (year_end - current_date).days
            days_in_year = 365  # Simplified, ignoring leap years
            year_fraction_remaining = days_remaining / days_in_year
        else:
            year_fraction_remaining = 0.0
        
        # Get realized gains for the year so far
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        realized_gains = self.tax_manager.get_realized_gains(
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate short-term and long-term gains/losses
        short_term_gains = sum(gain["gain_loss"] for gain in realized_gains 
                              if not gain["long_term"] and gain["gain_loss"] > 0)
        short_term_losses = sum(-gain["gain_loss"] for gain in realized_gains 
                               if not gain["long_term"] and gain["gain_loss"] < 0)
        long_term_gains = sum(gain["gain_loss"] for gain in realized_gains 
                             if gain["long_term"] and gain["gain_loss"] > 0)
        long_term_losses = sum(-gain["gain_loss"] for gain in realized_gains 
                              if gain["long_term"] and gain["gain_loss"] < 0)
        
        # Add expected additional gains/losses
        short_term_gains += max(0, expected_additional_gains * 0.5)  # Assume half are short-term
        short_term_losses += max(0, expected_additional_losses * 0.5)  # Assume half are short-term
        long_term_gains += max(0, expected_additional_gains * 0.5)  # Assume half are long-term
        long_term_losses += max(0, expected_additional_losses * 0.5)  # Assume half are long-term
        
        # Calculate unrealized gains if requested
        unrealized_short_term_gains = 0.0
        unrealized_long_term_gains = 0.0
        
        if include_unrealized:
            # Get all tax lots
            all_tax_lots = self.tax_manager.get_tax_lots()
            
            for symbol, lots in all_tax_lots.items():
                for lot in lots:
                    # Get current price (this is simplified, in reality would need market data)
                    current_price = lot["purchase_price"] * (1 + price_appreciation_rate) ** (
                        (current_date - lot["purchase_date"].date()).days / 365
                    )
                    
                    # Calculate unrealized gain/loss
                    unrealized_gain = lot["quantity"] * (current_price - lot["purchase_price"])
                    
                    # Determine if long-term or short-term
                    holding_period = (current_date - lot["purchase_date"].date()).days
                    is_long_term = holding_period > 365
                    
                    if is_long_term:
                        unrealized_long_term_gains += max(0, unrealized_gain)
                    else:
                        unrealized_short_term_gains += max(0, unrealized_gain)
        
        # Apply tax loss harvesting rules
        # 1. Short-term losses offset short-term gains
        # 2. Long-term losses offset long-term gains
        # 3. Net short-term losses offset net long-term gains
        # 4. Net long-term losses offset net short-term gains
        # 5. Remaining losses can offset up to $3,000 of ordinary income
        # 6. Excess losses are carried forward
        
        net_short_term = short_term_gains - short_term_losses
        net_long_term = long_term_gains - long_term_losses
        
        if net_short_term < 0 and net_long_term > 0:
            # Short-term losses offsetting long-term gains
            offset = min(abs(net_short_term), net_long_term)
            net_long_term -= offset
            net_short_term += offset
        elif net_long_term < 0 and net_short_term > 0:
            # Long-term losses offsetting short-term gains
            offset = min(abs(net_long_term), net_short_term)
            net_short_term -= offset
            net_long_term += offset
        
        # Calculate remaining losses that can offset ordinary income
        remaining_loss = abs(min(0, net_short_term + net_long_term))
        income_offset = min(remaining_loss, 3000)  # Max $3,000 per year
        loss_carryforward = remaining_loss - income_offset
        
        # Calculate taxable income
        taxable_income = max(0, expected_income - expected_deductions - income_offset)
        
        # Calculate income tax
        income_tax = self._calculate_income_tax(taxable_income, filing_status)
        
        # Calculate capital gains tax
        capital_gains_tax = max(0, net_short_term) * self.short_term_tax_rate + \
                           max(0, net_long_term) * self.long_term_tax_rate
        
        # Calculate total tax
        total_tax = income_tax + capital_gains_tax
        
        # If forecasting for future years, include unrealized gains that might be realized
        if year > current_year and include_unrealized:
            # Assume some percentage of unrealized gains are realized each year
            # This is a simplified model and could be made more sophisticated
            annual_realization_rate = 0.2  # 20% of unrealized gains realized each year
            years_in_future = year - current_year
            
            # Calculate how much unrealized gain would be realized by that year
            # This is a simplified model assuming compound growth
            realized_from_unrealized_short = unrealized_short_term_gains * \
                                           (1 - (1 - annual_realization_rate) ** years_in_future)
            realized_from_unrealized_long = unrealized_long_term_gains * \
                                          (1 - (1 - annual_realization_rate) ** years_in_future)
            
            # Add tax on these realized gains
            additional_tax = realized_from_unrealized_short * self.short_term_tax_rate + \
                           realized_from_unrealized_long * self.long_term_tax_rate
            
            total_tax += additional_tax
        
        # Record the forecast
        forecast = {
            "year": year,
            "expected_income": expected_income,
            "expected_deductions": expected_deductions,
            "filing_status": filing_status,
            "short_term_gains": short_term_gains,
            "short_term_losses": short_term_losses,
            "long_term_gains": long_term_gains,
            "long_term_losses": long_term_losses,
            "net_short_term": net_short_term,
            "net_long_term": net_long_term,
            "income_offset": income_offset,
            "loss_carryforward": loss_carryforward,
            "taxable_income": taxable_income,
            "income_tax": income_tax,
            "capital_gains_tax": capital_gains_tax,
            "total_tax": total_tax,
            "effective_tax_rate": total_tax / expected_income if expected_income > 0 else 0,
            "timestamp": datetime.now()
        }
        
        if include_unrealized:
            forecast.update({
                "unrealized_short_term_gains": unrealized_short_term_gains,
                "unrealized_long_term_gains": unrealized_long_term_gains,
                "price_appreciation_rate": price_appreciation_rate
            })
        
        self.planning_history.append({
            "type": "forecast",
            "data": forecast
        })
        
        return forecast
    
    def generate_year_end_plan(self,
                             year: int = None,
                             expected_income: float = 0.0,
                             expected_deductions: float = 0.0,
                             filing_status: str = "single",
                             max_net_capital_loss: float = 3000.0,
                             target_capital_gains: float = 0.0) -> Dict[str, Any]:
        """
        Generate a year-end tax plan with recommendations.
        
        Args:
            year: Tax year (default: current year)
            expected_income: Expected ordinary income
            expected_deductions: Expected deductions
            filing_status: Tax filing status
            max_net_capital_loss: Maximum net capital loss to realize
            target_capital_gains: Target capital gains amount
            
        Returns:
            Dict with year-end tax plan details and recommendations
        """
        if year is None:
            year = datetime.now().year
            
        # Get current date
        current_date = datetime.now().date()
        
        # Calculate days remaining in year
        year_end = date(year, 12, 31)
        days_remaining = (year_end - current_date).days
        
        if days_remaining < 0:
            return {"error": f"Cannot generate year-end plan for past year {year}"}
        
        # Get realized gains for the year so far
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        realized_gains = self.tax_manager.get_realized_gains(
            start_date=start_date,
            end_date=current_date
        )
        
        # Calculate short-term and long-term gains/losses so far
        short_term_gains = sum(gain["gain_loss"] for gain in realized_gains 
                              if not gain["long_term"] and gain["gain_loss"] > 0)
        short_term_losses = sum(-gain["gain_loss"] for gain in realized_gains 
                               if not gain["long_term"] and gain["gain_loss"] < 0)
        long_term_gains = sum(gain["gain_loss"] for gain in realized_gains 
                             if gain["long_term"] and gain["gain_loss"] > 0)
        long_term_losses = sum(-gain["gain_loss"] for gain in realized_gains 
                              if gain["long_term"] and gain["gain_loss"] < 0)
        
        net_short_term = short_term_gains - short_term_losses
        net_long_term = long_term_gains - long_term_losses
        net_capital_gain = net_short_term + net_long_term
        
        # Get all current tax lots
        all_tax_lots = self.tax_manager.get_tax_lots()
        
        # Identify potential tax-loss harvesting opportunities
        harvesting_opportunities = []
        gain_realization_opportunities = []
        
        for symbol, lots in all_tax_lots.items():
            for lot in lots:
                # Get current price (this is simplified, in reality would need market data)
                # For demonstration, assume 5% annual growth from purchase price
                days_held = (current_date - lot["purchase_date"].date()).days
                annual_growth_rate = 0.05  # 5% annual growth
                current_price = lot["purchase_price"] * (1 + annual_growth_rate) ** (days_held / 365)
                
                # Calculate unrealized gain/loss
                unrealized_gain = lot["quantity"] * (current_price - lot["purchase_price"])
                
                # Determine if long-term or short-term
                is_long_term = days_held > 365
                
                # Add to appropriate opportunity list
                if unrealized_gain < 0:  # Loss
                    harvesting_opportunities.append({
                        "symbol": symbol,
                        "lot_id": lot["lot_id"],
                        "quantity": lot["quantity"],
                        "purchase_price": lot["purchase_price"],
                        "purchase_date": lot["purchase_date"],
                        "current_price": current_price,
                        "unrealized_loss": -unrealized_gain,
                        "is_long_term": is_long_term,
                        "days_held": days_held
                    })
                else:  # Gain
                    gain_realization_opportunities.append({
                        "symbol": symbol,
                        "lot_id": lot["lot_id"],
                        "quantity": lot["quantity"],
                        "purchase_price": lot["purchase_price"],
                        "purchase_date": lot["purchase_date"],
                        "current_price": current_price,
                        "unrealized_gain": unrealized_gain,
                        "is_long_term": is_long_term,
                        "days_held": days_held
                    })
        
        # Sort opportunities
        harvesting_opportunities.sort(key=lambda x: x["unrealized_loss"], reverse=True)  # Largest losses first
        gain_realization_opportunities.sort(key=lambda x: (x["is_long_term"], x["unrealized_gain"]))  # Long-term, smallest gains first
        
        # Generate recommendations
        recommendations = []
        
        # 1. Tax-loss harvesting recommendations
        if net_capital_gain > target_capital_gains or net_capital_gain > 0:
            # We want to realize losses to offset gains
            loss_target = net_capital_gain - target_capital_gains
            
            if loss_target > 0:
                recommendations.append({
                    "type": "tax_loss_harvesting",
                    "description": f"Harvest losses to offset {loss_target:.2f} of capital gains",
                    "opportunities": harvesting_opportunities[:10]  # Top 10 opportunities
                })
        elif net_capital_gain < -max_net_capital_loss:
            # We already have more than the maximum deductible loss
            recommendations.append({
                "type": "defer_loss_harvesting",
                "description": f"Consider deferring additional loss harvesting as you already have {-net_capital_gain:.2f} in net capital losses (maximum deductible is {max_net_capital_loss:.2f})"
            })
        else:
            # We can realize more losses up to the maximum deductible amount
            additional_loss_target = max_net_capital_loss + net_capital_gain
            
            if additional_loss_target > 0 and harvesting_opportunities:
                recommendations.append({
                    "type": "tax_loss_harvesting",
                    "description": f"Harvest up to {additional_loss_target:.2f} in losses to offset ordinary income",
                    "opportunities": harvesting_opportunities[:10]  # Top 10 opportunities
                })
        
        # 2. Gain realization recommendations
        if net_capital_gain < target_capital_gains:
            # We want to realize some gains to use up lower tax brackets
            gain_target = target_capital_gains - net_capital_gain
            
            if gain_target > 0 and gain_realization_opportunities:
                # Filter for long-term gains if possible
                long_term_opportunities = [opp for opp in gain_realization_opportunities if opp["is_long_term"]]
                
                if long_term_opportunities:
                    recommendations.append({
                        "type": "realize_long_term_gains",
                        "description": f"Realize up to {gain_target:.2f} in long-term capital gains to utilize lower tax brackets",
                        "opportunities": long_term_opportunities[:10]  # Top 10 opportunities
                    })
                else:
                    recommendations.append({
                        "type": "realize_gains",
                        "description": f"Consider realizing up to {gain_target:.2f} in capital gains to utilize lower tax brackets",
                        "opportunities": gain_realization_opportunities[:10]  # Top 10 opportunities
                    })
        
        # 3. Long-term vs short-term holding recommendations
        near_long_term_lots = []
        
        for symbol, lots in all_tax_lots.items():
            for lot in lots:
                days_held = (current_date - lot["purchase_date"].date()).days
                days_to_long_term = max(0, 365 - days_held)
                
                if 0 < days_to_long_term <= 60:  # Within 60 days of long-term status
                    # Get current price (simplified)
                    annual_growth_rate = 0.05  # 5% annual growth
                    current_price = lot["purchase_price"] * (1 + annual_growth_rate) ** (days_held / 365)
                    
                    # Calculate unrealized gain/loss
                    unrealized_gain = lot["quantity"] * (current_price - lot["purchase_price"])
                    
                    if unrealized_gain > 0:  # Only include gains
                        near_long_term_lots.append({
                            "symbol": symbol,
                            "lot_id": lot["lot_id"],
                            "quantity": lot["quantity"],
                            "purchase_price": lot["purchase_price"],
                            "purchase_date": lot["purchase_date"],
                            "current_price": current_price,
                            "unrealized_gain": unrealized_gain,
                            "days_to_long_term": days_to_long_term
                        })
        
        if near_long_term_lots:
            near_long_term_lots.sort(key=lambda x: x["days_to_long_term"])  # Sort by days to long-term status
            
            recommendations.append({
                "type": "hold_for_long_term",
                "description": "Consider holding these positions until they qualify for long-term capital gains treatment",
                "opportunities": near_long_term_lots[:10]  # Top 10 opportunities
            })
        
        # 4. Wash sale warnings
        wash_sales = self.tax_manager.get_wash_sales(
            start_date=start_date,
            end_date=current_date
        )
        
        if wash_sales:
            # Group wash sales by symbol
            wash_sale_symbols = {}
            for wash_sale in wash_sales:
                symbol = wash_sale["symbol"]
                if symbol not in wash_sale_symbols:
                    wash_sale_symbols[symbol] = 0
                wash_sale_symbols[symbol] += wash_sale["disallowed_loss"]
            
            # Create warning for symbols with significant wash sales
            significant_wash_sales = [(symbol, amount) for symbol, amount in wash_sale_symbols.items() if amount > 1000]
            
            if significant_wash_sales:
                significant_wash_sales.sort(key=lambda x: x[1], reverse=True)  # Sort by amount, largest first
                
                recommendations.append({
                    "type": "wash_sale_warning",
                    "description": "Be cautious of wash sales when tax-loss harvesting these securities",
                    "symbols": significant_wash_sales
                })
        
        # Create the year-end plan
        plan = {
            "year": year,
            "days_remaining": days_remaining,
            "current_tax_situation": {
                "short_term_gains": short_term_gains,
                "short_term_losses": short_term_losses,
                "long_term_gains": long_term_gains,
                "long_term_losses": long_term_losses,
                "net_short_term": net_short_term,
                "net_long_term": net_long_term,
                "net_capital_gain": net_capital_gain
            },
            "expected_income": expected_income,
            "expected_deductions": expected_deductions,
            "filing_status": filing_status,
            "recommendations": recommendations,
            "timestamp": datetime.now()
        }
        
        # Add to planning history
        self.planning_history.append({
            "type": "year_end_plan",
            "data": plan
        })
        
        return plan
    
    def analyze_tax_efficiency(self, 
                             year: int = None,
                             include_unrealized: bool = True) -> Dict[str, Any]:
        """
        Analyze the tax efficiency of the portfolio.
        
        Args:
            year: Tax year to analyze (default: current year)
            include_unrealized: Whether to include unrealized gains in the analysis
            
        Returns:
            Dict with tax efficiency analysis
        """
        if year is None:
            year = datetime.now().year
            
        # Get current date
        current_date = datetime.now().date()
        
        # Get realized gains for the year
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        realized_gains = self.tax_manager.get_realized_gains(
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate total realized gains and losses
        total_realized_gains = sum(max(0, gain["gain_loss"]) for gain in realized_gains)
        total_realized_losses = sum(max(0, -gain["gain_loss"]) for gain in realized_gains)
        net_realized = total_realized_gains - total_realized_losses
        
        # Calculate short-term and long-term breakdown
        short_term_gains = sum(gain["gain_loss"] for gain in realized_gains 
                              if not gain["long_term"] and gain["gain_loss"] > 0)
        short_term_losses = sum(-gain["gain_loss"] for gain in realized_gains 
                               if not gain["long_term"] and gain["gain_loss"] < 0)
        long_term_gains = sum(gain["gain_loss"] for gain in realized_gains 
                             if gain["long_term"] and gain["gain_loss"] > 0)
        long_term_losses = sum(-gain["gain_loss"] for gain in realized_gains 
                              if gain["long_term"] and gain["gain_loss"] < 0)
        
        # Calculate tax impact
        short_term_tax = max(0, short_term_gains - short_term_losses) * self.short_term_tax_rate
        long_term_tax = max(0, long_term_gains - long_term_losses) * self.long_term_tax_rate
        total_tax = short_term_tax + long_term_tax
        
        # Calculate unrealized gains and losses if requested
        unrealized_gains = 0.0
        unrealized_losses = 0.0
        unrealized_short_term_gains = 0.0
        unrealized_long_term_gains = 0.0
        unrealized_short_term_losses = 0.0
        unrealized_long_term_losses = 0.0
        potential_tax_on_unrealized = 0.0
        
        if include_unrealized:
            # Get all tax lots
            all_tax_lots = self.tax_manager.get_tax_lots()
            
            for symbol, lots in all_tax_lots.items():
                for lot in lots:
                    # Get current price (simplified)
                    days_held = (current_date - lot["purchase_date"].date()).days
                    annual_growth_rate = 0.05  # 5% annual growth
                    current_price = lot["purchase_price"] * (1 + annual_growth_rate) ** (days_held / 365)
                    
                    # Calculate unrealized gain/loss
                    unrealized_gain = lot["quantity"] * (current_price - lot["purchase_price"])
                    
                    # Determine if long-term or short-term
                    is_long_term = days_held > 365
                    
                    if unrealized_gain > 0:  # Gain
                        unrealized_gains += unrealized_gain
                        if is_long_term:
                            unrealized_long_term_gains += unrealized_gain
                            potential_tax_on_unrealized += unrealized_gain * self.long_term_tax_rate
                        else:
                            unrealized_short_term_gains += unrealized_gain
                            potential_tax_on_unrealized += unrealized_gain * self.short_term_tax_rate
                    else:  # Loss
                        unrealized_losses += -unrealized_gain
                        if is_long_term:
                            unrealized_long_term_losses += -unrealized_gain
                        else:
                            unrealized_short_term_losses += -unrealized_gain
        
        # Calculate tax efficiency metrics
        total_gains = total_realized_gains + unrealized_gains
        total_losses = total_realized_losses + unrealized_losses
        
        # Tax efficiency ratio (lower is better)
        tax_efficiency_ratio = total_tax / total_realized_gains if total_realized_gains > 0 else 0
        
        # Long-term gain ratio (higher is better)
        long_term_gain_ratio = long_term_gains / total_realized_gains if total_realized_gains > 0 else 0
        
        # Loss harvesting efficiency (higher is better)
        loss_harvesting_efficiency = total_realized_losses / (total_realized_losses + unrealized_losses) if (total_realized_losses + unrealized_losses) > 0 else 0
        
        # Tax deferral ratio (higher is better)
        tax_deferral_ratio = unrealized_gains / (unrealized_gains + total_realized_gains) if (unrealized_gains + total_realized_gains) > 0 else 0
        
        # Create the analysis
        analysis = {
            "year": year,
            "realized": {
                "total_gains": total_realized_gains,
                "total_losses": total_realized_losses,
                "net": net_realized,
                "short_term_gains": short_term_gains,
                "short_term_losses": short_term_losses,
                "long_term_gains": long_term_gains,
                "long_term_losses": long_term_losses,
                "tax_impact": total_tax
            },
            "tax_efficiency_metrics": {
                "tax_efficiency_ratio": tax_efficiency_ratio,
                "long_term_gain_ratio": long_term_gain_ratio,
                "loss_harvesting_efficiency": loss_harvesting_efficiency,
                "tax_deferral_ratio": tax_deferral_ratio
            },
            "timestamp": datetime.now()
        }
        
        if include_unrealized:
            analysis["unrealized"] = {
                "total_gains": unrealized_gains,
                "total_losses": unrealized_losses,
                "net": unrealized_gains - unrealized_losses,
                "short_term_gains": unrealized_short_term_gains,
                "short_term_losses": unrealized_short_term_losses,
                "long_term_gains": unrealized_long_term_gains,
                "long_term_losses": unrealized_long_term_losses,
                "potential_tax": potential_tax_on_unrealized
            }
        
        # Add to planning history
        self.planning_history.append({
            "type": "tax_efficiency_analysis",
            "data": analysis
        })
        
        return analysis
    
    def create_scenario(self,
                      name: str,
                      description: str,
                      expected_income: float,
                      expected_deductions: float,
                      filing_status: str = "single",
                      planning_horizon: TaxPlanningHorizon = None,
                      custom_horizon_years: int = None,
                      expected_annual_return: float = 0.07,
                      expected_annual_income_growth: float = 0.03,
                      expected_annual_withdrawal: float = 0.0,
                      tax_loss_harvesting_enabled: bool = True,
                      tax_gain_harvesting_enabled: bool = False) -> Dict[str, Any]:
        """
        Create a tax planning scenario for what-if analysis.
        
        Args:
            name: Scenario name
            description: Scenario description
            expected_income: Expected annual income
            expected_deductions: Expected annual deductions
            filing_status: Tax filing status
            planning_horizon: Tax planning horizon
            custom_horizon_years: Custom horizon in years (if planning_horizon is CUSTOM)
            expected_annual_return: Expected annual portfolio return
            expected_annual_income_growth: Expected annual income growth rate
            expected_annual_withdrawal: Expected annual portfolio withdrawal
            tax_loss_harvesting_enabled: Whether tax-loss harvesting is enabled
            tax_gain_harvesting_enabled: Whether tax-gain harvesting is enabled
            
        Returns:
            Dict with scenario details
        """
        if planning_horizon is None:
            planning_horizon = self.default_planning_horizon
            
        if planning_horizon == TaxPlanningHorizon.CUSTOM and custom_horizon_years is None:
            custom_horizon_years = 10  # Default to 10 years for custom horizon
        
        # Create the scenario
        scenario = {
            "name": name,
            "description": description,
            "expected_income": expected_income,
            "expected_deductions": expected_deductions,
            "filing_status": filing_status,
            "planning_horizon": planning_horizon.value,
            "custom_horizon_years": custom_horizon_years,
            "expected_annual_return": expected_annual_return,
            "expected_annual_income_growth": expected_annual_income_growth,
            "expected_annual_withdrawal": expected_annual_withdrawal,
            "tax_loss_harvesting_enabled": tax_loss_harvesting_enabled,
            "tax_gain_harvesting_enabled": tax_gain_harvesting_enabled,
            "created_at": datetime.now()
        }
        
        # Store the scenario
        self.scenarios[name] = scenario
        
        return scenario
    
    def run_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Run a tax planning scenario and generate projections.
        
        Args:
            scenario_name: Name of the scenario to run
            
        Returns:
            Dict with scenario projection results
        """
        if scenario_name not in self.scenarios:
            return {"error": f"Scenario '{scenario_name}' not found"}
            
        scenario = self.scenarios[scenario_name]
        
        # Get current date and year
        current_date = datetime.now().date()
        current_year = current_date.year
        
        # Determine the number of years to project
        if scenario["planning_horizon"] == TaxPlanningHorizon.YEAR_END.value:
            projection_years = 1
        elif scenario["planning_horizon"] == TaxPlanningHorizon.ONE_YEAR.value:
            projection_years = 1
        elif scenario["planning_horizon"] == TaxPlanningHorizon.THREE_YEAR.value:
            projection_years = 3
        elif scenario["planning_horizon"] == TaxPlanningHorizon.FIVE_YEAR.value:
            projection_years = 5
        elif scenario["planning_horizon"] == TaxPlanningHorizon.CUSTOM.value:
            projection_years = scenario["custom_horizon_years"]
        else:
            projection_years = 1
        
        # Get current portfolio value (simplified)
        # In a real implementation, this would come from the portfolio manager
        portfolio_value = 1000000.0  # Example value
        
        # Generate projections for each year
        projections = []
        
        income = scenario["expected_income"]
        deductions = scenario["expected_deductions"]
        
        for year_offset in range(projection_years):
            year = current_year + year_offset
            
            # Calculate expected portfolio value for this year
            if year_offset > 0:
                # Apply annual return and withdrawal
                portfolio_value = portfolio_value * (1 + scenario["expected_annual_return"]) - scenario["expected_annual_withdrawal"]
                
                # Apply income growth
                income = income * (1 + scenario["expected_annual_income_growth"])
            
            # Estimate capital gains for the year
            estimated_capital_gains = portfolio_value * 0.02  # Assume 2% of portfolio value is realized as gains
            
            # Estimate tax-loss harvesting benefit if enabled
            tax_loss_harvesting_benefit = 0.0
            if scenario["tax_loss_harvesting_enabled"]:
                # Assume tax-loss harvesting can offset 1% of portfolio value
                tax_loss_harvesting_benefit = portfolio_value * 0.01
            
            # Estimate tax-gain harvesting benefit if enabled
            tax_gain_harvesting_benefit = 0.0
            if scenario["tax_gain_harvesting_enabled"]:
                # Assume tax-gain harvesting can optimize 0.5% of portfolio value
                tax_gain_harvesting_benefit = portfolio_value * 0.005
            
            # Calculate tax liability
            forecast = self.forecast_tax_liability(
                year=year,
                expected_income=income,
                expected_deductions=deductions,
                filing_status=scenario["filing_status"],
                include_unrealized=False,
                expected_additional_gains=estimated_capital_gains,
                expected_additional_losses=tax_loss_harvesting_benefit
            )
            
            # Add projection for this year
            projection = {
                "year": year,
                "portfolio_value": portfolio_value,
                "income": income,
                "deductions": deductions,
                "estimated_capital_gains": estimated_capital_gains,
                "tax_loss_harvesting_benefit": tax_loss_harvesting_benefit,
                "tax_gain_harvesting_benefit": tax_gain_harvesting_benefit,
                "tax_liability": forecast["total_tax"],
                "after_tax_income": income - forecast["income_tax"],
                "effective_tax_rate": forecast["effective_tax_rate"]
            }
            
            projections.append(projection)
        
        # Calculate summary metrics
        total_tax = sum(proj["tax_liability"] for proj in projections)
        average_tax_rate = sum(proj["effective_tax_rate"] for proj in projections) / len(projections)
        total_after_tax_income = sum(proj["after_tax_income"] for proj in projections)
        
        # Create the result
        result = {
            "scenario_name": scenario_name,
            "scenario": scenario,
            "projections": projections,
            "summary": {
                "total_tax": total_tax,
                "average_tax_rate": average_tax_rate,
                "total_after_tax_income": total_after_tax_income,
                "final_portfolio_value": projections[-1]["portfolio_value"] if projections else 0
            },
            "timestamp": datetime.now()
        }
        
        # Add to planning history
        self.planning_history.append({
            "type": "scenario_projection",
            "data": result
        })
        
        return result
    
    def compare_scenarios(self, scenario_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple tax planning scenarios.
        
        Args:
            scenario_names: List of scenario names to compare
            
        Returns:
            Dict with scenario comparison results
        """
        if not scenario_names:
            return {"error": "No scenarios provided for comparison"}
            
        # Check if all scenarios exist
        missing_scenarios = [name for name in scenario_names if name not in self.scenarios]
        if missing_scenarios:
            return {"error": f"Scenarios not found: {', '.join(missing_scenarios)}"}
        
        # Run each scenario
        scenario_results = {}
        for name in scenario_names:
            scenario_results[name] = self.run_scenario(name)
        
        # Create comparison metrics
        comparison = {
            "scenarios": scenario_names,
            "metrics": {
                "total_tax": {},
                "average_tax_rate": {},
                "total_after_tax_income": {},
                "final_portfolio_value": {}
            },
            "best_scenario": {
                "lowest_tax": None,
                "lowest_tax_rate": None,
                "highest_after_tax_income": None,
                "highest_final_portfolio_value": None
            },
            "timestamp": datetime.now()
        }
        
        # Calculate metrics for each scenario
        for name, result in scenario_results.items():
            if "error" in result:
                continue
                
            comparison["metrics"]["total_tax"][name] = result["summary"]["total_tax"]
            comparison["metrics"]["average_tax_rate"][name] = result["summary"]["average_tax_rate"]
            comparison["metrics"]["total_after_tax_income"][name] = result["summary"]["total_after_tax_income"]
            comparison["metrics"]["final_portfolio_value"][name] = result["summary"]["final_portfolio_value"]
        
        # Determine best scenarios
        if comparison["metrics"]["total_tax"]:
            comparison["best_scenario"]["lowest_tax"] = min(
                comparison["metrics"]["total_tax"].items(),
                key=lambda x: x[1]
            )[0]
            
        if comparison["metrics"]["average_tax_rate"]:
            comparison["best_scenario"]["lowest_tax_rate"] = min(
                comparison["metrics"]["average_tax_rate"].items(),
                key=lambda x: x[1]
            )[0]
            
        if comparison["metrics"]["total_after_tax_income"]:
            comparison["best_scenario"]["highest_after_tax_income"] = max(
                comparison["metrics"]["total_after_tax_income"].items(),
                key=lambda x: x[1]
            )[0]
            
        if comparison["metrics"]["final_portfolio_value"]:
            comparison["best_scenario"]["highest_final_portfolio_value"] = max(
                comparison["metrics"]["final_portfolio_value"].items(),
                key=lambda x: x[1]
            )[0]
        
        # Add to planning history
        self.planning_history.append({
            "type": "scenario_comparison",
            "data": comparison
        })
        
        return comparison
    
    def get_planning_history(self,
                           history_type: Optional[str] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get tax planning history with optional filtering.
        
        Args:
            history_type: Filter by history type (forecast, year_end_plan, etc.)
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of planning history entries
        """
        filtered_history = self.planning_history
        
        if history_type:
            filtered_history = [h for h in filtered_history if h["type"] == history_type]
            
        if start_date:
            filtered_history = [h for h in filtered_history if h["data"]["timestamp"] >= start_date]
            
        if end_date:
            filtered_history = [h for h in filtered_history if h["data"]["timestamp"] <= end_date]
            
        return filtered_history
    
    def _calculate_income_tax(self, taxable_income: float, filing_status: str) -> float:
        """
        Calculate income tax based on tax brackets.
        
        Args:
            taxable_income: Taxable income amount
            filing_status: Tax filing status
            
        Returns:
            Calculated income tax
        """
        if filing_status not in self.income_tax_brackets:
            # Default to single if filing status not found
            filing_status = "single"
            
        brackets = self.income_tax_brackets[filing_status]
        
        # Sort brackets by income threshold
        brackets = sorted(brackets, key=lambda x: x[0])
        
        tax = 0.0
        prev_threshold = 0.0
        
        for threshold, rate in brackets:
            if taxable_income <= threshold:
                tax += (taxable_income - prev_threshold) * rate
                break
            else:
                tax += (threshold - prev_threshold) * rate
                prev_threshold = threshold
        else:
            # If we've gone through all brackets, calculate tax on the remaining income
            tax += (taxable_income - prev_threshold) * brackets[-1][1]
        
        return tax
    
    def reset(self) -> None:
        """
        Reset the tax planner.
        """
        self.scenarios = {}
        self.planning_history = []
        
        logger.info("Tax Planner reset")