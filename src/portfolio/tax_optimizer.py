import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, date, timedelta
from enum import Enum

from portfolio.tax_manager import TaxManager
from portfolio.tax_lot_selector import TaxLotSelector, TaxLotSelectionStrategy
from portfolio.tax_loss_harvester import TaxLossHarvester, HarvestingStrategy
from portfolio.tax_aware_rebalancer import TaxAwareRebalancer, TaxAwareRebalanceMethod
from portfolio.tax_planner import TaxPlanner, TaxPlanningHorizon
from portfolio.tax_reporting import TaxReporting

# Optional import for visualization
try:
    from analytics.visualization import TaxVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class TaxOptimizationStrategy(Enum):
    """Strategies for overall tax optimization."""
    MINIMIZE_CURRENT_TAX = "Minimize Current Tax"  # Focus on minimizing current year tax
    MAXIMIZE_AFTER_TAX_RETURN = "Maximize After-Tax Return"  # Focus on long-term after-tax return
    BALANCED = "Balanced"  # Balance current tax minimization with long-term optimization
    CUSTOM = "Custom"  # Custom optimization strategy

class TaxOptimizer:
    """
    Tax Optimizer for comprehensive tax optimization.
    
    This class integrates all tax optimization components:
    - Tax lot selection strategies
    - Automated tax-loss harvesting
    - Tax-aware rebalancing
    - Tax planning and forecasting
    - Comprehensive tax reporting
    
    It provides a unified interface for optimizing after-tax returns
    through sophisticated tax strategies and algorithms.
    """
    
    def __init__(self, 
                 tax_manager: TaxManager,
                 tax_lot_selector: Optional[TaxLotSelector] = None,
                 tax_loss_harvester: Optional[TaxLossHarvester] = None,
                 tax_aware_rebalancer: Optional[TaxAwareRebalancer] = None,
                 tax_planner: Optional[TaxPlanner] = None,
                 tax_reporting: Optional[TaxReporting] = None,
                 optimization_strategy: TaxOptimizationStrategy = TaxOptimizationStrategy.BALANCED,
                 short_term_tax_rate: float = 0.35,
                 long_term_tax_rate: float = 0.15,
                 tax_alpha_benchmark: Optional[str] = None):
        """
        Initialize the Tax Optimizer.
        
        Args:
            tax_manager: The tax manager instance
            tax_lot_selector: The tax lot selector instance (optional)
            tax_loss_harvester: The tax loss harvester instance (optional)
            tax_aware_rebalancer: The tax-aware rebalancer instance (optional)
            tax_planner: The tax planner instance (optional)
            tax_reporting: The tax reporting instance (optional)
            optimization_strategy: Overall tax optimization strategy
            short_term_tax_rate: Default tax rate for short-term gains
            long_term_tax_rate: Default tax rate for long-term gains
            tax_alpha_benchmark: Benchmark for tax alpha calculation
        """
        self.tax_manager = tax_manager
        self.optimization_strategy = optimization_strategy
        self.short_term_tax_rate = short_term_tax_rate
        self.long_term_tax_rate = long_term_tax_rate
        self.tax_alpha_benchmark = tax_alpha_benchmark
        
        # Initialize components if not provided
        if tax_lot_selector is None:
            self.tax_lot_selector = TaxLotSelector(
                tax_manager=tax_manager,
                default_strategy=TaxLotSelectionStrategy.MIN_TAX,
                short_term_tax_rate=short_term_tax_rate,
                long_term_tax_rate=long_term_tax_rate
            )
        else:
            self.tax_lot_selector = tax_lot_selector
            
        if tax_loss_harvester is None:
            self.tax_loss_harvester = TaxLossHarvester(
                tax_manager=tax_manager,
                harvesting_strategy=HarvestingStrategy.THRESHOLD_BASED,
                min_loss_threshold=500.0,  # $500 minimum loss
                min_loss_threshold_percent=0.05,  # 5% minimum loss
                wash_sale_window_days=30
            )
        else:
            self.tax_loss_harvester = tax_loss_harvester
            
        if tax_aware_rebalancer is None:
            self.tax_aware_rebalancer = None  # Will be initialized when allocation_manager is provided
        else:
            self.tax_aware_rebalancer = tax_aware_rebalancer
            
        if tax_planner is None:
            self.tax_planner = TaxPlanner(
                tax_manager=tax_manager,
                tax_loss_harvester=self.tax_loss_harvester,
                tax_lot_selector=self.tax_lot_selector,
                short_term_tax_rate=short_term_tax_rate,
                long_term_tax_rate=long_term_tax_rate
            )
        else:
            self.tax_planner = tax_planner
            
        if tax_reporting is None:
            self.tax_reporting = TaxReporting(
                tax_manager=tax_manager
            )
        else:
            self.tax_reporting = tax_reporting
            
        # Initialize visualization component if available
        if VISUALIZATION_AVAILABLE:
            try:
                self.visualizer = TaxVisualizer()
            except Exception as e:
                logger.warning(f"Failed to initialize TaxVisualizer: {e}")
                self.visualizer = None
            
        # Optimization configuration
        self.config = {
            "account_types": {
                "taxable": {
                    "tax_lot_selection": {
                        "default_method": "min_tax",
                        "allowed_methods": ["fifo", "lifo", "hifo", "lofo", "min_tax", "max_tax", "tax_efficient"],
                        "dynamic_selection": True
                    },
                    "tax_loss_harvesting": {
                        "enabled": True,
                        "threshold_absolute": 500.0,  # Minimum loss in dollars
                        "threshold_relative": 0.05,  # Minimum loss as percentage
                        "wash_sale_window": 30,  # Days
                        "replacement_strategy": "similar_etf",
                        "max_harvest_per_year": 50000.0,
                        "harvest_schedule": "weekly"
                    },
                    "capital_gains": {
                        "short_term_rate": short_term_tax_rate,
                        "long_term_rate": long_term_tax_rate,
                        "budget_enabled": True,
                        "annual_budget": 25000.0
                    }
                },
                "ira": {
                    "tax_lot_selection": {
                        "default_method": "fifo",
                        "allowed_methods": ["fifo"],
                        "dynamic_selection": False
                    },
                    "tax_loss_harvesting": {
                        "enabled": False
                    }
                }
            },
            "asset_location": {
                "enabled": True,
                "preferences": {
                    "high_yield_bonds": "ira",
                    "reits": "ira",
                    "growth_stocks": "taxable",
                    "municipal_bonds": "taxable"
                }
            },
            "rebalancing": {
                "tax_aware": True,
                "max_tax_impact": 0.1,  # Maximum tax cost as percentage of portfolio
                "prioritize_harvesting": True,
                "use_cash_flows": True
            },
            "reporting": {
                "realized_gains": True,
                "unrealized_gains": True,
                "tax_efficiency": True,
                "tax_alpha": True,
                "tax_drag": True
            },
            "planning": {
                "forecast_horizon": 5,  # Years
                "year_end_planning": True,
                "distribution_analysis": True,
                "withdrawal_strategy": "tax_efficient"
            }
        }
        
        # Optimization history
        self.optimization_history = []
        
        logger.info(f"Tax Optimizer initialized with {optimization_strategy.value} strategy")
    
    def set_allocation_manager(self, allocation_manager: Any) -> None:
        """
        Set the allocation manager for tax-aware rebalancing.
        
        Args:
            allocation_manager: The allocation manager instance
        """
        if self.tax_aware_rebalancer is None:
            self.tax_aware_rebalancer = TaxAwareRebalancer(
                allocation_manager=allocation_manager,
                tax_manager=self.tax_manager,
                tax_aware_method=TaxAwareRebalanceMethod.MINIMIZE_TAX_IMPACT,
                capital_gains_budget=self.config["account_types"]["taxable"]["capital_gains"]["annual_budget"],
                long_term_preference_factor=0.5,
                harvest_losses_first=True
            )
        else:
            self.tax_aware_rebalancer.allocation_manager = allocation_manager
            
        logger.info("Allocation manager set for tax-aware rebalancing")
    
    def set_optimization_strategy(self, strategy: TaxOptimizationStrategy) -> None:
        """
        Set the overall tax optimization strategy.
        
        Args:
            strategy: The tax optimization strategy
        """
        self.optimization_strategy = strategy
        
        # Update component configurations based on strategy
        if strategy == TaxOptimizationStrategy.MINIMIZE_CURRENT_TAX:
            # Configure for current tax minimization
            self.tax_lot_selector.default_strategy = TaxLotSelectionStrategy.MIN_TAX
            self.tax_loss_harvester.min_loss_threshold = 100.0  # Lower threshold to harvest more losses
            self.tax_loss_harvester.min_loss_threshold_percent = 0.01  # 1% threshold
            
            if self.tax_aware_rebalancer:
                self.tax_aware_rebalancer.tax_aware_method = TaxAwareRebalanceMethod.MINIMIZE_TAX_IMPACT
                self.tax_aware_rebalancer.capital_gains_budget = 0.0  # Minimize gains
                
        elif strategy == TaxOptimizationStrategy.MAXIMIZE_AFTER_TAX_RETURN:
            # Configure for long-term after-tax return
            self.tax_lot_selector.default_strategy = TaxLotSelectionStrategy.TAX_EFFICIENT
            self.tax_loss_harvester.min_loss_threshold = 500.0  # Standard threshold
            self.tax_loss_harvester.min_loss_threshold_percent = 0.05  # 5% threshold
            
            if self.tax_aware_rebalancer:
                self.tax_aware_rebalancer.tax_aware_method = TaxAwareRebalanceMethod.MULTI_PERIOD
                self.tax_aware_rebalancer.capital_gains_budget = 25000.0  # Allow more gains for better allocation
                
        elif strategy == TaxOptimizationStrategy.BALANCED:
            # Configure for balanced approach
            self.tax_lot_selector.default_strategy = TaxLotSelectionStrategy.MIN_TAX
            self.tax_loss_harvester.min_loss_threshold = 300.0  # Moderate threshold
            self.tax_loss_harvester.min_loss_threshold_percent = 0.03  # 3% threshold
            
            if self.tax_aware_rebalancer:
                self.tax_aware_rebalancer.tax_aware_method = TaxAwareRebalanceMethod.CAPITAL_GAINS_BUDGET
                self.tax_aware_rebalancer.capital_gains_budget = 10000.0  # Moderate gains budget
        
        logger.info(f"Tax optimization strategy set to {strategy.value}")
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update the tax optimization configuration.
        
        Args:
            config_updates: Dict with configuration updates
        """
        # Recursively update the configuration
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_dict(d[k], v)
                else:
                    d[k] = v
        
        update_dict(self.config, config_updates)
        
        # Update component configurations based on new config
        self._apply_config_to_components()
        
        logger.info("Tax optimization configuration updated")
    
    def _apply_config_to_components(self) -> None:
        """
        Apply the current configuration to all tax optimization components.
        """
        # Update tax lot selector
        taxable_config = self.config["account_types"]["taxable"]
        self.tax_lot_selector.default_strategy = getattr(
            TaxLotSelectionStrategy, 
            taxable_config["tax_lot_selection"]["default_method"].upper()
        )
        
        # Update tax loss harvester
        harvesting_config = taxable_config["tax_loss_harvesting"]
        self.tax_loss_harvester.min_loss_threshold = harvesting_config["threshold_absolute"]
        self.tax_loss_harvester.min_loss_threshold_percent = harvesting_config["threshold_relative"]
        self.tax_loss_harvester.wash_sale_window_days = harvesting_config["wash_sale_window"]
        
        # Update tax-aware rebalancer if available
        if self.tax_aware_rebalancer:
            rebalancing_config = self.config["rebalancing"]
            self.tax_aware_rebalancer.capital_gains_budget = taxable_config["capital_gains"]["annual_budget"]
            self.tax_aware_rebalancer.harvest_losses_first = rebalancing_config["prioritize_harvesting"]
    
    def optimize_tax_lots_for_sale(self, 
                                 symbol: str, 
                                 quantity: float, 
                                 current_price: float,
                                 account_type: str = "taxable",
                                 strategy: Optional[TaxLotSelectionStrategy] = None) -> Dict[str, Any]:
        """
        Optimize tax lot selection for a sale.
        
        Args:
            symbol: The security symbol
            quantity: Quantity to sell
            current_price: Current price per share/unit
            account_type: Account type (taxable, ira, etc.)
            strategy: Specific tax lot selection strategy (optional)
            
        Returns:
            Dict with optimized tax lot selection details
        """
        # Get account-specific configuration
        account_config = self.config["account_types"].get(account_type, self.config["account_types"]["taxable"])
        
        # Determine strategy to use
        if strategy is None:
            if account_config["tax_lot_selection"]["dynamic_selection"]:
                # Use dynamic selection based on current tax situation
                strategy = self._determine_dynamic_lot_strategy(symbol, quantity, current_price, account_type)
            else:
                # Use account default strategy
                strategy_name = account_config["tax_lot_selection"]["default_method"].upper()
                strategy = getattr(TaxLotSelectionStrategy, strategy_name)
        
        # Select lots using the determined strategy
        selected_lots = self.tax_lot_selector.select_lots(
            symbol=symbol,
            quantity=quantity,
            current_price=current_price,
            strategy=strategy
        )
        
        # Calculate tax impact
        tax_impact = self.tax_lot_selector.calculate_tax_impact(
            selected_lots=selected_lots,
            current_price=current_price
        )
        
        # Record the optimization
        optimization = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "quantity": quantity,
            "current_price": current_price,
            "account_type": account_type,
            "strategy": strategy.value,
            "selected_lots": selected_lots,
            "tax_impact": tax_impact
        }
        
        self.optimization_history.append({
            "type": "tax_lot_selection",
            "data": optimization
        })
        
        return optimization
    
    def _determine_dynamic_lot_strategy(self, 
                                     symbol: str, 
                                     quantity: float, 
                                     current_price: float,
                                     account_type: str) -> TaxLotSelectionStrategy:
        """
        Dynamically determine the optimal tax lot selection strategy based on current tax situation.
        
        Args:
            symbol: The security symbol
            quantity: Quantity to sell
            current_price: Current price per share/unit
            account_type: Account type
            
        Returns:
            The optimal tax lot selection strategy
        """
        # For non-taxable accounts, use FIFO
        if account_type != "taxable":
            return TaxLotSelectionStrategy.FIFO
        
        # Get current year's realized gains/losses
        current_year = datetime.now().year
        start_date = date(current_year, 1, 1)
        end_date = date(current_year, 12, 31)
        
        realized_gains = self.tax_manager.get_realized_gains(
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate net realized gain/loss for the year
        net_gain = sum(gain["gain_loss"] for gain in realized_gains)
        
        # Compare different strategies
        comparison = self.tax_lot_selector.compare_strategies(
            symbol=symbol,
            quantity=quantity,
            current_price=current_price
        )
        
        # If we have net losses for the year, consider realizing gains
        if net_gain < -3000:  # More than $3000 in losses (max deduction)
            # Look for strategy with highest gains or lowest losses
            strategies_by_gain = sorted(
                comparison["strategies"],
                key=lambda x: x["total_gain_loss"],
                reverse=True
            )
            
            # Return the strategy with highest gain (or lowest loss)
            return getattr(TaxLotSelectionStrategy, strategies_by_gain[0]["strategy"])
            
        # If we have significant gains, look to minimize additional gains
        elif net_gain > 10000:  # More than $10,000 in gains
            # Use MIN_TAX strategy to minimize additional tax impact
            return TaxLotSelectionStrategy.MIN_TAX
            
        # For moderate gain/loss situation, use TAX_EFFICIENT
        else:
            return TaxLotSelectionStrategy.TAX_EFFICIENT
    
    def identify_tax_optimization_opportunities(self, 
                                            portfolio_values: Dict[str, float],
                                            current_prices: Dict[str, float],
                                            account_types: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Identify comprehensive tax optimization opportunities.
        
        Args:
            portfolio_values: Dict of {symbol: value}
            current_prices: Dict of {symbol: price}
            account_types: Dict of {symbol: account_type}
            
        Returns:
            Dict with tax optimization opportunities
        """
        timestamp = datetime.now()
        
        # Default all securities to taxable if account_types not provided
        if account_types is None:
            account_types = {symbol: "taxable" for symbol in portfolio_values.keys()}
        
        # Get current year's realized gains/losses
        current_year = timestamp.year
        start_date = date(current_year, 1, 1)
        end_date = date(current_year, 12, 31)
        
        realized_gains = self.tax_manager.get_realized_gains(
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate net realized gain/loss for the year
        net_short_term_gain = sum(gain["gain_loss"] for gain in realized_gains if not gain["long_term"])
        net_long_term_gain = sum(gain["gain_loss"] for gain in realized_gains if gain["long_term"])
        net_gain = net_short_term_gain + net_long_term_gain
        
        # Identify tax-loss harvesting opportunities
        harvesting_opportunities = []
        taxable_symbols = [symbol for symbol, account in account_types.items() if account == "taxable"]
        
        taxable_portfolio = {symbol: portfolio_values[symbol] for symbol in taxable_symbols if symbol in portfolio_values}
        taxable_prices = {symbol: current_prices[symbol] for symbol in taxable_symbols if symbol in current_prices}
        
        if taxable_portfolio and taxable_prices:
            harvesting_opportunities = self.tax_loss_harvester.find_harvesting_opportunities(
                portfolio_values=taxable_portfolio,
                current_prices=taxable_prices,
                timestamp=timestamp
            )
        
        # Identify tax-efficient rebalancing opportunities
        rebalancing_opportunities = None
        if self.tax_aware_rebalancer and self.tax_aware_rebalancer.allocation_manager:
            # Get current allocation
            current_allocation = self.tax_aware_rebalancer.allocation_manager.get_current_allocation(
                portfolio_values=portfolio_values
            )
            
            # Get target allocation
            target_allocation = self.tax_aware_rebalancer.allocation_manager.get_target_allocation()
            
            # Calculate drift
            drift = {}
            for symbol, current in current_allocation.items():
                if symbol in target_allocation:
                    drift[symbol] = current - target_allocation[symbol]
            
            # Identify symbols that need rebalancing
            rebalancing_opportunities = {
                "overweight": [],
                "underweight": []
            }
            
            for symbol, drift_value in drift.items():
                if drift_value > 0.02:  # More than 2% overweight
                    rebalancing_opportunities["overweight"].append({
                        "symbol": symbol,
                        "drift": drift_value,
                        "account_type": account_types.get(symbol, "taxable")
                    })
                elif drift_value < -0.02:  # More than 2% underweight
                    rebalancing_opportunities["underweight"].append({
                        "symbol": symbol,
                        "drift": drift_value,
                        "account_type": account_types.get(symbol, "taxable")
                    })
        
        # Identify year-end planning opportunities
        year_end_opportunities = None
        days_to_year_end = (date(current_year, 12, 31) - date.today()).days
        
        if days_to_year_end <= 60:  # Within 60 days of year-end
            year_end_opportunities = self.tax_planner.generate_year_end_plan(
                year=current_year,
                expected_income=100000.0,  # Example value, would be provided by user
                expected_deductions=24000.0,  # Example value, would be provided by user
                filing_status="single"  # Example value, would be provided by user
            )
        
        # Compile all opportunities
        opportunities = {
            "timestamp": timestamp,
            "net_realized_gains": {
                "short_term": net_short_term_gain,
                "long_term": net_long_term_gain,
                "total": net_gain
            },
            "tax_loss_harvesting": {
                "opportunities": len(harvesting_opportunities),
                "potential_tax_savings": sum(opp["estimated_tax_benefit"] for opp in harvesting_opportunities) if harvesting_opportunities else 0,
                "details": harvesting_opportunities
            }
        }
        
        if rebalancing_opportunities:
            opportunities["tax_efficient_rebalancing"] = rebalancing_opportunities
            
        if year_end_opportunities:
            opportunities["year_end_planning"] = year_end_opportunities
        
        # Record the optimization
        self.optimization_history.append({
            "type": "opportunity_identification",
            "data": opportunities
        })
        
        return opportunities
    
    def execute_tax_optimization_plan(self, 
                                   plan: Dict[str, Any],
                                   execution_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a tax optimization plan.
        
        Args:
            plan: The tax optimization plan to execute
            execution_context: Additional context for execution (e.g., prices, constraints)
            
        Returns:
            Dict with execution results
        """
        timestamp = datetime.now()
        results = {
            "timestamp": timestamp,
            "plan_type": plan.get("type", "unknown"),
            "executed_actions": []
        }
        
        # Execute based on plan type
        if plan.get("type") == "tax_loss_harvesting":
            # Execute tax-loss harvesting plan
            if "harvests" in plan:
                harvest_results = self.tax_loss_harvester.execute_harvest_plan(
                    plan=plan,
                    current_prices=execution_context.get("current_prices", {}),
                    timestamp=timestamp
                )
                
                results["harvest_results"] = harvest_results
                results["executed_actions"].extend([f"Harvested {h['symbol']}" for h in plan.get("harvests", [])])
                
        elif plan.get("type") == "tax_aware_rebalancing":
            # Execute tax-aware rebalancing plan
            if self.tax_aware_rebalancer and "trades" in plan:
                # In a real implementation, this would execute the trades
                # For now, we'll just record the planned trades
                results["rebalance_results"] = {
                    "trades": plan["trades"],
                    "estimated_tax_impact": plan.get("tax_impact_analysis", {}).get("total_tax_impact", 0)
                }
                
                results["executed_actions"].extend(
                    [f"{t['action']} {t['name']}" for t in plan.get("trades", [])]
                )
        
        # Record the execution
        self.optimization_history.append({
            "type": "plan_execution",
            "data": results
        })
        
        return results
    
    def calculate_tax_efficiency_metrics(self, 
                                      start_date: Optional[date] = None,
                                      end_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Calculate tax efficiency metrics for the portfolio.
        
        Args:
            start_date: Start date for the calculation period
            end_date: End date for the calculation period
            
        Returns:
            Dict with tax efficiency metrics
        """
        # Set default dates if not provided
        if start_date is None:
            start_date = date(datetime.now().year, 1, 1)
            
        if end_date is None:
            end_date = date.today()
        
        # Get realized gains for the period
        realized_gains = self.tax_manager.get_realized_gains(
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate metrics
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
        
        # Calculate tax efficiency metrics
        tax_efficiency_ratio = total_tax / total_realized_gains if total_realized_gains > 0 else 0
        long_term_gain_ratio = long_term_gains / total_realized_gains if total_realized_gains > 0 else 0
        loss_harvesting_efficiency = total_realized_losses / (total_realized_losses + total_realized_gains) if (total_realized_losses + total_realized_gains) > 0 else 0
        
        # Get tax-loss harvesting history
        if hasattr(self.tax_loss_harvester, 'get_harvest_history'):
            harvest_history = self.tax_loss_harvester.get_harvest_history(
                start_date=start_date,
                end_date=end_date
            )
            
            total_harvested = sum(h["realized_loss"] for h in harvest_history)
            tax_benefit_from_harvesting = total_harvested * self.short_term_tax_rate
        else:
            total_harvested = 0
            tax_benefit_from_harvesting = 0
        
        # Create the metrics
        metrics = {
            "period": {
                "start_date": start_date,
                "end_date": end_date
            },
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
            "tax_efficiency": {
                "tax_efficiency_ratio": tax_efficiency_ratio,
                "long_term_gain_ratio": long_term_gain_ratio,
                "loss_harvesting_efficiency": loss_harvesting_efficiency,
                "tax_benefit_from_harvesting": tax_benefit_from_harvesting
            },
            "timestamp": datetime.now()
        }
        
        # Record the metrics
        self.optimization_history.append({
            "type": "tax_efficiency_metrics",
            "data": metrics
        })
        
        return metrics
    
    def generate_tax_optimization_report(self, 
                                      year: Optional[int] = None,
                                      report_type: str = "comprehensive",
                                      format: str = "dict") -> Any:
        """
        Generate a comprehensive tax optimization report.
        
        Args:
            year: Tax year for the report (default: current year)
            report_type: Type of report (comprehensive, summary, harvesting, etc.)
            format: Output format (dict, dataframe, json, csv, excel)
            
        Returns:
            Tax optimization report in the specified format
        """
        if year is None:
            year = datetime.now().year
            
        # Set date range for the report
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        # Generate the appropriate report based on type
        if report_type == "comprehensive":
            # Generate a comprehensive tax report
            tax_report = self.tax_reporting.generate_tax_report(
                year=year,
                include_wash_sales=True,
                format=format
            )
            
            # Calculate tax efficiency metrics
            efficiency_metrics = self.calculate_tax_efficiency_metrics(
                start_date=start_date,
                end_date=end_date
            )
            
            # Get tax-loss harvesting summary
            if hasattr(self.tax_loss_harvester, 'get_yearly_harvest_summary'):
                harvesting_summary = self.tax_loss_harvester.get_yearly_harvest_summary(year=year)
            else:
                harvesting_summary = None
            
            # Generate visualizations if requested
            visualizations = None
            if format in ['dict', 'json'] and VISUALIZATION_AVAILABLE and hasattr(self, 'visualizer') and self.visualizer is not None:
                try:
                    from analytics.visualization import TaxVisualizer
                    
                    # Create visualizer if not already available
                    if not hasattr(self, 'visualizer'):
                        self.visualizer = TaxVisualizer()
                    
                    # Get optimization history for visualization
                    optimization_history = self.get_optimization_history(
                        type_filter='tax_efficiency_metrics',
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Generate tax optimization metrics visualization
                    if optimization_history:
                        tax_optimization_viz = self.visualizer.plot_tax_optimization_metrics(
                            optimization_history=optimization_history,
                            title=f"Tax Optimization Metrics for {year}",
                            interactive=True
                        )
                        
                        # Extract tax efficiency metrics for radar chart
                        tax_efficiency_viz = self.visualizer.plot_tax_efficiency_metrics(
                            tax_metrics=efficiency_metrics['tax_efficiency'],
                            title=f"Tax Efficiency Metrics for {year}",
                            interactive=True
                        )
                        
                        visualizations = {
                            "tax_optimization_metrics": tax_optimization_viz,
                            "tax_efficiency_metrics": tax_efficiency_viz
                        }
                except ImportError:
                    # Visualization module not available
                    visualizations = None
            
            # Compile the comprehensive report
            report = {
                "year": year,
                "tax_report": tax_report,
                "tax_efficiency": efficiency_metrics,
                "tax_loss_harvesting": harvesting_summary,
                "visualizations": visualizations,
                "timestamp": datetime.now()
            }
            
        elif report_type == "harvesting":
            # Generate a tax-loss harvesting report
            if hasattr(self.tax_loss_harvester, 'get_harvest_history'):
                harvest_history = self.tax_loss_harvester.get_harvest_history(
                    start_date=start_date,
                    end_date=end_date
                )
                
                if hasattr(self.tax_loss_harvester, 'get_yearly_harvest_summary'):
                    harvest_summary = self.tax_loss_harvester.get_yearly_harvest_summary(year=year)
                else:
                    harvest_summary = None
                
                report = {
                    "year": year,
                    "harvest_history": harvest_history,
                    "harvest_summary": harvest_summary,
                    "timestamp": datetime.now()
                }
            else:
                report = {
                    "year": year,
                    "error": "Tax loss harvester does not support harvest history",
                    "timestamp": datetime.now()
                }
                
        elif report_type == "planning":
            # Generate a tax planning report
            if hasattr(self.tax_planner, 'forecast_tax_liability'):
                forecast = self.tax_planner.forecast_tax_liability(
                    year=year,
                    expected_income=100000.0,  # Example value, would be provided by user
                    expected_deductions=24000.0,  # Example value, would be provided by user
                    filing_status="single"  # Example value, would be provided by user
                )
                
                year_end_plan = self.tax_planner.generate_year_end_plan(
                    year=year,
                    expected_income=100000.0,  # Example value, would be provided by user
                    expected_deductions=24000.0,  # Example value, would be provided by user
                    filing_status="single"  # Example value, would be provided by user
                )
                
                report = {
                    "year": year,
                    "tax_forecast": forecast,
                    "year_end_plan": year_end_plan,
                    "timestamp": datetime.now()
                }
            else:
                report = {
                    "year": year,
                    "error": "Tax planner does not support tax forecasting",
                    "timestamp": datetime.now()
                }
        else:
            # Default to basic report
            report = {
                "year": year,
                "realized_gains": self.tax_manager.get_realized_gains(
                    start_date=start_date,
                    end_date=end_date
                ),
                "timestamp": datetime.now()
            }
        
        # Convert to requested format
        if format == "dataframe" and report_type != "comprehensive":
            return pd.DataFrame(report)
        elif format == "json":
            return pd.json.dumps(report)
        elif format == "csv" and report_type != "comprehensive":
            return pd.DataFrame(report).to_csv()
        elif format == "excel" and report_type != "comprehensive":
            # In a real implementation, this would return an Excel file
            return pd.DataFrame(report)
        else:
            return report
    
    def get_optimization_history(self,
                               history_type: Optional[str] = None,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get tax optimization history with optional filtering.
        
        Args:
            history_type: Filter by history type
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of optimization history entries
        """
        filtered_history = self.optimization_history
        
        if history_type:
            filtered_history = [h for h in filtered_history if h["type"] == history_type]
            
        if start_date:
            filtered_history = [h for h in filtered_history if h["data"]["timestamp"] >= start_date]
            
        if end_date:
            filtered_history = [h for h in filtered_history if h["data"]["timestamp"] <= end_date]
            
        return filtered_history
    
    def reset(self) -> None:
        """
        Reset the tax optimizer and all its components.
        """
        # Reset all components
        self.tax_lot_selector.reset()
        self.tax_loss_harvester.reset()
        
        if self.tax_aware_rebalancer:
            self.tax_aware_rebalancer.rebalance_history = []
            
        if hasattr(self.tax_planner, 'reset'):
            self.tax_planner.reset()
            
        if hasattr(self.tax_reporting, 'reset'):
            self.tax_reporting.reset()
        
        # Reset optimization history
        self.optimization_history = []
        
        logger.info("Tax Optimizer reset")