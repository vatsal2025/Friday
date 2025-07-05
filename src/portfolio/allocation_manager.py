import logging
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class RebalanceMethod(Enum):
    """Methods for portfolio rebalancing."""
    THRESHOLD = "Threshold"  # Rebalance when allocations drift beyond threshold
    CALENDAR = "Calendar"    # Rebalance on a fixed schedule
    CONSTANT_MIX = "Constant Mix"  # Continuous rebalancing
    BUY_AND_HOLD = "Buy and Hold"  # No rebalancing

class AllocationTarget:
    """Represents a target allocation for a portfolio component."""

    def __init__(self,
                 name: str,
                 target_percentage: float,
                 min_percentage: Optional[float] = None,
                 max_percentage: Optional[float] = None,
                 category: Optional[str] = None):
        """
        Initialize an allocation target.

        Args:
            name: Name of the component (symbol, sector, asset class, etc.)
            target_percentage: Target allocation percentage (0-100)
            min_percentage: Minimum acceptable percentage (optional)
            max_percentage: Maximum acceptable percentage (optional)
            category: Category for grouping (e.g., asset class, sector)
        """
        self.name = name
        self.target_percentage = target_percentage
        self.min_percentage = min_percentage if min_percentage is not None else target_percentage * 0.9
        self.max_percentage = max_percentage if max_percentage is not None else target_percentage * 1.1
        self.category = category

    def __repr__(self) -> str:
        return (f"AllocationTarget(name={self.name}, target={self.target_percentage:.2f}%, "
                f"min={self.min_percentage:.2f}%, max={self.max_percentage:.2f}%)")

class AllocationManager:
    """
    Allocation Manager for portfolio asset allocation.

    This class provides functionality for:
    - Setting and tracking target allocations
    - Monitoring allocation drift
    - Generating rebalancing recommendations
    - Supporting different rebalancing methods
    - Analyzing allocation by various dimensions (asset class, sector, etc.)
    """

    def __init__(self,
                rebalance_method: RebalanceMethod = RebalanceMethod.THRESHOLD,
                default_threshold: float = 5.0,
                rebalance_frequency_days: int = 90):
        """
        Initialize the Allocation Manager.

        Args:
            rebalance_method: Method for rebalancing
            default_threshold: Default threshold percentage for rebalancing (used with THRESHOLD method)
            rebalance_frequency_days: Days between rebalances (used with CALENDAR method)
        """
        self.rebalance_method = rebalance_method
        self.default_threshold = default_threshold
        self.rebalance_frequency_days = rebalance_frequency_days
        self.last_rebalance_date = None

        # Allocation targets by category
        self.allocation_targets = {}

        # Current allocations
        self.current_allocations = {}

        # Historical allocations
        self.allocation_history = []

        # Rebalance history
        self.rebalance_history = []

        logger.info(f"Allocation Manager initialized with {rebalance_method.name} method")

    def set_allocation_target(self,
                             name: str,
                             target_percentage: float,
                             min_percentage: Optional[float] = None,
                             max_percentage: Optional[float] = None,
                             category: str = "default") -> AllocationTarget:
        """
        Set a target allocation for a portfolio component.

        Args:
            name: Name of the component (symbol, sector, asset class, etc.)
            target_percentage: Target allocation percentage (0-100)
            min_percentage: Minimum acceptable percentage (optional)
            max_percentage: Maximum acceptable percentage (optional)
            category: Category for grouping (e.g., asset class, sector)

        Returns:
            The created AllocationTarget
        """
        # Create the allocation target
        target = AllocationTarget(name, target_percentage, min_percentage, max_percentage, category)

        # Add to the category's targets
        if category not in self.allocation_targets:
            self.allocation_targets[category] = {}

        self.allocation_targets[category][name] = target

        logger.debug(f"Set allocation target: {target}")
        return target

    def set_multiple_allocation_targets(self,
                                       targets: List[Dict[str, Any]],
                                       category: str = "default") -> List[AllocationTarget]:
        """
        Set multiple allocation targets at once.

        Args:
            targets: List of dicts with target information
            category: Category for all targets

        Returns:
            List of created AllocationTargets
        """
        created_targets = []

        for target_info in targets:
            target = self.set_allocation_target(
                name=target_info["name"],
                target_percentage=target_info["target_percentage"],
                min_percentage=target_info.get("min_percentage"),
                max_percentage=target_info.get("max_percentage"),
                category=target_info.get("category", category)
            )
            created_targets.append(target)

        # Validate that targets sum to approximately 100% for each category
        self._validate_category_targets()

        return created_targets

    def _validate_category_targets(self) -> None:
        """
        Validate that allocation targets sum to approximately 100% for each category.
        """
        for category, targets in self.allocation_targets.items():
            total = sum(target.target_percentage for target in targets.values())
            if not (99.0 <= total <= 101.0):  # Allow for small rounding errors
                logger.warning(f"Allocation targets for category '{category}' sum to {total:.2f}%, "
                             f"which is not close to 100%")

    def update_current_allocations(self,
                                 allocations: Dict[str, Dict[str, float]],
                                 timestamp: Optional[datetime] = None) -> None:
        """
        Update the current allocations.

        Args:
            allocations: Dict of {category: {name: percentage}}
            timestamp: Timestamp for the update (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.current_allocations = allocations

        # Add to history
        history_entry = {
            "timestamp": timestamp,
            "allocations": allocations
        }
        self.allocation_history.append(history_entry)

        # Check if rebalancing is needed
        if self.rebalance_method == RebalanceMethod.CONSTANT_MIX:
            self.check_rebalance_needed()

    def update_allocation_from_portfolio(self,
                                       portfolio_values: Dict[str, float],
                                       categories: Optional[Dict[str, str]] = None,
                                       timestamp: Optional[datetime] = None) -> Dict[str, Dict[str, float]]:
        """
        Update allocations based on portfolio values.

        Args:
            portfolio_values: Dict of {name: value}
            categories: Dict of {name: category} (optional)
            timestamp: Timestamp for the update (default: now)

        Returns:
            The calculated allocations
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Calculate total portfolio value
        total_value = sum(portfolio_values.values())

        if total_value == 0:
            logger.warning("Total portfolio value is zero, cannot calculate allocations")
            return {}

        # Calculate percentages
        percentages = {name: (value / total_value) * 100 for name, value in portfolio_values.items()}

        # Organize by category
        allocations = {"default": percentages}

        if categories:
            category_allocations = {}

            for name, value in portfolio_values.items():
                category = categories.get(name, "default")

                if category not in category_allocations:
                    category_allocations[category] = {}

                category_allocations[category][name] = (value / total_value) * 100

            allocations = category_allocations

        # Update current allocations
        self.update_current_allocations(allocations, timestamp)

        return allocations

    def check_rebalance_needed(self,
                             timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Check if portfolio rebalancing is needed.

        Args:
            timestamp: Timestamp for the check (default: now)

        Returns:
            Dict with rebalance information
        """
        if timestamp is None:
            timestamp = datetime.now()

        rebalance_needed = False
        drift_components = []

        # Check based on rebalance method
        if self.rebalance_method == RebalanceMethod.BUY_AND_HOLD:
            # Never rebalance
            return {
                "rebalance_needed": False,
                "method": self.rebalance_method.value,
                "timestamp": timestamp,
                "drift_components": []
            }

        elif self.rebalance_method == RebalanceMethod.CALENDAR:
            # Check if enough time has passed since last rebalance
            if self.last_rebalance_date is None:
                rebalance_needed = True
            else:
                days_since_last = (timestamp - self.last_rebalance_date).days
                rebalance_needed = days_since_last >= self.rebalance_frequency_days

        # For THRESHOLD and CONSTANT_MIX, check allocation drift
        if self.rebalance_method in [RebalanceMethod.THRESHOLD, RebalanceMethod.CONSTANT_MIX]:
            for category, targets in self.allocation_targets.items():
                if category not in self.current_allocations:
                    continue

                current = self.current_allocations[category]

                for name, target in targets.items():
                    if name not in current:
                        continue

                    current_pct = current[name]
                    drift = current_pct - target.target_percentage
                    drift_pct = abs(drift)

                    # Check if outside min/max bounds
                    if current_pct < target.min_percentage or current_pct > target.max_percentage:
                        rebalance_needed = True

                        drift_components.append({
                            "category": category,
                            "name": name,
                            "current_percentage": current_pct,
                            "target_percentage": target.target_percentage,
                            "drift": drift,
                            "drift_percentage": drift_pct,
                            "outside_bounds": True
                        })

                    # For THRESHOLD method, also check against default threshold
                    elif self.rebalance_method == RebalanceMethod.THRESHOLD and drift_pct > self.default_threshold:
                        rebalance_needed = True

                        drift_components.append({
                            "category": category,
                            "name": name,
                            "current_percentage": current_pct,
                            "target_percentage": target.target_percentage,
                            "drift": drift,
                            "drift_percentage": drift_pct,
                            "outside_bounds": False
                        })

                    # For CONSTANT_MIX, any drift triggers rebalance
                    elif self.rebalance_method == RebalanceMethod.CONSTANT_MIX and drift_pct > 0:
                        rebalance_needed = True

                        drift_components.append({
                            "category": category,
                            "name": name,
                            "current_percentage": current_pct,
                            "target_percentage": target.target_percentage,
                            "drift": drift,
                            "drift_percentage": drift_pct,
                            "outside_bounds": False
                        })

        result = {
            "rebalance_needed": rebalance_needed,
            "method": self.rebalance_method.value,
            "timestamp": timestamp,
            "drift_components": drift_components
        }

        # If calendar-based and rebalance is needed, also check for drift
        if self.rebalance_method == RebalanceMethod.CALENDAR and rebalance_needed:
            # Add drift information even though the decision is calendar-based
            for category, targets in self.allocation_targets.items():
                if category not in self.current_allocations:
                    continue

                current = self.current_allocations[category]

                for name, target in targets.items():
                    if name not in current:
                        continue

                    current_pct = current[name]
                    drift = current_pct - target.target_percentage
                    drift_pct = abs(drift)

                    drift_components.append({
                        "category": category,
                        "name": name,
                        "current_percentage": current_pct,
                        "target_percentage": target.target_percentage,
                        "drift": drift,
                        "drift_percentage": drift_pct,
                        "outside_bounds": current_pct < target.min_percentage or current_pct > target.max_percentage
                    })

            result["drift_components"] = drift_components

        return result

    def generate_rebalance_plan(self,
                              portfolio_values: Dict[str, float],
                              categories: Optional[Dict[str, str]] = None,
                              cash_available: float = 0.0,
                              timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate a rebalancing plan.

        Args:
            portfolio_values: Dict of {name: value}
            categories: Dict of {name: category} (optional)
            cash_available: Additional cash available for rebalancing
            timestamp: Timestamp for the plan (default: now)

        Returns:
            Dict with rebalance plan
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Update allocations first
        self.update_allocation_from_portfolio(portfolio_values, categories, timestamp)

        # Check if rebalance is needed
        rebalance_check = self.check_rebalance_needed(timestamp)

        if not rebalance_check["rebalance_needed"]:
            return {
                "rebalance_needed": False,
                "timestamp": timestamp,
                "message": "No rebalancing needed at this time"
            }

        # Calculate total portfolio value including available cash
        total_value = sum(portfolio_values.values()) + cash_available

        # Generate the plan
        trades = []

        for category, targets in self.allocation_targets.items():
            # Skip categories not in current portfolio
            if category not in self.current_allocations and category != "default":
                continue

            for name, target in targets.items():
                # Calculate target value
                target_value = (target.target_percentage / 100) * total_value

                # Get current value
                current_value = portfolio_values.get(name, 0.0)

                # Calculate difference
                diff = target_value - current_value

                if abs(diff) > 0.01:  # Ignore very small differences
                    trades.append({
                        "name": name,
                        "category": category,
                        "current_value": current_value,
                        "target_value": target_value,
                        "difference": diff,
                        "action": "BUY" if diff > 0 else "SELL",
                        "amount": abs(diff)
                    })

        # Record the rebalance plan
        rebalance_plan = {
            "rebalance_needed": True,
            "timestamp": timestamp,
            "method": self.rebalance_method.value,
            "total_portfolio_value": total_value,
            "cash_available": cash_available,
            "trades": trades,
            "drift_components": rebalance_check["drift_components"]
        }

        self.rebalance_history.append(rebalance_plan)

        return rebalance_plan

    def record_rebalance_execution(self,
                                 executed_trades: List[Dict[str, Any]],
                                 timestamp: Optional[datetime] = None) -> None:
        """
        Record that a rebalance was executed.

        Args:
            executed_trades: List of executed trades
            timestamp: Timestamp for the execution (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.last_rebalance_date = timestamp

        # Record the execution
        execution_record = {
            "timestamp": timestamp,
            "executed_trades": executed_trades
        }

        # Add to the last rebalance plan
        if self.rebalance_history:
            self.rebalance_history[-1]["execution"] = execution_record

        logger.info(f"Recorded rebalance execution with {len(executed_trades)} trades")

    def get_allocation_targets(self, category: Optional[str] = None) -> Dict[str, Dict[str, AllocationTarget]]:
        """
        Get allocation targets.

        Args:
            category: Filter by category (optional)

        Returns:
            Dict of allocation targets
        """
        if category:
            return {category: self.allocation_targets.get(category, {})}
        return self.allocation_targets

    def get_current_allocations(self, category: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Get current allocations.

        Args:
            category: Filter by category (optional)

        Returns:
            Dict of current allocations
        """
        if category:
            return {category: self.current_allocations.get(category, {})}
        return self.current_allocations

    def get_allocation_drift(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get allocation drift information.

        Args:
            category: Filter by category (optional)

        Returns:
            List of drift components
        """
        drift_components = []

        categories_to_check = [category] if category else self.allocation_targets.keys()

        for cat in categories_to_check:
            if cat not in self.allocation_targets or cat not in self.current_allocations:
                continue

            targets = self.allocation_targets[cat]
            current = self.current_allocations[cat]

            for name, target in targets.items():
                if name not in current:
                    continue

                current_pct = current[name]
                drift = current_pct - target.target_percentage
                drift_pct = abs(drift)

                drift_components.append({
                    "category": cat,
                    "name": name,
                    "current_percentage": current_pct,
                    "target_percentage": target.target_percentage,
                    "drift": drift,
                    "drift_percentage": drift_pct,
                    "outside_bounds": current_pct < target.min_percentage or current_pct > target.max_percentage
                })

        return drift_components

    def get_allocation_history(self,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get allocation history with optional date filtering.

        Args:
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)

        Returns:
            List of allocation history entries
        """
        filtered_history = self.allocation_history

        if start_date:
            filtered_history = [h for h in filtered_history if h["timestamp"] >= start_date]

        if end_date:
            filtered_history = [h for h in filtered_history if h["timestamp"] <= end_date]

        return filtered_history

    def get_rebalance_history(self,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get rebalance history with optional date filtering.

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

    def get_allocation_targets_dataframe(self, category: Optional[str] = None) -> pd.DataFrame:
        """
        Get allocation targets as a pandas DataFrame.

        Args:
            category: Filter by category (optional)

        Returns:
            DataFrame with allocation target information
        """
        data = []

        categories_to_include = [category] if category else self.allocation_targets.keys()

        for cat in categories_to_include:
            if cat not in self.allocation_targets:
                continue

            for name, target in self.allocation_targets[cat].items():
                data.append({
                    "category": cat,
                    "name": name,
                    "target_percentage": target.target_percentage,
                    "min_percentage": target.min_percentage,
                    "max_percentage": target.max_percentage
                })

        if not data:
            return pd.DataFrame(columns=["category", "name", "target_percentage",
                                       "min_percentage", "max_percentage"])

        return pd.DataFrame(data)

    def get_current_allocations_dataframe(self, category: Optional[str] = None) -> pd.DataFrame:
        """
        Get current allocations as a pandas DataFrame.

        Args:
            category: Filter by category (optional)

        Returns:
            DataFrame with current allocation information
        """
        data = []

        categories_to_include = [category] if category else self.current_allocations.keys()

        for cat in categories_to_include:
            if cat not in self.current_allocations:
                continue

            for name, percentage in self.current_allocations[cat].items():
                # Get target if available
                target_pct = None
                min_pct = None
                max_pct = None
                drift = None

                if cat in self.allocation_targets and name in self.allocation_targets[cat]:
                    target = self.allocation_targets[cat][name]
                    target_pct = target.target_percentage
                    min_pct = target.min_percentage
                    max_pct = target.max_percentage
                    drift = percentage - target_pct

                data.append({
                    "category": cat,
                    "name": name,
                    "current_percentage": percentage,
                    "target_percentage": target_pct,
                    "min_percentage": min_pct,
                    "max_percentage": max_pct,
                    "drift": drift
                })

        if not data:
            return pd.DataFrame(columns=["category", "name", "current_percentage",
                                       "target_percentage", "min_percentage",
                                       "max_percentage", "drift"])

        return pd.DataFrame(data)

    def reset(self) -> None:
        """
        Reset the allocation manager.
        """
        self.allocation_targets = {}
        self.current_allocations = {}
        self.allocation_history = []
        self.rebalance_history = []
        self.last_rebalance_date = None

        logger.info("Allocation Manager reset")
