import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
from .performance_optimizer import PerformanceOptimizer

logger = logging.getLogger(__name__)

class TaxLotMethod(Enum):
    """Tax lot selection methods."""
    FIFO = "First-In-First-Out"  # Oldest lots sold first
    LIFO = "Last-In-First-Out"   # Newest lots sold first
    HIFO = "Highest-In-First-Out"  # Highest cost lots sold first
    LOFO = "Lowest-In-First-Out"   # Lowest cost lots sold first
    SPECIFIC = "Specific Identification"  # Manually specified lots

class TaxLot:
    """Represents a tax lot for a security."""

    def __init__(self,
                 quantity: float,
                 purchase_price: float,
                 purchase_date: datetime,
                 lot_id: Optional[str] = None):
        """
        Initialize a tax lot.

        Args:
            quantity: Number of shares/units
            purchase_price: Price per share/unit
            purchase_date: Date of purchase
            lot_id: Optional identifier for the lot
        """
        self.quantity = quantity
        self.purchase_price = purchase_price
        self.purchase_date = purchase_date
        self.lot_id = lot_id or f"{purchase_date.strftime('%Y%m%d%H%M%S')}-{id(self)}"
        self.cost_basis = quantity * purchase_price

    def __repr__(self) -> str:
        return (f"TaxLot(id={self.lot_id}, quantity={self.quantity}, "
                f"price=${self.purchase_price:.2f}, date={self.purchase_date.strftime('%Y-%m-%d')})")

class TaxManager:
    """
    Tax Manager for tax-aware trading and reporting.

    This class provides functionality for:
    - Tax lot tracking and management
    - Tax-efficient trading strategies
    - Capital gains/losses calculation
    - Tax reporting
    - Wash sale detection
    """

    def __init__(self, default_method: TaxLotMethod = TaxLotMethod.FIFO, wash_sale_window_days: int = 30):
        """
        Initialize the Tax Manager.

        Args:
            default_method: Default tax lot selection method
            wash_sale_window_days: Number of days to look for wash sales
        """
        self.default_method = default_method
        # Add an alias for compatibility with tests
        self.default_tax_lot_method = default_method
        self.wash_sale_window_days = wash_sale_window_days
        self.tax_lots = {}  # {symbol: [TaxLot, ...]}
        self.realized_gains = []  # List of realized gain/loss records
        self.wash_sales = []  # List of detected wash sales
        self.symbol_methods = {}  # {symbol: TaxLotMethod}

        logger.info(f"Tax Manager initialized with {default_method.name} method and {wash_sale_window_days} day wash sale window")

    def add_tax_lot(self,
                   symbol: str,
                   quantity: float,
                   purchase_price: float,
                   purchase_date: Optional[datetime] = None,
                   lot_id: Optional[str] = None) -> TaxLot:
        """
        Add a new tax lot for a security.

        Args:
            symbol: Security symbol
            quantity: Number of shares/units
            purchase_price: Price per share/unit
            purchase_date: Date of purchase (default: now)
            lot_id: Optional identifier for the lot

        Returns:
            The created TaxLot
        """
        if purchase_date is None:
            purchase_date = datetime.now()

        # Create the tax lot
        tax_lot = TaxLot(quantity, purchase_price, purchase_date, lot_id)

        # Add to the symbol's tax lots
        if symbol not in self.tax_lots:
            self.tax_lots[symbol] = []

        self.tax_lots[symbol].append(tax_lot)

        logger.debug(f"Added tax lot for {symbol}: {tax_lot}")
        return tax_lot

    def sell_tax_lots(self,
                     symbol: str,
                     quantity: float,
                     sale_price: float,
                     sale_date: Optional[datetime] = None,
                     method: Optional[TaxLotMethod] = None,
                     specific_lots: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Sell tax lots for a security and calculate gains/losses.

        Args:
            symbol: Security symbol
            quantity: Number of shares/units to sell
            sale_price: Price per share/unit
            sale_date: Date of sale (default: now)
            method: Tax lot selection method (default: use symbol's method or default)
            specific_lots: List of lot IDs for specific identification method

        Returns:
            Dict with sale details including realized gains/losses
        """
        if sale_date is None:
            sale_date = datetime.now()

        # Check if we have tax lots for this symbol
        if symbol not in self.tax_lots or not self.tax_lots[symbol]:
            logger.warning(f"No tax lots found for {symbol}")
            return {"error": f"No tax lots found for {symbol}"}

        # Determine the method to use
        if method is None:
            method = self.symbol_methods.get(symbol, self.default_method)

        # Get the lots to sell based on the method
        remaining_quantity = quantity
        sold_lots = []
        realized_gain = 0.0

        # Make a copy of the tax lots to avoid modifying during iteration
        available_lots = self.tax_lots[symbol].copy()

        if method == TaxLotMethod.SPECIFIC and specific_lots:
            # Sell specific lots
            for lot_id in specific_lots:
                if remaining_quantity <= 0:
                    break

                # Find the lot with this ID
                lot_index = next((i for i, lot in enumerate(available_lots)
                                 if lot.lot_id == lot_id), None)

                if lot_index is None:
                    logger.warning(f"Lot ID {lot_id} not found for {symbol}")
                    continue

                lot = available_lots[lot_index]

                # Determine how much to sell from this lot
                sell_quantity = min(remaining_quantity, lot.quantity)
                remaining_quantity -= sell_quantity

                # Calculate gain/loss
                proceeds = sell_quantity * sale_price
                cost = sell_quantity * lot.purchase_price
                gain = proceeds - cost
                realized_gain += gain

                # Record the sale
                sold_lot = {
                    "lot_id": lot.lot_id,
                    "quantity": sell_quantity,
                    "purchase_price": lot.purchase_price,
                    "purchase_date": lot.purchase_date,
                    "sale_price": sale_price,
                    "sale_date": sale_date,
                    "proceeds": proceeds,
                    "cost": cost,
                    "gain": gain,
                    "holding_period_days": (sale_date - lot.purchase_date).days
                }
                sold_lots.append(sold_lot)

                # Update or remove the lot
                if sell_quantity >= lot.quantity:
                    # Remove the lot
                    self.tax_lots[symbol].remove(lot)
                else:
                    # Update the lot
                    lot.quantity -= sell_quantity
                    lot.cost_basis = lot.quantity * lot.purchase_price
        else:
            # Sort the lots based on the method
            if method == TaxLotMethod.FIFO:
                # Oldest first
                available_lots.sort(key=lambda lot: lot.purchase_date)
            elif method == TaxLotMethod.LIFO:
                # Newest first
                available_lots.sort(key=lambda lot: lot.purchase_date, reverse=True)
            elif method == TaxLotMethod.HIFO:
                # Highest cost first
                available_lots.sort(key=lambda lot: lot.purchase_price, reverse=True)
            elif method == TaxLotMethod.LOFO:
                # Lowest cost first
                available_lots.sort(key=lambda lot: lot.purchase_price)

            # Sell the lots in order
            for lot in available_lots:
                if remaining_quantity <= 0:
                    break

                # Determine how much to sell from this lot
                sell_quantity = min(remaining_quantity, lot.quantity)
                remaining_quantity -= sell_quantity

                # Calculate gain/loss
                proceeds = sell_quantity * sale_price
                cost = sell_quantity * lot.purchase_price
                gain = proceeds - cost
                realized_gain += gain

                # Record the sale
                sold_lot = {
                    "lot_id": lot.lot_id,
                    "quantity": sell_quantity,
                    "purchase_price": lot.purchase_price,
                    "purchase_date": lot.purchase_date,
                    "sale_price": sale_price,
                    "sale_date": sale_date,
                    "proceeds": proceeds,
                    "cost": cost,
                    "gain": gain,
                    "holding_period_days": (sale_date - lot.purchase_date).days
                }
                sold_lots.append(sold_lot)

                # Update or remove the lot
                if sell_quantity >= lot.quantity:
                    # Remove the lot
                    self.tax_lots[symbol].remove(lot)
                else:
                    # Update the lot
                    lot.quantity -= sell_quantity
                    lot.cost_basis = lot.quantity * lot.purchase_price

        # Check if we sold all requested quantity
        if remaining_quantity > 0:
            logger.warning(f"Could not sell all requested quantity for {symbol}. "
                          f"Remaining: {remaining_quantity}")

        # Record the realized gains
        for sold_lot in sold_lots:
            self.realized_gains.append({
                "symbol": symbol,
                "lot_id": sold_lot["lot_id"],
                "quantity": sold_lot["quantity"],
                "purchase_price": sold_lot["purchase_price"],
                "purchase_date": sold_lot["purchase_date"],
                "sale_price": sold_lot["sale_price"],
                "sale_date": sold_lot["sale_date"],
                "proceeds": sold_lot["proceeds"],
                "cost": sold_lot["cost"],
                "gain": sold_lot["gain"],
                "holding_period_days": sold_lot["holding_period_days"],
                "long_term": sold_lot["holding_period_days"] > 365
            })

        # Check for wash sales
        self._check_wash_sales(symbol, sale_date)

        return {
            "symbol": symbol,
            "quantity_sold": quantity - remaining_quantity,
            "sale_price": sale_price,
            "sale_date": sale_date,
            "method": method.value,
            "realized_gain": realized_gain,
            "sold_lots": sold_lots,
            "remaining_quantity": remaining_quantity
        }

    def _check_wash_sales(self, symbol: str, sale_date: datetime) -> None:
        """
        Check for wash sales (buying the same security within 30 days of a loss sale).

        Args:
            symbol: Security symbol
            sale_date: Date of the sale
        """
        # Get recent loss sales for this symbol
        recent_losses = [g for g in self.realized_gains
                        if g["symbol"] == symbol and
                        g["gain"] < 0 and
                        abs((sale_date - g["sale_date"]).days) <= 30]

        # Check if we have any tax lots purchased within 30 days of the loss sales
        for loss in recent_losses:
            for lot in self.tax_lots.get(symbol, []):
                days_diff = abs((lot.purchase_date - loss["sale_date"]).days)

                if days_diff <= 30:
                    # This is a wash sale
                    wash_sale = {
                        "symbol": symbol,
                        "loss_lot_id": loss["lot_id"],
                        "loss_sale_date": loss["sale_date"],
                        "loss_amount": loss["gain"],
                        "replacement_lot_id": lot.lot_id,
                        "replacement_purchase_date": lot.purchase_date,
                        "days_between": days_diff
                    }

                    self.wash_sales.append(wash_sale)
                    logger.warning(f"Wash sale detected for {symbol}: {wash_sale}")

    def set_tax_lot_method(self, symbol: str, method: TaxLotMethod) -> None:
        """
        Set the tax lot selection method for a specific symbol.

        Args:
            symbol: Security symbol
            method: Tax lot selection method
        """
        self.symbol_methods[symbol] = method
        # If setting the default method (empty symbol), update both attributes
        if symbol == "":
            self.default_method = method
            self.default_tax_lot_method = method
        logger.info(f"Set tax lot method for {symbol} to {method.name}")

    def get_tax_lots(self, symbol: Optional[str] = None) -> Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Get tax lots for a symbol or all symbols.

        Args:
            symbol: Security symbol (optional)

        Returns:
            If symbol is provided: List of tax lot dictionaries for that symbol
            If symbol is None: Dict of {symbol: [tax_lot_dict, ...]} for all symbols
        """
        if symbol:
            # Convert TaxLot objects to dictionaries for the specified symbol
            return [{
                "lot_id": lot.lot_id,
                "quantity": lot.quantity,
                "purchase_price": lot.purchase_price,
                "purchase_date": lot.purchase_date,
                "cost_basis": lot.cost_basis
            } for lot in self.tax_lots.get(symbol, [])]
        
        # Convert all TaxLot objects to dictionaries for all symbols
        result = {}
        for sym, lots in self.tax_lots.items():
            result[sym] = [{
                "lot_id": lot.lot_id,
                "quantity": lot.quantity,
                "purchase_price": lot.purchase_price,
                "purchase_date": lot.purchase_date,
                "cost_basis": lot.cost_basis
            } for lot in lots]
        return result

    def get_realized_gains(self,
                          symbol: Optional[str] = None,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          long_term_only: bool = False,
                          short_term_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get realized gains/losses with optional filtering.

        Args:
            symbol: Filter by symbol (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)
            long_term_only: Filter for long-term gains only
            short_term_only: Filter for short-term gains only

        Returns:
            List of realized gain/loss records
        """
        filtered_gains = self.realized_gains

        if symbol:
            filtered_gains = [g for g in filtered_gains if g["symbol"] == symbol]

        if start_date:
            filtered_gains = [g for g in filtered_gains if g["sale_date"] >= start_date]

        if end_date:
            filtered_gains = [g for g in filtered_gains if g["sale_date"] <= end_date]

        if long_term_only:
            filtered_gains = [g for g in filtered_gains if g["long_term"]]

        if short_term_only:
            filtered_gains = [g for g in filtered_gains if not g["long_term"]]

        return filtered_gains

    def get_wash_sales(self,
                      symbol: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get detected wash sales with optional filtering.

        Args:
            symbol: Filter by symbol (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)

        Returns:
            List of wash sale records
        """
        filtered_wash_sales = self.wash_sales

        if symbol:
            filtered_wash_sales = [w for w in filtered_wash_sales if w["symbol"] == symbol]

        if start_date:
            filtered_wash_sales = [w for w in filtered_wash_sales if w["loss_sale_date"] >= start_date]

        if end_date:
            filtered_wash_sales = [w for w in filtered_wash_sales if w["loss_sale_date"] <= end_date]

        return filtered_wash_sales

    def generate_tax_report(self,
                           tax_year: int,
                           include_wash_sales: bool = True) -> Dict[str, Any]:
        """
        Generate a tax report for a specific year.

        Args:
            tax_year: The tax year
            include_wash_sales: Whether to include wash sale information

        Returns:
            Dict with tax report data
        """
        # Define the tax year period
        start_date = datetime(tax_year, 1, 1)
        end_date = datetime(tax_year, 12, 31, 23, 59, 59)

        # Get realized gains for the tax year
        gains = self.get_realized_gains(start_date=start_date, end_date=end_date)

        # Separate long-term and short-term gains
        long_term_gains = [g for g in gains if g["long_term"]]
        short_term_gains = [g for g in gains if not g["long_term"]]

        # Calculate totals
        total_long_term = sum(g["gain"] for g in long_term_gains)
        total_short_term = sum(g["gain"] for g in short_term_gains)
        total_gains = total_long_term + total_short_term

        # Get wash sales if requested
        wash_sales = []
        if include_wash_sales:
            wash_sales = self.get_wash_sales(start_date=start_date, end_date=end_date)

        # Organize by symbol
        symbols = set(g["symbol"] for g in gains)
        symbol_summary = {}

        for symbol in symbols:
            symbol_gains = [g for g in gains if g["symbol"] == symbol]
            symbol_long_term = [g for g in symbol_gains if g["long_term"]]
            symbol_short_term = [g for g in symbol_gains if not g["long_term"]]

            symbol_summary[symbol] = {
                "total_gain": sum(g["gain"] for g in symbol_gains),
                "long_term_gain": sum(g["gain"] for g in symbol_long_term),
                "short_term_gain": sum(g["gain"] for g in symbol_short_term),
                "transactions": len(symbol_gains),
                "long_term_transactions": len(symbol_long_term),
                "short_term_transactions": len(symbol_short_term)
            }

        return {
            "tax_year": tax_year,
            "total_gains": total_gains,
            "long_term_gains": total_long_term,
            "short_term_gains": total_short_term,
            "total_transactions": len(gains),
            "long_term_transactions": len(long_term_gains),
            "short_term_transactions": len(short_term_gains),
            "symbol_summary": symbol_summary,
            "detailed_gains": gains,
            "wash_sales": wash_sales if include_wash_sales else []
        }

    def get_tax_lots_dataframe(self) -> pd.DataFrame:
        """
        Get all tax lots as a pandas DataFrame.

        Returns:
            DataFrame with tax lot information
        """
        data = []

        for symbol, lots in self.tax_lots.items():
            for lot in lots:
                data.append({
                    "symbol": symbol,
                    "lot_id": lot.lot_id,
                    "quantity": lot.quantity,
                    "purchase_price": lot.purchase_price,
                    "purchase_date": lot.purchase_date,
                    "cost_basis": lot.cost_basis,
                    "holding_period_days": (datetime.now() - lot.purchase_date).days,
                    "long_term": (datetime.now() - lot.purchase_date).days > 365
                })

        if not data:
            return pd.DataFrame(columns=["symbol", "lot_id", "quantity", "purchase_price",
                                       "purchase_date", "cost_basis", "holding_period_days",
                                       "long_term"])

        return pd.DataFrame(data)

    def get_realized_gains_dataframe(self) -> pd.DataFrame:
        """
        Get realized gains/losses as a pandas DataFrame.

        Returns:
            DataFrame with realized gain/loss information
        """
        if not self.realized_gains:
            return pd.DataFrame(columns=["symbol", "lot_id", "quantity", "purchase_price",
                                       "purchase_date", "sale_price", "sale_date", "proceeds",
                                       "cost", "gain", "holding_period_days", "long_term"])

        return pd.DataFrame(self.realized_gains)

    def reset(self) -> None:
        """
        Reset the tax manager.
        """
        # Use performance monitoring if optimization is enabled
        if self.enable_optimization and self.performance_optimizer:
            self.performance_optimizer.monitor_performance("tax_manager")(self._reset)()
        else:
            self._reset()
    
    def _reset(self) -> None:
        """
        Internal method to reset the tax manager.
        """
        self.tax_lots = {}
        self.realized_gains = []
        self.wash_sales = []
        self.symbol_methods = {}

        logger.info("Tax Manager reset")
