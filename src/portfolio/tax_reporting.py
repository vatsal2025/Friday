import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, date
from enum import Enum
import json
import os
from decimal import Decimal

from portfolio.tax_manager import TaxManager

logger = logging.getLogger(__name__)

class ReportFormat(Enum):
    """Formats for tax reports."""
    CSV = "csv"
    JSON = "json"
    PDF = "pdf"
    EXCEL = "excel"

class ReportType(Enum):
    """Types of tax reports."""
    REALIZED_GAINS = "realized_gains"
    UNREALIZED_GAINS = "unrealized_gains"
    TAX_LOTS = "tax_lots"
    INCOME = "income"
    WASH_SALES = "wash_sales"
    HARVESTING = "harvesting"
    YEAR_END = "year_end_summary"
    FORM_1099 = "form_1099"
    CUSTOM = "custom"

class TaxReporting:
    """
    Comprehensive tax reporting module for investment portfolios.
    
    This class provides functionality for generating various tax reports,
    including realized gains/losses, unrealized gains/losses, tax lot details,
    income reporting, wash sale tracking, tax-loss harvesting summaries,
    and year-end tax summaries.
    """
    
    def __init__(self, tax_manager: TaxManager, output_dir: str = "tax_reports"):
        """
        Initialize the Tax Reporting module.
        
        Args:
            tax_manager: The tax manager instance
            output_dir: Directory to save generated reports
        """
        self.tax_manager = tax_manager
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Report generation history
        self.report_history = []
        
        logger.info(f"Tax Reporting module initialized with output directory: {output_dir}")
    
    def generate_report(self,
                       report_type: ReportType,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       accounts: Optional[List[str]] = None,
                       symbols: Optional[List[str]] = None,
                       format: ReportFormat = ReportFormat.CSV,
                       include_details: bool = True,
                       filename: Optional[str] = None) -> str:
        """
        Generate a tax report based on specified parameters.
        
        Args:
            report_type: Type of report to generate
            start_date: Start date for the report period
            end_date: End date for the report period
            accounts: List of account IDs to include (None for all)
            symbols: List of symbols to include (None for all)
            format: Output format for the report
            include_details: Whether to include detailed information
            filename: Custom filename for the report
            
        Returns:
            Path to the generated report file
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
            
        if start_date is None:
            # Default to beginning of the year
            start_date = datetime(end_date.year, 1, 1)
        
        # Generate the appropriate report based on type
        if report_type == ReportType.REALIZED_GAINS:
            report_data = self._generate_realized_gains_report(start_date, end_date, accounts, symbols)
        elif report_type == ReportType.UNREALIZED_GAINS:
            report_data = self._generate_unrealized_gains_report(end_date, accounts, symbols)
        elif report_type == ReportType.TAX_LOTS:
            report_data = self._generate_tax_lots_report(end_date, accounts, symbols)
        elif report_type == ReportType.INCOME:
            report_data = self._generate_income_report(start_date, end_date, accounts, symbols)
        elif report_type == ReportType.WASH_SALES:
            report_data = self._generate_wash_sales_report(start_date, end_date, accounts, symbols)
        elif report_type == ReportType.HARVESTING:
            report_data = self._generate_harvesting_report(start_date, end_date, accounts, symbols)
        elif report_type == ReportType.YEAR_END:
            report_data = self._generate_year_end_report(end_date.year, accounts)
        elif report_type == ReportType.FORM_1099:
            report_data = self._generate_1099_report(end_date.year, accounts)
        elif report_type == ReportType.CUSTOM:
            # Custom reports require additional parameters
            logger.error("Custom reports require additional parameters")
            raise ValueError("Custom reports require additional parameters")
        else:
            logger.error(f"Unsupported report type: {report_type}")
            raise ValueError(f"Unsupported report type: {report_type}")
        
        # Generate filename if not provided
        if filename is None:
            date_str = end_date.strftime("%Y%m%d")
            filename = f"{report_type.value}_{date_str}"
        
        # Save the report in the specified format
        file_path = self._save_report(report_data, filename, format)
        
        # Record in history
        self.report_history.append({
            "timestamp": datetime.now(),
            "report_type": report_type.value,
            "start_date": start_date,
            "end_date": end_date,
            "accounts": accounts,
            "symbols": symbols,
            "format": format.value,
            "file_path": file_path
        })
        
        return file_path
    
    def _generate_realized_gains_report(self,
                                      start_date: datetime,
                                      end_date: datetime,
                                      accounts: Optional[List[str]] = None,
                                      symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate a report of realized gains and losses.
        
        Args:
            start_date: Start date for the report period
            end_date: End date for the report period
            accounts: List of account IDs to include (None for all)
            symbols: List of symbols to include (None for all)
            
        Returns:
            DataFrame with realized gains/losses information
        """
        # Get realized gains from tax manager
        realized_gains = self.tax_manager.get_realized_gains(
            start_date=start_date,
            end_date=end_date,
            account_ids=accounts,
            symbols=symbols
        )
        
        # Convert to DataFrame for easier manipulation
        if not realized_gains:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "symbol", "account_id", "quantity", "purchase_date", "sale_date",
                "holding_period_days", "cost_basis", "proceeds", "gain_amount",
                "gain_percentage", "is_long_term", "wash_sale_adjustment"
            ])
        
        df = pd.DataFrame(realized_gains)
        
        # Calculate additional metrics
        df["holding_period_days"] = (df["sale_date"] - df["purchase_date"]).dt.days
        df["is_long_term"] = df["holding_period_days"] >= 365
        df["gain_percentage"] = (df["gain_amount"] / df["cost_basis"]) * 100
        
        # Add tax rate information based on long-term/short-term status
        # This is a simplified approach - actual tax rates depend on income brackets
        df["estimated_tax_rate"] = df["is_long_term"].apply(lambda x: 20.0 if x else 37.0)
        df["estimated_tax"] = df.apply(
            lambda row: max(0, row["gain_amount"] * row["estimated_tax_rate"] / 100) 
            if row["gain_amount"] > 0 else 0, axis=1
        )
        
        # Sort by sale date
        df = df.sort_values(by="sale_date", ascending=False)
        
        # Calculate summary statistics
        summary = {
            "total_gains": df[df["gain_amount"] > 0]["gain_amount"].sum(),
            "total_losses": df[df["gain_amount"] < 0]["gain_amount"].sum(),
            "net_gain_loss": df["gain_amount"].sum(),
            "total_proceeds": df["proceeds"].sum(),
            "total_cost_basis": df["cost_basis"].sum(),
            "long_term_gains": df[(df["is_long_term"]) & (df["gain_amount"] > 0)]["gain_amount"].sum(),
            "short_term_gains": df[(~df["is_long_term"]) & (df["gain_amount"] > 0)]["gain_amount"].sum(),
            "long_term_losses": df[(df["is_long_term"]) & (df["gain_amount"] < 0)]["gain_amount"].sum(),
            "short_term_losses": df[(~df["is_long_term"]) & (df["gain_amount"] < 0)]["gain_amount"].sum(),
            "estimated_tax": df["estimated_tax"].sum()
        }
        
        # Add summary as a metadata attribute
        df.attrs["summary"] = summary
        
        return df
    
    def _generate_unrealized_gains_report(self,
                                        as_of_date: datetime,
                                        accounts: Optional[List[str]] = None,
                                        symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate a report of unrealized gains and losses.
        
        Args:
            as_of_date: Date for valuation
            accounts: List of account IDs to include (None for all)
            symbols: List of symbols to include (None for all)
            
        Returns:
            DataFrame with unrealized gains/losses information
        """
        # Get current positions and tax lots
        positions = self.tax_manager.get_positions(account_ids=accounts, symbols=symbols)
        
        # Prepare data for the report
        unrealized_data = []
        
        for position in positions:
            symbol = position.get("symbol")
            account_id = position.get("account_id")
            current_price = position.get("price", 0)
            
            # Get tax lots for this position
            tax_lots = self.tax_manager.get_tax_lots(symbol=symbol, account_id=account_id)
            
            for lot in tax_lots:
                # Calculate unrealized gain/loss
                cost_basis = lot.purchase_price * lot.quantity
                market_value = current_price * lot.quantity
                unrealized_gain = market_value - cost_basis
                holding_period_days = (as_of_date - lot.purchase_date).days
                
                unrealized_data.append({
                    "symbol": symbol,
                    "account_id": account_id,
                    "quantity": lot.quantity,
                    "purchase_date": lot.purchase_date,
                    "purchase_price": lot.purchase_price,
                    "current_price": current_price,
                    "cost_basis": cost_basis,
                    "market_value": market_value,
                    "unrealized_gain": unrealized_gain,
                    "gain_percentage": (unrealized_gain / cost_basis * 100) if cost_basis > 0 else 0,
                    "holding_period_days": holding_period_days,
                    "is_long_term": holding_period_days >= 365
                })
        
        # Convert to DataFrame
        if not unrealized_data:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "symbol", "account_id", "quantity", "purchase_date", "purchase_price",
                "current_price", "cost_basis", "market_value", "unrealized_gain",
                "gain_percentage", "holding_period_days", "is_long_term"
            ])
        
        df = pd.DataFrame(unrealized_data)
        
        # Sort by unrealized gain (descending)
        df = df.sort_values(by="unrealized_gain", ascending=False)
        
        # Calculate summary statistics
        summary = {
            "total_unrealized_gains": df[df["unrealized_gain"] > 0]["unrealized_gain"].sum(),
            "total_unrealized_losses": df[df["unrealized_gain"] < 0]["unrealized_gain"].sum(),
            "net_unrealized": df["unrealized_gain"].sum(),
            "total_market_value": df["market_value"].sum(),
            "total_cost_basis": df["cost_basis"].sum(),
            "long_term_unrealized_gains": df[(df["is_long_term"]) & (df["unrealized_gain"] > 0)]["unrealized_gain"].sum(),
            "short_term_unrealized_gains": df[(~df["is_long_term"]) & (df["unrealized_gain"] > 0)]["unrealized_gain"].sum(),
            "long_term_unrealized_losses": df[(df["is_long_term"]) & (df["unrealized_gain"] < 0)]["unrealized_gain"].sum(),
            "short_term_unrealized_losses": df[(~df["is_long_term"]) & (df["unrealized_gain"] < 0)]["unrealized_gain"].sum(),
        }
        
        # Add summary as a metadata attribute
        df.attrs["summary"] = summary
        
        return df
    
    def _generate_tax_lots_report(self,
                                as_of_date: datetime,
                                accounts: Optional[List[str]] = None,
                                symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate a detailed report of all tax lots.
        
        Args:
            as_of_date: Date for valuation
            accounts: List of account IDs to include (None for all)
            symbols: List of symbols to include (None for all)
            
        Returns:
            DataFrame with tax lot information
        """
        # Get positions and tax lots
        positions = self.tax_manager.get_positions(account_ids=accounts, symbols=symbols)
        
        # Prepare data for the report
        tax_lot_data = []
        
        for position in positions:
            symbol = position.get("symbol")
            account_id = position.get("account_id")
            current_price = position.get("price", 0)
            
            # Get tax lots for this position
            tax_lots = self.tax_manager.get_tax_lots(symbol=symbol, account_id=account_id)
            
            for lot in tax_lots:
                # Calculate metrics
                cost_basis = lot.purchase_price * lot.quantity
                market_value = current_price * lot.quantity
                unrealized_gain = market_value - cost_basis
                holding_period_days = (as_of_date - lot.purchase_date).days
                days_to_long_term = max(0, 365 - holding_period_days)
                
                tax_lot_data.append({
                    "symbol": symbol,
                    "account_id": account_id,
                    "lot_id": lot.lot_id,
                    "quantity": lot.quantity,
                    "purchase_date": lot.purchase_date,
                    "purchase_price": lot.purchase_price,
                    "current_price": current_price,
                    "cost_basis": cost_basis,
                    "market_value": market_value,
                    "unrealized_gain": unrealized_gain,
                    "gain_percentage": (unrealized_gain / cost_basis * 100) if cost_basis > 0 else 0,
                    "holding_period_days": holding_period_days,
                    "is_long_term": holding_period_days >= 365,
                    "days_to_long_term": days_to_long_term,
                    "tax_lot_method": self.tax_manager.get_tax_lot_method(symbol)
                })
        
        # Convert to DataFrame
        if not tax_lot_data:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "symbol", "account_id", "lot_id", "quantity", "purchase_date",
                "purchase_price", "current_price", "cost_basis", "market_value",
                "unrealized_gain", "gain_percentage", "holding_period_days",
                "is_long_term", "days_to_long_term", "tax_lot_method"
            ])
        
        df = pd.DataFrame(tax_lot_data)
        
        # Sort by symbol and purchase date
        df = df.sort_values(by=["symbol", "purchase_date"])
        
        return df
    
    def _generate_income_report(self,
                              start_date: datetime,
                              end_date: datetime,
                              accounts: Optional[List[str]] = None,
                              symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate a report of investment income (dividends, interest, etc.).
        
        Args:
            start_date: Start date for the report period
            end_date: End date for the report period
            accounts: List of account IDs to include (None for all)
            symbols: List of symbols to include (None for all)
            
        Returns:
            DataFrame with income information
        """
        # Get income data from tax manager
        income_data = self.tax_manager.get_income(
            start_date=start_date,
            end_date=end_date,
            account_ids=accounts,
            symbols=symbols
        )
        
        # Convert to DataFrame
        if not income_data:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "date", "symbol", "account_id", "income_type", "amount",
                "is_qualified", "foreign_tax_paid", "description"
            ])
        
        df = pd.DataFrame(income_data)
        
        # Sort by date (descending)
        df = df.sort_values(by="date", ascending=False)
        
        # Calculate summary by income type
        income_summary = df.groupby("income_type")["amount"].sum().to_dict()
        
        # Calculate qualified vs non-qualified dividends
        qualified_dividends = df[df["is_qualified"]]["amount"].sum() if "is_qualified" in df.columns else 0
        non_qualified_dividends = df[(df["income_type"] == "dividend") & (~df["is_qualified"])]["amount"].sum() \
            if "is_qualified" in df.columns else 0
        
        # Add summary as a metadata attribute
        df.attrs["summary"] = {
            "total_income": df["amount"].sum(),
            "income_by_type": income_summary,
            "qualified_dividends": qualified_dividends,
            "non_qualified_dividends": non_qualified_dividends,
            "foreign_tax_paid": df["foreign_tax_paid"].sum() if "foreign_tax_paid" in df.columns else 0
        }
        
        return df
    
    def _generate_wash_sales_report(self,
                                  start_date: datetime,
                                  end_date: datetime,
                                  accounts: Optional[List[str]] = None,
                                  symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate a report of wash sales.
        
        Args:
            start_date: Start date for the report period
            end_date: End date for the report period
            accounts: List of account IDs to include (None for all)
            symbols: List of symbols to include (None for all)
            
        Returns:
            DataFrame with wash sale information
        """
        # Get wash sale data from tax manager
        wash_sales = self.tax_manager.get_wash_sales(
            start_date=start_date,
            end_date=end_date,
            account_ids=accounts,
            symbols=symbols
        )
        
        # Convert to DataFrame
        if not wash_sales:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "symbol", "account_id", "sale_date", "loss_amount", "disallowed_amount",
                "repurchase_date", "repurchase_quantity", "original_lot_id", "new_lot_id"
            ])
        
        df = pd.DataFrame(wash_sales)
        
        # Sort by sale date (descending)
        df = df.sort_values(by="sale_date", ascending=False)
        
        # Calculate summary statistics
        summary = {
            "total_wash_sales": len(df),
            "total_disallowed_losses": df["disallowed_amount"].sum(),
            "symbols_with_wash_sales": df["symbol"].nunique(),
            "accounts_with_wash_sales": df["account_id"].nunique()
        }
        
        # Add summary as a metadata attribute
        df.attrs["summary"] = summary
        
        return df
    
    def _generate_harvesting_report(self,
                                  start_date: datetime,
                                  end_date: datetime,
                                  accounts: Optional[List[str]] = None,
                                  symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate a report of tax-loss harvesting activities.
        
        Args:
            start_date: Start date for the report period
            end_date: End date for the report period
            accounts: List of account IDs to include (None for all)
            symbols: List of symbols to include (None for all)
            
        Returns:
            DataFrame with tax-loss harvesting information
        """
        # Get harvesting data from tax manager
        harvesting_data = self.tax_manager.get_harvesting_history(
            start_date=start_date,
            end_date=end_date,
            account_ids=accounts,
            symbols=symbols
        )
        
        # Convert to DataFrame
        if not harvesting_data:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "harvest_date", "symbol", "account_id", "quantity", "sale_price",
                "cost_basis", "realized_loss", "replacement_symbol", "replacement_quantity",
                "replacement_price", "wash_sale_disallowed", "net_tax_benefit"
            ])
        
        df = pd.DataFrame(harvesting_data)
        
        # Sort by harvest date (descending)
        df = df.sort_values(by="harvest_date", ascending=False)
        
        # Calculate net tax benefit (assuming 37% tax rate for short-term losses)
        if "net_tax_benefit" not in df.columns:
            df["net_tax_benefit"] = df.apply(
                lambda row: abs(row["realized_loss"]) * 0.37 if row["realized_loss"] < 0 else 0, 
                axis=1
            )
        
        # Calculate summary statistics
        summary = {
            "total_harvesting_transactions": len(df),
            "total_realized_losses": df["realized_loss"].sum(),
            "total_wash_sale_disallowed": df["wash_sale_disallowed"].sum() if "wash_sale_disallowed" in df.columns else 0,
            "net_harvested_losses": df["realized_loss"].sum() - 
                (df["wash_sale_disallowed"].sum() if "wash_sale_disallowed" in df.columns else 0),
            "estimated_tax_benefit": df["net_tax_benefit"].sum()
        }
        
        # Add summary as a metadata attribute
        df.attrs["summary"] = summary
        
        return df
    
    def _generate_year_end_report(self, tax_year: int, accounts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive year-end tax summary.
        
        Args:
            tax_year: The tax year to generate the report for
            accounts: List of account IDs to include (None for all)
            
        Returns:
            Dictionary with year-end tax summary information
        """
        # Set date range for the tax year
        start_date = datetime(tax_year, 1, 1)
        end_date = datetime(tax_year, 12, 31, 23, 59, 59)
        
        # Generate component reports
        realized_gains_df = self._generate_realized_gains_report(start_date, end_date, accounts)
        income_df = self._generate_income_report(start_date, end_date, accounts)
        wash_sales_df = self._generate_wash_sales_report(start_date, end_date, accounts)
        harvesting_df = self._generate_harvesting_report(start_date, end_date, accounts)
        
        # Get unrealized gains as of year end
        unrealized_gains_df = self._generate_unrealized_gains_report(end_date, accounts)
        
        # Compile year-end summary
        year_end_summary = {
            "tax_year": tax_year,
            "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accounts_included": accounts if accounts else "All accounts",
            
            # Realized gains summary
            "realized_gains_summary": realized_gains_df.attrs.get("summary", {}),
            
            # Income summary
            "income_summary": income_df.attrs.get("summary", {}),
            
            # Wash sales summary
            "wash_sales_summary": wash_sales_df.attrs.get("summary", {}),
            
            # Harvesting summary
            "harvesting_summary": harvesting_df.attrs.get("summary", {}),
            
            # Unrealized gains summary (for informational purposes)
            "unrealized_gains_summary": unrealized_gains_df.attrs.get("summary", {}),
            
            # Tax liability estimate
            "estimated_tax_liability": {
                "short_term_gains": realized_gains_df.attrs.get("summary", {}).get("short_term_gains", 0),
                "long_term_gains": realized_gains_df.attrs.get("summary", {}).get("long_term_gains", 0),
                "qualified_dividends": income_df.attrs.get("summary", {}).get("qualified_dividends", 0),
                "non_qualified_dividends": income_df.attrs.get("summary", {}).get("non_qualified_dividends", 0),
                "interest_income": income_df.attrs.get("summary", {}).get("income_by_type", {}).get("interest", 0),
                "foreign_tax_paid": income_df.attrs.get("summary", {}).get("foreign_tax_paid", 0),
                "estimated_total_tax": realized_gains_df.attrs.get("summary", {}).get("estimated_tax", 0) + 
                                      (income_df.attrs.get("summary", {}).get("qualified_dividends", 0) * 0.20) + 
                                      (income_df.attrs.get("summary", {}).get("non_qualified_dividends", 0) * 0.37) + 
                                      (income_df.attrs.get("summary", {}).get("income_by_type", {}).get("interest", 0) * 0.37)
            },
            
            # Tax optimization opportunities
            "tax_optimization_opportunities": self._identify_tax_optimization_opportunities(realized_gains_df, unrealized_gains_df)
        }
        
        return year_end_summary
    
    def _generate_1099_report(self, tax_year: int, accounts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a report with information needed for Form 1099-B, 1099-DIV, and 1099-INT.
        
        Args:
            tax_year: The tax year to generate the report for
            accounts: List of account IDs to include (None for all)
            
        Returns:
            Dictionary with 1099 form information
        """
        # Set date range for the tax year
        start_date = datetime(tax_year, 1, 1)
        end_date = datetime(tax_year, 12, 31, 23, 59, 59)
        
        # Generate component reports
        realized_gains_df = self._generate_realized_gains_report(start_date, end_date, accounts)
        income_df = self._generate_income_report(start_date, end_date, accounts)
        
        # Prepare 1099-B data (realized gains/losses)
        form_1099b_data = {
            "short_term_transactions": [],
            "long_term_transactions": [],
            "summary": {
                "short_term_proceeds": 0,
                "short_term_cost_basis": 0,
                "short_term_gain_loss": 0,
                "long_term_proceeds": 0,
                "long_term_cost_basis": 0,
                "long_term_gain_loss": 0,
                "wash_sale_loss_disallowed": 0
            }
        }
        
        if not realized_gains_df.empty:
            # Process short-term transactions
            short_term_df = realized_gains_df[~realized_gains_df["is_long_term"]]
            for _, row in short_term_df.iterrows():
                form_1099b_data["short_term_transactions"].append({
                    "description": f"{row['symbol']} - {row['quantity']} shares",
                    "date_acquired": row["purchase_date"].strftime("%m/%d/%Y"),
                    "date_sold": row["sale_date"].strftime("%m/%d/%Y"),
                    "proceeds": row["proceeds"],
                    "cost_basis": row["cost_basis"],
                    "gain_loss": row["gain_amount"],
                    "wash_sale_loss_disallowed": row.get("wash_sale_adjustment", 0)
                })
            
            # Process long-term transactions
            long_term_df = realized_gains_df[realized_gains_df["is_long_term"]]
            for _, row in long_term_df.iterrows():
                form_1099b_data["long_term_transactions"].append({
                    "description": f"{row['symbol']} - {row['quantity']} shares",
                    "date_acquired": row["purchase_date"].strftime("%m/%d/%Y"),
                    "date_sold": row["sale_date"].strftime("%m/%d/%Y"),
                    "proceeds": row["proceeds"],
                    "cost_basis": row["cost_basis"],
                    "gain_loss": row["gain_amount"],
                    "wash_sale_loss_disallowed": row.get("wash_sale_adjustment", 0)
                })
            
            # Update summary
            form_1099b_data["summary"] = {
                "short_term_proceeds": short_term_df["proceeds"].sum(),
                "short_term_cost_basis": short_term_df["cost_basis"].sum(),
                "short_term_gain_loss": short_term_df["gain_amount"].sum(),
                "long_term_proceeds": long_term_df["proceeds"].sum(),
                "long_term_cost_basis": long_term_df["cost_basis"].sum(),
                "long_term_gain_loss": long_term_df["gain_amount"].sum(),
                "wash_sale_loss_disallowed": realized_gains_df["wash_sale_adjustment"].sum() 
                    if "wash_sale_adjustment" in realized_gains_df.columns else 0
            }
        
        # Prepare 1099-DIV data (dividends)
        form_1099div_data = {
            "total_ordinary_dividends": 0,
            "qualified_dividends": 0,
            "total_capital_gain_distributions": 0,
            "section_199A_dividends": 0,
            "foreign_tax_paid": 0,
            "foreign_country": "",
            "transactions": []
        }
        
        if not income_df.empty:
            # Filter for dividend income
            dividend_df = income_df[income_df["income_type"] == "dividend"]
            
            # Process dividend transactions
            for _, row in dividend_df.iterrows():
                form_1099div_data["transactions"].append({
                    "date": row["date"].strftime("%m/%d/%Y"),
                    "symbol": row["symbol"],
                    "description": row.get("description", f"{row['symbol']} Dividend"),
                    "amount": row["amount"],
                    "qualified": row.get("is_qualified", False),
                    "foreign_tax": row.get("foreign_tax_paid", 0)
                })
            
            # Update summary
            form_1099div_data.update({
                "total_ordinary_dividends": dividend_df["amount"].sum(),
                "qualified_dividends": dividend_df[dividend_df.get("is_qualified", False)]["amount"].sum() 
                    if "is_qualified" in dividend_df.columns else 0,
                "foreign_tax_paid": dividend_df["foreign_tax_paid"].sum() 
                    if "foreign_tax_paid" in dividend_df.columns else 0
            })
        
        # Prepare 1099-INT data (interest)
        form_1099int_data = {
            "interest_income": 0,
            "early_withdrawal_penalty": 0,
            "us_savings_bonds": 0,
            "federal_tax_withheld": 0,
            "tax_exempt_interest": 0,
            "transactions": []
        }
        
        if not income_df.empty:
            # Filter for interest income
            interest_df = income_df[income_df["income_type"] == "interest"]
            
            # Process interest transactions
            for _, row in interest_df.iterrows():
                form_1099int_data["transactions"].append({
                    "date": row["date"].strftime("%m/%d/%Y"),
                    "payer": row.get("symbol", "Unknown"),
                    "description": row.get("description", "Interest Income"),
                    "amount": row["amount"],
                    "tax_exempt": row.get("tax_exempt", False)
                })
            
            # Update summary
            form_1099int_data.update({
                "interest_income": interest_df[~interest_df.get("tax_exempt", False)]["amount"].sum() 
                    if "tax_exempt" in interest_df.columns else interest_df["amount"].sum(),
                "tax_exempt_interest": interest_df[interest_df.get("tax_exempt", False)]["amount"].sum() 
                    if "tax_exempt" in interest_df.columns else 0
            })
        
        # Combine all 1099 data
        form_1099_report = {
            "tax_year": tax_year,
            "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accounts_included": accounts if accounts else "All accounts",
            "form_1099b": form_1099b_data,
            "form_1099div": form_1099div_data,
            "form_1099int": form_1099int_data
        }
        
        return form_1099_report
    
    def _identify_tax_optimization_opportunities(self, realized_gains_df: pd.DataFrame, unrealized_gains_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify potential tax optimization opportunities based on realized and unrealized gains/losses.
        
        Args:
            realized_gains_df: DataFrame with realized gains/losses
            unrealized_gains_df: DataFrame with unrealized gains/losses
            
        Returns:
            List of tax optimization opportunities
        """
        opportunities = []
        
        # Get summary data
        realized_summary = realized_gains_df.attrs.get("summary", {})
        unrealized_summary = unrealized_gains_df.attrs.get("summary", {})
        
        # Check for tax-loss harvesting opportunities
        if unrealized_summary.get("total_unrealized_losses", 0) < 0:
            # Significant unrealized losses that could be harvested
            opportunities.append({
                "type": "tax_loss_harvesting",
                "description": "Consider harvesting unrealized losses to offset gains",
                "potential_benefit": abs(unrealized_summary.get("total_unrealized_losses", 0)),
                "details": {
                    "short_term_losses": abs(unrealized_summary.get("short_term_unrealized_losses", 0)),
                    "long_term_losses": abs(unrealized_summary.get("long_term_unrealized_losses", 0))
                }
            })
        
        # Check for capital gains budget opportunities
        if realized_summary.get("net_gain_loss", 0) > 0:
            # Already realized gains - may want to realize some losses to offset
            opportunities.append({
                "type": "gain_offset",
                "description": "Consider realizing losses to offset existing gains",
                "potential_benefit": min(
                    abs(unrealized_summary.get("total_unrealized_losses", 0)),
                    realized_summary.get("net_gain_loss", 0)
                ),
                "details": {
                    "realized_gains": realized_summary.get("net_gain_loss", 0),
                    "available_unrealized_losses": abs(unrealized_summary.get("total_unrealized_losses", 0))
                }
            })
        
        # Check for long-term vs short-term optimization
        short_term_near_long = []
        if not unrealized_gains_df.empty and "days_to_long_term" in unrealized_gains_df.columns:
            # Find positions with gains that are close to long-term status
            near_long_term = unrealized_gains_df[
                (unrealized_gains_df["unrealized_gain"] > 0) & 
                (~unrealized_gains_df["is_long_term"]) & 
                (unrealized_gains_df["days_to_long_term"] < 60) & 
                (unrealized_gains_df["days_to_long_term"] > 0)
            ]
            
            if not near_long_term.empty:
                for _, row in near_long_term.iterrows():
                    short_term_near_long.append({
                        "symbol": row["symbol"],
                        "days_to_long_term": row["days_to_long_term"],
                        "unrealized_gain": row["unrealized_gain"],
                        "potential_tax_savings": row["unrealized_gain"] * 0.17  # Approximate difference between short and long-term rates
                    })
        
        if short_term_near_long:
            opportunities.append({
                "type": "short_to_long_term",
                "description": "Consider holding these positions until they qualify for long-term capital gains rates",
                "potential_benefit": sum(item["potential_tax_savings"] for item in short_term_near_long),
                "details": {
                    "positions": short_term_near_long
                }
            })
        
        # Check for year-end planning opportunities
        current_date = datetime.now()
        days_to_year_end = (datetime(current_date.year, 12, 31) - current_date).days
        
        if days_to_year_end < 60:
            # Year-end tax planning opportunities
            if realized_summary.get("net_gain_loss", 0) > 0:
                # Have net gains for the year - consider loss harvesting
                opportunities.append({
                    "type": "year_end_loss_harvesting",
                    "description": "Consider harvesting losses before year-end to offset gains",
                    "potential_benefit": min(
                        abs(unrealized_summary.get("total_unrealized_losses", 0)),
                        realized_summary.get("net_gain_loss", 0)
                    ),
                    "details": {
                        "days_to_year_end": days_to_year_end,
                        "net_realized_gains": realized_summary.get("net_gain_loss", 0),
                        "available_unrealized_losses": abs(unrealized_summary.get("total_unrealized_losses", 0))
                    }
                })
            elif realized_summary.get("net_gain_loss", 0) < 0:
                # Have net losses for the year - consider gain harvesting up to the loss amount
                # plus $3,000 (standard capital loss deduction)
                loss_offset_capacity = abs(realized_summary.get("net_gain_loss", 0)) + 3000
                
                opportunities.append({
                    "type": "year_end_gain_harvesting",
                    "description": "Consider harvesting gains before year-end to utilize existing losses",
                    "potential_benefit": min(
                        unrealized_summary.get("total_unrealized_gains", 0),
                        loss_offset_capacity
                    ),
                    "details": {
                        "days_to_year_end": days_to_year_end,
                        "net_realized_losses": abs(realized_summary.get("net_gain_loss", 0)),
                        "loss_offset_capacity": loss_offset_capacity,
                        "available_unrealized_gains": unrealized_summary.get("total_unrealized_gains", 0)
                    }
                })
        
        return opportunities
    
    def _save_report(self, report_data: Union[pd.DataFrame, Dict[str, Any]], filename: str, format: ReportFormat) -> str:
        """
        Save a report in the specified format.
        
        Args:
            report_data: The report data (DataFrame or Dict)
            filename: Base filename for the report
            format: Output format
            
        Returns:
            Path to the saved report file
        """
        # Create full path
        if not filename.endswith(f".{format.value}"):
            filename = f"{filename}.{format.value}"
            
        file_path = os.path.join(self.output_dir, filename)
        
        # Save in the appropriate format
        if isinstance(report_data, pd.DataFrame):
            if format == ReportFormat.CSV:
                report_data.to_csv(file_path, index=False)
            elif format == ReportFormat.JSON:
                # Convert DataFrame to JSON with summary metadata
                json_data = {
                    "data": json.loads(report_data.to_json(orient="records")),
                    "summary": report_data.attrs.get("summary", {})
                }
                with open(file_path, 'w') as f:
                    json.dump(json_data, f, indent=2, default=self._json_serializer)
            elif format == ReportFormat.EXCEL:
                # Save to Excel with summary on a separate sheet
                with pd.ExcelWriter(file_path) as writer:
                    report_data.to_excel(writer, sheet_name="Data", index=False)
                    
                    # Add summary sheet if available
                    if "summary" in report_data.attrs:
                        summary_df = pd.DataFrame(report_data.attrs["summary"].items(), columns=["Metric", "Value"])
                        summary_df.to_excel(writer, sheet_name="Summary", index=False)
            elif format == ReportFormat.PDF:
                # PDF generation would require additional libraries like reportlab
                # This is a simplified placeholder
                with open(file_path, 'w') as f:
                    f.write("PDF generation not implemented in this version.")
                logger.warning("PDF generation not fully implemented")
        else:
            # Handle dictionary data
            if format == ReportFormat.JSON:
                with open(file_path, 'w') as f:
                    json.dump(report_data, f, indent=2, default=self._json_serializer)
            elif format == ReportFormat.CSV:
                # For dictionary data, convert to DataFrame if possible
                try:
                    if "data" in report_data:
                        pd.DataFrame(report_data["data"]).to_csv(file_path, index=False)
                    else:
                        pd.DataFrame([report_data]).to_csv(file_path, index=False)
                except Exception as e:
                    logger.error(f"Error converting dictionary to CSV: {e}")
                    with open(file_path, 'w') as f:
                        f.write(str(report_data))
            elif format == ReportFormat.EXCEL:
                # For dictionary data, create Excel with multiple sheets
                with pd.ExcelWriter(file_path) as writer:
                    # Try to convert each top-level key to a sheet
                    for key, value in report_data.items():
                        if isinstance(value, list):
                            pd.DataFrame(value).to_excel(writer, sheet_name=key[:31], index=False)  # Excel sheet name length limit
                        elif isinstance(value, dict):
                            pd.DataFrame([value]).to_excel(writer, sheet_name=key[:31], index=False)
                        else:
                            pd.DataFrame([{key: value}]).to_excel(writer, sheet_name="Summary"[:31], index=False)
            elif format == ReportFormat.PDF:
                # PDF generation placeholder
                with open(file_path, 'w') as f:
                    f.write("PDF generation not implemented in this version.")
                logger.warning("PDF generation not fully implemented")
        
        logger.info(f"Report saved to {file_path}")
        return file_path
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for objects not serializable by default json code."""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if pd.isna(obj):
            return None
        raise TypeError(f"Type {type(obj)} not serializable")
    
    def get_report_history(self,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         report_type: Optional[ReportType] = None) -> List[Dict[str, Any]]:
        """
        Get history of generated reports with optional filtering.
        
        Args:
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)
            report_type: Filter by report type (optional)
            
        Returns:
            List of report history entries
        """
        filtered_history = self.report_history
        
        if start_date:
            filtered_history = [h for h in filtered_history if h["timestamp"] >= start_date]
            
        if end_date:
            filtered_history = [h for h in filtered_history if h["timestamp"] <= end_date]
            
        if report_type:
            filtered_history = [h for h in filtered_history if h["report_type"] == report_type.value]
            
        return filtered_history
    
    def generate_custom_report(self,
                             query: Dict[str, Any],
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             format: ReportFormat = ReportFormat.CSV,
                             filename: Optional[str] = None) -> str:
        """
        Generate a custom report based on a query specification.
        
        Args:
            query: Dictionary with query parameters and aggregations
            start_date: Start date for the report period
            end_date: End date for the report period
            format: Output format for the report
            filename: Custom filename for the report
            
        Returns:
            Path to the generated report file
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
            
        if start_date is None:
            # Default to beginning of the year
            start_date = datetime(end_date.year, 1, 1)
        
        # Extract query parameters
        report_type = query.get("report_type", "custom")
        accounts = query.get("accounts")
        symbols = query.get("symbols")
        groupby = query.get("groupby", [])
        metrics = query.get("metrics", [])
        filters = query.get("filters", {})
        sort_by = query.get("sort_by")
        limit = query.get("limit")
        
        # Generate base report based on report type
        if report_type == "realized_gains":
            base_df = self._generate_realized_gains_report(start_date, end_date, accounts, symbols)
        elif report_type == "unrealized_gains":
            base_df = self._generate_unrealized_gains_report(end_date, accounts, symbols)
        elif report_type == "tax_lots":
            base_df = self._generate_tax_lots_report(end_date, accounts, symbols)
        elif report_type == "income":
            base_df = self._generate_income_report(start_date, end_date, accounts, symbols)
        elif report_type == "wash_sales":
            base_df = self._generate_wash_sales_report(start_date, end_date, accounts, symbols)
        elif report_type == "harvesting":
            base_df = self._generate_harvesting_report(start_date, end_date, accounts, symbols)
        else:
            logger.error(f"Unsupported report type for custom query: {report_type}")
            raise ValueError(f"Unsupported report type for custom query: {report_type}")
        
        # Apply filters
        for column, filter_value in filters.items():
            if column in base_df.columns:
                if isinstance(filter_value, dict):
                    # Range filter
                    if "min" in filter_value:
                        base_df = base_df[base_df[column] >= filter_value["min"]]
                    if "max" in filter_value:
                        base_df = base_df[base_df[column] <= filter_value["max"]]
                    if "in" in filter_value:
                        base_df = base_df[base_df[column].isin(filter_value["in"])]
                    if "not_in" in filter_value:
                        base_df = base_df[~base_df[column].isin(filter_value["not_in"])]
                else:
                    # Exact match filter
                    base_df = base_df[base_df[column] == filter_value]
        
        # Apply groupby and aggregations if specified
        if groupby and metrics:
            # Prepare aggregation dictionary
            agg_dict = {}
            for metric in metrics:
                if isinstance(metric, dict):
                    # Custom aggregation
                    col = metric.get("column")
                    func = metric.get("function", "sum")
                    name = metric.get("name", f"{func}_{col}")
                    agg_dict[col] = func
                else:
                    # Simple column aggregation (default to sum)
                    agg_dict[metric] = "sum"
            
            # Apply groupby and aggregation
            result_df = base_df.groupby(groupby).agg(agg_dict).reset_index()
        else:
            result_df = base_df
        
        # Apply sorting if specified
        if sort_by:
            if isinstance(sort_by, list):
                # Multiple sort columns
                sort_cols = []
                ascending = []
                for sort_item in sort_by:
                    if isinstance(sort_item, dict):
                        sort_cols.append(sort_item.get("column"))
                        ascending.append(not sort_item.get("desc", False))
                    else:
                        sort_cols.append(sort_item)
                        ascending.append(True)
                result_df = result_df.sort_values(by=sort_cols, ascending=ascending)
            else:
                # Single sort column
                result_df = result_df.sort_values(by=sort_by)
        
        # Apply limit if specified
        if limit and isinstance(limit, int) and limit > 0:
            result_df = result_df.head(limit)
        
        # Generate filename if not provided
        if filename is None:
            date_str = end_date.strftime("%Y%m%d")
            filename = f"custom_{report_type}_{date_str}"
        
        # Save the report in the specified format
        file_path = self._save_report(result_df, filename, format)
        
        # Record in history
        self.report_history.append({
            "timestamp": datetime.now(),
            "report_type": "custom",
            "query": query,
            "start_date": start_date,
            "end_date": end_date,
            "format": format.value,
            "file_path": file_path
        })
        
        return file_path
    
    def reset(self) -> None:
        """
        Reset the tax reporting module.
        """
        self.report_history = []
        logger.info("Tax Reporting module reset")