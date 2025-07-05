"""Example configuration for multi-portfolio support.

This module demonstrates how to configure and use the multi-portfolio support
features, including creating multiple portfolios, organizing them into groups,
and performing cross-portfolio analysis.
"""

import sys
import os
import logging
from datetime import datetime

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.portfolio.multi_portfolio_integration import MultiPortfolioIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Run the multi-portfolio configuration example."""
    logger.info("Starting multi-portfolio configuration example")
    
    # Create the multi-portfolio integration
    multi_integration = MultiPortfolioIntegration(
        default_config={
            "risk_management": {
                "max_position_size": 0.05,  # 5% of portfolio
                "max_sector_exposure": 0.25,  # 25% of portfolio
                "stop_loss_percentage": 0.15  # 15% stop loss
            },
            "tax_settings": {
                "tax_rate_short_term": 0.35,  # 35% short-term tax rate
                "tax_rate_long_term": 0.15,  # 15% long-term tax rate
                "tax_loss_harvesting": True
            },
            "performance_settings": {
                "benchmark": "SPY",  # S&P 500 ETF as benchmark
                "risk_free_rate": 0.02  # 2% risk-free rate
            }
        }
    )
    
    # Create multiple portfolios with different strategies
    growth_portfolio_id = multi_integration.create_portfolio(
        name="Growth Portfolio",
        initial_capital=100000.0,
        description="Aggressive growth strategy focusing on technology and emerging markets",
        tags=["growth", "high-risk", "technology"],
        config={
            "risk_management": {
                "max_position_size": 0.08,  # 8% of portfolio
                "max_sector_exposure": 0.35,  # 35% of portfolio
                "stop_loss_percentage": 0.20  # 20% stop loss
            }
        }
    )
    
    income_portfolio_id = multi_integration.create_portfolio(
        name="Income Portfolio",
        initial_capital=200000.0,
        description="Income-focused strategy with dividend stocks and bonds",
        tags=["income", "low-risk", "dividend"],
        config={
            "risk_management": {
                "max_position_size": 0.04,  # 4% of portfolio
                "max_sector_exposure": 0.20,  # 20% of portfolio
                "stop_loss_percentage": 0.10  # 10% stop loss
            }
        }
    )
    
    value_portfolio_id = multi_integration.create_portfolio(
        name="Value Portfolio",
        initial_capital=150000.0,
        description="Value investing strategy focusing on undervalued companies",
        tags=["value", "medium-risk"],
        config={
            "risk_management": {
                "max_position_size": 0.06,  # 6% of portfolio
                "max_sector_exposure": 0.25,  # 25% of portfolio
                "stop_loss_percentage": 0.15  # 15% stop loss
            }
        }
    )
    
    esg_portfolio_id = multi_integration.create_portfolio(
        name="ESG Portfolio",
        initial_capital=120000.0,
        description="Environmental, Social, and Governance focused investments",
        tags=["esg", "sustainable", "medium-risk"],
        config={
            "risk_management": {
                "max_position_size": 0.05,  # 5% of portfolio
                "max_sector_exposure": 0.30,  # 30% of portfolio
                "stop_loss_percentage": 0.15  # 15% stop loss
            }
        }
    )
    
    # Create portfolio groups
    equity_group_id = multi_integration.create_portfolio_group(
        name="Equity Strategies",
        portfolio_ids=[growth_portfolio_id, value_portfolio_id],
        description="Portfolios focused on equity investments",
        allocation={
            growth_portfolio_id: 0.6,  # 60% allocation to growth
            value_portfolio_id: 0.4    # 40% allocation to value
        }
    )
    
    conservative_group_id = multi_integration.create_portfolio_group(
        name="Conservative Strategies",
        portfolio_ids=[income_portfolio_id, esg_portfolio_id],
        description="More conservative investment approaches",
        allocation={
            income_portfolio_id: 0.7,  # 70% allocation to income
            esg_portfolio_id: 0.3      # 30% allocation to ESG
        }
    )
    
    # Create a master group containing all portfolios
    master_group_id = multi_integration.create_portfolio_group(
        name="All Strategies",
        portfolio_ids=[
            growth_portfolio_id, 
            income_portfolio_id, 
            value_portfolio_id, 
            esg_portfolio_id
        ],
        description="All investment strategies",
        allocation={
            growth_portfolio_id: 0.25,  # 25% allocation
            income_portfolio_id: 0.35,  # 35% allocation
            value_portfolio_id: 0.20,   # 20% allocation
            esg_portfolio_id: 0.20      # 20% allocation
        }
    )
    
    # Set the active portfolio
    multi_integration.set_active_portfolio(growth_portfolio_id)
    logger.info(f"Active portfolio: {multi_integration.get_active_portfolio_id()}")
    
    # Demonstrate cross-portfolio analysis
    logger.info("Performing cross-portfolio analysis")
    
    # Compare performance metrics across portfolios
    comparison = multi_integration.compare_portfolios(
        portfolio_ids=[growth_portfolio_id, income_portfolio_id, value_portfolio_id, esg_portfolio_id],
        metrics=["returns", "volatility", "sharpe_ratio", "max_drawdown"]
    )
    logger.info(f"Portfolio comparison: {comparison}")
    
    # Calculate correlation between portfolios
    correlation = multi_integration.calculate_correlation(
        portfolio_ids=[growth_portfolio_id, income_portfolio_id, value_portfolio_id, esg_portfolio_id]
    )
    logger.info(f"Portfolio correlation: {correlation}")
    
    # Analyze diversification across portfolios
    diversification = multi_integration.analyze_diversification(
        portfolio_ids=[growth_portfolio_id, income_portfolio_id, value_portfolio_id, esg_portfolio_id]
    )
    logger.info(f"Diversification analysis: {diversification}")
    
    # Generate a consolidated report for all portfolios
    report = multi_integration.generate_consolidated_report()
    logger.info(f"Consolidated report: {report}")
    
    # Demonstrate portfolio group operations
    logger.info("Performing portfolio group operations")
    
    # Add a new portfolio to a group
    international_portfolio_id = multi_integration.create_portfolio(
        name="International Portfolio",
        initial_capital=180000.0,
        description="International markets focused strategy",
        tags=["international", "medium-risk", "diversified"],
        config={
            "risk_management": {
                "max_position_size": 0.05,  # 5% of portfolio
                "max_sector_exposure": 0.25,  # 25% of portfolio
                "stop_loss_percentage": 0.15  # 15% stop loss
            }
        }
    )
    
    # Add the new portfolio to the equity group
    multi_integration.add_to_group(equity_group_id, international_portfolio_id, allocation=0.3)
    
    # Update allocations in the equity group
    # Note: This would typically be done through a method in the PortfolioGroup class
    # For this example, we're recreating the group with updated allocations
    multi_integration.delete_portfolio_group(equity_group_id)
    equity_group_id = multi_integration.create_portfolio_group(
        name="Equity Strategies",
        portfolio_ids=[growth_portfolio_id, value_portfolio_id, international_portfolio_id],
        description="Portfolios focused on equity investments",
        allocation={
            growth_portfolio_id: 0.4,        # 40% allocation to growth
            value_portfolio_id: 0.3,         # 30% allocation to value
            international_portfolio_id: 0.3   # 30% allocation to international
        }
    )
    
    # Demonstrate portfolio lifecycle management
    logger.info("Demonstrating portfolio lifecycle management")
    
    # Deactivate a portfolio
    multi_integration.deactivate_portfolio(esg_portfolio_id)
    logger.info(f"Deactivated portfolio: {esg_portfolio_id}")
    
    # Delete a portfolio
    multi_integration.delete_portfolio(international_portfolio_id)
    logger.info(f"Deleted portfolio: {international_portfolio_id}")
    
    # Clean up event subscriptions
    multi_integration.cleanup_subscriptions()
    logger.info("Cleaned up event subscriptions")
    
    logger.info("Multi-portfolio configuration example completed")


if __name__ == "__main__":
    main()