"""Example of using external risk models in Friday AI Trading System.

This script demonstrates how to integrate external risk models into the
Friday AI Trading System's risk management framework.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the necessary classes
from src.risk.risk_model_integrator import RiskModelIntegrator
from src.risk.adapters.bloomberg_risk_adapter import BloombergRiskAdapter
from src.risk.adapters.msci_risk_adapter import MSCIRiskAdapter
from src.risk.adapters.factset_risk_adapter import FactSetRiskAdapter


def create_sample_portfolio():
    """Create a sample portfolio for demonstration purposes.
    
    Returns:
        Dictionary containing portfolio data.
    """
    # Create a sample portfolio with tech stocks
    portfolio = {
        "positions": {
            "AAPL": {
                "quantity": 100,
                "market_value": 15000,
                "weight": 0.15,
                "sector": "Technology",
                "country": "US",
            },
            "MSFT": {
                "quantity": 80,
                "market_value": 20000,
                "weight": 0.20,
                "sector": "Technology",
                "country": "US",
            },
            "GOOGL": {
                "quantity": 25,
                "market_value": 25000,
                "weight": 0.25,
                "sector": "Communication Services",
                "country": "US",
            },
            "AMZN": {
                "quantity": 30,
                "market_value": 30000,
                "weight": 0.30,
                "sector": "Consumer Discretionary",
                "country": "US",
            },
            "META": {
                "quantity": 50,
                "market_value": 10000,
                "weight": 0.10,
                "sector": "Communication Services",
                "country": "US",
            },
        },
        "total_value": 100000,
        "benchmark": "SPY",
        "as_of_date": datetime.now().strftime("%Y-%m-%d"),
        "historical_returns": {
            # Mock historical returns data
            "dates": [f"2023-{month:02d}-01" for month in range(1, 13)],
            "returns": {
                "AAPL": [0.05, -0.02, 0.03, 0.01, -0.01, 0.04, 0.02, -0.03, 0.01, 0.03, -0.02, 0.04],
                "MSFT": [0.04, -0.01, 0.02, 0.03, -0.02, 0.03, 0.01, -0.02, 0.02, 0.04, -0.01, 0.03],
                "GOOGL": [0.03, -0.03, 0.04, 0.02, -0.01, 0.02, 0.03, -0.04, 0.03, 0.02, -0.03, 0.05],
                "AMZN": [0.06, -0.04, 0.05, 0.03, -0.02, 0.05, 0.04, -0.05, 0.02, 0.04, -0.03, 0.06],
                "META": [0.07, -0.05, 0.04, 0.02, -0.03, 0.06, 0.03, -0.06, 0.04, 0.05, -0.04, 0.07],
            },
        },
    }
    
    return portfolio


def pretty_print_dict(data, indent=0):
    """Pretty print a dictionary with indentation.
    
    Args:
        data: Dictionary to print.
        indent: Indentation level.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            pretty_print_dict(value, indent + 1)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            print("  " * indent + f"{key}:")
            for item in value:
                pretty_print_dict(item, indent + 1)
                print("  " * indent + "---")
        else:
            print("  " * indent + f"{key}: {value}")


def main():
    """Main function to demonstrate external risk model integration."""
    logger.info("Starting external risk model integration example")
    
    # Create a sample portfolio
    portfolio = create_sample_portfolio()
    logger.info(f"Created sample portfolio with {len(portfolio['positions'])} positions")
    
    # Create the risk model integrator
    integrator = RiskModelIntegrator()
    logger.info("Created risk model integrator")
    
    # Create and register the risk model adapters
    # In a real-world scenario, you would provide actual API keys
    bloomberg_adapter = BloombergRiskAdapter(api_key="your_bloomberg_api_key_here")
    msci_adapter = MSCIRiskAdapter(api_key="your_msci_api_key_here")
    factset_adapter = FactSetRiskAdapter(api_key="your_factset_api_key_here")
    
    integrator.register_model("bloomberg", bloomberg_adapter)
    integrator.register_model("msci", msci_adapter)
    integrator.register_model("factset", factset_adapter)
    logger.info("Registered Bloomberg, MSCI, and FactSet risk model adapters")
    
    # Activate all models
    integrator.activate_model("bloomberg")
    integrator.activate_model("msci")
    integrator.activate_model("factset")
    logger.info("Activated all risk models")
    
    # Set model weights (50% Bloomberg, 30% MSCI, 20% FactSet)
    integrator.set_model_weights({"bloomberg": 0.5, "msci": 0.3, "factset": 0.2})
    logger.info("Set model weights: 50% Bloomberg, 30% MSCI, 20% FactSet")
    
    # Get combined risk metrics
    logger.info("Getting combined risk metrics...")
    risk_metrics = integrator.get_combined_risk_metrics(portfolio)
    
    print("\n=== Combined Risk Metrics ===\n")
    pretty_print_dict(risk_metrics)
    
    # Get combined factor exposures
    logger.info("Getting combined factor exposures...")
    factor_exposures = integrator.get_combined_factor_exposures(portfolio)
    
    print("\n=== Combined Factor Exposures ===\n")
    # Only print the top 10 factor exposures for brevity
    top_factors = {k: v for k, v in sorted(
        {k: v for k, v in factor_exposures.items() if k not in ["model_sources", "model_weights"]}.items(), 
        key=lambda item: abs(item[1]), 
        reverse=True
    )[:10]}
    pretty_print_dict({**top_factors, "model_sources": factor_exposures["model_sources"]})
    
    # Get stress test results for a financial crisis scenario
    logger.info("Getting stress test results for 2008 financial crisis scenario...")
    stress_results = integrator.get_stress_test_results(portfolio, "2008_financial_crisis")
    
    print("\n=== Stress Test Results (2008 Financial Crisis) ===\n")
    pretty_print_dict(stress_results)
    
    # Get consensus correlation matrix
    logger.info("Getting consensus correlation matrix...")
    assets = list(portfolio["positions"].keys())
    correlation_matrix = integrator.get_consensus_correlation_matrix(assets)
    
    print("\n=== Consensus Correlation Matrix ===\n")
    print(correlation_matrix.round(2))
    
    # Get combined VaR contribution
    logger.info("Getting combined VaR contribution...")
    var_contribution = integrator.get_combined_var_contribution(portfolio)
    
    print("\n=== Combined VaR Contribution ===\n")
    # Sort by contribution
    sorted_var = {k: v for k, v in sorted(
        {k: v for k, v in var_contribution.items() if k != "model_sources"}.items(), 
        key=lambda item: item[1], 
        reverse=True
    )}
    pretty_print_dict({**sorted_var, "model_sources": var_contribution["model_sources"]})
    
    # Example of using only one model
    logger.info("Deactivating MSCI and FactSet models to demonstrate using only Bloomberg...")
    integrator.deactivate_model("msci")
    integrator.deactivate_model("factset")
    
    # Get risk metrics from just Bloomberg
    logger.info("Getting risk metrics from Bloomberg only...")
    bloomberg_metrics = integrator.get_combined_risk_metrics(portfolio)
    
    print("\n=== Bloomberg Risk Metrics ===\n")
    pretty_print_dict(bloomberg_metrics)
    
    # Example of comparing different models
    logger.info("Reactivating all models to compare their results...")
    integrator.activate_model("msci")
    integrator.activate_model("factset")
    
    # Set equal weights for comparison
    integrator.set_model_weights({"bloomberg": 0.33, "msci": 0.33, "factset": 0.34})
    logger.info("Set equal weights for all models")
    
    # Compare VaR estimates from different models
    bloomberg_var = bloomberg_adapter.get_risk_metrics(portfolio)["var_95"]
    msci_var = msci_adapter.get_risk_metrics(portfolio)["var_95"]
    factset_var = factset_adapter.get_risk_metrics(portfolio)["var_95"]
    combined_var = integrator.get_combined_risk_metrics(portfolio)["var_95"]
    
    print("\n=== VaR Comparison (95% Confidence) ===\n")
    print(f"Bloomberg VaR: {bloomberg_var:.4f}")
    print(f"MSCI VaR: {msci_var:.4f}")
    print(f"FactSet VaR: {factset_var:.4f}")
    print(f"Combined VaR: {combined_var:.4f}")
    
    logger.info("External risk model integration example completed")


if __name__ == "__main__":
    main()