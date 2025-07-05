"""Example usage of the Factor-Based Risk Model.

This example demonstrates how to use the FactorRiskModel and FactorRiskModelAdapter
classes for portfolio risk analysis.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.risk.factor_risk_model import FactorRiskModel, FactorRiskModelAdapter
from src.risk.risk_model_integrator import RiskModelIntegrator


def create_sample_portfolio():
    """Create a sample portfolio for demonstration purposes."""
    return {
        "positions": [
            {"ticker": "AAPL", "weight": 0.15, "sector": "Technology", "region": "North America"},
            {"ticker": "MSFT", "weight": 0.12, "sector": "Technology", "region": "North America"},
            {"ticker": "AMZN", "weight": 0.10, "sector": "Consumer Discretionary", "region": "North America"},
            {"ticker": "GOOGL", "weight": 0.08, "sector": "Communication Services", "region": "North America"},
            {"ticker": "META", "weight": 0.07, "sector": "Communication Services", "region": "North America"},
            {"ticker": "BRK.B", "weight": 0.06, "sector": "Financials", "region": "North America"},
            {"ticker": "JNJ", "weight": 0.05, "sector": "Healthcare", "region": "North America"},
            {"ticker": "JPM", "weight": 0.05, "sector": "Financials", "region": "North America"},
            {"ticker": "V", "weight": 0.04, "sector": "Financials", "region": "North America"},
            {"ticker": "PG", "weight": 0.04, "sector": "Consumer Staples", "region": "North America"},
            {"ticker": "NESN.SW", "weight": 0.03, "sector": "Consumer Staples", "region": "Europe"},
            {"ticker": "ASML.AS", "weight": 0.03, "sector": "Technology", "region": "Europe"},
            {"ticker": "NOVN.SW", "weight": 0.03, "sector": "Healthcare", "region": "Europe"},
            {"ticker": "ROG.SW", "weight": 0.02, "sector": "Healthcare", "region": "Europe"},
            {"ticker": "SAP.DE", "weight": 0.02, "sector": "Technology", "region": "Europe"},
            {"ticker": "7203.T", "weight": 0.02, "sector": "Consumer Discretionary", "region": "Asia Pacific"},
            {"ticker": "9984.T", "weight": 0.02, "sector": "Communication Services", "region": "Asia Pacific"},
            {"ticker": "9988.HK", "weight": 0.02, "sector": "Consumer Discretionary", "region": "Asia Pacific"},
            {"ticker": "0700.HK", "weight": 0.02, "sector": "Communication Services", "region": "Asia Pacific"},
            {"ticker": "RELIANCE.NS", "weight": 0.03, "sector": "Energy", "region": "Emerging Markets"},
        ],
        "total_value": 1000000.0,
        "currency": "USD",
        "as_of_date": datetime.now().strftime("%Y-%m-%d"),
    }


def demonstrate_factor_risk_model():
    """Demonstrate the usage of the FactorRiskModel class."""
    print("\n=== Demonstrating Factor Risk Model ===\n")
    
    # Create a sample portfolio
    portfolio = create_sample_portfolio()
    print(f"Sample portfolio with {len(portfolio['positions'])} positions")
    
    # Create instances of different factor risk models
    fundamental_model = FactorRiskModel(model_type="fundamental")
    statistical_model = FactorRiskModel(model_type="statistical")
    macroeconomic_model = FactorRiskModel(model_type="macroeconomic")
    
    print("\n--- Factor Exposures ---")
    
    # Get factor exposures from each model
    fundamental_exposures = fundamental_model.get_factor_exposures(portfolio)
    statistical_exposures = statistical_model.get_factor_exposures(portfolio)
    macroeconomic_exposures = macroeconomic_model.get_factor_exposures(portfolio)
    
    # Print top 5 factor exposures for each model
    print("\nTop 5 Fundamental Factor Exposures:")
    for factor, exposure in sorted(fundamental_exposures.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {factor}: {exposure:.4f}")
    
    print("\nTop 5 Statistical Factor Exposures:")
    for factor, exposure in sorted(statistical_exposures.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {factor}: {exposure:.4f}")
    
    print("\nTop 5 Macroeconomic Factor Exposures:")
    for factor, exposure in sorted(macroeconomic_exposures.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {factor}: {exposure:.4f}")
    
    print("\n--- Factor Returns ---")
    
    # Define date range for factor returns
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Get factor returns from each model
    fundamental_returns = fundamental_model.get_factor_returns(start_date, end_date)
    statistical_returns = statistical_model.get_factor_returns(start_date, end_date)
    macroeconomic_returns = macroeconomic_model.get_factor_returns(start_date, end_date)
    
    # Print top 5 factor returns for each model
    print("\nTop 5 Fundamental Factor Returns:")
    for factor, factor_return in sorted(fundamental_returns.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {factor}: {factor_return:.4f}")
    
    print("\nTop 5 Statistical Factor Returns:")
    for factor, factor_return in sorted(statistical_returns.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {factor}: {factor_return:.4f}")
    
    print("\nTop 5 Macroeconomic Factor Returns:")
    for factor, factor_return in sorted(macroeconomic_returns.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {factor}: {factor_return:.4f}")
    
    print("\n--- Factor Risk Metrics ---")
    
    # Calculate factor risk metrics for each model
    fundamental_risk = fundamental_model.calculate_factor_risk(portfolio)
    statistical_risk = statistical_model.calculate_factor_risk(portfolio)
    macroeconomic_risk = macroeconomic_model.calculate_factor_risk(portfolio)
    
    # Print risk metrics for each model
    print("\nFundamental Model Risk Metrics:")
    print(f"  Factor Risk: {fundamental_risk['factor_risk']:.4f}")
    print(f"  Specific Risk: {fundamental_risk['specific_risk']:.4f}")
    print(f"  Total Risk: {fundamental_risk['total_risk']:.4f}")
    print(f"  Factor VaR (95%): {fundamental_risk['factor_var_95']:.4f}")
    print(f"  Factor Expected Shortfall (95%): {fundamental_risk['factor_es_95']:.4f}")
    
    print("\nStatistical Model Risk Metrics:")
    print(f"  Factor Risk: {statistical_risk['factor_risk']:.4f}")
    print(f"  Specific Risk: {statistical_risk['specific_risk']:.4f}")
    print(f"  Total Risk: {statistical_risk['total_risk']:.4f}")
    print(f"  Factor VaR (95%): {statistical_risk['factor_var_95']:.4f}")
    print(f"  Factor Expected Shortfall (95%): {statistical_risk['factor_es_95']:.4f}")
    
    print("\nMacroeconomic Model Risk Metrics:")
    print(f"  Factor Risk: {macroeconomic_risk['factor_risk']:.4f}")
    print(f"  Specific Risk: {macroeconomic_risk['specific_risk']:.4f}")
    print(f"  Total Risk: {macroeconomic_risk['total_risk']:.4f}")
    print(f"  Factor VaR (95%): {macroeconomic_risk['factor_var_95']:.4f}")
    print(f"  Factor Expected Shortfall (95%): {macroeconomic_risk['factor_es_95']:.4f}")
    
    print("\n--- Top 5 Risk Contributors ---")
    
    # Print top 5 risk contributors for each model
    print("\nFundamental Model Risk Contributors:")
    for factor, contribution in sorted(fundamental_risk['risk_contribution'].items(), 
                                      key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {factor}: {contribution:.6f}")
    
    print("\nStatistical Model Risk Contributors:")
    for factor, contribution in sorted(statistical_risk['risk_contribution'].items(), 
                                      key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {factor}: {contribution:.6f}")
    
    print("\nMacroeconomic Model Risk Contributors:")
    for factor, contribution in sorted(macroeconomic_risk['risk_contribution'].items(), 
                                      key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {factor}: {contribution:.6f}")
    
    print("\n--- Stress Test Results ---")
    
    # Define stress test scenarios
    financial_crisis_scenario = {
        "market": -0.40,  # 40% market decline
        "credit": 0.30,  # 30% credit spread widening
        "liquidity": -0.35,  # 35% liquidity decline
        "volatility": 0.50,  # 50% volatility increase
        "value": -0.25,  # 25% value factor decline
        "momentum": -0.15,  # 15% momentum factor decline
        "quality": 0.10,  # 10% quality factor increase
    }
    
    covid_crash_scenario = {
        "market": -0.30,  # 30% market decline
        "credit": 0.25,  # 25% credit spread widening
        "liquidity": -0.30,  # 30% liquidity decline
        "volatility": 0.40,  # 40% volatility increase
        "value": -0.20,  # 20% value factor decline
        "momentum": -0.10,  # 10% momentum factor decline
        "quality": 0.15,  # 15% quality factor increase
    }
    
    # Run stress tests for each model
    fundamental_financial_crisis = fundamental_model.run_factor_stress_test(
        portfolio, financial_crisis_scenario)
    fundamental_covid_crash = fundamental_model.run_factor_stress_test(
        portfolio, covid_crash_scenario)
    
    macroeconomic_financial_crisis = macroeconomic_model.run_factor_stress_test(
        portfolio, financial_crisis_scenario)
    macroeconomic_covid_crash = macroeconomic_model.run_factor_stress_test(
        portfolio, covid_crash_scenario)
    
    # Print stress test results
    print("\nFundamental Model - Financial Crisis Scenario:")
    print(f"  Portfolio Return: {fundamental_financial_crisis['portfolio_return']:.4f}")
    print(f"  Factor Risk Increase: {fundamental_financial_crisis['stressed_risk_metrics']['factor_risk'] / fundamental_risk['factor_risk']:.2f}x")
    print(f"  Total Risk Increase: {fundamental_financial_crisis['stressed_risk_metrics']['total_risk'] / fundamental_risk['total_risk']:.2f}x")
    
    print("\nFundamental Model - COVID Crash Scenario:")
    print(f"  Portfolio Return: {fundamental_covid_crash['portfolio_return']:.4f}")
    print(f"  Factor Risk Increase: {fundamental_covid_crash['stressed_risk_metrics']['factor_risk'] / fundamental_risk['factor_risk']:.2f}x")
    print(f"  Total Risk Increase: {fundamental_covid_crash['stressed_risk_metrics']['total_risk'] / fundamental_risk['total_risk']:.2f}x")
    
    print("\nMacroeconomic Model - Financial Crisis Scenario:")
    print(f"  Portfolio Return: {macroeconomic_financial_crisis['portfolio_return']:.4f}")
    print(f"  Factor Risk Increase: {macroeconomic_financial_crisis['stressed_risk_metrics']['factor_risk'] / macroeconomic_risk['factor_risk']:.2f}x")
    print(f"  Total Risk Increase: {macroeconomic_financial_crisis['stressed_risk_metrics']['total_risk'] / macroeconomic_risk['total_risk']:.2f}x")
    
    print("\nMacroeconomic Model - COVID Crash Scenario:")
    print(f"  Portfolio Return: {macroeconomic_covid_crash['portfolio_return']:.4f}")
    print(f"  Factor Risk Increase: {macroeconomic_covid_crash['stressed_risk_metrics']['factor_risk'] / macroeconomic_risk['factor_risk']:.2f}x")
    print(f"  Total Risk Increase: {macroeconomic_covid_crash['stressed_risk_metrics']['total_risk'] / macroeconomic_risk['total_risk']:.2f}x")


def demonstrate_factor_risk_model_adapter():
    """Demonstrate the usage of the FactorRiskModelAdapter class."""
    print("\n=== Demonstrating Factor Risk Model Adapter ===\n")
    
    # Create a sample portfolio
    portfolio = create_sample_portfolio()
    
    # Create instances of different factor risk model adapters
    fundamental_adapter = FactorRiskModelAdapter(model_type="fundamental")
    statistical_adapter = FactorRiskModelAdapter(model_type="statistical")
    macroeconomic_adapter = FactorRiskModelAdapter(model_type="macroeconomic")
    
    print("\n--- Risk Metrics from Adapters ---")
    
    # Get risk metrics from each adapter
    fundamental_metrics = fundamental_adapter.get_risk_metrics(portfolio)
    statistical_metrics = statistical_adapter.get_risk_metrics(portfolio)
    macroeconomic_metrics = macroeconomic_adapter.get_risk_metrics(portfolio)
    
    # Print risk metrics for each adapter
    print("\nFundamental Adapter Risk Metrics:")
    print(f"  VaR (95%): {fundamental_metrics['var_95']:.4f}")
    print(f"  Expected Shortfall (95%): {fundamental_metrics['expected_shortfall_95']:.4f}")
    print(f"  Volatility: {fundamental_metrics['volatility']:.4f}")
    print(f"  Factor Risk: {fundamental_metrics['factor_risk']:.4f}")
    print(f"  Specific Risk: {fundamental_metrics['specific_risk']:.4f}")
    print(f"  Total Risk: {fundamental_metrics['total_risk']:.4f}")
    
    print("\nStatistical Adapter Risk Metrics:")
    print(f"  VaR (95%): {statistical_metrics['var_95']:.4f}")
    print(f"  Expected Shortfall (95%): {statistical_metrics['expected_shortfall_95']:.4f}")
    print(f"  Volatility: {statistical_metrics['volatility']:.4f}")
    print(f"  Factor Risk: {statistical_metrics['factor_risk']:.4f}")
    print(f"  Specific Risk: {statistical_metrics['specific_risk']:.4f}")
    print(f"  Total Risk: {statistical_metrics['total_risk']:.4f}")
    
    print("\nMacroeconomic Adapter Risk Metrics:")
    print(f"  VaR (95%): {macroeconomic_metrics['var_95']:.4f}")
    print(f"  Expected Shortfall (95%): {macroeconomic_metrics['expected_shortfall_95']:.4f}")
    print(f"  Volatility: {macroeconomic_metrics['volatility']:.4f}")
    print(f"  Factor Risk: {macroeconomic_metrics['factor_risk']:.4f}")
    print(f"  Specific Risk: {macroeconomic_metrics['specific_risk']:.4f}")
    print(f"  Total Risk: {macroeconomic_metrics['total_risk']:.4f}")
    
    print("\n--- Factor Exposures from Adapters ---")
    
    # Get factor exposures from each adapter
    fundamental_exposures = fundamental_adapter.get_factor_exposures(portfolio)
    statistical_exposures = statistical_adapter.get_factor_exposures(portfolio)
    macroeconomic_exposures = macroeconomic_adapter.get_factor_exposures(portfolio)
    
    # Print top 5 factor exposures for each adapter
    print("\nTop 5 Fundamental Adapter Factor Exposures:")
    for factor, exposure in sorted(fundamental_exposures.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {factor}: {exposure:.4f}")
    
    print("\nTop 5 Statistical Adapter Factor Exposures:")
    for factor, exposure in sorted(statistical_exposures.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {factor}: {exposure:.4f}")
    
    print("\nTop 5 Macroeconomic Adapter Factor Exposures:")
    for factor, exposure in sorted(macroeconomic_exposures.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {factor}: {exposure:.4f}")
    
    print("\n--- Stress Test Results from Adapters ---")
    
    # Get stress test results from each adapter
    fundamental_financial_crisis = fundamental_adapter.get_stress_test_results(
        portfolio, "2008_financial_crisis")
    fundamental_covid_crash = fundamental_adapter.get_stress_test_results(
        portfolio, "2020_covid_crash")
    
    macroeconomic_financial_crisis = macroeconomic_adapter.get_stress_test_results(
        portfolio, "2008_financial_crisis")
    macroeconomic_covid_crash = macroeconomic_adapter.get_stress_test_results(
        portfolio, "2020_covid_crash")
    
    # Print stress test results for each adapter
    print("\nFundamental Adapter - Financial Crisis Scenario:")
    print(f"  Portfolio Return: {fundamental_financial_crisis['portfolio_return']:.4f}")
    print(f"  VaR Increase: {fundamental_financial_crisis['var_increase']:.2f}x")
    print(f"  Risk Increase: {fundamental_financial_crisis['risk_increase']:.2f}x")
    
    print("\nFundamental Adapter - COVID Crash Scenario:")
    print(f"  Portfolio Return: {fundamental_covid_crash['portfolio_return']:.4f}")
    print(f"  VaR Increase: {fundamental_covid_crash['var_increase']:.2f}x")
    print(f"  Risk Increase: {fundamental_covid_crash['risk_increase']:.2f}x")
    
    print("\nMacroeconomic Adapter - Financial Crisis Scenario:")
    print(f"  Portfolio Return: {macroeconomic_financial_crisis['portfolio_return']:.4f}")
    print(f"  VaR Increase: {macroeconomic_financial_crisis['var_increase']:.2f}x")
    print(f"  Risk Increase: {macroeconomic_financial_crisis['risk_increase']:.2f}x")
    
    print("\nMacroeconomic Adapter - COVID Crash Scenario:")
    print(f"  Portfolio Return: {macroeconomic_covid_crash['portfolio_return']:.4f}")
    print(f"  VaR Increase: {macroeconomic_covid_crash['var_increase']:.2f}x")
    print(f"  Risk Increase: {macroeconomic_covid_crash['risk_increase']:.2f}x")
    
    print("\n--- Correlation Matrix from Adapters ---")
    
    # Get correlation matrix from each adapter
    fundamental_corr = fundamental_adapter.get_correlation_matrix(portfolio)
    
    # Print correlation matrix dimensions
    print(f"\nFundamental Adapter Correlation Matrix Shape: {fundamental_corr.shape}")
    print(f"Number of Assets: {len(portfolio['positions'])}")
    
    # Print a small subset of the correlation matrix
    print("\nCorrelation Matrix Subset (first 5x5):")
    print(fundamental_corr[:5, :5])
    
    print("\n--- VaR Contribution from Adapters ---")
    
    # Get VaR contribution from each adapter
    fundamental_var_contrib = fundamental_adapter.get_var_contribution(portfolio)
    
    # Print top 5 VaR contributors
    print("\nTop 5 VaR Contributors:")
    for ticker, contribution in sorted(fundamental_var_contrib.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {ticker}: {contribution:.6f}")


def demonstrate_risk_model_integrator():
    """Demonstrate the usage of the RiskModelIntegrator with factor risk models."""
    print("\n=== Demonstrating Risk Model Integrator with Factor Risk Models ===\n")
    
    # Create a sample portfolio
    portfolio = create_sample_portfolio()
    
    # Create instances of different factor risk model adapters
    fundamental_adapter = FactorRiskModelAdapter(model_type="fundamental")
    macroeconomic_adapter = FactorRiskModelAdapter(model_type="macroeconomic")
    
    # Create a risk model integrator
    integrator = RiskModelIntegrator()
    
    # Register the factor risk model adapters
    integrator.register_model("fundamental", fundamental_adapter)
    integrator.register_model("macro", macroeconomic_adapter)
    
    # Activate the models
    integrator.activate_model("fundamental")
    integrator.activate_model("macro")
    
    print("\n--- Combined Risk Metrics ---")
    
    # Get combined risk metrics
    combined_metrics = integrator.get_combined_risk_metrics(portfolio)
    
    # Print combined risk metrics
    print("\nCombined Risk Metrics:")
    for metric, value in combined_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"  {metric}: {len(value)} items")
        else:
            print(f"  {metric}: {value}")
    
    print("\n--- Combined Factor Exposures ---")
    
    # Get combined factor exposures
    combined_exposures = integrator.get_combined_factor_exposures(portfolio)
    
    # Print number of factors from each model
    print(f"\nNumber of Combined Factors: {len(combined_exposures)}")
    print(f"Number of Fundamental Factors: {len([f for f in combined_exposures if f.startswith('fundamental_')])}")
    print(f"Number of Macro Factors: {len([f for f in combined_exposures if f.startswith('macro_')])}")
    
    # Print top 5 combined factor exposures
    print("\nTop 5 Combined Factor Exposures:")
    for factor, exposure in sorted(combined_exposures.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {factor}: {exposure:.4f}")
    
    print("\n--- Combined Stress Test Results ---")
    
    # Get combined stress test results
    combined_financial_crisis = integrator.get_combined_stress_test_results(
        portfolio, "2008_financial_crisis")
    combined_covid_crash = integrator.get_combined_stress_test_results(
        portfolio, "2020_covid_crash")
    
    # Print combined stress test results
    print("\nCombined - Financial Crisis Scenario:")
    for metric, value in combined_financial_crisis.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"  {metric}: {len(value)} items")
        else:
            print(f"  {metric}: {value}")
    
    print("\nCombined - COVID Crash Scenario:")
    for metric, value in combined_covid_crash.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"  {metric}: {len(value)} items")
        else:
            print(f"  {metric}: {value}")


def plot_factor_exposures(model, portfolio, title):
    """Plot factor exposures for a given model and portfolio."""
    # Get factor exposures
    exposures = model.get_factor_exposures(portfolio)
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(list(exposures.items()), columns=['Factor', 'Exposure'])
    
    # Sort by absolute exposure
    df['AbsExposure'] = df['Exposure'].abs()
    df = df.sort_values('AbsExposure', ascending=False).head(10)
    
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df['Factor'], df['Exposure'])
    
    # Color bars based on exposure value
    for i, bar in enumerate(bars):
        if df['Exposure'].iloc[i] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.title(title)
    plt.xlabel('Exposure')
    plt.ylabel('Factor')
    plt.tight_layout()
    plt.show()


def plot_risk_contribution(model, portfolio, title):
    """Plot risk contribution for a given model and portfolio."""
    # Calculate factor risk
    risk_metrics = model.calculate_factor_risk(portfolio)
    
    # Get risk contribution
    risk_contrib = risk_metrics['risk_contribution']
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(list(risk_contrib.items()), columns=['Factor', 'Contribution'])
    
    # Sort by absolute contribution
    df['AbsContribution'] = df['Contribution'].abs()
    df = df.sort_values('AbsContribution', ascending=False).head(10)
    
    # Calculate percentage contribution
    total_contrib = df['Contribution'].abs().sum()
    df['Percentage'] = df['Contribution'].abs() / total_contrib * 100
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.pie(df['Percentage'], labels=df['Factor'], autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_stress_test_comparison(model, portfolio, title):
    """Plot stress test comparison for a given model and portfolio."""
    # Define stress test scenarios
    scenarios = {
        "2008 Financial Crisis": {
            "market": -0.40,
            "credit": 0.30,
            "liquidity": -0.35,
            "volatility": 0.50,
        },
        "2020 COVID Crash": {
            "market": -0.30,
            "credit": 0.25,
            "liquidity": -0.30,
            "volatility": 0.40,
        },
        "Rate Hike 100bps": {
            "interest_rate": 0.10,
            "financials": 0.05,
            "utilities": -0.08,
            "real_estate": -0.07,
        },
        "Inflation Shock": {
            "inflation": 0.15,
            "interest_rate": 0.08,
            "energy": 0.10,
            "materials": 0.07,
        },
    }
    
    # Run stress tests for each scenario
    results = {}
    for scenario_name, scenario_shocks in scenarios.items():
        results[scenario_name] = model.run_factor_stress_test(portfolio, scenario_shocks)
    
    # Extract portfolio returns for each scenario
    returns = {scenario: result['portfolio_return'] for scenario, result in results.items()}
    
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(returns.keys(), returns.values())
    
    # Color bars based on return value
    for i, bar in enumerate(bars):
        if list(returns.values())[i] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title(title)
    plt.ylabel('Portfolio Return')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def demonstrate_visualizations():
    """Demonstrate visualizations of factor risk model results."""
    print("\n=== Demonstrating Factor Risk Model Visualizations ===\n")
    
    # Create a sample portfolio
    portfolio = create_sample_portfolio()
    
    # Create instances of different factor risk models
    fundamental_model = FactorRiskModel(model_type="fundamental")
    macroeconomic_model = FactorRiskModel(model_type="macroeconomic")
    
    # Plot factor exposures
    plot_factor_exposures(fundamental_model, portfolio, "Top 10 Fundamental Factor Exposures")
    plot_factor_exposures(macroeconomic_model, portfolio, "Top 10 Macroeconomic Factor Exposures")
    
    # Plot risk contribution
    plot_risk_contribution(fundamental_model, portfolio, "Fundamental Factor Risk Contribution")
    plot_risk_contribution(macroeconomic_model, portfolio, "Macroeconomic Factor Risk Contribution")
    
    # Plot stress test comparison
    plot_stress_test_comparison(fundamental_model, portfolio, "Fundamental Model Stress Test Comparison")
    plot_stress_test_comparison(macroeconomic_model, portfolio, "Macroeconomic Model Stress Test Comparison")


if __name__ == "__main__":
    # Demonstrate the usage of the FactorRiskModel class
    demonstrate_factor_risk_model()
    
    # Demonstrate the usage of the FactorRiskModelAdapter class
    demonstrate_factor_risk_model_adapter()
    
    # Demonstrate the usage of the RiskModelIntegrator with factor risk models
    demonstrate_risk_model_integrator()
    
    # Demonstrate visualizations (uncomment to run)
    # demonstrate_visualizations()