"""Factor-Based Risk Model for Friday AI Trading System.

This module provides a factor-based risk model implementation for the Friday AI Trading System.
It allows for the calculation of factor exposures, factor returns, and factor-based risk metrics.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from abc import ABC, abstractmethod

# Import the abstract base class
from .external_risk_model import ExternalRiskModel


class FactorRiskModel:
    """Factor-based risk model for portfolio risk analysis.
    
    This class implements a factor-based risk model that can be used to analyze
    portfolio risk based on factor exposures and factor covariances. It supports
    multiple factor model types (e.g., fundamental, statistical, macroeconomic)
    and provides methods for calculating factor exposures, factor returns, and
    factor-based risk metrics.
    """
    
    def __init__(self, model_type: str = "fundamental", use_cache: bool = True, 
                 cache_expiry: int = 3600):
        """Initialize the Factor Risk Model.
        
        Args:
            model_type: Type of factor model to use. Options are "fundamental",
                       "statistical", or "macroeconomic".
            use_cache: Whether to cache results to reduce computation time.
            cache_expiry: Cache expiry time in seconds.
        """
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type.lower()
        self.use_cache = use_cache
        self.cache_expiry = cache_expiry
        self.cache = {}
        self.cache_timestamps = {}
        
        # Initialize factor definitions based on model type
        self.factors = self._initialize_factors()
        
        # Initialize factor covariance matrix
        self.factor_covariance = self._initialize_factor_covariance()
        
        self.logger.info(f"Factor Risk Model initialized with model type: {model_type}")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if the cached data for the given key is still valid.
        
        Args:
            cache_key: The cache key to check.
            
        Returns:
            True if the cache is valid, False otherwise.
        """
        if not self.use_cache or cache_key not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[cache_key]
        current_time = datetime.now().timestamp()
        return (current_time - cache_time) < self.cache_expiry
    
    def _update_cache(self, cache_key: str, data: Any) -> None:
        """Update the cache with new data.
        
        Args:
            cache_key: The cache key to update.
            data: The data to cache.
        """
        if self.use_cache:
            self.cache[cache_key] = data
            self.cache_timestamps[cache_key] = datetime.now().timestamp()
    
    def _initialize_factors(self) -> Dict[str, Dict[str, Any]]:
        """Initialize factor definitions based on the model type.
        
        Returns:
            Dictionary of factor definitions.
        """
        if self.model_type == "fundamental":
            return {
                # Style factors
                "value": {"description": "Value factor based on P/E, P/B, etc.", "category": "style"},
                "size": {"description": "Size factor based on market capitalization", "category": "style"},
                "momentum": {"description": "Momentum factor based on price trends", "category": "style"},
                "volatility": {"description": "Volatility factor based on price volatility", "category": "style"},
                "quality": {"description": "Quality factor based on profitability, etc.", "category": "style"},
                "yield": {"description": "Yield factor based on dividend yield", "category": "style"},
                "growth": {"description": "Growth factor based on earnings growth", "category": "style"},
                "liquidity": {"description": "Liquidity factor based on trading volume", "category": "style"},
                
                # Industry factors
                "technology": {"description": "Technology sector exposure", "category": "industry"},
                "healthcare": {"description": "Healthcare sector exposure", "category": "industry"},
                "financials": {"description": "Financials sector exposure", "category": "industry"},
                "consumer_discretionary": {"description": "Consumer Discretionary sector exposure", "category": "industry"},
                "industrials": {"description": "Industrials sector exposure", "category": "industry"},
                "communication_services": {"description": "Communication Services sector exposure", "category": "industry"},
                "consumer_staples": {"description": "Consumer Staples sector exposure", "category": "industry"},
                "energy": {"description": "Energy sector exposure", "category": "industry"},
                "utilities": {"description": "Utilities sector exposure", "category": "industry"},
                "materials": {"description": "Materials sector exposure", "category": "industry"},
                "real_estate": {"description": "Real Estate sector exposure", "category": "industry"},
                
                # Region factors
                "north_america": {"description": "North America exposure", "category": "region"},
                "europe": {"description": "Europe exposure", "category": "region"},
                "asia_pacific": {"description": "Asia Pacific exposure", "category": "region"},
                "emerging_markets": {"description": "Emerging Markets exposure", "category": "region"},
            }
        elif self.model_type == "statistical":
            # For statistical factors, we would typically derive these from PCA
            # or other statistical methods. For this implementation, we'll use
            # placeholder factors.
            return {
                "factor_1": {"description": "Statistical factor 1", "category": "statistical"},
                "factor_2": {"description": "Statistical factor 2", "category": "statistical"},
                "factor_3": {"description": "Statistical factor 3", "category": "statistical"},
                "factor_4": {"description": "Statistical factor 4", "category": "statistical"},
                "factor_5": {"description": "Statistical factor 5", "category": "statistical"},
                "factor_6": {"description": "Statistical factor 6", "category": "statistical"},
                "factor_7": {"description": "Statistical factor 7", "category": "statistical"},
                "factor_8": {"description": "Statistical factor 8", "category": "statistical"},
                "factor_9": {"description": "Statistical factor 9", "category": "statistical"},
                "factor_10": {"description": "Statistical factor 10", "category": "statistical"},
            }
        elif self.model_type == "macroeconomic":
            return {
                # Macro factors
                "market": {"description": "Market factor (beta)", "category": "macro"},
                "interest_rate": {"description": "Interest rate sensitivity", "category": "macro"},
                "credit": {"description": "Credit spread sensitivity", "category": "macro"},
                "inflation": {"description": "Inflation sensitivity", "category": "macro"},
                "oil_price": {"description": "Oil price sensitivity", "category": "macro"},
                "currency": {"description": "Currency sensitivity", "category": "macro"},
                "economic_growth": {"description": "Economic growth sensitivity", "category": "macro"},
                "monetary_policy": {"description": "Monetary policy sensitivity", "category": "macro"},
                "fiscal_policy": {"description": "Fiscal policy sensitivity", "category": "macro"},
                "geopolitical_risk": {"description": "Geopolitical risk sensitivity", "category": "macro"},
            }
        else:
            self.logger.warning(f"Unknown model type: {self.model_type}. Using fundamental factors.")
            # Default to fundamental factors
            return self._initialize_factors()
    
    def _initialize_factor_covariance(self) -> np.ndarray:
        """Initialize the factor covariance matrix.
        
        In a real implementation, this would be estimated from historical factor returns.
        For this implementation, we'll use a placeholder covariance matrix.
        
        Returns:
            Factor covariance matrix as a numpy array.
        """
        # Get the number of factors
        n_factors = len(self.factors)
        
        # Create a placeholder covariance matrix
        # In a real implementation, this would be estimated from historical factor returns
        np.random.seed(42)  # For reproducibility
        cov_matrix = np.random.randn(n_factors, n_factors)
        cov_matrix = np.dot(cov_matrix, cov_matrix.T) / 100.0  # Make it positive semi-definite
        
        # Set diagonal elements (variances) to reasonable values
        np.fill_diagonal(cov_matrix, np.random.uniform(0.01, 0.05, n_factors))
        
        return cov_matrix
    
    def get_factor_exposures(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate factor exposures for the given portfolio.
        
        Args:
            portfolio_data: Dictionary containing portfolio data including positions,
                           historical returns, and other relevant information.
                           
        Returns:
            Dictionary mapping factor names to exposure values.
        """
        cache_key = f"factor_exposures_{self.model_type}_{hash(str(portfolio_data))}"  # Simplified for demo
        
        if self._is_cache_valid(cache_key):
            self.logger.info(f"Using cached factor exposures for {self.model_type} model")
            return self.cache[cache_key]
        
        self.logger.info(f"Calculating factor exposures for {self.model_type} model")
        
        # In a real implementation, we would calculate factor exposures based on
        # portfolio holdings and factor loadings. For this implementation, we'll
        # use placeholder exposures.
        
        # Get the list of factor names
        factor_names = list(self.factors.keys())
        
        # Generate random exposures for demonstration purposes
        # In a real implementation, these would be calculated based on portfolio holdings
        np.random.seed(hash(str(portfolio_data)) % 2**32)  # Seed based on portfolio data
        exposures = {}
        
        for factor in factor_names:
            # Generate a random exposure between -0.5 and 0.5
            # In a real implementation, this would be calculated based on portfolio holdings
            exposure = np.random.uniform(-0.5, 0.5)
            exposures[factor] = exposure
        
        self._update_cache(cache_key, exposures)
        
        return exposures
    
    def get_factor_returns(self, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Get historical factor returns for the specified period.
        
        Args:
            start_date: Start date for the period.
            end_date: End date for the period.
            
        Returns:
            Dictionary mapping factor names to return values.
        """
        cache_key = f"factor_returns_{self.model_type}_{start_date.isoformat()}_{end_date.isoformat()}"
        
        if self._is_cache_valid(cache_key):
            self.logger.info(f"Using cached factor returns for {self.model_type} model")
            return self.cache[cache_key]
        
        self.logger.info(f"Calculating factor returns for {self.model_type} model")
        
        # In a real implementation, we would retrieve historical factor returns
        # from a database or calculate them based on historical data. For this
        # implementation, we'll use placeholder returns.
        
        # Get the list of factor names
        factor_names = list(self.factors.keys())
        
        # Generate random returns for demonstration purposes
        # In a real implementation, these would be retrieved from a database
        np.random.seed(hash(start_date.isoformat() + end_date.isoformat()) % 2**32)  # Seed based on dates
        returns = {}
        
        for factor in factor_names:
            # Generate a random return between -0.1 and 0.1
            # In a real implementation, this would be retrieved from a database
            factor_return = np.random.uniform(-0.1, 0.1)
            returns[factor] = factor_return
        
        self._update_cache(cache_key, returns)
        
        return returns
    
    def calculate_factor_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate factor-based risk metrics for the given portfolio.
        
        Args:
            portfolio_data: Dictionary containing portfolio data including positions,
                           historical returns, and other relevant information.
                           
        Returns:
            Dictionary containing factor-based risk metrics.
        """
        cache_key = f"factor_risk_{self.model_type}_{hash(str(portfolio_data))}"  # Simplified for demo
        
        if self._is_cache_valid(cache_key):
            self.logger.info(f"Using cached factor risk metrics for {self.model_type} model")
            return self.cache[cache_key]
        
        self.logger.info(f"Calculating factor risk metrics for {self.model_type} model")
        
        # Get factor exposures for the portfolio
        exposures = self.get_factor_exposures(portfolio_data)
        
        # Convert exposures to a numpy array in the same order as the covariance matrix
        factor_names = list(self.factors.keys())
        exposure_vector = np.array([exposures[factor] for factor in factor_names])
        
        # Calculate factor risk (systematic risk)
        factor_risk = np.sqrt(exposure_vector.T @ self.factor_covariance @ exposure_vector)
        
        # Calculate specific risk (idiosyncratic risk)
        # In a real implementation, this would be calculated based on residual returns
        # For this implementation, we'll use a placeholder value
        specific_risk = np.random.uniform(0.005, 0.015)  # Between 0.5% and 1.5%
        
        # Calculate total risk
        total_risk = np.sqrt(factor_risk**2 + specific_risk**2)
        
        # Calculate factor contribution to risk
        risk_contribution = {}
        for i, factor in enumerate(factor_names):
            # Calculate marginal contribution to risk
            mcr = exposure_vector[i] * (self.factor_covariance[i] @ exposure_vector) / factor_risk
            # Calculate contribution to risk
            cr = mcr * exposure_vector[i]
            risk_contribution[factor] = cr
        
        # Calculate factor VaR (Value at Risk)
        # Assuming normal distribution and 95% confidence level
        factor_var_95 = factor_risk * 1.645  # 1.645 is the z-score for 95% confidence
        
        # Calculate factor Expected Shortfall (Conditional VaR)
        # Assuming normal distribution and 95% confidence level
        factor_es_95 = factor_risk * 2.063  # 2.063 is the expected shortfall multiplier for 95% confidence
        
        # Prepare the results
        risk_metrics = {
            "factor_risk": factor_risk,
            "specific_risk": specific_risk,
            "total_risk": total_risk,
            "factor_var_95": factor_var_95,
            "factor_es_95": factor_es_95,
            "risk_contribution": risk_contribution,
            "factor_exposures": exposures,
            "model_type": self.model_type,
            "timestamp": datetime.now().isoformat(),
        }
        
        self._update_cache(cache_key, risk_metrics)
        
        return risk_metrics
    
    def run_factor_stress_test(self, portfolio_data: Dict[str, Any], 
                              scenario: Dict[str, float]) -> Dict[str, Any]:
        """Run a factor-based stress test for the given portfolio.
        
        Args:
            portfolio_data: Dictionary containing portfolio data.
            scenario: Dictionary mapping factor names to stress test shocks.
            
        Returns:
            Dictionary containing stress test results.
        """
        cache_key = f"stress_test_{self.model_type}_{hash(str(portfolio_data))}_{hash(str(scenario))}"
        
        if self._is_cache_valid(cache_key):
            self.logger.info(f"Using cached stress test results for {self.model_type} model")
            return self.cache[cache_key]
        
        self.logger.info(f"Running factor stress test for {self.model_type} model")
        
        # Get factor exposures for the portfolio
        exposures = self.get_factor_exposures(portfolio_data)
        
        # Calculate portfolio return under the stress scenario
        portfolio_return = 0.0
        for factor, shock in scenario.items():
            if factor in exposures:
                portfolio_return += exposures[factor] * shock
        
        # Calculate stressed risk metrics
        # In a real implementation, we would adjust the factor covariance matrix
        # based on the stress scenario. For this implementation, we'll use a
        # simple scaling approach.
        base_risk_metrics = self.calculate_factor_risk(portfolio_data)
        stress_multiplier = 1.5  # Assume risk increases by 50% under stress
        
        stressed_risk_metrics = {
            "factor_risk": base_risk_metrics["factor_risk"] * stress_multiplier,
            "specific_risk": base_risk_metrics["specific_risk"] * stress_multiplier,
            "total_risk": base_risk_metrics["total_risk"] * stress_multiplier,
            "factor_var_95": base_risk_metrics["factor_var_95"] * stress_multiplier,
            "factor_es_95": base_risk_metrics["factor_es_95"] * stress_multiplier,
        }
        
        # Prepare the results
        stress_test_results = {
            "portfolio_return": portfolio_return,
            "stressed_risk_metrics": stressed_risk_metrics,
            "scenario": scenario,
            "model_type": self.model_type,
            "timestamp": datetime.now().isoformat(),
        }
        
        self._update_cache(cache_key, stress_test_results)
        
        return stress_test_results


class FactorRiskModelAdapter(ExternalRiskModel):
    """Adapter for the FactorRiskModel to implement the ExternalRiskModel interface.
    
    This class adapts the FactorRiskModel to the ExternalRiskModel interface,
    allowing it to be used with the RiskModelIntegrator.
    """
    
    def __init__(self, model_type: str = "fundamental", use_cache: bool = True,
                 cache_expiry: int = 3600):
        """Initialize the FactorRiskModelAdapter.
        
        Args:
            model_type: Type of factor model to use. Options are "fundamental",
                       "statistical", or "macroeconomic".
            use_cache: Whether to cache results to reduce computation time.
            cache_expiry: Cache expiry time in seconds.
        """
        self.logger = logging.getLogger(__name__)
        self.model = FactorRiskModel(model_type, use_cache, cache_expiry)
        self.logger.info(f"FactorRiskModelAdapter initialized with model type: {model_type}")
    
    def get_risk_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get risk metrics from the factor risk model.
        
        Args:
            portfolio_data: Dictionary containing portfolio data.
            
        Returns:
            Dictionary containing risk metrics.
        """
        self.logger.info("Getting risk metrics from factor risk model")
        
        # Calculate factor-based risk metrics
        factor_risk_metrics = self.model.calculate_factor_risk(portfolio_data)
        
        # Convert to the format expected by the ExternalRiskModel interface
        risk_metrics = {
            "var_95": factor_risk_metrics["factor_var_95"],
            "expected_shortfall_95": factor_risk_metrics["factor_es_95"],
            "volatility": factor_risk_metrics["total_risk"],
            "factor_risk": factor_risk_metrics["factor_risk"],
            "specific_risk": factor_risk_metrics["specific_risk"],
            "total_risk": factor_risk_metrics["total_risk"],
            "model_type": factor_risk_metrics["model_type"],
            "model_timestamp": factor_risk_metrics["timestamp"],
        }
        
        return risk_metrics
    
    def get_factor_exposures(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """Get factor exposures from the factor risk model.
        
        Args:
            portfolio_data: Dictionary containing portfolio data.
            
        Returns:
            Dictionary mapping factor names to exposure values.
        """
        self.logger.info("Getting factor exposures from factor risk model")
        return self.model.get_factor_exposures(portfolio_data)
    
    def get_stress_test_results(self, portfolio_data: Dict[str, Any], 
                               scenario: str) -> Dict[str, float]:
        """Get stress test results from the factor risk model.
        
        Args:
            portfolio_data: Dictionary containing portfolio data.
            scenario: Name of the stress test scenario to run.
            
        Returns:
            Dictionary containing stress test results.
        """
        self.logger.info(f"Getting stress test results from factor risk model for scenario: {scenario}")
        
        # Define stress test scenarios
        scenarios = {
            "2008_financial_crisis": {
                "market": -0.40,  # 40% market decline
                "credit": 0.30,  # 30% credit spread widening
                "liquidity": -0.35,  # 35% liquidity decline
                "volatility": 0.50,  # 50% volatility increase
                "value": -0.25,  # 25% value factor decline
                "momentum": -0.15,  # 15% momentum factor decline
                "quality": 0.10,  # 10% quality factor increase
            },
            "2020_covid_crash": {
                "market": -0.30,  # 30% market decline
                "credit": 0.25,  # 25% credit spread widening
                "liquidity": -0.30,  # 30% liquidity decline
                "volatility": 0.40,  # 40% volatility increase
                "value": -0.20,  # 20% value factor decline
                "momentum": -0.10,  # 10% momentum factor decline
                "quality": 0.15,  # 15% quality factor increase
            },
            "rate_hike_100bps": {
                "interest_rate": 0.10,  # 10% interest rate sensitivity
                "financials": 0.05,  # 5% financials sector increase
                "utilities": -0.08,  # 8% utilities sector decline
                "real_estate": -0.07,  # 7% real estate sector decline
                "yield": -0.05,  # 5% yield factor decline
            },
            "inflation_shock": {
                "inflation": 0.15,  # 15% inflation sensitivity
                "interest_rate": 0.08,  # 8% interest rate sensitivity
                "energy": 0.10,  # 10% energy sector increase
                "materials": 0.07,  # 7% materials sector increase
                "consumer_staples": -0.05,  # 5% consumer staples sector decline
                "technology": -0.08,  # 8% technology sector decline
            },
        }
        
        # Check if the scenario exists
        if scenario not in scenarios:
            self.logger.warning(f"Unknown scenario: {scenario}. Using default scenario.")
            # Create a default scenario with small shocks to all factors
            default_scenario = {}
            for factor in self.model.factors.keys():
                default_scenario[factor] = np.random.uniform(-0.1, 0.1)
            scenario_shocks = default_scenario
        else:
            scenario_shocks = scenarios[scenario]
        
        # Run the stress test
        stress_test_results = self.model.run_factor_stress_test(portfolio_data, scenario_shocks)
        
        # Convert to the format expected by the ExternalRiskModel interface
        results = {
            "portfolio_return": stress_test_results["portfolio_return"],
            "var_increase": stress_test_results["stressed_risk_metrics"]["factor_var_95"] / 
                           self.model.calculate_factor_risk(portfolio_data)["factor_var_95"],
            "risk_increase": stress_test_results["stressed_risk_metrics"]["total_risk"] / 
                            self.model.calculate_factor_risk(portfolio_data)["total_risk"],
            "scenario": scenario,
            "model_type": stress_test_results["model_type"],
            "timestamp": stress_test_results["timestamp"],
        }
        
        return results
    
    def get_correlation_matrix(self, portfolio_data: Dict[str, Any]) -> np.ndarray:
        """Get the correlation matrix for the portfolio assets.
        
        Args:
            portfolio_data: Dictionary containing portfolio data.
            
        Returns:
            Correlation matrix as a numpy array.
        """
        self.logger.info("Getting correlation matrix from factor risk model")
        
        # In a real implementation, we would calculate the correlation matrix
        # based on the factor model. For this implementation, we'll use a
        # placeholder correlation matrix.
        
        # Get the number of assets in the portfolio
        if "positions" in portfolio_data:
            n_assets = len(portfolio_data["positions"])
        else:
            n_assets = 10  # Default to 10 assets if positions are not provided
        
        # Create a placeholder correlation matrix
        # In a real implementation, this would be calculated based on the factor model
        np.random.seed(hash(str(portfolio_data)) % 2**32)  # Seed based on portfolio data
        corr_matrix = np.random.rand(n_assets, n_assets)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make it symmetric
        np.fill_diagonal(corr_matrix, 1.0)  # Set diagonal to 1.0
        
        # Ensure it's a valid correlation matrix (positive semi-definite)
        # This is a simple approach and may not work for all cases
        min_eig = np.min(np.linalg.eigvals(corr_matrix))
        if min_eig < 0:
            corr_matrix -= 1.1 * min_eig * np.eye(n_assets)
        
        return corr_matrix
    
    def get_var_contribution(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """Get the VaR contribution for each asset in the portfolio.
        
        Args:
            portfolio_data: Dictionary containing portfolio data.
            
        Returns:
            Dictionary mapping asset names to VaR contribution values.
        """
        self.logger.info("Getting VaR contribution from factor risk model")
        
        # In a real implementation, we would calculate the VaR contribution
        # based on the factor model. For this implementation, we'll use
        # placeholder values.
        
        # Get the assets in the portfolio
        if "positions" in portfolio_data:
            assets = [position["ticker"] for position in portfolio_data["positions"]]
        else:
            assets = [f"Asset_{i}" for i in range(10)]  # Default to 10 assets if positions are not provided
        
        # Calculate factor risk metrics
        factor_risk_metrics = self.model.calculate_factor_risk(portfolio_data)
        
        # Get factor exposures
        exposures = self.model.get_factor_exposures(portfolio_data)
        
        # Calculate VaR contribution for each asset
        # In a real implementation, this would be calculated based on the factor model
        # For this implementation, we'll use placeholder values
        var_contribution = {}
        total_var = factor_risk_metrics["factor_var_95"]
        
        for asset in assets:
            # Generate a random contribution between 0 and 1
            # In a real implementation, this would be calculated based on the factor model
            contribution = np.random.uniform(0, 1)
            var_contribution[asset] = contribution
        
        # Normalize the contributions to sum to the total VaR
        total_contribution = sum(var_contribution.values())
        for asset in var_contribution:
            var_contribution[asset] = var_contribution[asset] / total_contribution * total_var
        
        return var_contribution