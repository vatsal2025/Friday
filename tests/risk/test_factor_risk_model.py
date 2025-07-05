"""Tests for the Factor-Based Risk Model.

This module contains tests for the FactorRiskModel and FactorRiskModelAdapter classes.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
from src.risk.factor_risk_model import FactorRiskModel, FactorRiskModelAdapter


class TestFactorRiskModel(unittest.TestCase):
    """Test cases for the FactorRiskModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample portfolio for testing
        self.sample_portfolio = {
            "positions": [
                {"ticker": "AAPL", "weight": 0.15, "sector": "Technology"},
                {"ticker": "MSFT", "weight": 0.12, "sector": "Technology"},
                {"ticker": "AMZN", "weight": 0.10, "sector": "Consumer Discretionary"},
                {"ticker": "GOOGL", "weight": 0.08, "sector": "Communication Services"},
                {"ticker": "FB", "weight": 0.07, "sector": "Communication Services"},
                {"ticker": "BRK.B", "weight": 0.06, "sector": "Financials"},
                {"ticker": "JNJ", "weight": 0.05, "sector": "Healthcare"},
                {"ticker": "JPM", "weight": 0.05, "sector": "Financials"},
                {"ticker": "V", "weight": 0.04, "sector": "Financials"},
                {"ticker": "PG", "weight": 0.04, "sector": "Consumer Staples"},
                {"ticker": "UNH", "weight": 0.04, "sector": "Healthcare"},
                {"ticker": "HD", "weight": 0.03, "sector": "Consumer Discretionary"},
                {"ticker": "MA", "weight": 0.03, "sector": "Financials"},
                {"ticker": "INTC", "weight": 0.03, "sector": "Technology"},
                {"ticker": "VZ", "weight": 0.03, "sector": "Communication Services"},
                {"ticker": "T", "weight": 0.02, "sector": "Communication Services"},
                {"ticker": "PFE", "weight": 0.02, "sector": "Healthcare"},
                {"ticker": "CSCO", "weight": 0.02, "sector": "Technology"},
                {"ticker": "KO", "weight": 0.01, "sector": "Consumer Staples"},
                {"ticker": "PEP", "weight": 0.01, "sector": "Consumer Staples"},
            ],
            "total_value": 1000000.0,
            "currency": "USD",
            "as_of_date": datetime.now().strftime("%Y-%m-%d"),
        }
        
        # Create instances of the models for testing
        self.fundamental_model = FactorRiskModel(model_type="fundamental")
        self.statistical_model = FactorRiskModel(model_type="statistical")
        self.macroeconomic_model = FactorRiskModel(model_type="macroeconomic")
        
        # Create instances of the adapters for testing
        self.fundamental_adapter = FactorRiskModelAdapter(model_type="fundamental")
        self.statistical_adapter = FactorRiskModelAdapter(model_type="statistical")
        self.macroeconomic_adapter = FactorRiskModelAdapter(model_type="macroeconomic")
    
    def test_initialization(self):
        """Test that the models initialize correctly."""
        # Test fundamental model initialization
        self.assertEqual(self.fundamental_model.model_type, "fundamental")
        self.assertTrue(self.fundamental_model.use_cache)
        self.assertEqual(self.fundamental_model.cache_expiry, 3600)
        
        # Test statistical model initialization
        self.assertEqual(self.statistical_model.model_type, "statistical")
        self.assertTrue(self.statistical_model.use_cache)
        self.assertEqual(self.statistical_model.cache_expiry, 3600)
        
        # Test macroeconomic model initialization
        self.assertEqual(self.macroeconomic_model.model_type, "macroeconomic")
        self.assertTrue(self.macroeconomic_model.use_cache)
        self.assertEqual(self.macroeconomic_model.cache_expiry, 3600)
    
    def test_factor_initialization(self):
        """Test that the factors are initialized correctly."""
        # Test fundamental model factors
        self.assertIn("value", self.fundamental_model.factors)
        self.assertIn("size", self.fundamental_model.factors)
        self.assertIn("momentum", self.fundamental_model.factors)
        self.assertIn("technology", self.fundamental_model.factors)
        self.assertIn("healthcare", self.fundamental_model.factors)
        self.assertIn("north_america", self.fundamental_model.factors)
        
        # Test statistical model factors
        self.assertIn("factor_1", self.statistical_model.factors)
        self.assertIn("factor_2", self.statistical_model.factors)
        self.assertIn("factor_3", self.statistical_model.factors)
        
        # Test macroeconomic model factors
        self.assertIn("market", self.macroeconomic_model.factors)
        self.assertIn("interest_rate", self.macroeconomic_model.factors)
        self.assertIn("credit", self.macroeconomic_model.factors)
    
    def test_factor_covariance_initialization(self):
        """Test that the factor covariance matrix is initialized correctly."""
        # Test fundamental model factor covariance
        self.assertIsInstance(self.fundamental_model.factor_covariance, np.ndarray)
        self.assertEqual(self.fundamental_model.factor_covariance.shape, 
                         (len(self.fundamental_model.factors), len(self.fundamental_model.factors)))
        
        # Test statistical model factor covariance
        self.assertIsInstance(self.statistical_model.factor_covariance, np.ndarray)
        self.assertEqual(self.statistical_model.factor_covariance.shape, 
                         (len(self.statistical_model.factors), len(self.statistical_model.factors)))
        
        # Test macroeconomic model factor covariance
        self.assertIsInstance(self.macroeconomic_model.factor_covariance, np.ndarray)
        self.assertEqual(self.macroeconomic_model.factor_covariance.shape, 
                         (len(self.macroeconomic_model.factors), len(self.macroeconomic_model.factors)))
    
    def test_get_factor_exposures(self):
        """Test that the get_factor_exposures method returns the expected format."""
        # Test fundamental model factor exposures
        fundamental_exposures = self.fundamental_model.get_factor_exposures(self.sample_portfolio)
        self.assertIsInstance(fundamental_exposures, dict)
        self.assertEqual(len(fundamental_exposures), len(self.fundamental_model.factors))
        for factor, exposure in fundamental_exposures.items():
            self.assertIn(factor, self.fundamental_model.factors)
            self.assertIsInstance(exposure, float)
            self.assertTrue(-0.5 <= exposure <= 0.5)  # Check range based on implementation
        
        # Test statistical model factor exposures
        statistical_exposures = self.statistical_model.get_factor_exposures(self.sample_portfolio)
        self.assertIsInstance(statistical_exposures, dict)
        self.assertEqual(len(statistical_exposures), len(self.statistical_model.factors))
        for factor, exposure in statistical_exposures.items():
            self.assertIn(factor, self.statistical_model.factors)
            self.assertIsInstance(exposure, float)
            self.assertTrue(-0.5 <= exposure <= 0.5)  # Check range based on implementation
        
        # Test macroeconomic model factor exposures
        macroeconomic_exposures = self.macroeconomic_model.get_factor_exposures(self.sample_portfolio)
        self.assertIsInstance(macroeconomic_exposures, dict)
        self.assertEqual(len(macroeconomic_exposures), len(self.macroeconomic_model.factors))
        for factor, exposure in macroeconomic_exposures.items():
            self.assertIn(factor, self.macroeconomic_model.factors)
            self.assertIsInstance(exposure, float)
            self.assertTrue(-0.5 <= exposure <= 0.5)  # Check range based on implementation
    
    def test_get_factor_returns(self):
        """Test that the get_factor_returns method returns the expected format."""
        # Define date range for testing
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Test fundamental model factor returns
        fundamental_returns = self.fundamental_model.get_factor_returns(start_date, end_date)
        self.assertIsInstance(fundamental_returns, dict)
        self.assertEqual(len(fundamental_returns), len(self.fundamental_model.factors))
        for factor, factor_return in fundamental_returns.items():
            self.assertIn(factor, self.fundamental_model.factors)
            self.assertIsInstance(factor_return, float)
            self.assertTrue(-0.1 <= factor_return <= 0.1)  # Check range based on implementation
        
        # Test statistical model factor returns
        statistical_returns = self.statistical_model.get_factor_returns(start_date, end_date)
        self.assertIsInstance(statistical_returns, dict)
        self.assertEqual(len(statistical_returns), len(self.statistical_model.factors))
        for factor, factor_return in statistical_returns.items():
            self.assertIn(factor, self.statistical_model.factors)
            self.assertIsInstance(factor_return, float)
            self.assertTrue(-0.1 <= factor_return <= 0.1)  # Check range based on implementation
        
        # Test macroeconomic model factor returns
        macroeconomic_returns = self.macroeconomic_model.get_factor_returns(start_date, end_date)
        self.assertIsInstance(macroeconomic_returns, dict)
        self.assertEqual(len(macroeconomic_returns), len(self.macroeconomic_model.factors))
        for factor, factor_return in macroeconomic_returns.items():
            self.assertIn(factor, self.macroeconomic_model.factors)
            self.assertIsInstance(factor_return, float)
            self.assertTrue(-0.1 <= factor_return <= 0.1)  # Check range based on implementation
    
    def test_calculate_factor_risk(self):
        """Test that the calculate_factor_risk method returns the expected format."""
        # Test fundamental model factor risk
        fundamental_risk = self.fundamental_model.calculate_factor_risk(self.sample_portfolio)
        self.assertIsInstance(fundamental_risk, dict)
        self.assertIn("factor_risk", fundamental_risk)
        self.assertIn("specific_risk", fundamental_risk)
        self.assertIn("total_risk", fundamental_risk)
        self.assertIn("factor_var_95", fundamental_risk)
        self.assertIn("factor_es_95", fundamental_risk)
        self.assertIn("risk_contribution", fundamental_risk)
        self.assertIn("factor_exposures", fundamental_risk)
        self.assertIn("model_type", fundamental_risk)
        self.assertIn("timestamp", fundamental_risk)
        
        # Test statistical model factor risk
        statistical_risk = self.statistical_model.calculate_factor_risk(self.sample_portfolio)
        self.assertIsInstance(statistical_risk, dict)
        self.assertIn("factor_risk", statistical_risk)
        self.assertIn("specific_risk", statistical_risk)
        self.assertIn("total_risk", statistical_risk)
        self.assertIn("factor_var_95", statistical_risk)
        self.assertIn("factor_es_95", statistical_risk)
        self.assertIn("risk_contribution", statistical_risk)
        self.assertIn("factor_exposures", statistical_risk)
        self.assertIn("model_type", statistical_risk)
        self.assertIn("timestamp", statistical_risk)
        
        # Test macroeconomic model factor risk
        macroeconomic_risk = self.macroeconomic_model.calculate_factor_risk(self.sample_portfolio)
        self.assertIsInstance(macroeconomic_risk, dict)
        self.assertIn("factor_risk", macroeconomic_risk)
        self.assertIn("specific_risk", macroeconomic_risk)
        self.assertIn("total_risk", macroeconomic_risk)
        self.assertIn("factor_var_95", macroeconomic_risk)
        self.assertIn("factor_es_95", macroeconomic_risk)
        self.assertIn("risk_contribution", macroeconomic_risk)
        self.assertIn("factor_exposures", macroeconomic_risk)
        self.assertIn("model_type", macroeconomic_risk)
        self.assertIn("timestamp", macroeconomic_risk)
    
    def test_run_factor_stress_test(self):
        """Test that the run_factor_stress_test method returns the expected format."""
        # Define a sample stress scenario
        scenario = {
            "market": -0.20,  # 20% market decline
            "credit": 0.15,  # 15% credit spread widening
            "liquidity": -0.10,  # 10% liquidity decline
            "volatility": 0.25,  # 25% volatility increase
        }
        
        # Test fundamental model stress test
        fundamental_stress = self.fundamental_model.run_factor_stress_test(self.sample_portfolio, scenario)
        self.assertIsInstance(fundamental_stress, dict)
        self.assertIn("portfolio_return", fundamental_stress)
        self.assertIn("stressed_risk_metrics", fundamental_stress)
        self.assertIn("scenario", fundamental_stress)
        self.assertIn("model_type", fundamental_stress)
        self.assertIn("timestamp", fundamental_stress)
        
        # Test statistical model stress test
        statistical_stress = self.statistical_model.run_factor_stress_test(self.sample_portfolio, scenario)
        self.assertIsInstance(statistical_stress, dict)
        self.assertIn("portfolio_return", statistical_stress)
        self.assertIn("stressed_risk_metrics", statistical_stress)
        self.assertIn("scenario", statistical_stress)
        self.assertIn("model_type", statistical_stress)
        self.assertIn("timestamp", statistical_stress)
        
        # Test macroeconomic model stress test
        macroeconomic_stress = self.macroeconomic_model.run_factor_stress_test(self.sample_portfolio, scenario)
        self.assertIsInstance(macroeconomic_stress, dict)
        self.assertIn("portfolio_return", macroeconomic_stress)
        self.assertIn("stressed_risk_metrics", macroeconomic_stress)
        self.assertIn("scenario", macroeconomic_stress)
        self.assertIn("model_type", macroeconomic_stress)
        self.assertIn("timestamp", macroeconomic_stress)


class TestFactorRiskModelAdapter(unittest.TestCase):
    """Test cases for the FactorRiskModelAdapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample portfolio for testing
        self.sample_portfolio = {
            "positions": [
                {"ticker": "AAPL", "weight": 0.15, "sector": "Technology"},
                {"ticker": "MSFT", "weight": 0.12, "sector": "Technology"},
                {"ticker": "AMZN", "weight": 0.10, "sector": "Consumer Discretionary"},
                {"ticker": "GOOGL", "weight": 0.08, "sector": "Communication Services"},
                {"ticker": "FB", "weight": 0.07, "sector": "Communication Services"},
                {"ticker": "BRK.B", "weight": 0.06, "sector": "Financials"},
                {"ticker": "JNJ", "weight": 0.05, "sector": "Healthcare"},
                {"ticker": "JPM", "weight": 0.05, "sector": "Financials"},
                {"ticker": "V", "weight": 0.04, "sector": "Financials"},
                {"ticker": "PG", "weight": 0.04, "sector": "Consumer Staples"},
            ],
            "total_value": 1000000.0,
            "currency": "USD",
            "as_of_date": datetime.now().strftime("%Y-%m-%d"),
        }
        
        # Create instances of the adapters for testing
        self.fundamental_adapter = FactorRiskModelAdapter(model_type="fundamental")
        self.statistical_adapter = FactorRiskModelAdapter(model_type="statistical")
        self.macroeconomic_adapter = FactorRiskModelAdapter(model_type="macroeconomic")
    
    def test_get_risk_metrics(self):
        """Test that the get_risk_metrics method returns the expected format."""
        # Test fundamental adapter risk metrics
        fundamental_metrics = self.fundamental_adapter.get_risk_metrics(self.sample_portfolio)
        self.assertIsInstance(fundamental_metrics, dict)
        self.assertIn("var_95", fundamental_metrics)
        self.assertIn("expected_shortfall_95", fundamental_metrics)
        self.assertIn("volatility", fundamental_metrics)
        self.assertIn("factor_risk", fundamental_metrics)
        self.assertIn("specific_risk", fundamental_metrics)
        self.assertIn("total_risk", fundamental_metrics)
        self.assertIn("model_type", fundamental_metrics)
        self.assertIn("model_timestamp", fundamental_metrics)
        
        # Test statistical adapter risk metrics
        statistical_metrics = self.statistical_adapter.get_risk_metrics(self.sample_portfolio)
        self.assertIsInstance(statistical_metrics, dict)
        self.assertIn("var_95", statistical_metrics)
        self.assertIn("expected_shortfall_95", statistical_metrics)
        self.assertIn("volatility", statistical_metrics)
        self.assertIn("factor_risk", statistical_metrics)
        self.assertIn("specific_risk", statistical_metrics)
        self.assertIn("total_risk", statistical_metrics)
        self.assertIn("model_type", statistical_metrics)
        self.assertIn("model_timestamp", statistical_metrics)
        
        # Test macroeconomic adapter risk metrics
        macroeconomic_metrics = self.macroeconomic_adapter.get_risk_metrics(self.sample_portfolio)
        self.assertIsInstance(macroeconomic_metrics, dict)
        self.assertIn("var_95", macroeconomic_metrics)
        self.assertIn("expected_shortfall_95", macroeconomic_metrics)
        self.assertIn("volatility", macroeconomic_metrics)
        self.assertIn("factor_risk", macroeconomic_metrics)
        self.assertIn("specific_risk", macroeconomic_metrics)
        self.assertIn("total_risk", macroeconomic_metrics)
        self.assertIn("model_type", macroeconomic_metrics)
        self.assertIn("model_timestamp", macroeconomic_metrics)
    
    def test_get_factor_exposures(self):
        """Test that the get_factor_exposures method returns the expected format."""
        # Test fundamental adapter factor exposures
        fundamental_exposures = self.fundamental_adapter.get_factor_exposures(self.sample_portfolio)
        self.assertIsInstance(fundamental_exposures, dict)
        self.assertTrue(len(fundamental_exposures) > 0)
        for factor, exposure in fundamental_exposures.items():
            self.assertIsInstance(exposure, float)
        
        # Test statistical adapter factor exposures
        statistical_exposures = self.statistical_adapter.get_factor_exposures(self.sample_portfolio)
        self.assertIsInstance(statistical_exposures, dict)
        self.assertTrue(len(statistical_exposures) > 0)
        for factor, exposure in statistical_exposures.items():
            self.assertIsInstance(exposure, float)
        
        # Test macroeconomic adapter factor exposures
        macroeconomic_exposures = self.macroeconomic_adapter.get_factor_exposures(self.sample_portfolio)
        self.assertIsInstance(macroeconomic_exposures, dict)
        self.assertTrue(len(macroeconomic_exposures) > 0)
        for factor, exposure in macroeconomic_exposures.items():
            self.assertIsInstance(exposure, float)
    
    def test_get_stress_test_results(self):
        """Test that the get_stress_test_results method returns the expected format."""
        # Test fundamental adapter stress test results
        fundamental_stress = self.fundamental_adapter.get_stress_test_results(
            self.sample_portfolio, "2008_financial_crisis")
        self.assertIsInstance(fundamental_stress, dict)
        self.assertIn("portfolio_return", fundamental_stress)
        self.assertIn("var_increase", fundamental_stress)
        self.assertIn("risk_increase", fundamental_stress)
        self.assertIn("scenario", fundamental_stress)
        self.assertIn("model_type", fundamental_stress)
        self.assertIn("timestamp", fundamental_stress)
        
        # Test statistical adapter stress test results
        statistical_stress = self.statistical_adapter.get_stress_test_results(
            self.sample_portfolio, "2020_covid_crash")
        self.assertIsInstance(statistical_stress, dict)
        self.assertIn("portfolio_return", statistical_stress)
        self.assertIn("var_increase", statistical_stress)
        self.assertIn("risk_increase", statistical_stress)
        self.assertIn("scenario", statistical_stress)
        self.assertIn("model_type", statistical_stress)
        self.assertIn("timestamp", statistical_stress)
        
        # Test macroeconomic adapter stress test results
        macroeconomic_stress = self.macroeconomic_adapter.get_stress_test_results(
            self.sample_portfolio, "rate_hike_100bps")
        self.assertIsInstance(macroeconomic_stress, dict)
        self.assertIn("portfolio_return", macroeconomic_stress)
        self.assertIn("var_increase", macroeconomic_stress)
        self.assertIn("risk_increase", macroeconomic_stress)
        self.assertIn("scenario", macroeconomic_stress)
        self.assertIn("model_type", macroeconomic_stress)
        self.assertIn("timestamp", macroeconomic_stress)
    
    def test_get_correlation_matrix(self):
        """Test that the get_correlation_matrix method returns the expected format."""
        # Test fundamental adapter correlation matrix
        fundamental_corr = self.fundamental_adapter.get_correlation_matrix(self.sample_portfolio)
        self.assertIsInstance(fundamental_corr, np.ndarray)
        self.assertEqual(fundamental_corr.shape, (len(self.sample_portfolio["positions"]), 
                                                len(self.sample_portfolio["positions"])))
        self.assertTrue(np.all(np.diag(fundamental_corr) == 1.0))  # Check diagonal is 1.0
        
        # Test statistical adapter correlation matrix
        statistical_corr = self.statistical_adapter.get_correlation_matrix(self.sample_portfolio)
        self.assertIsInstance(statistical_corr, np.ndarray)
        self.assertEqual(statistical_corr.shape, (len(self.sample_portfolio["positions"]), 
                                                len(self.sample_portfolio["positions"])))
        self.assertTrue(np.all(np.diag(statistical_corr) == 1.0))  # Check diagonal is 1.0
        
        # Test macroeconomic adapter correlation matrix
        macroeconomic_corr = self.macroeconomic_adapter.get_correlation_matrix(self.sample_portfolio)
        self.assertIsInstance(macroeconomic_corr, np.ndarray)
        self.assertEqual(macroeconomic_corr.shape, (len(self.sample_portfolio["positions"]), 
                                                  len(self.sample_portfolio["positions"])))
        self.assertTrue(np.all(np.diag(macroeconomic_corr) == 1.0))  # Check diagonal is 1.0
    
    def test_get_var_contribution(self):
        """Test that the get_var_contribution method returns the expected format."""
        # Test fundamental adapter VaR contribution
        fundamental_var = self.fundamental_adapter.get_var_contribution(self.sample_portfolio)
        self.assertIsInstance(fundamental_var, dict)
        self.assertEqual(len(fundamental_var), len(self.sample_portfolio["positions"]))
        for ticker, contribution in fundamental_var.items():
            self.assertIsInstance(contribution, float)
        
        # Test statistical adapter VaR contribution
        statistical_var = self.statistical_adapter.get_var_contribution(self.sample_portfolio)
        self.assertIsInstance(statistical_var, dict)
        self.assertEqual(len(statistical_var), len(self.sample_portfolio["positions"]))
        for ticker, contribution in statistical_var.items():
            self.assertIsInstance(contribution, float)
        
        # Test macroeconomic adapter VaR contribution
        macroeconomic_var = self.macroeconomic_adapter.get_var_contribution(self.sample_portfolio)
        self.assertIsInstance(macroeconomic_var, dict)
        self.assertEqual(len(macroeconomic_var), len(self.sample_portfolio["positions"]))
        for ticker, contribution in macroeconomic_var.items():
            self.assertIsInstance(contribution, float)


if __name__ == "__main__":
    unittest.main()