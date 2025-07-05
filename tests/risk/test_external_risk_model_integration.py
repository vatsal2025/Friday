"""Tests for external risk model integration in Friday AI Trading System.

This module contains tests for the external risk model integration,
including the RiskModelIntegrator and the various risk model adapters.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import the classes to test
from src.risk.external_risk_model import ExternalRiskModel
from src.risk.risk_model_integrator import RiskModelIntegrator
from src.risk.adapters.bloomberg_risk_adapter import BloombergRiskAdapter
from src.risk.adapters.msci_risk_adapter import MSCIRiskAdapter
from src.risk.adapters.factset_risk_adapter import FactSetRiskAdapter


class TestExternalRiskModelIntegration(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        # Create a sample portfolio for testing
        self.sample_portfolio = {
            "positions": [
                {"ticker": "AAPL", "quantity": 100, "price": 150.0},
                {"ticker": "MSFT", "quantity": 50, "price": 250.0},
                {"ticker": "GOOGL", "quantity": 25, "price": 2800.0}
            ],
            "cash": 10000.0
        }
        
        # Create the integrator
        self.integrator = RiskModelIntegrator()
        
        # Create mock risk model adapters
        self.bloomberg_adapter = BloombergRiskAdapter(api_key="mock_key")
        self.msci_adapter = MSCIRiskAdapter(api_key="mock_key")
        self.factset_adapter = FactSetRiskAdapter(api_key="mock_key")
    
    def test_register_and_activate_models(self):
        """Test registering and activating risk models."""
        # Register the models
        self.integrator.register_model("bloomberg", self.bloomberg_adapter)
        self.integrator.register_model("msci", self.msci_adapter)
        self.integrator.register_model("factset", self.factset_adapter)
        
        # Check that the models are registered
        self.assertIn("bloomberg", self.integrator.registered_models)
        self.assertIn("msci", self.integrator.registered_models)
        self.assertIn("factset", self.integrator.registered_models)
        
        # Activate the models
        self.integrator.activate_model("bloomberg")
        self.integrator.activate_model("msci")
        self.integrator.activate_model("factset")
        
        # Check that the models are active
        self.assertIn("bloomberg", self.integrator.active_models)
        self.assertIn("msci", self.integrator.active_models)
        self.assertIn("factset", self.integrator.active_models)
        
        # Deactivate a model
        self.integrator.deactivate_model("msci")
        
        # Check that the model is deactivated
        self.assertNotIn("msci", self.integrator.active_models)
        self.assertIn("bloomberg", self.integrator.active_models)
        self.assertIn("factset", self.integrator.active_models)
    
    def test_get_combined_risk_metrics(self):
        """Test getting combined risk metrics."""
        # Register and activate the models
        self.integrator.register_model("bloomberg", self.bloomberg_adapter)
        self.integrator.register_model("msci", self.msci_adapter)
        self.integrator.register_model("factset", self.factset_adapter)
        self.integrator.activate_model("bloomberg")
        self.integrator.activate_model("msci")
        self.integrator.activate_model("factset")
        
        # Get the combined risk metrics
        combined_metrics = self.integrator.get_combined_risk_metrics(self.sample_portfolio)
        
        # Check that the combined metrics have the expected structure
        self.assertIn("var_95", combined_metrics)
        self.assertIn("var_99", combined_metrics)
        self.assertIn("expected_shortfall", combined_metrics)
        self.assertIn("volatility", combined_metrics)
        self.assertIn("model_sources", combined_metrics)
        
        # Check that the model sources are included
        self.assertIn("bloomberg", combined_metrics["model_sources"])
        self.assertIn("msci", combined_metrics["model_sources"])
        self.assertIn("factset", combined_metrics["model_sources"])
    
    def test_get_factor_exposures(self):
        """Test getting factor exposures."""
        # Register and activate the models
        self.integrator.register_model("bloomberg", self.bloomberg_adapter)
        self.integrator.register_model("msci", self.msci_adapter)
        self.integrator.register_model("factset", self.factset_adapter)
        self.integrator.activate_model("bloomberg")
        self.integrator.activate_model("msci")
        self.integrator.activate_model("factset")
        
        # Get the combined factor exposures
        combined_exposures = self.integrator.get_factor_exposures(self.sample_portfolio)
        
        # Check that the combined exposures have the expected structure
        self.assertIn("factors", combined_exposures)
        self.assertIn("exposures", combined_exposures)
        self.assertIn("model_sources", combined_exposures)
        
        # Check that the model sources are included
        self.assertIn("bloomberg", combined_exposures["model_sources"])
        self.assertIn("msci", combined_exposures["model_sources"])
        self.assertIn("factset", combined_exposures["model_sources"])
    
    def test_get_stress_test_results(self):
        """Test getting stress test results."""
        # Register and activate the models
        self.integrator.register_model("bloomberg", self.bloomberg_adapter)
        self.integrator.register_model("msci", self.msci_adapter)
        self.integrator.register_model("factset", self.factset_adapter)
        self.integrator.activate_model("bloomberg")
        self.integrator.activate_model("msci")
        self.integrator.activate_model("factset")
        
        # Define stress scenarios
        scenarios = ["market_crash", "interest_rate_hike", "tech_sector_downturn"]
        
        # Get the stress test results
        stress_results = self.integrator.get_stress_test_results(self.sample_portfolio, scenarios)
        
        # Check that the stress results have the expected structure
        self.assertIn("scenarios", stress_results)
        self.assertIn("impacts", stress_results)
        self.assertIn("model_sources", stress_results)
        
        # Check that all scenarios are included
        for scenario in scenarios:
            self.assertIn(scenario, stress_results["scenarios"])
        
        # Check that the model sources are included
        self.assertIn("bloomberg", stress_results["model_sources"])
        self.assertIn("msci", stress_results["model_sources"])
        self.assertIn("factset", stress_results["model_sources"])
    
    def test_get_consensus_correlation_matrix(self):
        """Test getting consensus correlation matrix."""
        # Register and activate the models
        self.integrator.register_model("bloomberg", self.bloomberg_adapter)
        self.integrator.register_model("msci", self.msci_adapter)
        self.integrator.register_model("factset", self.factset_adapter)
        self.integrator.activate_model("bloomberg")
        self.integrator.activate_model("msci")
        self.integrator.activate_model("factset")
        
        # Get the consensus correlation matrix
        correlation_matrix = self.integrator.get_consensus_correlation_matrix(self.sample_portfolio)
        
        # Check that the correlation matrix has the expected structure
        self.assertIn("assets", correlation_matrix)
        self.assertIn("matrix", correlation_matrix)
        self.assertIn("model_sources", correlation_matrix)
        
        # Check that all assets are included
        for position in self.sample_portfolio["positions"]:
            self.assertIn(position["ticker"], correlation_matrix["assets"])
        
        # Check that the matrix has the correct dimensions
        num_assets = len(self.sample_portfolio["positions"])
        self.assertEqual(len(correlation_matrix["matrix"]), num_assets)
        for row in correlation_matrix["matrix"]:
            self.assertEqual(len(row), num_assets)
    
    def test_get_combined_var_contribution(self):
        """Test getting combined VaR contribution."""
        # Register and activate the models
        self.integrator.register_model("bloomberg", self.bloomberg_adapter)
        self.integrator.register_model("msci", self.msci_adapter)
        self.integrator.register_model("factset", self.factset_adapter)
        self.integrator.activate_model("bloomberg")
        self.integrator.activate_model("msci")
        self.integrator.activate_model("factset")
        
        # Get the combined VaR contribution
        combined_var = self.integrator.get_combined_var_contribution(self.sample_portfolio)
        
        # Check that the combined VaR has the expected structure
        self.assertIn("assets", combined_var)
        self.assertIn("contributions", combined_var)
        self.assertIn("total_var", combined_var)
        
        # Check that all assets are included
        for position in self.sample_portfolio["positions"]:
            self.assertIn(position["ticker"], combined_var["assets"])
        
        # Check that the contributions sum up to the total VaR (approximately)
        total_contribution = sum(combined_var["contributions"])
        self.assertAlmostEqual(total_contribution, combined_var["total_var"], delta=0.01)
        
        # Check that the model sources are included
        self.assertIn("model_sources", combined_var)
        self.assertIn("bloomberg", combined_var["model_sources"])
        self.assertIn("msci", combined_var["model_sources"])
        self.assertIn("factset", combined_var["model_sources"])
    
    def test_model_weights(self):
        """Test setting and using model weights."""
        # Register and activate the models
        self.integrator.register_model("bloomberg", self.bloomberg_adapter)
        self.integrator.register_model("msci", self.msci_adapter)
        self.integrator.register_model("factset", self.factset_adapter)
        self.integrator.activate_model("bloomberg")
        self.integrator.activate_model("msci")
        self.integrator.activate_model("factset")
        
        # Set model weights
        self.integrator.set_model_weights({"bloomberg": 0.5, "msci": 0.3, "factset": 0.2})
        
        # Get the combined risk metrics
        combined_metrics = self.integrator.get_combined_risk_metrics(self.sample_portfolio)
        
        # Check that the model weights are included
        self.assertIn("model_weights", combined_metrics)
        self.assertEqual(combined_metrics["model_weights"]["bloomberg"], 0.5)
        self.assertEqual(combined_metrics["model_weights"]["msci"], 0.3)
        self.assertEqual(combined_metrics["model_weights"]["factset"], 0.2)
    
    def test_error_handling(self):
        """Test error handling when a model fails."""
        # Create a mock model that raises an exception
        mock_model = MagicMock(spec=ExternalRiskModel)
        mock_model.get_risk_metrics.side_effect = Exception("Test exception")
        
        # Register and activate the models
        self.integrator.register_model("bloomberg", self.bloomberg_adapter)
        self.integrator.register_model("failing_model", mock_model)
        self.integrator.activate_model("bloomberg")
        self.integrator.activate_model("failing_model")
        
        # Get the combined risk metrics - should not raise an exception
        combined_metrics = self.integrator.get_combined_risk_metrics(self.sample_portfolio)
        
        # Check that the combined metrics only contain data from the working model
        self.assertIn("model_sources", combined_metrics)
        self.assertIn("bloomberg", combined_metrics["model_sources"])
        self.assertNotIn("failing_model", combined_metrics["model_sources"])
        
        # Check that the errors are recorded
        self.assertIn("model_errors", combined_metrics)
        self.assertIn("failing_model", combined_metrics["model_errors"])


    def test_model_comparison(self):
        """Test comparing risk metrics from different models."""
        # Register and activate the models
        self.integrator.register_model("bloomberg", self.bloomberg_adapter)
        self.integrator.register_model("msci", self.msci_adapter)
        self.integrator.register_model("factset", self.factset_adapter)
        self.integrator.activate_model("bloomberg")
        self.integrator.activate_model("msci")
        self.integrator.activate_model("factset")
        
        # Get risk metrics from each model individually
        bloomberg_metrics = self.bloomberg_adapter.get_risk_metrics(self.sample_portfolio)
        msci_metrics = self.msci_adapter.get_risk_metrics(self.sample_portfolio)
        factset_metrics = self.factset_adapter.get_risk_metrics(self.sample_portfolio)
        
        # Get combined risk metrics
        combined_metrics = self.integrator.get_combined_risk_metrics(self.sample_portfolio)
        
        # Check that each model provides VaR estimates
        self.assertIn("var_95", bloomberg_metrics)
        self.assertIn("var_95", msci_metrics)
        self.assertIn("var_95", factset_metrics)
        self.assertIn("var_95", combined_metrics)
        
        # Check that the combined VaR is within the range of individual VaRs
        min_var = min(bloomberg_metrics["var_95"], msci_metrics["var_95"], factset_metrics["var_95"])
        max_var = max(bloomberg_metrics["var_95"], msci_metrics["var_95"], factset_metrics["var_95"])
        self.assertGreaterEqual(combined_metrics["var_95"], min_var)
        self.assertLessEqual(combined_metrics["var_95"], max_var)


if __name__ == "__main__":
    unittest.main()