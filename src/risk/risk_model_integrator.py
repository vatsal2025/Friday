"""Risk Model Integrator for Friday AI Trading System.

This module provides the RiskModelIntegrator class for integrating external risk models
into the Friday AI Trading System's risk management framework.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Type
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from .external_risk_model import ExternalRiskModel


class RiskModelIntegrator:
    """Class for integrating external risk models with the internal risk management system.
    
    This class serves as a bridge between external risk models and the Friday AI Trading System's
    risk management components. It provides methods for registering external risk models,
    retrieving risk metrics from them, and combining them with internal risk metrics.
    """
    
    def __init__(self):
        """Initialize the RiskModelIntegrator."""
        self.logger = logging.getLogger(__name__)
        self.external_models: Dict[str, ExternalRiskModel] = {}
        self.active_models: List[str] = []
        self.logger.info("RiskModelIntegrator initialized")
    
    def register_model(self, model_id: str, model: ExternalRiskModel) -> None:
        """Register an external risk model.
        
        Args:
            model_id: Unique identifier for the model.
            model: Instance of an ExternalRiskModel implementation.
        """
        if model_id in self.external_models:
            self.logger.warning(f"Model with ID {model_id} already registered. Overwriting.")
        
        self.external_models[model_id] = model
        self.logger.info(f"Registered external risk model: {model_id}")
    
    def activate_model(self, model_id: str) -> bool:
        """Activate an external risk model for use.
        
        Args:
            model_id: Identifier of the model to activate.
            
        Returns:
            True if the model was activated successfully, False otherwise.
        """
        if model_id not in self.external_models:
            self.logger.error(f"Cannot activate model {model_id}: not registered")
            return False
        
        if model_id not in self.active_models:
            self.active_models.append(model_id)
            self.logger.info(f"Activated external risk model: {model_id}")
        
        return True
    
    def deactivate_model(self, model_id: str) -> bool:
        """Deactivate an external risk model.
        
        Args:
            model_id: Identifier of the model to deactivate.
            
        Returns:
            True if the model was deactivated successfully, False otherwise.
        """
        if model_id in self.active_models:
            self.active_models.remove(model_id)
            self.logger.info(f"Deactivated external risk model: {model_id}")
            return True
        
        self.logger.warning(f"Model {model_id} was not active")
        return False
    
    def get_combined_risk_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get combined risk metrics from all active external models.
        
        Args:
            portfolio_data: Dictionary containing portfolio data.
            
        Returns:
            Dictionary containing combined risk metrics from all active external models.
        """
        combined_metrics = {}
        
        for model_id in self.active_models:
            try:
                model = self.external_models[model_id]
                metrics = model.get_risk_metrics(portfolio_data)
                
                # Add model identifier to metric keys to avoid conflicts
                prefixed_metrics = {f"{model_id}_{key}": value for key, value in metrics.items()}
                combined_metrics.update(prefixed_metrics)
                
                # Also store the raw metrics under the model ID
                combined_metrics[model_id] = metrics
            except Exception as e:
                self.logger.error(f"Error getting risk metrics from model {model_id}: {str(e)}")
        
        return combined_metrics
    
    def get_combined_factor_exposures(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """Get combined factor exposures from all active external models.
        
        Args:
            portfolio_data: Dictionary containing portfolio data.
            
        Returns:
            Dictionary mapping factor names to exposure values.
        """
        combined_exposures = {}
        
        for model_id in self.active_models:
            try:
                model = self.external_models[model_id]
                exposures = model.get_factor_exposures(portfolio_data)
                
                # Add model identifier to factor names to avoid conflicts
                prefixed_exposures = {f"{model_id}_{factor}": value 
                                     for factor, value in exposures.items()}
                combined_exposures.update(prefixed_exposures)
                
                # Also store the raw exposures under the model ID
                combined_exposures[model_id] = exposures
            except Exception as e:
                self.logger.error(f"Error getting factor exposures from model {model_id}: {str(e)}")
        
        return combined_exposures
    
    def get_stress_test_results(self, portfolio_data: Dict[str, Any], 
                               scenario: str) -> Dict[str, Dict[str, float]]:
        """Get stress test results from all active external models.
        
        Args:
            portfolio_data: Dictionary containing portfolio data.
            scenario: Name of the stress test scenario to run.
            
        Returns:
            Dictionary mapping model IDs to their stress test results.
        """
        results = {}
        
        for model_id in self.active_models:
            try:
                model = self.external_models[model_id]
                model_results = model.get_stress_test_results(portfolio_data, scenario)
                results[model_id] = model_results
            except Exception as e:
                self.logger.error(f"Error getting stress test results from model {model_id}: {str(e)}")
        
        return results
    
    def get_consensus_correlation_matrix(self, assets: List[str]) -> pd.DataFrame:
        """Get a consensus correlation matrix from all active external models.
        
        This method combines correlation matrices from all active models by taking
        a weighted average.
        
        Args:
            assets: List of asset identifiers.
            
        Returns:
            Pandas DataFrame containing the consensus correlation matrix.
        """
        if not self.active_models:
            self.logger.warning("No active external models for correlation matrix")
            # Return an identity matrix if no models are active
            return pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets)
        
        matrices = []
        
        for model_id in self.active_models:
            try:
                model = self.external_models[model_id]
                matrix = model.get_correlation_matrix(assets)
                matrices.append(matrix)
            except Exception as e:
                self.logger.error(f"Error getting correlation matrix from model {model_id}: {str(e)}")
        
        if not matrices:
            self.logger.warning("Failed to get correlation matrices from any active models")
            # Return an identity matrix if all models failed
            return pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets)
        
        # Simple average of all matrices
        # Could be enhanced with weighted averaging based on model confidence/accuracy
        consensus_matrix = sum(matrices) / len(matrices)
        
        return consensus_matrix
    
    def get_combined_var_contribution(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """Get combined VaR contribution from all active external models.
        
        Args:
            portfolio_data: Dictionary containing portfolio data.
            
        Returns:
            Dictionary mapping asset identifiers to combined VaR contribution values.
        """
        contributions = {}
        model_count = 0
        
        for model_id in self.active_models:
            try:
                model = self.external_models[model_id]
                model_contributions = model.get_var_contribution(portfolio_data)
                
                # Initialize contributions dict with first model
                if not contributions:
                    contributions = {asset: 0.0 for asset in model_contributions.keys()}
                
                # Add contributions from this model
                for asset, value in model_contributions.items():
                    if asset in contributions:
                        contributions[asset] += value
                    else:
                        self.logger.warning(f"Asset {asset} not found in previous models")
                        contributions[asset] = value
                
                model_count += 1
            except Exception as e:
                self.logger.error(f"Error getting VaR contribution from model {model_id}: {str(e)}")
        
        # Average the contributions if we have any models
        if model_count > 0:
            for asset in contributions:
                contributions[asset] /= model_count
        
        return contributions