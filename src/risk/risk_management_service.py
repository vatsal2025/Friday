import logging
import os
import json
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
import pickle

from src.risk.advanced_risk_manager import AdvancedRiskManager
from src.risk.production_config import RiskManagementProductionConfig, load_production_config
from src.risk.risk_management_factory import RiskManagementFactory

logger = logging.getLogger(__name__)

class RiskManagementService:
    """
    Production-ready risk management service with persistence and monitoring.

    This service wraps the AdvancedRiskManager and adds production features like:
    - Periodic risk metrics calculation and monitoring
    - State persistence to disk
    - Automatic recovery from persistence
    - Alert notifications
    - Health checks
    """

    def __init__(self, config: Optional[RiskManagementProductionConfig] = None):
        """
        Initialize the risk management service.

        Args:
            config: The production configuration to use. If None, loads the default.
        """
        self.config = config if config is not None else load_production_config()
        self.factory = RiskManagementFactory(self.config)
        self.risk_manager = None
        self.alert_callbacks = []
        self.is_running = False
        self.metrics_thread = None
        self.persistence_thread = None
        self.last_metrics = {}
        self.last_metrics_time = None
        self.last_persistence_time = None
        self.health_status = {"status": "initializing"}

        # Configure logging
        log_level = getattr(logging, self.config.log_level, logging.INFO)
        logging.basicConfig(level=log_level)

        logger.info("Initialized RiskManagementService")

    def start(self, emergency_handlers: Optional[List[Any]] = None) -> None:
        """
        Start the risk management service.

        Args:
            emergency_handlers: Optional list of emergency handlers to notify when circuit breakers trigger.
        """
        if self.is_running:
            logger.warning("RiskManagementService is already running")
            return

        # Try to recover from persistence first
        recovered = self._recover_from_persistence()

        # If recovery failed, create a new risk manager
        if not recovered:
            logger.info("Creating new AdvancedRiskManager instance")
            self.risk_manager = self.factory.create_advanced_risk_manager(emergency_handlers)

        # Register alert callbacks
        if self.config.alert_notification_enabled:
            self.risk_manager.register_alert_callback(self._handle_risk_alert)

        # Start monitoring threads
        self.is_running = True
        self._start_metrics_thread()

        if self.config.persistence_enabled:
            self._start_persistence_thread()

        self.health_status = {"status": "running", "start_time": datetime.now().isoformat()}
        logger.info("RiskManagementService started successfully")

    def stop(self) -> None:
        """
        Stop the risk management service.
        """
        if not self.is_running:
            logger.warning("RiskManagementService is not running")
            return

        # Stop monitoring threads
        self.is_running = False

        if self.metrics_thread and self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=5.0)

        if self.persistence_thread and self.persistence_thread.is_alive():
            self.persistence_thread.join(timeout=5.0)

        # Persist state one last time
        if self.config.persistence_enabled and self.risk_manager:
            self._persist_state()

        self.health_status = {"status": "stopped", "stop_time": datetime.now().isoformat()}
        logger.info("RiskManagementService stopped successfully")

    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback to be notified of risk alerts.

        Args:
            callback: A function that takes a risk alert dictionary as input.
        """
        self.alert_callbacks.append(callback)
        logger.info(f"Registered new alert callback, total callbacks: {len(self.alert_callbacks)}")

    def get_risk_manager(self) -> Optional[AdvancedRiskManager]:
        """
        Get the underlying risk manager instance.

        Returns:
            AdvancedRiskManager: The risk manager instance, or None if not initialized.
        """
        return self.risk_manager

    def get_latest_metrics(self) -> Dict[str, Any]:
        """
        Get the latest risk metrics.

        Returns:
            Dict[str, Any]: The latest risk metrics, or an empty dict if none available.
        """
        return self.last_metrics

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get the health status of the risk management service.

        Returns:
            Dict[str, Any]: The health status information.
        """
        if self.is_running:
            self.health_status["status"] = "running"
        else:
            self.health_status["status"] = "stopped"

        self.health_status["last_metrics_time"] = self.last_metrics_time
        self.health_status["last_persistence_time"] = self.last_persistence_time
        self.health_status["metrics_available"] = len(self.last_metrics) > 0

        return self.health_status

    def _start_metrics_thread(self) -> None:
        """
        Start the thread that periodically calculates risk metrics.
        """
        def metrics_worker():
            logger.info("Risk metrics monitoring thread started")
            while self.is_running:
                try:
                    if self.risk_manager:
                        # Calculate and store risk metrics
                        metrics = self.risk_manager.get_risk_metrics()
                        self.last_metrics = metrics
                        self.last_metrics_time = datetime.now().isoformat()

                        # Check for any risk limit breaches
                        self._check_risk_limits(metrics)

                        logger.debug(f"Updated risk metrics: {metrics}")
                except Exception as e:
                    logger.error(f"Error calculating risk metrics: {e}")

                # Sleep until next calculation interval
                time.sleep(self.config.risk_metrics_calculation_interval)

            logger.info("Risk metrics monitoring thread stopped")

        self.metrics_thread = threading.Thread(target=metrics_worker, daemon=True)
        self.metrics_thread.start()

    def _start_persistence_thread(self) -> None:
        """
        Start the thread that periodically persists the risk manager state.
        """
        def persistence_worker():
            logger.info("Risk management persistence thread started")
            while self.is_running:
                try:
                    self._persist_state()
                except Exception as e:
                    logger.error(f"Error persisting risk management state: {e}")

                # Sleep until next persistence interval
                time.sleep(self.config.persistence_interval)

            logger.info("Risk management persistence thread stopped")

        self.persistence_thread = threading.Thread(target=persistence_worker, daemon=True)
        self.persistence_thread.start()

    def _persist_state(self) -> bool:
        """
        Persist the current state of the risk manager to disk.

        Returns:
            bool: True if persistence was successful, False otherwise.
        """
        if not self.risk_manager:
            return False

        try:
            # Create persistence directory if it doesn't exist
            os.makedirs(self.config.persistence_path, exist_ok=True)

            # Serialize the risk manager state
            state_path = os.path.join(self.config.persistence_path, "risk_manager_state.pkl")
            with open(state_path, 'wb') as f:
                pickle.dump(self.risk_manager, f)

            # Also save the latest metrics as JSON for easier inspection
            metrics_path = os.path.join(self.config.persistence_path, "latest_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.last_metrics, f, indent=4)

            self.last_persistence_time = datetime.now().isoformat()
            logger.info(f"Persisted risk management state to {state_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to persist risk management state: {e}")
            return False

    def _recover_from_persistence(self) -> bool:
        """
        Try to recover the risk manager from a persisted state.

        Returns:
            bool: True if recovery was successful, False otherwise.
        """
        if not self.config.persistence_enabled:
            return False

        state_path = os.path.join(self.config.persistence_path, "risk_manager_state.pkl")
        if not os.path.exists(state_path):
            logger.info(f"No persisted state found at {state_path}")
            return False

        try:
            with open(state_path, 'rb') as f:
                self.risk_manager = pickle.load(f)

            # Load the latest metrics
            metrics_path = os.path.join(self.config.persistence_path, "latest_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.last_metrics = json.load(f)

            logger.info(f"Successfully recovered risk manager from {state_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to recover risk manager from persistence: {e}")
            return False

    def _check_risk_limits(self, metrics: Dict[str, Any]) -> None:
        """
        Check if any risk limits are breached and generate alerts if needed.

        Args:
            metrics: The current risk metrics.
        """
        alerts = []

        # Check VaR limit
        if "var_percent" in metrics and metrics["var_percent"] > self.config.max_portfolio_var_percent:
            alerts.append({
                "type": "risk_limit_breach",
                "severity": "high",
                "metric": "var_percent",
                "value": metrics["var_percent"],
                "limit": self.config.max_portfolio_var_percent,
                "timestamp": datetime.now().isoformat()
            })

        # Check drawdown limit
        if "drawdown_percent" in metrics and metrics["drawdown_percent"] > self.config.max_drawdown_percent:
            alerts.append({
                "type": "risk_limit_breach",
                "severity": "high",
                "metric": "drawdown_percent",
                "value": metrics["drawdown_percent"],
                "limit": self.config.max_drawdown_percent,
                "timestamp": datetime.now().isoformat()
            })

        # Check sector exposure limits
        if "sector_exposure" in metrics:
            for sector, exposure in metrics["sector_exposure"].items():
                if exposure > self.config.max_sector_exposure:
                    alerts.append({
                        "type": "risk_limit_breach",
                        "severity": "medium",
                        "metric": "sector_exposure",
                        "sector": sector,
                        "value": exposure,
                        "limit": self.config.max_sector_exposure,
                        "timestamp": datetime.now().isoformat()
                    })

        # Check asset exposure limits
        if "asset_exposure" in metrics:
            for asset, exposure in metrics["asset_exposure"].items():
                if exposure > self.config.max_asset_exposure:
                    alerts.append({
                        "type": "risk_limit_breach",
                        "severity": "medium",
                        "metric": "asset_exposure",
                        "asset": asset,
                        "value": exposure,
                        "limit": self.config.max_asset_exposure,
                        "timestamp": datetime.now().isoformat()
                    })

        # Process any alerts
        for alert in alerts:
            self._handle_risk_alert(alert)

    def _handle_risk_alert(self, alert: Dict[str, Any]) -> None:
        """
        Handle a risk alert by logging it and notifying registered callbacks.

        Args:
            alert: The risk alert information.
        """
        # Log the alert
        log_level = logging.WARNING
        if alert.get("severity") == "high":
            log_level = logging.ERROR
        elif alert.get("severity") == "low":
            log_level = logging.INFO

        logger.log(log_level, f"Risk alert: {alert}")

        # Notify all registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
