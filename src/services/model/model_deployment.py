from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set
import uuid
import json
import os

from infrastructure.security.audit_logging import (
    SecurityAuditLogger,
    log_model_deployment_started,
    log_model_deployment_completed,
    log_model_deployment_failed,
    log_model_undeployed,
    log_model_deployment_status_changed
)


class DeploymentStatus(Enum):
    """Status of a model deployment."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    UNDEPLOYING = "undeploying"
    UNDEPLOYED = "undeployed"


class DeploymentEnvironment(Enum):
    """Environment where a model can be deployed."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DeploymentConfig:
    """Configuration for a model deployment."""
    environment: DeploymentEnvironment
    resources: Dict[str, str] = field(default_factory=dict)
    scaling: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 300
    retries: int = 3
    enable_monitoring: bool = True
    enable_logging: bool = True
    enable_metrics: bool = True
    custom_settings: Dict[str, str] = field(default_factory=dict)


@dataclass
class ModelDeployment:
    """Represents a model deployment."""
    deployment_id: str
    model_id: str
    model_name: str
    model_version: str
    environment: DeploymentEnvironment
    status: DeploymentStatus
    config: DeploymentConfig
    created_by: str
    created_at: datetime
    updated_at: datetime
    deployment_url: Optional[str] = None
    status_history: List[Dict[str, str]] = field(default_factory=list)
    health_status: str = "unknown"
    metrics: Dict[str, float] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert the deployment to a dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "environment": self.environment.value,
            "status": self.status.value,
            "config": {
                "environment": self.config.environment.value,
                "resources": self.config.resources,
                "scaling": self.config.scaling,
                "timeout_seconds": self.config.timeout_seconds,
                "retries": self.config.retries,
                "enable_monitoring": self.config.enable_monitoring,
                "enable_logging": self.config.enable_logging,
                "enable_metrics": self.config.enable_metrics,
                "custom_settings": self.config.custom_settings
            },
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "deployment_url": self.deployment_url,
            "status_history": self.status_history,
            "health_status": self.health_status,
            "metrics": self.metrics
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelDeployment':
        """Create a deployment from a dictionary."""
        config_data = data.get("config", {})
        config = DeploymentConfig(
            environment=DeploymentEnvironment(config_data.get("environment")),
            resources=config_data.get("resources", {}),
            scaling=config_data.get("scaling", {}),
            timeout_seconds=config_data.get("timeout_seconds", 300),
            retries=config_data.get("retries", 3),
            enable_monitoring=config_data.get("enable_monitoring", True),
            enable_logging=config_data.get("enable_logging", True),
            enable_metrics=config_data.get("enable_metrics", True),
            custom_settings=config_data.get("custom_settings", {})
        )
        
        return cls(
            deployment_id=data.get("deployment_id"),
            model_id=data.get("model_id"),
            model_name=data.get("model_name"),
            model_version=data.get("model_version"),
            environment=DeploymentEnvironment(data.get("environment")),
            status=DeploymentStatus(data.get("status")),
            config=config,
            created_by=data.get("created_by"),
            created_at=datetime.fromisoformat(data.get("created_at")),
            updated_at=datetime.fromisoformat(data.get("updated_at")),
            deployment_url=data.get("deployment_url"),
            status_history=data.get("status_history", []),
            health_status=data.get("health_status", "unknown"),
            metrics=data.get("metrics", {}),
            logs=data.get("logs", [])
        )


class ModelDeploymentManager:
    """Manages model deployments across different environments."""
    
    def __init__(self, base_directory: str, audit_logger: SecurityAuditLogger):
        """Initialize the deployment manager.
        
        Args:
            base_directory: The base directory for storing deployment information.
            audit_logger: The security audit logger for logging deployment events.
        """
        self.base_directory = base_directory
        self.audit_logger = audit_logger
        self.deployments_directory = os.path.join(base_directory, "deployments")
        os.makedirs(self.deployments_directory, exist_ok=True)
        
        # Create environment-specific directories
        for env in DeploymentEnvironment:
            os.makedirs(os.path.join(self.deployments_directory, env.value), exist_ok=True)
    
    def _get_deployment_path(self, deployment_id: str) -> str:
        """Get the file path for a deployment.
        
        Args:
            deployment_id: The ID of the deployment.
            
        Returns:
            The file path for the deployment.
        """
        return os.path.join(self.deployments_directory, f"{deployment_id}.json")
    
    def _save_deployment(self, deployment: ModelDeployment) -> None:
        """Save a deployment to disk.
        
        Args:
            deployment: The deployment to save.
        """
        deployment_path = self._get_deployment_path(deployment.deployment_id)
        with open(deployment_path, "w") as f:
            json.dump(deployment.to_dict(), f, indent=2)
    
    def _load_deployment(self, deployment_id: str) -> Optional[ModelDeployment]:
        """Load a deployment from disk.
        
        Args:
            deployment_id: The ID of the deployment.
            
        Returns:
            The loaded deployment, or None if it doesn't exist.
        """
        deployment_path = self._get_deployment_path(deployment_id)
        if not os.path.exists(deployment_path):
            return None
        
        with open(deployment_path, "r") as f:
            data = json.load(f)
        
        return ModelDeployment.from_dict(data)
    
    def create_deployment(self, model_id: str, model_name: str, model_version: str,
                         environment: DeploymentEnvironment, config: DeploymentConfig,
                         username: str) -> ModelDeployment:
        """Create a new deployment for a model.
        
        Args:
            model_id: The ID of the model.
            model_name: The name of the model.
            model_version: The version of the model.
            environment: The environment to deploy to.
            config: The deployment configuration.
            username: The username of the user creating the deployment.
            
        Returns:
            The created deployment.
        """
        deployment_id = str(uuid.uuid4())
        now = datetime.now()
        
        deployment = ModelDeployment(
            deployment_id=deployment_id,
            model_id=model_id,
            model_name=model_name,
            model_version=model_version,
            environment=environment,
            status=DeploymentStatus.PENDING,
            config=config,
            created_by=username,
            created_at=now,
            updated_at=now,
            status_history=[{"status": DeploymentStatus.PENDING.value, "timestamp": now.isoformat()}]
        )
        
        self._save_deployment(deployment)
        
        return deployment
    
    def start_deployment(self, deployment_id: str, username: str) -> ModelDeployment:
        """Start the deployment process for a model.
        
        Args:
            deployment_id: The ID of the deployment.
            username: The username of the user starting the deployment.
            
        Returns:
            The updated deployment.
            
        Raises:
            ValueError: If the deployment doesn't exist or is not in PENDING status.
        """
        deployment = self._load_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        if deployment.status != DeploymentStatus.PENDING:
            raise ValueError(f"Deployment {deployment_id} is not in PENDING status")
        
        # Update status to DEPLOYING
        old_status = deployment.status
        deployment.status = DeploymentStatus.DEPLOYING
        deployment.updated_at = datetime.now()
        deployment.status_history.append({
            "status": DeploymentStatus.DEPLOYING.value,
            "timestamp": deployment.updated_at.isoformat(),
            "username": username
        })
        
        self._save_deployment(deployment)
        
        # Log the deployment started event
        log_model_deployment_started(
            self.audit_logger,
            username,
            deployment.model_id,
            deployment.model_name,
            deployment.environment.value,
            deployment_id
        )
        
        # Log the status change
        log_model_deployment_status_changed(
            self.audit_logger,
            username,
            deployment.model_id,
            deployment.model_name,
            deployment.environment.value,
            deployment_id,
            old_status.value,
            deployment.status.value
        )
        
        return deployment
    
    def complete_deployment(self, deployment_id: str, username: str, deployment_url: str) -> ModelDeployment:
        """Mark a deployment as completed.
        
        Args:
            deployment_id: The ID of the deployment.
            username: The username of the user completing the deployment.
            deployment_url: The URL where the deployed model is accessible.
            
        Returns:
            The updated deployment.
            
        Raises:
            ValueError: If the deployment doesn't exist or is not in DEPLOYING status.
        """
        deployment = self._load_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        if deployment.status != DeploymentStatus.DEPLOYING:
            raise ValueError(f"Deployment {deployment_id} is not in DEPLOYING status")
        
        # Update status to DEPLOYED
        old_status = deployment.status
        deployment.status = DeploymentStatus.DEPLOYED
        deployment.updated_at = datetime.now()
        deployment.deployment_url = deployment_url
        deployment.status_history.append({
            "status": DeploymentStatus.DEPLOYED.value,
            "timestamp": deployment.updated_at.isoformat(),
            "username": username
        })
        
        self._save_deployment(deployment)
        
        # Log the deployment completed event
        log_model_deployment_completed(
            self.audit_logger,
            username,
            deployment.model_id,
            deployment.model_name,
            deployment.environment.value,
            deployment_id,
            deployment_url
        )
        
        # Log the status change
        log_model_deployment_status_changed(
            self.audit_logger,
            username,
            deployment.model_id,
            deployment.model_name,
            deployment.environment.value,
            deployment_id,
            old_status.value,
            deployment.status.value
        )
        
        return deployment
    
    def fail_deployment(self, deployment_id: str, username: str, reason: str) -> ModelDeployment:
        """Mark a deployment as failed.
        
        Args:
            deployment_id: The ID of the deployment.
            username: The username of the user failing the deployment.
            reason: The reason for the failure.
            
        Returns:
            The updated deployment.
            
        Raises:
            ValueError: If the deployment doesn't exist or is not in DEPLOYING status.
        """
        deployment = self._load_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        if deployment.status != DeploymentStatus.DEPLOYING:
            raise ValueError(f"Deployment {deployment_id} is not in DEPLOYING status")
        
        # Update status to FAILED
        old_status = deployment.status
        deployment.status = DeploymentStatus.FAILED
        deployment.updated_at = datetime.now()
        deployment.status_history.append({
            "status": DeploymentStatus.FAILED.value,
            "timestamp": deployment.updated_at.isoformat(),
            "username": username,
            "reason": reason
        })
        
        self._save_deployment(deployment)
        
        # Log the deployment failed event
        log_model_deployment_failed(
            self.audit_logger,
            username,
            deployment.model_id,
            deployment.model_name,
            deployment.environment.value,
            deployment_id,
            reason
        )
        
        # Log the status change
        log_model_deployment_status_changed(
            self.audit_logger,
            username,
            deployment.model_id,
            deployment.model_name,
            deployment.environment.value,
            deployment_id,
            old_status.value,
            deployment.status.value
        )
        
        return deployment
    
    def undeploy(self, deployment_id: str, username: str) -> ModelDeployment:
        """Undeploy a model.
        
        Args:
            deployment_id: The ID of the deployment.
            username: The username of the user undeploying the model.
            
        Returns:
            The updated deployment.
            
        Raises:
            ValueError: If the deployment doesn't exist or is not in DEPLOYED status.
        """
        deployment = self._load_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        if deployment.status != DeploymentStatus.DEPLOYED:
            raise ValueError(f"Deployment {deployment_id} is not in DEPLOYED status")
        
        # Update status to UNDEPLOYING
        old_status = deployment.status
        deployment.status = DeploymentStatus.UNDEPLOYING
        deployment.updated_at = datetime.now()
        deployment.status_history.append({
            "status": DeploymentStatus.UNDEPLOYING.value,
            "timestamp": deployment.updated_at.isoformat(),
            "username": username
        })
        
        self._save_deployment(deployment)
        
        # Log the status change
        log_model_deployment_status_changed(
            self.audit_logger,
            username,
            deployment.model_id,
            deployment.model_name,
            deployment.environment.value,
            deployment_id,
            old_status.value,
            deployment.status.value
        )
        
        # Update status to UNDEPLOYED
        old_status = deployment.status
        deployment.status = DeploymentStatus.UNDEPLOYED
        deployment.updated_at = datetime.now()
        deployment.status_history.append({
            "status": DeploymentStatus.UNDEPLOYED.value,
            "timestamp": deployment.updated_at.isoformat(),
            "username": username
        })
        
        self._save_deployment(deployment)
        
        # Log the undeployment event
        log_model_undeployed(
            self.audit_logger,
            username,
            deployment.model_id,
            deployment.model_name,
            deployment.environment.value,
            deployment_id
        )
        
        # Log the status change
        log_model_deployment_status_changed(
            self.audit_logger,
            username,
            deployment.model_id,
            deployment.model_name,
            deployment.environment.value,
            deployment_id,
            old_status.value,
            deployment.status.value
        )
        
        return deployment
    
    def get_deployment(self, deployment_id: str) -> Optional[ModelDeployment]:
        """Get a deployment by ID.
        
        Args:
            deployment_id: The ID of the deployment.
            
        Returns:
            The deployment, or None if it doesn't exist.
        """
        return self._load_deployment(deployment_id)
    
    def get_deployments_by_model(self, model_id: str) -> List[ModelDeployment]:
        """Get all deployments for a model.
        
        Args:
            model_id: The ID of the model.
            
        Returns:
            A list of deployments for the model.
        """
        deployments = []
        for filename in os.listdir(self.deployments_directory):
            if filename.endswith(".json"):
                deployment_id = filename[:-5]  # Remove .json extension
                deployment = self._load_deployment(deployment_id)
                if deployment and deployment.model_id == model_id:
                    deployments.append(deployment)
        
        return deployments
    
    def get_deployments_by_environment(self, environment: DeploymentEnvironment) -> List[ModelDeployment]:
        """Get all deployments in an environment.
        
        Args:
            environment: The environment to get deployments for.
            
        Returns:
            A list of deployments in the environment.
        """
        deployments = []
        for filename in os.listdir(self.deployments_directory):
            if filename.endswith(".json"):
                deployment_id = filename[:-5]  # Remove .json extension
                deployment = self._load_deployment(deployment_id)
                if deployment and deployment.environment == environment:
                    deployments.append(deployment)
        
        return deployments
    
    def get_active_deployments(self) -> List[ModelDeployment]:
        """Get all active deployments (DEPLOYING or DEPLOYED).
        
        Returns:
            A list of active deployments.
        """
        deployments = []
        for filename in os.listdir(self.deployments_directory):
            if filename.endswith(".json"):
                deployment_id = filename[:-5]  # Remove .json extension
                deployment = self._load_deployment(deployment_id)
                if deployment and deployment.status in [DeploymentStatus.DEPLOYING, DeploymentStatus.DEPLOYED]:
                    deployments.append(deployment)
        
        return deployments
    
    def update_deployment_health(self, deployment_id: str, health_status: str, metrics: Dict[str, float]) -> ModelDeployment:
        """Update the health status and metrics of a deployment.
        
        Args:
            deployment_id: The ID of the deployment.
            health_status: The health status of the deployment.
            metrics: The metrics of the deployment.
            
        Returns:
            The updated deployment.
            
        Raises:
            ValueError: If the deployment doesn't exist.
        """
        deployment = self._load_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment.health_status = health_status
        deployment.metrics = metrics
        deployment.updated_at = datetime.now()
        
        self._save_deployment(deployment)
        
        return deployment
    
    def add_deployment_log(self, deployment_id: str, log_message: str) -> ModelDeployment:
        """Add a log message to a deployment.
        
        Args:
            deployment_id: The ID of the deployment.
            log_message: The log message to add.
            
        Returns:
            The updated deployment.
            
        Raises:
            ValueError: If the deployment doesn't exist.
        """
        deployment = self._load_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        timestamp = datetime.now().isoformat()
        deployment.logs.append(f"[{timestamp}] {log_message}")
        deployment.updated_at = datetime.now()
        
        self._save_deployment(deployment)
        
        return deployment