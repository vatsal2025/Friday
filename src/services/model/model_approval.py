"""Model Approval Workflow for Friday AI Trading System.

This module provides functionality for implementing model approval workflows.
"""

import os
import json
import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum, auto
from dataclasses import dataclass

from src.infrastructure.logging import get_logger
from src.services.model.model_registry import ModelRegistry
from src.services.model.model_versioning import ModelVersioning
from src.orchestration.trading_engine.model_versioning import ModelVersionStatus
from src.services.model.config.model_registry_config import ModelStatus
from src.services.security.audit_logging import log_model_approval_status_change, log_model_approved, log_model_rejected

# Create logger
logger = get_logger(__name__)


class ApprovalStatus(Enum):
    """Enum representing the approval status of a model."""
    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()
    NEEDS_REVISION = auto()


@dataclass
class ApprovalRequest:
    """Represents a model approval request.
    
    Attributes:
        model_id: ID of the model.
        model_name: Name of the model.
        model_version: Version of the model.
        requested_by: Username of the requester.
        requested_at: Timestamp of the request.
        status: Current approval status.
        approvers: List of usernames who can approve the request.
        comments: List of comments on the request.
        approval_history: History of approval status changes.
    """
    model_id: str
    model_name: str
    model_version: str
    requested_by: str
    requested_at: datetime.datetime
    status: ApprovalStatus
    approvers: List[str]
    comments: List[Dict[str, Any]]
    approval_history: List[Dict[str, Any]]


class ModelApprovalWorkflow:
    """Model Approval Workflow for managing model approvals.

    This class provides functionality for implementing model approval workflows.

    Attributes:
        registry: The model registry instance.
        versioning: The model versioning instance.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None, versioning: Optional[ModelVersioning] = None):
        """Initialize the model approval workflow.

        Args:
            registry: The model registry. If None, a new one will be created.
            versioning: The model versioning. If None, a new one will be created.
        """
        self.registry = registry or ModelRegistry()
        self.versioning = versioning or ModelVersioning(registry=self.registry)
        logger.info("Initialized ModelApprovalWorkflow")

    def request_approval(self, model_id: str, requested_by: str, approvers: List[str], comment: Optional[str] = None) -> str:
        """Request approval for a model.

        Args:
            model_id: ID of the model.
            requested_by: Username of the requester.
            approvers: List of usernames who can approve the request.
            comment: Optional comment on the request.

        Returns:
            str: ID of the approval request.

        Raises:
            ValueError: If the model is not found or not in a valid state for approval.
        """
        # Get model metadata
        try:
            model_metadata = self.registry.get_model_metadata(model_id)
        except ValueError as e:
            logger.error(f"Error getting model metadata: {e}")
            raise

        model_name = model_metadata["model_name"]
        model_version = model_metadata["version"]

        # Check if model is in a valid state for approval
        if "status" in model_metadata:
            current_status = model_metadata["status"]
            if current_status not in [ModelStatus.DEVELOPMENT.name, ModelStatus.TESTING.name]:
                raise ValueError(f"Model {model_id} is in {current_status} status, which is not valid for approval")

        # Check if model has already been validated
        if "validation_status" in model_metadata:
            validation_status = model_metadata["validation_status"]
            if validation_status != ModelVersionStatus.VALIDATED.name:
                raise ValueError(f"Model {model_id} has not been validated yet")

        # Create approval request
        request_id = f"approval_{model_id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        requested_at = datetime.datetime.now()

        approval_request = ApprovalRequest(
            model_id=model_id,
            model_name=model_name,
            model_version=model_version,
            requested_by=requested_by,
            requested_at=requested_at,
            status=ApprovalStatus.PENDING,
            approvers=approvers,
            comments=[],
            approval_history=[],
        )

        # Add initial comment if provided
        if comment:
            approval_request.comments.append({
                "user": requested_by,
                "timestamp": requested_at,
                "comment": comment,
            })

        # Add initial history entry
        approval_request.approval_history.append({
            "timestamp": requested_at,
            "user": requested_by,
            "status": ApprovalStatus.PENDING.name,
            "comment": "Approval requested",
        })

        # Update model metadata with approval request
        if "approval" not in model_metadata:
            model_metadata["approval"] = {}

        model_metadata["approval"]["request_id"] = request_id
        model_metadata["approval"]["status"] = ApprovalStatus.PENDING.name
        model_metadata["approval"]["requested_by"] = requested_by
        model_metadata["approval"]["requested_at"] = requested_at.isoformat()
        model_metadata["approval"]["approvers"] = approvers
        model_metadata["approval"]["comments"] = approval_request.comments
        model_metadata["approval"]["approval_history"] = approval_request.approval_history

        # Update model metadata in registry
        self.registry.update_model_metadata(model_id, model_metadata)

        # Log the approval request
        log_model_approval_status_change(
            username=requested_by,
            model_id=model_id,
            model_name=model_name,
            old_status=None,
            new_status=ApprovalStatus.PENDING.name,
            comment="Approval requested",
        )

        logger.info(f"Created approval request {request_id} for model {model_id}")
        return request_id

    def approve_model(self, model_id: str, approved_by: str, comment: Optional[str] = None) -> None:
        """Approve a model.

        Args:
            model_id: ID of the model.
            approved_by: Username of the approver.
            comment: Optional comment on the approval.

        Raises:
            ValueError: If the model is not found, not pending approval, or the user is not an approver.
        """
        # Get model metadata
        try:
            model_metadata = self.registry.get_model_metadata(model_id)
        except ValueError as e:
            logger.error(f"Error getting model metadata: {e}")
            raise

        model_name = model_metadata["model_name"]

        # Check if model has an approval request
        if "approval" not in model_metadata:
            raise ValueError(f"Model {model_id} does not have an approval request")

        approval = model_metadata["approval"]

        # Check if model is pending approval
        if approval["status"] != ApprovalStatus.PENDING.name:
            raise ValueError(f"Model {model_id} is not pending approval")

        # Check if user is an approver
        if approved_by not in approval["approvers"]:
            raise ValueError(f"User {approved_by} is not an approver for model {model_id}")

        # Update approval status
        old_status = approval["status"]
        approval["status"] = ApprovalStatus.APPROVED.name

        # Add comment if provided
        timestamp = datetime.datetime.now()
        if comment:
            if "comments" not in approval:
                approval["comments"] = []
            approval["comments"].append({
                "user": approved_by,
                "timestamp": timestamp.isoformat(),
                "comment": comment,
            })

        # Add history entry
        if "approval_history" not in approval:
            approval["approval_history"] = []
        approval["approval_history"].append({
            "timestamp": timestamp.isoformat(),
            "user": approved_by,
            "status": ApprovalStatus.APPROVED.name,
            "comment": comment or "Model approved",
        })

        # Update model status to STAGING
        model_metadata["status"] = ModelStatus.STAGING.name

        # Update model metadata in registry
        self.registry.update_model_metadata(model_id, model_metadata)

        # Log the approval
        log_model_approved(
            username=approved_by,
            model_id=model_id,
            model_name=model_name,
            comment=comment or "Model approved",
        )

        # Log the status change
        log_model_approval_status_change(
            username=approved_by,
            model_id=model_id,
            model_name=model_name,
            old_status=old_status,
            new_status=ApprovalStatus.APPROVED.name,
            comment=comment or "Model approved",
        )

        logger.info(f"Model {model_id} approved by {approved_by}")

    def reject_model(self, model_id: str, rejected_by: str, comment: Optional[str] = None) -> None:
        """Reject a model.

        Args:
            model_id: ID of the model.
            rejected_by: Username of the rejector.
            comment: Optional comment on the rejection.

        Raises:
            ValueError: If the model is not found, not pending approval, or the user is not an approver.
        """
        # Get model metadata
        try:
            model_metadata = self.registry.get_model_metadata(model_id)
        except ValueError as e:
            logger.error(f"Error getting model metadata: {e}")
            raise

        model_name = model_metadata["model_name"]

        # Check if model has an approval request
        if "approval" not in model_metadata:
            raise ValueError(f"Model {model_id} does not have an approval request")

        approval = model_metadata["approval"]

        # Check if model is pending approval
        if approval["status"] != ApprovalStatus.PENDING.name:
            raise ValueError(f"Model {model_id} is not pending approval")

        # Check if user is an approver
        if rejected_by not in approval["approvers"]:
            raise ValueError(f"User {rejected_by} is not an approver for model {model_id}")

        # Update approval status
        old_status = approval["status"]
        approval["status"] = ApprovalStatus.REJECTED.name

        # Add comment if provided
        timestamp = datetime.datetime.now()
        if comment:
            if "comments" not in approval:
                approval["comments"] = []
            approval["comments"].append({
                "user": rejected_by,
                "timestamp": timestamp.isoformat(),
                "comment": comment,
            })

        # Add history entry
        if "approval_history" not in approval:
            approval["approval_history"] = []
        approval["approval_history"].append({
            "timestamp": timestamp.isoformat(),
            "user": rejected_by,
            "status": ApprovalStatus.REJECTED.name,
            "comment": comment or "Model rejected",
        })

        # Update model metadata in registry
        self.registry.update_model_metadata(model_id, model_metadata)

        # Log the rejection
        log_model_rejected(
            username=rejected_by,
            model_id=model_id,
            model_name=model_name,
            comment=comment or "Model rejected",
        )

        # Log the status change
        log_model_approval_status_change(
            username=rejected_by,
            model_id=model_id,
            model_name=model_name,
            old_status=old_status,
            new_status=ApprovalStatus.REJECTED.name,
            comment=comment or "Model rejected",
        )

        logger.info(f"Model {model_id} rejected by {rejected_by}")

    def request_revision(self, model_id: str, requested_by: str, comment: str) -> None:
        """Request revision for a model.

        Args:
            model_id: ID of the model.
            requested_by: Username of the requester.
            comment: Comment on what needs to be revised.

        Raises:
            ValueError: If the model is not found, not pending approval, or the user is not an approver.
        """
        # Get model metadata
        try:
            model_metadata = self.registry.get_model_metadata(model_id)
        except ValueError as e:
            logger.error(f"Error getting model metadata: {e}")
            raise

        model_name = model_metadata["model_name"]

        # Check if model has an approval request
        if "approval" not in model_metadata:
            raise ValueError(f"Model {model_id} does not have an approval request")

        approval = model_metadata["approval"]

        # Check if model is pending approval
        if approval["status"] != ApprovalStatus.PENDING.name:
            raise ValueError(f"Model {model_id} is not pending approval")

        # Check if user is an approver
        if requested_by not in approval["approvers"]:
            raise ValueError(f"User {requested_by} is not an approver for model {model_id}")

        # Update approval status
        old_status = approval["status"]
        approval["status"] = ApprovalStatus.NEEDS_REVISION.name

        # Add comment
        timestamp = datetime.datetime.now()
        if "comments" not in approval:
            approval["comments"] = []
        approval["comments"].append({
            "user": requested_by,
            "timestamp": timestamp.isoformat(),
            "comment": comment,
        })

        # Add history entry
        if "approval_history" not in approval:
            approval["approval_history"] = []
        approval["approval_history"].append({
            "timestamp": timestamp.isoformat(),
            "user": requested_by,
            "status": ApprovalStatus.NEEDS_REVISION.name,
            "comment": comment,
        })

        # Update model metadata in registry
        self.registry.update_model_metadata(model_id, model_metadata)

        # Log the status change
        log_model_approval_status_change(
            username=requested_by,
            model_id=model_id,
            model_name=model_name,
            old_status=old_status,
            new_status=ApprovalStatus.NEEDS_REVISION.name,
            comment=comment,
        )

        logger.info(f"Revision requested for model {model_id} by {requested_by}")

    def add_comment(self, model_id: str, user: str, comment: str) -> None:
        """Add a comment to a model approval request.

        Args:
            model_id: ID of the model.
            user: Username of the commenter.
            comment: Comment text.

        Raises:
            ValueError: If the model is not found or does not have an approval request.
        """
        # Get model metadata
        try:
            model_metadata = self.registry.get_model_metadata(model_id)
        except ValueError as e:
            logger.error(f"Error getting model metadata: {e}")
            raise

        # Check if model has an approval request
        if "approval" not in model_metadata:
            raise ValueError(f"Model {model_id} does not have an approval request")

        approval = model_metadata["approval"]

        # Add comment
        timestamp = datetime.datetime.now()
        if "comments" not in approval:
            approval["comments"] = []
        approval["comments"].append({
            "user": user,
            "timestamp": timestamp.isoformat(),
            "comment": comment,
        })

        # Update model metadata in registry
        self.registry.update_model_metadata(model_id, model_metadata)

        logger.info(f"Comment added to model {model_id} by {user}")

    def get_approval_request(self, model_id: str) -> Dict[str, Any]:
        """Get the approval request for a model.

        Args:
            model_id: ID of the model.

        Returns:
            Dict[str, Any]: The approval request.

        Raises:
            ValueError: If the model is not found or does not have an approval request.
        """
        # Get model metadata
        try:
            model_metadata = self.registry.get_model_metadata(model_id)
        except ValueError as e:
            logger.error(f"Error getting model metadata: {e}")
            raise

        # Check if model has an approval request
        if "approval" not in model_metadata:
            raise ValueError(f"Model {model_id} does not have an approval request")

        return model_metadata["approval"]

    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests.

        Returns:
            List[Dict[str, Any]]: List of pending approval requests.
        """
        pending_approvals = []

        # Get all models
        models = self.registry.get_models()

        # Filter models with pending approval requests
        for model in models:
            model_id = model["model_id"]
            try:
                model_metadata = self.registry.get_model_metadata(model_id)
                if "approval" in model_metadata and model_metadata["approval"]["status"] == ApprovalStatus.PENDING.name:
                    pending_approvals.append({
                        "model_id": model_id,
                        "model_name": model["model_name"],
                        "model_version": model_metadata["version"],
                        "approval": model_metadata["approval"],
                    })
            except ValueError:
                continue

        return pending_approvals

    def get_approvals_by_user(self, username: str) -> List[Dict[str, Any]]:
        """Get all approval requests for a user.

        Args:
            username: Username of the approver.

        Returns:
            List[Dict[str, Any]]: List of approval requests for the user.
        """
        user_approvals = []

        # Get all models
        models = self.registry.get_models()

        # Filter models with approval requests for the user
        for model in models:
            model_id = model["model_id"]
            try:
                model_metadata = self.registry.get_model_metadata(model_id)
                if "approval" in model_metadata and username in model_metadata["approval"]["approvers"]:
                    user_approvals.append({
                        "model_id": model_id,
                        "model_name": model["model_name"],
                        "model_version": model_metadata["version"],
                        "approval": model_metadata["approval"],
                    })
            except ValueError:
                continue

        return user_approvals