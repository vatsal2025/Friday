"""Audit Logging for Friday AI Trading System.

This module provides functionality for logging security-sensitive operations
for auditing and compliance purposes.
"""

import os
import json
import time
import uuid
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import datetime

from ..logging import get_logger

# Create logger
logger = get_logger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    # Authentication events
    USER_LOGIN = "user:login"
    USER_LOGOUT = "user:logout"
    USER_LOGIN_FAILED = "user:login_failed"
    PASSWORD_CHANGED = "user:password_changed"
    
    # Authorization events
    PERMISSION_GRANTED = "permission:granted"
    PERMISSION_REVOKED = "permission:revoked"
    ROLE_ASSIGNED = "role:assigned"
    ROLE_REVOKED = "role:revoked"
    ACCESS_DENIED = "access:denied"
    
    # Model security events
    MODEL_CREATED = "model:created"
    MODEL_UPDATED = "model:updated"
    MODEL_DELETED = "model:deleted"
    MODEL_DEPLOYED = "model:deployed"
    MODEL_SIGNED = "model:signed"
    MODEL_SIGNATURE_VERIFIED = "model:signature_verified"
    MODEL_SIGNATURE_FAILED = "model:signature_failed"
    MODEL_ENCRYPTED = "model:encrypted"
    MODEL_DECRYPTED = "model:decrypted"
    MODEL_LOADED = "model:loaded"
    MODEL_ACCESS_DENIED = "model:access_denied"
    MODEL_VERSION_CHANGED = "model:version_changed"
    
    # Model approval events
    MODEL_APPROVAL_REQUESTED = "model:approval_requested"
    MODEL_APPROVED = "model:approved"
    MODEL_REJECTED = "model:rejected"
    MODEL_APPROVAL_STATUS_CHANGED = "model:approval_status_changed"
    
    # Model deployment tracking events
    MODEL_DEPLOYMENT_STARTED = "model:deployment_started"
    MODEL_DEPLOYMENT_COMPLETED = "model:deployment_completed"
    MODEL_DEPLOYMENT_FAILED = "model:deployment_failed"
    MODEL_UNDEPLOYED = "model:undeployed"
    MODEL_DEPLOYMENT_STATUS_CHANGED = "model:deployment_status_changed"
    
    # System events
    SYSTEM_STARTUP = "system:startup"
    SYSTEM_SHUTDOWN = "system:shutdown"
    CONFIG_CHANGED = "system:config_changed"
    SECURITY_ALERT = "system:security_alert"


@dataclass
class AuditEvent:
    """Represents an audit event."""
    event_type: AuditEventType
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: Optional[str] = None
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None
    action: Optional[str] = None
    status: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    source_ip: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the audit event to a dictionary.
        
        Returns:
            Dict[str, Any]: The audit event as a dictionary.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "datetime": datetime.datetime.fromtimestamp(self.timestamp).isoformat(),
            "username": self.username,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "action": self.action,
            "status": self.status,
            "details": self.details,
            "source_ip": self.source_ip
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create an audit event from a dictionary.
        
        Args:
            data: The dictionary containing the audit event data.
            
        Returns:
            AuditEvent: The created audit event.
        """
        event_type = AuditEventType(data["event_type"])
        
        # Create a new instance with required fields
        event = cls(
            event_type=event_type,
            timestamp=data["timestamp"],
            event_id=data["event_id"]
        )
        
        # Set optional fields if they exist
        if "username" in data:
            event.username = data["username"]
        if "resource_id" in data:
            event.resource_id = data["resource_id"]
        if "resource_type" in data:
            event.resource_type = data["resource_type"]
        if "action" in data:
            event.action = data["action"]
        if "status" in data:
            event.status = data["status"]
        if "details" in data:
            event.details = data["details"]
        if "source_ip" in data:
            event.source_ip = data["source_ip"]
        
        return event


class SecurityAuditLogger:
    """Logger for security-sensitive operations."""
    
    def __init__(self, log_file: Optional[str] = None, db_connection=None):
        """Initialize the security audit logger.
        
        Args:
            log_file: Path to the audit log file. If None, logs will only be stored in memory.
            db_connection: Database connection for storing audit logs. If None, logs will not be stored in a database.
        """
        self.log_file = log_file
        self.db_connection = db_connection
        self.events: List[AuditEvent] = []
        logger.info("Initialized SecurityAuditLogger")
    
    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event.
        
        Args:
            event: The audit event to log.
        """
        # Store the event in memory
        self.events.append(event)
        
        # Log to file if a log file is specified
        if self.log_file:
            self._write_to_file(event)
        
        # Log to database if a database connection is specified
        if self.db_connection:
            self._write_to_database(event)
        
        # Also log to the application logger
        logger.info(f"Audit event: {event.event_type.value} - {event.username or 'unknown'} - {event.resource_id or 'N/A'}")
    
    def _write_to_file(self, event: AuditEvent) -> None:
        """Write an audit event to the log file.
        
        Args:
            event: The audit event to write.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            
            # Append the event to the log file
            with open(self.log_file, "a") as f:
                json.dump(event.to_dict(), f)
                f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write audit event to file: {e}")
    
    def _write_to_database(self, event: AuditEvent) -> None:
        """Write an audit event to the database.
        
        Args:
            event: The audit event to write.
        """
        try:
            # Implementation depends on the database being used
            # This is a placeholder for actual database implementation
            pass
        except Exception as e:
            logger.error(f"Failed to write audit event to database: {e}")
    
    def get_events(self, 
                  event_type: Optional[AuditEventType] = None, 
                  username: Optional[str] = None,
                  resource_id: Optional[str] = None,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None) -> List[AuditEvent]:
        """Get audit events matching the specified criteria.
        
        Args:
            event_type: Filter by event type.
            username: Filter by username.
            resource_id: Filter by resource ID.
            start_time: Filter by start time (timestamp).
            end_time: Filter by end time (timestamp).
            
        Returns:
            List[AuditEvent]: The matching audit events.
        """
        filtered_events = self.events
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if username:
            filtered_events = [e for e in filtered_events if e.username == username]
        
        if resource_id:
            filtered_events = [e for e in filtered_events if e.resource_id == resource_id]
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        return filtered_events
    
    def clear_events(self) -> None:
        """Clear all events from memory."""
        self.events = []
        logger.info("Cleared all audit events from memory")


# Model-specific audit logging functions
def log_model_created(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str, 
                     model_type: str, source_ip: Optional[str] = None) -> None:
    """Log a model creation event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who created the model.
        model_id: The ID of the created model.
        model_name: The name of the created model.
        model_type: The type of the created model.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_CREATED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="create",
        status="success",
        details={
            "model_name": model_name,
            "model_type": model_type
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_updated(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                     changes: Dict[str, Any], source_ip: Optional[str] = None) -> None:
    """Log a model update event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who updated the model.
        model_id: The ID of the updated model.
        model_name: The name of the updated model.
        changes: The changes made to the model.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_UPDATED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="update",
        status="success",
        details={
            "model_name": model_name,
            "changes": changes
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_deleted(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                     source_ip: Optional[str] = None) -> None:
    """Log a model deletion event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who deleted the model.
        model_id: The ID of the deleted model.
        model_name: The name of the deleted model.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_DELETED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="delete",
        status="success",
        details={
            "model_name": model_name
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_deployed(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                      environment: str, source_ip: Optional[str] = None) -> None:
    """Log a model deployment event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who deployed the model.
        model_id: The ID of the deployed model.
        model_name: The name of the deployed model.
        environment: The environment where the model was deployed.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_DEPLOYED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="deploy",
        status="success",
        details={
            "model_name": model_name,
            "environment": environment
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_signed(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                    signature_id: str, source_ip: Optional[str] = None) -> None:
    """Log a model signing event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who signed the model.
        model_id: The ID of the signed model.
        model_name: The name of the signed model.
        signature_id: The ID of the signature.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_SIGNED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="sign",
        status="success",
        details={
            "model_name": model_name,
            "signature_id": signature_id
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_signature_verified(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                               signature_id: str, source_ip: Optional[str] = None) -> None:
    """Log a model signature verification event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who verified the model signature.
        model_id: The ID of the model.
        model_name: The name of the model.
        signature_id: The ID of the signature.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_SIGNATURE_VERIFIED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="verify_signature",
        status="success",
        details={
            "model_name": model_name,
            "signature_id": signature_id
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_signature_failed(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                             signature_id: str, reason: str, source_ip: Optional[str] = None) -> None:
    """Log a failed model signature verification event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who attempted to verify the model signature.
        model_id: The ID of the model.
        model_name: The name of the model.
        signature_id: The ID of the signature.
        reason: The reason for the verification failure.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_SIGNATURE_FAILED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="verify_signature",
        status="failed",
        details={
            "model_name": model_name,
            "signature_id": signature_id,
            "reason": reason
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_encrypted(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                       source_ip: Optional[str] = None) -> None:
    """Log a model encryption event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who encrypted the model.
        model_id: The ID of the encrypted model.
        model_name: The name of the encrypted model.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_ENCRYPTED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="encrypt",
        status="success",
        details={
            "model_name": model_name
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_decrypted(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                       source_ip: Optional[str] = None) -> None:
    """Log a model decryption event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who decrypted the model.
        model_id: The ID of the decrypted model.
        model_name: The name of the decrypted model.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_DECRYPTED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="decrypt",
        status="success",
        details={
            "model_name": model_name
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_loaded(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                    environment: str, source_ip: Optional[str] = None) -> None:
    """Log a model loading event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who loaded the model.
        model_id: The ID of the loaded model.
        model_name: The name of the loaded model.
        environment: The environment where the model was loaded.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_LOADED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="load",
        status="success",
        details={
            "model_name": model_name,
            "environment": environment
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_access_denied(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                          action: str, reason: str, source_ip: Optional[str] = None) -> None:
    """Log a model access denied event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who was denied access.
        model_id: The ID of the model.
        model_name: The name of the model.
        action: The action that was denied.
        reason: The reason for the denial.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_ACCESS_DENIED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action=action,
        status="denied",
        details={
            "model_name": model_name,
            "reason": reason
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


# Model approval workflow logging functions
def log_model_approval_requested(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                               approvers: List[str], comment: Optional[str] = None, source_ip: Optional[str] = None) -> None:
    """Log a model approval request event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who requested approval.
        model_id: The ID of the model.
        model_name: The name of the model.
        approvers: List of usernames who can approve the request.
        comment: Optional comment on the request.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_APPROVAL_REQUESTED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="request_approval",
        status="pending",
        details={
            "model_name": model_name,
            "approvers": approvers,
            "comment": comment
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_approved(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                      comment: Optional[str] = None, source_ip: Optional[str] = None) -> None:
    """Log a model approval event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who approved the model.
        model_id: The ID of the model.
        model_name: The name of the model.
        comment: Optional comment on the approval.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_APPROVED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="approve",
        status="success",
        details={
            "model_name": model_name,
            "comment": comment
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_rejected(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                      comment: Optional[str] = None, source_ip: Optional[str] = None) -> None:
    """Log a model rejection event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who rejected the model.
        model_id: The ID of the model.
        model_name: The name of the model.
        comment: Optional comment on the rejection.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_REJECTED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="reject",
        status="rejected",
        details={
            "model_name": model_name,
            "comment": comment
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_approval_status_change(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                                   old_status: str, new_status: str, comment: Optional[str] = None,
                                   source_ip: Optional[str] = None) -> None:
    """Log a model approval status change event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who changed the status.
        model_id: The ID of the model.
        model_name: The name of the model.
        old_status: The old approval status.
        new_status: The new approval status.
        comment: Optional comment on the status change.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_APPROVAL_STATUS_CHANGED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="change_approval_status",
        status=new_status,
        details={
            "model_name": model_name,
            "old_status": old_status,
            "new_status": new_status,
            "comment": comment
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


# Model deployment tracking logging functions
def log_model_deployment_started(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                              environment: str, deployment_id: str, source_ip: Optional[str] = None) -> None:
    """Log a model deployment started event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who started the deployment.
        model_id: The ID of the model.
        model_name: The name of the model.
        environment: The environment where the model is being deployed.
        deployment_id: The ID of the deployment.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_DEPLOYMENT_STARTED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="start_deployment",
        status="in_progress",
        details={
            "model_name": model_name,
            "environment": environment,
            "deployment_id": deployment_id
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_deployment_completed(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                                environment: str, deployment_id: str, deployment_url: Optional[str] = None,
                                source_ip: Optional[str] = None) -> None:
    """Log a model deployment completed event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who completed the deployment.
        model_id: The ID of the model.
        model_name: The name of the model.
        environment: The environment where the model was deployed.
        deployment_id: The ID of the deployment.
        deployment_url: The URL where the deployed model is accessible.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_DEPLOYMENT_COMPLETED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="complete_deployment",
        status="success",
        details={
            "model_name": model_name,
            "environment": environment,
            "deployment_id": deployment_id,
            "deployment_url": deployment_url
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_deployment_failed(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                             environment: str, deployment_id: str, reason: str, source_ip: Optional[str] = None) -> None:
    """Log a model deployment failed event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who attempted the deployment.
        model_id: The ID of the model.
        model_name: The name of the model.
        environment: The environment where the model was being deployed.
        deployment_id: The ID of the deployment.
        reason: The reason for the failure.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_DEPLOYMENT_FAILED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="deploy",
        status="failed",
        details={
            "model_name": model_name,
            "environment": environment,
            "deployment_id": deployment_id,
            "reason": reason
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_undeployed(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                       environment: str, deployment_id: str, source_ip: Optional[str] = None) -> None:
    """Log a model undeployment event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who undeployed the model.
        model_id: The ID of the model.
        model_name: The name of the model.
        environment: The environment from which the model was undeployed.
        deployment_id: The ID of the deployment.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_UNDEPLOYED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="undeploy",
        status="success",
        details={
            "model_name": model_name,
            "environment": environment,
            "deployment_id": deployment_id
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_deployment_status_changed(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                                      environment: str, deployment_id: str, old_status: str, new_status: str,
                                      source_ip: Optional[str] = None) -> None:
    """Log a model deployment status change event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who changed the status.
        model_id: The ID of the model.
        model_name: The name of the model.
        environment: The environment where the model is deployed.
        deployment_id: The ID of the deployment.
        old_status: The old deployment status.
        new_status: The new deployment status.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_DEPLOYMENT_STATUS_CHANGED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="change_deployment_status",
        status=new_status,
        details={
            "model_name": model_name,
            "environment": environment,
            "deployment_id": deployment_id,
            "old_status": old_status,
            "new_status": new_status
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_version_changed(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                            old_version: str, new_version: str, change_type: str, changes: Dict[str, Any],
                            source_ip: Optional[str] = None) -> None:
    """Log a model version change event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who changed the model version.
        model_id: The ID of the model.
        model_name: The name of the model.
        old_version: The old semantic version (e.g., "1.0.0").
        new_version: The new semantic version (e.g., "1.1.0").
        change_type: The type of version change (MAJOR, MINOR, PATCH).
        changes: Dictionary containing details about what changed.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_VERSION_CHANGED,
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="change_version",
        status="success",
        details={
            "model_name": model_name,
            "old_version": old_version,
            "new_version": new_version,
            "change_type": change_type,
            "changes": changes
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)


def log_model_loaded(audit_logger: SecurityAuditLogger, username: str, model_id: str, model_name: str,
                    model_version: str, environment: str, source_ip: Optional[str] = None) -> None:
    """Log a model loaded event.
    
    Args:
        audit_logger: The security audit logger.
        username: The username of the user who loaded the model.
        model_id: The ID of the model.
        model_name: The name of the model.
        model_version: The version of the model.
        environment: The environment where the model was loaded.
        source_ip: The source IP address of the request.
    """
    event = AuditEvent(
        event_type=AuditEventType.MODEL_DEPLOYED,  # Using MODEL_DEPLOYED as a close match
        username=username,
        resource_id=model_id,
        resource_type="model",
        action="load",
        status="success",
        details={
            "model_name": model_name,
            "model_version": model_version,
            "environment": environment,
            "operation": "load"
        },
        source_ip=source_ip
    )
    audit_logger.log_event(event)