"""Access Control for Friday AI Trading System.

This module provides functionality for controlling access to system resources
based on user roles and permissions.
"""

import os
import json
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Any, Union
from dataclasses import dataclass, field

from ..logging import get_logger

# Create logger
logger = get_logger(__name__)


class Permission(Enum):
    """Permissions for system resources."""
    # Model operations
    MODEL_READ = "model:read"  # View model details and metadata
    MODEL_CREATE = "model:create"  # Register new models
    MODEL_UPDATE = "model:update"  # Update model metadata
    MODEL_DELETE = "model:delete"  # Delete models
    MODEL_DEPLOY = "model:deploy"  # Deploy models to production
    MODEL_SIGN = "model:sign"  # Sign models for authenticity
    MODEL_VERIFY = "model:verify"  # Verify model signatures
    MODEL_ENCRYPT = "model:encrypt"  # Encrypt sensitive models
    MODEL_DECRYPT = "model:decrypt"  # Decrypt encrypted models
    
    # Trading operations
    TRADE_READ = "trade:read"  # View trade details
    TRADE_CREATE = "trade:create"  # Create new trades
    TRADE_UPDATE = "trade:update"  # Update trade details
    TRADE_CANCEL = "trade:cancel"  # Cancel trades
    
    # System operations
    SYSTEM_ADMIN = "system:admin"  # Full system administration
    SYSTEM_CONFIG = "system:config"  # Configure system settings
    SYSTEM_MONITOR = "system:monitor"  # Monitor system status
    SYSTEM_AUDIT = "system:audit"  # View audit logs


class Role(Enum):
    """User roles in the system."""
    ADMIN = "admin"  # System administrator
    MODEL_DEVELOPER = "model_developer"  # Develops and tests models
    MODEL_REVIEWER = "model_reviewer"  # Reviews and approves models
    TRADER = "trader"  # Creates and manages trades
    RISK_MANAGER = "risk_manager"  # Monitors and manages risk
    AUDITOR = "auditor"  # Reviews audit logs and compliance
    VIEWER = "viewer"  # Read-only access to system


@dataclass
class AccessPolicy:
    """Access policy for controlling permissions."""
    role_permissions: Dict[Role, Set[Permission]] = field(default_factory=dict)
    user_roles: Dict[str, Set[Role]] = field(default_factory=dict)
    user_permissions: Dict[str, Set[Permission]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default role permissions."""
        # Admin role has all permissions
        admin_permissions = set(Permission)
        
        # Model developer permissions
        model_developer_permissions = {
            Permission.MODEL_READ,
            Permission.MODEL_CREATE,
            Permission.MODEL_UPDATE,
            Permission.SYSTEM_MONITOR
        }
        
        # Model reviewer permissions
        model_reviewer_permissions = {
            Permission.MODEL_READ,
            Permission.MODEL_UPDATE,
            Permission.MODEL_VERIFY,
            Permission.MODEL_DEPLOY,
            Permission.SYSTEM_MONITOR
        }
        
        # Trader permissions
        trader_permissions = {
            Permission.MODEL_READ,
            Permission.TRADE_READ,
            Permission.TRADE_CREATE,
            Permission.TRADE_UPDATE,
            Permission.TRADE_CANCEL,
            Permission.SYSTEM_MONITOR
        }
        
        # Risk manager permissions
        risk_manager_permissions = {
            Permission.MODEL_READ,
            Permission.TRADE_READ,
            Permission.TRADE_CANCEL,
            Permission.SYSTEM_MONITOR,
            Permission.SYSTEM_AUDIT
        }
        
        # Auditor permissions
        auditor_permissions = {
            Permission.MODEL_READ,
            Permission.TRADE_READ,
            Permission.SYSTEM_MONITOR,
            Permission.SYSTEM_AUDIT
        }
        
        # Viewer permissions
        viewer_permissions = {
            Permission.MODEL_READ,
            Permission.TRADE_READ,
            Permission.SYSTEM_MONITOR
        }
        
        # Set default role permissions
        self.role_permissions = {
            Role.ADMIN: admin_permissions,
            Role.MODEL_DEVELOPER: model_developer_permissions,
            Role.MODEL_REVIEWER: model_reviewer_permissions,
            Role.TRADER: trader_permissions,
            Role.RISK_MANAGER: risk_manager_permissions,
            Role.AUDITOR: auditor_permissions,
            Role.VIEWER: viewer_permissions
        }
    
    def add_role_to_user(self, username: str, role: Role) -> None:
        """Add a role to a user.
        
        Args:
            username: The username.
            role: The role to add.
        """
        if username not in self.user_roles:
            self.user_roles[username] = set()
        
        self.user_roles[username].add(role)
        logger.info(f"Added role {role.value} to user {username}")
    
    def remove_role_from_user(self, username: str, role: Role) -> None:
        """Remove a role from a user.
        
        Args:
            username: The username.
            role: The role to remove.
        """
        if username in self.user_roles and role in self.user_roles[username]:
            self.user_roles[username].remove(role)
            logger.info(f"Removed role {role.value} from user {username}")
    
    def add_permission_to_user(self, username: str, permission: Permission) -> None:
        """Add a specific permission to a user.
        
        Args:
            username: The username.
            permission: The permission to add.
        """
        if username not in self.user_permissions:
            self.user_permissions[username] = set()
        
        self.user_permissions[username].add(permission)
        logger.info(f"Added permission {permission.value} to user {username}")
    
    def remove_permission_from_user(self, username: str, permission: Permission) -> None:
        """Remove a specific permission from a user.
        
        Args:
            username: The username.
            permission: The permission to remove.
        """
        if username in self.user_permissions and permission in self.user_permissions[username]:
            self.user_permissions[username].remove(permission)
            logger.info(f"Removed permission {permission.value} from user {username}")
    
    def get_user_permissions(self, username: str) -> Set[Permission]:
        """Get all permissions for a user based on roles and specific permissions.
        
        Args:
            username: The username.
            
        Returns:
            Set[Permission]: The set of permissions for the user.
        """
        permissions = set()
        
        # Add permissions from roles
        if username in self.user_roles:
            for role in self.user_roles[username]:
                if role in self.role_permissions:
                    permissions.update(self.role_permissions[role])
        
        # Add specific user permissions
        if username in self.user_permissions:
            permissions.update(self.user_permissions[username])
        
        return permissions
    
    def has_permission(self, username: str, permission: Permission) -> bool:
        """Check if a user has a specific permission.
        
        Args:
            username: The username.
            permission: The permission to check.
            
        Returns:
            bool: True if the user has the permission, False otherwise.
        """
        user_permissions = self.get_user_permissions(username)
        return permission in user_permissions
    
    def save_to_file(self, file_path: str) -> None:
        """Save the access policy to a file.
        
        Args:
            file_path: Path to the file.
        """
        # Convert sets to lists for JSON serialization
        data = {
            "role_permissions": {role.value: [perm.value for perm in perms] 
                               for role, perms in self.role_permissions.items()},
            "user_roles": {user: [role.value for role in roles] 
                          for user, roles in self.user_roles.items()},
            "user_permissions": {user: [perm.value for perm in perms] 
                                for user, perms in self.user_permissions.items()}
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved access policy to {file_path}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'AccessPolicy':
        """Load an access policy from a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            AccessPolicy: The loaded access policy.
            
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Access policy file {file_path} not found")
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        policy = cls()
        
        # Convert lists back to sets
        policy.role_permissions = {Role(role): {Permission(perm) for perm in perms} 
                                  for role, perms in data["role_permissions"].items()}
        policy.user_roles = {user: {Role(role) for role in roles} 
                             for user, roles in data["user_roles"].items()}
        policy.user_permissions = {user: {Permission(perm) for perm in perms} 
                                   for user, perms in data["user_permissions"].items()}
        
        logger.info(f"Loaded access policy from {file_path}")
        
        return policy


class AccessControl:
    """Access control for system resources."""
    
    def __init__(self, policy_file: Optional[str] = None):
        """Initialize the access control system.
        
        Args:
            policy_file: Path to the access policy file. If None, a default policy will be used.
        """
        if policy_file and os.path.exists(policy_file):
            self.policy = AccessPolicy.load_from_file(policy_file)
        else:
            self.policy = AccessPolicy()
        
        self.policy_file = policy_file
        logger.info("Initialized AccessControl system")
    
    def check_permission(self, username: str, permission: Permission) -> bool:
        """Check if a user has a specific permission.
        
        Args:
            username: The username.
            permission: The permission to check.
            
        Returns:
            bool: True if the user has the permission, False otherwise.
        """
        has_perm = self.policy.has_permission(username, permission)
        if not has_perm:
            logger.warning(f"Access denied: User {username} does not have permission {permission.value}")
        return has_perm
    
    def require_permission(self, username: str, permission: Permission) -> None:
        """Require a specific permission for a user.
        
        Args:
            username: The username.
            permission: The required permission.
            
        Raises:
            PermissionError: If the user does not have the required permission.
        """
        if not self.check_permission(username, permission):
            error_msg = f"Permission denied: User {username} does not have required permission {permission.value}"
            logger.error(error_msg)
            raise PermissionError(error_msg)
    
    def save_policy(self) -> None:
        """Save the current access policy to the policy file."""
        if self.policy_file:
            self.policy.save_to_file(self.policy_file)
        else:
            logger.warning("No policy file specified, policy not saved")