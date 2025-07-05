"""Webhook integration for external systems.

This module provides utilities for creating, managing, and processing webhooks
for integration with external systems, including webhook registration, validation,
and event handling.
"""

from typing import Dict, List, Optional, Callable, Any, Union, Tuple, Set
import time
import threading
import logging
from datetime import datetime, timedelta
from enum import Enum
import json
import os
import hashlib
import hmac
import uuid
import re
from urllib.parse import urlparse

from src.infrastructure.logging import get_logger
from src.infrastructure.error import FridayError
from src.infrastructure.event import EventSystem, Event
from src.infrastructure.config import ConfigManager
from src.infrastructure.security import generate_random_key

# Create logger
logger = get_logger(__name__)


class WebhookError(FridayError):
    """Exception raised for errors in webhook processing."""
    pass


class WebhookDirection(Enum):
    """Direction of webhook communication."""
    INBOUND = 'inbound'  # External system to Friday
    OUTBOUND = 'outbound'  # Friday to external system


class WebhookContentType(Enum):
    """Content type for webhook payloads."""
    JSON = 'application/json'
    FORM = 'application/x-www-form-urlencoded'
    XML = 'application/xml'
    TEXT = 'text/plain'


class WebhookSecurityType(Enum):
    """Security mechanism for webhooks."""
    NONE = 'none'
    BASIC_AUTH = 'basic_auth'
    API_KEY = 'api_key'
    HMAC = 'hmac'
    JWT = 'jwt'
    OAUTH = 'oauth'


class WebhookStatus(Enum):
    """Status of a webhook."""
    ACTIVE = 'active'
    INACTIVE = 'inactive'
    FAILED = 'failed'
    PENDING = 'pending'


class WebhookEvent(Event):
    """Event for webhook processing."""
    
    def __init__(self, webhook_id: str, payload: Dict[str, Any]):
        """Initialize a webhook event.
        
        Args:
            webhook_id: The ID of the webhook.
            payload: The webhook payload.
        """
        super().__init__(f"webhook.{webhook_id}")
        self.webhook_id = webhook_id
        self.payload = payload
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary.
        
        Returns:
            Dict[str, Any]: The event as a dictionary.
        """
        event_dict = super().to_dict()
        event_dict['webhook_id'] = self.webhook_id
        event_dict['payload'] = self.payload
        return event_dict


class WebhookReceivedEvent(WebhookEvent):
    """Event for a received webhook."""
    
    def __init__(self, webhook_id: str, payload: Dict[str, Any]):
        """Initialize a webhook received event.
        
        Args:
            webhook_id: The ID of the webhook.
            payload: The webhook payload.
        """
        super().__init__(webhook_id, payload)
        self.event_type = f"webhook.received.{webhook_id}"


class WebhookSentEvent(WebhookEvent):
    """Event for a sent webhook."""
    
    def __init__(self, webhook_id: str, payload: Dict[str, Any], response: Dict[str, Any]):
        """Initialize a webhook sent event.
        
        Args:
            webhook_id: The ID of the webhook.
            payload: The webhook payload.
            response: The response from the external system.
        """
        super().__init__(webhook_id, payload)
        self.event_type = f"webhook.sent.{webhook_id}"
        self.response = response
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary.
        
        Returns:
            Dict[str, Any]: The event as a dictionary.
        """
        event_dict = super().to_dict()
        event_dict['response'] = self.response
        return event_dict


class WebhookFailedEvent(WebhookEvent):
    """Event for a failed webhook."""
    
    def __init__(self, webhook_id: str, payload: Dict[str, Any], error: str):
        """Initialize a webhook failed event.
        
        Args:
            webhook_id: The ID of the webhook.
            payload: The webhook payload.
            error: The error message.
        """
        super().__init__(webhook_id, payload)
        self.event_type = f"webhook.failed.{webhook_id}"
        self.error = error
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary.
        
        Returns:
            Dict[str, Any]: The event as a dictionary.
        """
        event_dict = super().to_dict()
        event_dict['error'] = self.error
        return event_dict


class WebhookConfig:
    """Configuration for a webhook."""
    
    def __init__(self, webhook_id: str, name: str, system_id: str,
                 direction: WebhookDirection, url: str,
                 events: List[str], content_type: WebhookContentType = WebhookContentType.JSON,
                 security_type: WebhookSecurityType = WebhookSecurityType.NONE,
                 security_config: Optional[Dict[str, Any]] = None,
                 headers: Optional[Dict[str, str]] = None,
                 retry_config: Optional[Dict[str, Any]] = None,
                 transform_config: Optional[Dict[str, Any]] = None):
        """Initialize a webhook configuration.
        
        Args:
            webhook_id: The ID of the webhook.
            name: The name of the webhook.
            system_id: The ID of the external system.
            direction: The direction of the webhook.
            url: The URL for the webhook.
            events: The events that trigger the webhook.
            content_type: The content type for webhook payloads.
            security_type: The security mechanism for the webhook.
            security_config: Configuration for the security mechanism.
            headers: Additional headers to include in webhook requests.
            retry_config: Configuration for retry behavior.
            transform_config: Configuration for payload transformation.
        """
        self.webhook_id = webhook_id
        self.name = name
        self.system_id = system_id
        self.direction = direction
        self.url = url
        self.events = events
        self.content_type = content_type
        self.security_type = security_type
        self.security_config = security_config or {}
        self.headers = headers or {}
        self.retry_config = retry_config or {
            'max_retries': 3,
            'retry_delay': 5,
            'backoff_factor': 2.0
        }
        self.transform_config = transform_config or {}
        self.status = WebhookStatus.ACTIVE
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: The configuration as a dictionary.
        """
        return {
            'webhook_id': self.webhook_id,
            'name': self.name,
            'system_id': self.system_id,
            'direction': self.direction.value,
            'url': self.url,
            'events': self.events,
            'content_type': self.content_type.value,
            'security_type': self.security_type.value,
            'security_config': self.security_config,
            'headers': self.headers,
            'retry_config': self.retry_config,
            'transform_config': self.transform_config,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WebhookConfig':
        """Create a webhook configuration from a dictionary.
        
        Args:
            config_dict: The dictionary containing the configuration.
            
        Returns:
            WebhookConfig: The created webhook configuration.
        """
        webhook = cls(
            webhook_id=config_dict['webhook_id'],
            name=config_dict['name'],
            system_id=config_dict['system_id'],
            direction=WebhookDirection(config_dict['direction']),
            url=config_dict['url'],
            events=config_dict['events'],
            content_type=WebhookContentType(config_dict['content_type']),
            security_type=WebhookSecurityType(config_dict['security_type']),
            security_config=config_dict.get('security_config', {}),
            headers=config_dict.get('headers', {}),
            retry_config=config_dict.get('retry_config', {}),
            transform_config=config_dict.get('transform_config', {})
        )
        
        webhook.status = WebhookStatus(config_dict['status'])
        webhook.created_at = datetime.fromisoformat(config_dict['created_at'])
        webhook.updated_at = datetime.fromisoformat(config_dict['updated_at'])
        
        return webhook


class WebhookDelivery:
    """Record of a webhook delivery."""
    
    def __init__(self, delivery_id: str, webhook_id: str, event_type: str,
                 payload: Dict[str, Any], timestamp: datetime):
        """Initialize a webhook delivery record.
        
        Args:
            delivery_id: The ID of the delivery.
            webhook_id: The ID of the webhook.
            event_type: The type of event that triggered the webhook.
            payload: The webhook payload.
            timestamp: The timestamp of the delivery.
        """
        self.delivery_id = delivery_id
        self.webhook_id = webhook_id
        self.event_type = event_type
        self.payload = payload
        self.timestamp = timestamp
        self.status = 'pending'
        self.response: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.retry_count = 0
        self.completed_at: Optional[datetime] = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the delivery record to a dictionary.
        
        Returns:
            Dict[str, Any]: The delivery record as a dictionary.
        """
        return {
            'delivery_id': self.delivery_id,
            'webhook_id': self.webhook_id,
            'event_type': self.event_type,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status,
            'response': self.response,
            'error': self.error,
            'retry_count': self.retry_count,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }
        
    @classmethod
    def from_dict(cls, delivery_dict: Dict[str, Any]) -> 'WebhookDelivery':
        """Create a webhook delivery record from a dictionary.
        
        Args:
            delivery_dict: The dictionary containing the delivery record.
            
        Returns:
            WebhookDelivery: The created webhook delivery record.
        """
        delivery = cls(
            delivery_id=delivery_dict['delivery_id'],
            webhook_id=delivery_dict['webhook_id'],
            event_type=delivery_dict['event_type'],
            payload=delivery_dict['payload'],
            timestamp=datetime.fromisoformat(delivery_dict['timestamp'])
        )
        
        delivery.status = delivery_dict['status']
        delivery.response = delivery_dict.get('response')
        delivery.error = delivery_dict.get('error')
        delivery.retry_count = delivery_dict.get('retry_count', 0)
        
        if delivery_dict.get('completed_at'):
            delivery.completed_at = datetime.fromisoformat(delivery_dict['completed_at'])
            
        return delivery


class WebhookValidator:
    """Base class for webhook validators."""
    
    def validate(self, webhook_id: str, headers: Dict[str, str], payload: Any) -> bool:
        """Validate a webhook request.
        
        Args:
            webhook_id: The ID of the webhook.
            headers: The request headers.
            payload: The request payload.
            
        Returns:
            bool: True if the webhook is valid, False otherwise.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement validate()")


class HmacValidator(WebhookValidator):
    """Validator for HMAC-signed webhooks."""
    
    def __init__(self, secret: str, signature_header: str = 'X-Signature',
                 algorithm: str = 'sha256'):
        """Initialize an HMAC validator.
        
        Args:
            secret: The secret key for HMAC validation.
            signature_header: The header containing the signature.
            algorithm: The HMAC algorithm to use.
        """
        self.secret = secret
        self.signature_header = signature_header
        self.algorithm = algorithm
        
    def validate(self, webhook_id: str, headers: Dict[str, str], payload: Any) -> bool:
        """Validate a webhook request using HMAC.
        
        Args:
            webhook_id: The ID of the webhook.
            headers: The request headers.
            payload: The request payload.
            
        Returns:
            bool: True if the webhook is valid, False otherwise.
        """
        # Get the signature from headers
        signature = headers.get(self.signature_header)
        if not signature:
            logger.warning(f"Missing signature header {self.signature_header} for webhook {webhook_id}")
            return False
            
        # Convert payload to bytes if it's not already
        if isinstance(payload, dict):
            payload_bytes = json.dumps(payload).encode()
        elif isinstance(payload, str):
            payload_bytes = payload.encode()
        else:
            payload_bytes = payload
            
        # Calculate the expected signature
        if self.algorithm == 'sha256':
            digest = hmac.new(self.secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
        elif self.algorithm == 'sha1':
            digest = hmac.new(self.secret.encode(), payload_bytes, hashlib.sha1).hexdigest()
        else:
            logger.warning(f"Unsupported HMAC algorithm {self.algorithm} for webhook {webhook_id}")
            return False
            
        # Compare signatures
        return hmac.compare_digest(signature, digest)


class BasicAuthValidator(WebhookValidator):
    """Validator for Basic Authentication webhooks."""
    
    def __init__(self, username: str, password: str):
        """Initialize a Basic Authentication validator.
        
        Args:
            username: The expected username.
            password: The expected password.
        """
        self.username = username
        self.password = password
        
    def validate(self, webhook_id: str, headers: Dict[str, str], payload: Any) -> bool:
        """Validate a webhook request using Basic Authentication.
        
        Args:
            webhook_id: The ID of the webhook.
            headers: The request headers.
            payload: The request payload.
            
        Returns:
            bool: True if the webhook is valid, False otherwise.
        """
        import base64
        
        # Get the Authorization header
        auth_header = headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Basic '):
            logger.warning(f"Missing or invalid Authorization header for webhook {webhook_id}")
            return False
            
        # Decode the credentials
        try:
            encoded_credentials = auth_header[6:]  # Remove 'Basic '
            decoded_credentials = base64.b64decode(encoded_credentials).decode('utf-8')
            username, password = decoded_credentials.split(':', 1)
            
            # Compare credentials
            return username == self.username and password == self.password
        except Exception as e:
            logger.warning(f"Failed to decode Basic Authentication credentials for webhook {webhook_id}: {str(e)}")
            return False


class ApiKeyValidator(WebhookValidator):
    """Validator for API key webhooks."""
    
    def __init__(self, api_key: str, key_header: str = 'X-API-Key'):
        """Initialize an API key validator.
        
        Args:
            api_key: The expected API key.
            key_header: The header containing the API key.
        """
        self.api_key = api_key
        self.key_header = key_header
        
    def validate(self, webhook_id: str, headers: Dict[str, str], payload: Any) -> bool:
        """Validate a webhook request using an API key.
        
        Args:
            webhook_id: The ID of the webhook.
            headers: The request headers.
            payload: The request payload.
            
        Returns:
            bool: True if the webhook is valid, False otherwise.
        """
        # Get the API key from headers
        api_key = headers.get(self.key_header)
        if not api_key:
            logger.warning(f"Missing API key header {self.key_header} for webhook {webhook_id}")
            return False
            
        # Compare API keys
        return api_key == self.api_key


class JwtValidator(WebhookValidator):
    """Validator for JWT-signed webhooks."""
    
    def __init__(self, secret: str, token_header: str = 'Authorization',
                 token_prefix: str = 'Bearer '):
        """Initialize a JWT validator.
        
        Args:
            secret: The secret key for JWT validation.
            token_header: The header containing the JWT.
            token_prefix: The prefix for the JWT in the header.
        """
        self.secret = secret
        self.token_header = token_header
        self.token_prefix = token_prefix
        
    def validate(self, webhook_id: str, headers: Dict[str, str], payload: Any) -> bool:
        """Validate a webhook request using JWT.
        
        Args:
            webhook_id: The ID of the webhook.
            headers: The request headers.
            payload: The request payload.
            
        Returns:
            bool: True if the webhook is valid, False otherwise.
        """
        try:
            import jwt
            
            # Get the JWT from headers
            token_header = headers.get(self.token_header)
            if not token_header or not token_header.startswith(self.token_prefix):
                logger.warning(f"Missing or invalid JWT header {self.token_header} for webhook {webhook_id}")
                return False
                
            # Extract the token
            token = token_header[len(self.token_prefix):]
            
            # Verify the token
            try:
                jwt.decode(token, self.secret, algorithms=['HS256'])
                return True
            except jwt.InvalidTokenError as e:
                logger.warning(f"Invalid JWT for webhook {webhook_id}: {str(e)}")
                return False
        except ImportError:
            logger.warning("PyJWT package not found, JWT validation skipped")
            return False


class WebhookTransformer:
    """Base class for webhook payload transformers."""
    
    def transform(self, payload: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a webhook payload.
        
        Args:
            payload: The webhook payload.
            config: The transformation configuration.
            
        Returns:
            Dict[str, Any]: The transformed payload.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement transform()")


class JsonPathTransformer(WebhookTransformer):
    """Transformer for JSON Path transformations."""
    
    def transform(self, payload: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a webhook payload using JSON Path.
        
        Args:
            payload: The webhook payload.
            config: The transformation configuration.
            
        Returns:
            Dict[str, Any]: The transformed payload.
        """
        try:
            from jsonpath_ng import parse
            
            result = {}
            
            # Apply each mapping
            for target_key, path_expr in config.get('mappings', {}).items():
                try:
                    # Parse the JSON Path expression
                    jsonpath_expr = parse(path_expr)
                    
                    # Find all matches
                    matches = jsonpath_expr.find(payload)
                    
                    if matches:
                        # Use the first match
                        result[target_key] = matches[0].value
                except Exception as e:
                    logger.warning(f"Failed to apply JSON Path {path_expr}: {str(e)}")
                    
            # Add static values
            for key, value in config.get('static', {}).items():
                result[key] = value
                
            return result
        except ImportError:
            logger.warning("jsonpath-ng package not found, using identity transformation")
            return payload


class TemplateTransformer(WebhookTransformer):
    """Transformer for template-based transformations."""
    
    def transform(self, payload: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a webhook payload using templates.
        
        Args:
            payload: The webhook payload.
            config: The transformation configuration.
            
        Returns:
            Dict[str, Any]: The transformed payload.
        """
        try:
            import jinja2
            
            result = {}
            
            # Create Jinja2 environment
            env = jinja2.Environment()
            
            # Apply each template
            for target_key, template_str in config.get('templates', {}).items():
                try:
                    # Compile the template
                    template = env.from_string(template_str)
                    
                    # Render the template with the payload as context
                    result[target_key] = template.render(**payload)
                except Exception as e:
                    logger.warning(f"Failed to apply template for {target_key}: {str(e)}")
                    
            # Add static values
            for key, value in config.get('static', {}).items():
                result[key] = value
                
            return result
        except ImportError:
            logger.warning("jinja2 package not found, using identity transformation")
            return payload


class WebhookProcessor:
    """Processor for webhook requests and deliveries."""
    
    def __init__(self, config: WebhookConfig, event_system: Optional[EventSystem] = None):
        """Initialize a webhook processor.
        
        Args:
            config: The webhook configuration.
            event_system: The event system for publishing events.
        """
        self.config = config
        self.event_system = event_system
        self.validator = self._create_validator()
        self.transformer = self._create_transformer()
        
    def _create_validator(self) -> Optional[WebhookValidator]:
        """Create a validator based on the security configuration.
        
        Returns:
            Optional[WebhookValidator]: The created validator, or None if no validation is needed.
        """
        if self.config.security_type == WebhookSecurityType.NONE:
            return None
            
        security_config = self.config.security_config
        
        if self.config.security_type == WebhookSecurityType.HMAC:
            return HmacValidator(
                secret=security_config.get('secret', ''),
                signature_header=security_config.get('signature_header', 'X-Signature'),
                algorithm=security_config.get('algorithm', 'sha256')
            )
        elif self.config.security_type == WebhookSecurityType.BASIC_AUTH:
            return BasicAuthValidator(
                username=security_config.get('username', ''),
                password=security_config.get('password', '')
            )
        elif self.config.security_type == WebhookSecurityType.API_KEY:
            return ApiKeyValidator(
                api_key=security_config.get('api_key', ''),
                key_header=security_config.get('key_header', 'X-API-Key')
            )
        elif self.config.security_type == WebhookSecurityType.JWT:
            return JwtValidator(
                secret=security_config.get('secret', ''),
                token_header=security_config.get('token_header', 'Authorization'),
                token_prefix=security_config.get('token_prefix', 'Bearer ')
            )
            
        return None
        
    def _create_transformer(self) -> WebhookTransformer:
        """Create a transformer based on the transformation configuration.
        
        Returns:
            WebhookTransformer: The created transformer.
        """
        transform_type = self.config.transform_config.get('type', 'jsonpath')
        
        if transform_type == 'jsonpath':
            return JsonPathTransformer()
        elif transform_type == 'template':
            return TemplateTransformer()
            
        # Default to JSON Path transformer
        return JsonPathTransformer()
        
    def validate_request(self, headers: Dict[str, str], payload: Any) -> bool:
        """Validate a webhook request.
        
        Args:
            headers: The request headers.
            payload: The request payload.
            
        Returns:
            bool: True if the webhook is valid, False otherwise.
        """
        if self.validator is None:
            return True
            
        return self.validator.validate(self.config.webhook_id, headers, payload)
        
    def transform_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a webhook payload.
        
        Args:
            payload: The webhook payload.
            
        Returns:
            Dict[str, Any]: The transformed payload.
        """
        if not self.transformer or not self.config.transform_config:
            return payload
            
        return self.transformer.transform(payload, self.config.transform_config)
        
    def process_inbound(self, headers: Dict[str, str], payload: Any) -> Tuple[bool, Dict[str, Any]]:
        """Process an inbound webhook request.
        
        Args:
            headers: The request headers.
            payload: The request payload.
            
        Returns:
            Tuple[bool, Dict[str, Any]]: A tuple of (success, transformed_payload).
        """
        # Validate the request
        if not self.validate_request(headers, payload):
            logger.warning(f"Invalid webhook request for {self.config.webhook_id}")
            return False, {}
            
        # Transform the payload
        transformed_payload = self.transform_payload(payload)
        
        # Publish event
        if self.event_system:
            self.event_system.publish(WebhookReceivedEvent(
                self.config.webhook_id, transformed_payload
            ))
            
        return True, transformed_payload
        
    def prepare_outbound(self, event_type: str, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Prepare an outbound webhook delivery.
        
        Args:
            event_type: The type of event that triggered the webhook.
            payload: The event payload.
            
        Returns:
            Tuple[str, Dict[str, Any]]: A tuple of (delivery_id, transformed_payload).
        """
        # Generate a delivery ID
        delivery_id = str(uuid.uuid4())
        
        # Transform the payload
        transformed_payload = self.transform_payload(payload)
        
        return delivery_id, transformed_payload


class WebhookManager:
    """Manager for webhooks."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None,
                 event_system: Optional[EventSystem] = None):
        """Initialize a webhook manager.
        
        Args:
            config_manager: The configuration manager.
            event_system: The event system.
        """
        self.config_manager = config_manager
        self.event_system = event_system
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.processors: Dict[str, WebhookProcessor] = {}
        self.deliveries: Dict[str, WebhookDelivery] = {}
        self.delivery_queue: List[str] = []
        self.running = False
        self.delivery_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
    def register_webhook(self, config: WebhookConfig) -> str:
        """Register a webhook.
        
        Args:
            config: The webhook configuration.
            
        Returns:
            str: The webhook ID.
            
        Raises:
            WebhookError: If the webhook ID is already registered.
        """
        with self.lock:
            if config.webhook_id in self.webhooks:
                raise WebhookError(f"Webhook ID {config.webhook_id} is already registered")
                
            # Validate the URL
            if not self._validate_url(config.url):
                raise WebhookError(f"Invalid URL: {config.url}")
                
            # Add the webhook
            self.webhooks[config.webhook_id] = config
            
            # Create a processor
            self.processors[config.webhook_id] = WebhookProcessor(config, self.event_system)
            
            logger.info(f"Registered webhook {config.name} ({config.webhook_id})")
            
            return config.webhook_id
            
    def unregister_webhook(self, webhook_id: str) -> bool:
        """Unregister a webhook.
        
        Args:
            webhook_id: The ID of the webhook to unregister.
            
        Returns:
            bool: True if the webhook was unregistered, False otherwise.
        """
        with self.lock:
            if webhook_id not in self.webhooks:
                return False
                
            # Remove the webhook
            del self.webhooks[webhook_id]
            
            # Remove the processor
            if webhook_id in self.processors:
                del self.processors[webhook_id]
                
            logger.info(f"Unregistered webhook {webhook_id}")
            
            return True
            
    def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get a webhook configuration.
        
        Args:
            webhook_id: The ID of the webhook.
            
        Returns:
            Optional[WebhookConfig]: The webhook configuration, or None if not found.
        """
        with self.lock:
            return self.webhooks.get(webhook_id)
            
    def get_all_webhooks(self) -> Dict[str, WebhookConfig]:
        """Get all webhook configurations.
        
        Returns:
            Dict[str, WebhookConfig]: All webhook configurations.
        """
        with self.lock:
            return self.webhooks.copy()
            
    def update_webhook(self, webhook_id: str, **kwargs) -> bool:
        """Update a webhook configuration.
        
        Args:
            webhook_id: The ID of the webhook to update.
            **kwargs: The configuration options to update.
            
        Returns:
            bool: True if the webhook was updated, False otherwise.
        """
        with self.lock:
            webhook = self.webhooks.get(webhook_id)
            if webhook is None:
                return False
                
            # Update the webhook
            for key, value in kwargs.items():
                if hasattr(webhook, key):
                    setattr(webhook, key, value)
                    
            # Update the timestamp
            webhook.updated_at = datetime.now()
            
            # Recreate the processor
            self.processors[webhook_id] = WebhookProcessor(webhook, self.event_system)
            
            logger.info(f"Updated webhook {webhook.name} ({webhook_id})")
            
            return True
            
    def process_inbound(self, webhook_id: str, headers: Dict[str, str], payload: Any) -> Tuple[bool, Dict[str, Any]]:
        """Process an inbound webhook request.
        
        Args:
            webhook_id: The ID of the webhook.
            headers: The request headers.
            payload: The request payload.
            
        Returns:
            Tuple[bool, Dict[str, Any]]: A tuple of (success, transformed_payload).
            
        Raises:
            WebhookError: If the webhook is not found or is not an inbound webhook.
        """
        with self.lock:
            webhook = self.webhooks.get(webhook_id)
            if webhook is None:
                raise WebhookError(f"Webhook {webhook_id} not found")
                
            if webhook.direction != WebhookDirection.INBOUND:
                raise WebhookError(f"Webhook {webhook_id} is not an inbound webhook")
                
            processor = self.processors.get(webhook_id)
            if processor is None:
                raise WebhookError(f"Processor for webhook {webhook_id} not found")
                
            return processor.process_inbound(headers, payload)
            
    def queue_outbound(self, webhook_id: str, event_type: str, payload: Dict[str, Any]) -> str:
        """Queue an outbound webhook delivery.
        
        Args:
            webhook_id: The ID of the webhook.
            event_type: The type of event that triggered the webhook.
            payload: The event payload.
            
        Returns:
            str: The delivery ID.
            
        Raises:
            WebhookError: If the webhook is not found or is not an outbound webhook.
        """
        with self.lock:
            webhook = self.webhooks.get(webhook_id)
            if webhook is None:
                raise WebhookError(f"Webhook {webhook_id} not found")
                
            if webhook.direction != WebhookDirection.OUTBOUND:
                raise WebhookError(f"Webhook {webhook_id} is not an outbound webhook")
                
            # Check if the event type matches
            if event_type not in webhook.events and '*' not in webhook.events:
                logger.debug(f"Event type {event_type} does not match webhook {webhook_id} events")
                return ""
                
            processor = self.processors.get(webhook_id)
            if processor is None:
                raise WebhookError(f"Processor for webhook {webhook_id} not found")
                
            # Prepare the delivery
            delivery_id, transformed_payload = processor.prepare_outbound(event_type, payload)
            
            # Create a delivery record
            delivery = WebhookDelivery(
                delivery_id=delivery_id,
                webhook_id=webhook_id,
                event_type=event_type,
                payload=transformed_payload,
                timestamp=datetime.now()
            )
            
            # Add to deliveries and queue
            self.deliveries[delivery_id] = delivery
            self.delivery_queue.append(delivery_id)
            
            logger.debug(f"Queued outbound webhook {webhook_id} for event {event_type}")
            
            return delivery_id
            
    def get_delivery(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get a webhook delivery record.
        
        Args:
            delivery_id: The ID of the delivery.
            
        Returns:
            Optional[WebhookDelivery]: The delivery record, or None if not found.
        """
        with self.lock:
            return self.deliveries.get(delivery_id)
            
    def get_deliveries_for_webhook(self, webhook_id: str) -> List[WebhookDelivery]:
        """Get all delivery records for a webhook.
        
        Args:
            webhook_id: The ID of the webhook.
            
        Returns:
            List[WebhookDelivery]: The delivery records.
        """
        with self.lock:
            return [d for d in self.deliveries.values() if d.webhook_id == webhook_id]
            
    def start_delivery_thread(self):
        """Start the delivery thread."""
        with self.lock:
            if self.running:
                return
                
            self.running = True
            self.delivery_thread = threading.Thread(target=self._delivery_thread, daemon=True)
            self.delivery_thread.start()
            
            logger.info("Started webhook delivery thread")
            
    def stop_delivery_thread(self):
        """Stop the delivery thread."""
        with self.lock:
            self.running = False
            if self.delivery_thread is not None:
                self.delivery_thread.join(timeout=5.0)
                self.delivery_thread = None
                
            logger.info("Stopped webhook delivery thread")
            
    def _delivery_thread(self):
        """Thread for delivering outbound webhooks."""
        import requests
        
        while self.running:
            try:
                # Get the next delivery
                delivery_id = None
                with self.lock:
                    if self.delivery_queue:
                        delivery_id = self.delivery_queue.pop(0)
                        
                if delivery_id is None:
                    # No deliveries, sleep and try again
                    time.sleep(0.1)
                    continue
                    
                # Get the delivery and webhook
                delivery = self.get_delivery(delivery_id)
                if delivery is None:
                    continue
                    
                webhook = self.get_webhook(delivery.webhook_id)
                if webhook is None:
                    continue
                    
                # Deliver the webhook
                try:
                    # Prepare headers
                    headers = webhook.headers.copy()
                    
                    # Add content type
                    headers['Content-Type'] = webhook.content_type.value
                    
                    # Add security headers if needed
                    if webhook.security_type == WebhookSecurityType.HMAC:
                        # Add HMAC signature
                        secret = webhook.security_config.get('secret', '')
                        signature_header = webhook.security_config.get('signature_header', 'X-Signature')
                        algorithm = webhook.security_config.get('algorithm', 'sha256')
                        
                        payload_str = json.dumps(delivery.payload)
                        
                        if algorithm == 'sha256':
                            digest = hmac.new(secret.encode(), payload_str.encode(), hashlib.sha256).hexdigest()
                        elif algorithm == 'sha1':
                            digest = hmac.new(secret.encode(), payload_str.encode(), hashlib.sha1).hexdigest()
                        else:
                            digest = ''
                            
                        headers[signature_header] = digest
                    elif webhook.security_type == WebhookSecurityType.API_KEY:
                        # Add API key
                        api_key = webhook.security_config.get('api_key', '')
                        key_header = webhook.security_config.get('key_header', 'X-API-Key')
                        
                        headers[key_header] = api_key
                    elif webhook.security_type == WebhookSecurityType.BASIC_AUTH:
                        # Add Basic Authentication
                        import base64
                        
                        username = webhook.security_config.get('username', '')
                        password = webhook.security_config.get('password', '')
                        
                        credentials = f"{username}:{password}"
                        encoded_credentials = base64.b64encode(credentials.encode()).decode()
                        
                        headers['Authorization'] = f"Basic {encoded_credentials}"
                    elif webhook.security_type == WebhookSecurityType.JWT:
                        # Add JWT
                        try:
                            import jwt
                            
                            secret = webhook.security_config.get('secret', '')
                            token_header = webhook.security_config.get('token_header', 'Authorization')
                            token_prefix = webhook.security_config.get('token_prefix', 'Bearer ')
                            
                            # Create a JWT with the delivery ID and timestamp
                            payload = {
                                'delivery_id': delivery_id,
                                'timestamp': int(time.time()),
                                'webhook_id': webhook.webhook_id
                            }
                            
                            token = jwt.encode(payload, secret, algorithm='HS256')
                            
                            headers[token_header] = f"{token_prefix}{token}"
                        except ImportError:
                            logger.warning("PyJWT package not found, JWT security skipped")
                            
                    # Send the request
                    response = requests.post(
                        webhook.url,
                        json=delivery.payload,
                        headers=headers,
                        timeout=30.0
                    )
                    
                    # Update the delivery record
                    with self.lock:
                        delivery.status = 'completed' if response.ok else 'failed'
                        delivery.response = {
                            'status_code': response.status_code,
                            'headers': dict(response.headers),
                            'body': response.text
                        }
                        delivery.completed_at = datetime.now()
                        
                        if not response.ok:
                            delivery.error = f"HTTP error {response.status_code}: {response.text}"
                            
                    # Publish event
                    if self.event_system:
                        if response.ok:
                            self.event_system.publish(WebhookSentEvent(
                                webhook.webhook_id, delivery.payload, delivery.response
                            ))
                        else:
                            self.event_system.publish(WebhookFailedEvent(
                                webhook.webhook_id, delivery.payload, delivery.error
                            ))
                            
                    logger.debug(f"Delivered webhook {webhook.webhook_id} with status {response.status_code}")
                except Exception as e:
                    # Update the delivery record
                    with self.lock:
                        delivery.status = 'failed'
                        delivery.error = str(e)
                        delivery.retry_count += 1
                        
                        # Check if we should retry
                        max_retries = webhook.retry_config.get('max_retries', 3)
                        if delivery.retry_count < max_retries:
                            # Calculate retry delay with exponential backoff
                            retry_delay = webhook.retry_config.get('retry_delay', 5)
                            backoff_factor = webhook.retry_config.get('backoff_factor', 2.0)
                            delay = retry_delay * (backoff_factor ** (delivery.retry_count - 1))
                            
                            # Add back to the queue after a delay
                            threading.Timer(delay, self._requeue_delivery, args=[delivery_id]).start()
                            
                            logger.debug(f"Will retry webhook {webhook.webhook_id} in {delay} seconds")
                        else:
                            delivery.completed_at = datetime.now()
                            
                            # Publish event
                            if self.event_system:
                                self.event_system.publish(WebhookFailedEvent(
                                    webhook.webhook_id, delivery.payload, delivery.error
                                ))
                                
                            logger.warning(f"Failed to deliver webhook {webhook.webhook_id} after {max_retries} retries: {str(e)}")
            except Exception as e:
                logger.error(f"Error in webhook delivery thread: {str(e)}")
                
    def _requeue_delivery(self, delivery_id: str):
        """Requeue a delivery for retry.
        
        Args:
            delivery_id: The ID of the delivery to requeue.
        """
        with self.lock:
            if delivery_id in self.deliveries:
                self.delivery_queue.append(delivery_id)
                
    def _validate_url(self, url: str) -> bool:
        """Validate a URL.
        
        Args:
            url: The URL to validate.
            
        Returns:
            bool: True if the URL is valid, False otherwise.
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
            
    def load_webhooks_from_config(self):
        """Load webhooks from configuration.
        
        Raises:
            WebhookError: If the configuration is invalid or a webhook cannot be created.
        """
        if self.config_manager is None:
            raise WebhookError("Configuration manager is required to load webhooks from configuration")
            
        try:
            # Get webhook configurations
            webhook_configs = self.config_manager.get('integration.webhooks', {})
            
            # Register webhooks
            for config_id, config in webhook_configs.items():
                # Create webhook configuration
                webhook_config = WebhookConfig(
                    webhook_id=config.get('webhook_id', str(uuid.uuid4())),
                    name=config.get('name', f"Webhook {config_id}"),
                    system_id=config.get('system_id', ''),
                    direction=WebhookDirection(config.get('direction', 'inbound')),
                    url=config.get('url', ''),
                    events=config.get('events', []),
                    content_type=WebhookContentType(config.get('content_type', 'application/json')),
                    security_type=WebhookSecurityType(config.get('security_type', 'none')),
                    security_config=config.get('security_config', {}),
                    headers=config.get('headers', {}),
                    retry_config=config.get('retry_config', {}),
                    transform_config=config.get('transform_config', {})
                )
                
                # Register the webhook
                try:
                    self.register_webhook(webhook_config)
                except WebhookError as e:
                    logger.warning(f"Failed to register webhook {config_id}: {str(e)}")
                    
            logger.info(f"Loaded {len(self.webhooks)} webhooks from configuration")
        except Exception as e:
            raise WebhookError(f"Failed to load webhooks: {str(e)}") from e
            
    def save_webhooks_to_file(self, file_path: str):
        """Save webhooks to a file.
        
        Args:
            file_path: The path to save the webhooks to.
            
        Raises:
            WebhookError: If the webhooks cannot be saved.
        """
        try:
            # Get all webhooks
            webhooks = []
            with self.lock:
                for webhook in self.webhooks.values():
                    webhooks.append(webhook.to_dict())
                    
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the webhooks
            with open(file_path, 'w') as f:
                json.dump(webhooks, f, indent=2)
                
            logger.info(f"Saved {len(webhooks)} webhooks to {file_path}")
        except Exception as e:
            raise WebhookError(f"Failed to save webhooks: {str(e)}") from e
            
    def load_webhooks_from_file(self, file_path: str):
        """Load webhooks from a file.
        
        Args:
            file_path: The path to load the webhooks from.
            
        Raises:
            WebhookError: If the webhooks cannot be loaded.
        """
        try:
            # Load the webhooks
            with open(file_path, 'r') as f:
                webhooks_data = json.load(f)
                
            # Register webhooks
            for webhook_data in webhooks_data:
                webhook_config = WebhookConfig.from_dict(webhook_data)
                
                try:
                    self.register_webhook(webhook_config)
                except WebhookError as e:
                    logger.warning(f"Failed to register webhook {webhook_config.webhook_id}: {str(e)}")
                    
            logger.info(f"Loaded webhooks from {file_path}")
        except Exception as e:
            raise WebhookError(f"Failed to load webhooks: {str(e)}") from e
            
    def save_deliveries_to_file(self, file_path: str):
        """Save webhook deliveries to a file.
        
        Args:
            file_path: The path to save the deliveries to.
            
        Raises:
            WebhookError: If the deliveries cannot be saved.
        """
        try:
            # Get all deliveries
            deliveries = []
            with self.lock:
                for delivery in self.deliveries.values():
                    deliveries.append(delivery.to_dict())
                    
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the deliveries
            with open(file_path, 'w') as f:
                json.dump(deliveries, f, indent=2)
                
            logger.info(f"Saved {len(deliveries)} webhook deliveries to {file_path}")
        except Exception as e:
            raise WebhookError(f"Failed to save webhook deliveries: {str(e)}") from e
            
    def load_deliveries_from_file(self, file_path: str):
        """Load webhook deliveries from a file.
        
        Args:
            file_path: The path to load the deliveries from.
            
        Raises:
            WebhookError: If the deliveries cannot be loaded.
        """
        try:
            # Load the deliveries
            with open(file_path, 'r') as f:
                deliveries_data = json.load(f)
                
            # Add deliveries
            with self.lock:
                for delivery_data in deliveries_data:
                    delivery = WebhookDelivery.from_dict(delivery_data)
                    self.deliveries[delivery.delivery_id] = delivery
                    
                    # Add to queue if not completed
                    if delivery.status == 'pending':
                        self.delivery_queue.append(delivery.delivery_id)
                        
            logger.info(f"Loaded webhook deliveries from {file_path}")
        except Exception as e:
            raise WebhookError(f"Failed to load webhook deliveries: {str(e)}") from e


def create_webhook_config(name: str, system_id: str, direction: WebhookDirection,
                        url: str, events: List[str], **kwargs) -> WebhookConfig:
    """Create a webhook configuration.
    
    Args:
        name: The name of the webhook.
        system_id: The ID of the external system.
        direction: The direction of the webhook.
        url: The URL for the webhook.
        events: The events that trigger the webhook.
        **kwargs: Additional configuration options.
        
    Returns:
        WebhookConfig: The created webhook configuration.
    """
    webhook_id = kwargs.pop('webhook_id', str(uuid.uuid4()))
    return WebhookConfig(webhook_id, name, system_id, direction, url, events, **kwargs)


def get_webhook_manager() -> WebhookManager:
    """Get the global webhook manager instance.
    
    Returns:
        WebhookManager: The global webhook manager instance.
    """
    # Use a global variable to store the webhook manager instance
    global _webhook_manager
    
    # Create the webhook manager if it doesn't exist
    if '_webhook_manager' not in globals():
        from src.infrastructure.config import get_config_manager
        from src.infrastructure.event import get_event_system
        _webhook_manager = WebhookManager(get_config_manager(), get_event_system())
        
    return _webhook_manager


def generate_webhook_secret() -> str:
    """Generate a secure secret for webhook authentication.
    
    Returns:
        str: The generated secret.
    """
    return generate_random_key(32)


def validate_webhook_url(url: str) -> bool:
    """Validate a webhook URL.
    
    Args:
        url: The URL to validate.
        
    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False