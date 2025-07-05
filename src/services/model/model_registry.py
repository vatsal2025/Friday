"""Model Registry for Friday AI Trading System.

This module provides a model registry for storing and retrieving trained models.
"""

import os
import json
import uuid
import shutil
import datetime
import tempfile
import threading
import functools
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, TypeVar, Generic

from src.infrastructure.logging import get_logger
from src.services.model.model_serialization import ModelSerializer
from src.services.model.model_database import ModelMetadata, ModelTag, ModelMetric
from src.infrastructure.security.digital_signatures import ModelSigner
from src.infrastructure.security.model_encryption import ModelEncryptionManager
from src.infrastructure.security.access_control import AccessControl, Permission
from src.infrastructure.security.audit_logging import SecurityAuditLogger, log_model_created, log_model_updated, log_model_deleted, log_model_deployed, log_model_signed, log_model_signature_verified, log_model_signature_failed, log_model_encrypted, log_model_decrypted, log_model_access_denied, log_model_loaded, log_model_exported, log_model_decryption_failed

# Create logger
logger = get_logger(__name__)

# Type variable for generic model type
T = TypeVar('T')


class LazyModelProxy(Generic[T]):
    """A proxy class that lazily loads a model when it's accessed.
    
    This class acts as a transparent proxy to the underlying model, loading it only when
    methods or attributes are accessed. This helps reduce memory usage when multiple models
    are registered but not all are actively used.
    
    Attributes:
        _model: The actual model object, loaded on first access.
        _initialized: Whether the model has been loaded.
    """
    
    def __init__(self, model: T):
        """Initialize the lazy model proxy.
        
        Args:
            model: The model to proxy. This can be the actual model object or a callable
                  that returns the model when invoked.
        """
        self._model = model
        self._initialized = not callable(model)
    
    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the underlying model.
        
        This method is called when an attribute is accessed that doesn't exist on the proxy.
        It ensures the model is loaded before accessing the attribute.
        
        Args:
            name: The name of the attribute to get.
            
        Returns:
            The attribute value from the underlying model.
            
        Raises:
            AttributeError: If the attribute doesn't exist on the model.
        """
        if not self._initialized:
            self._model = self._model()
            self._initialized = True
        
        return getattr(self._model, name)
    
    def __call__(self, *args, **kwargs) -> Any:
        """Call the underlying model as a function.
        
        This method is called when the proxy is called as a function.
        It ensures the model is loaded before calling it.
        
        Args:
            *args: Positional arguments to pass to the model.
            **kwargs: Keyword arguments to pass to the model.
            
        Returns:
            The result of calling the model.
            
        Raises:
            TypeError: If the model is not callable.
        """
        if not self._initialized:
            self._model = self._model()
            self._initialized = True
        
        if not callable(self._model):
            raise TypeError("Model object is not callable")
        
        return self._model(*args, **kwargs)
    
    def __dir__(self) -> List[str]:
        """Get the list of attributes available on the proxy and the model.
        
        This method is called when dir() is called on the proxy.
        It ensures the model is loaded before getting its attributes.
        
        Returns:
            A list of attribute names.
        """
        if not self._initialized:
            self._model = self._model()
            self._initialized = True
        
        # Combine attributes from both the proxy and the model
        proxy_attrs = set(super().__dir__())
        model_attrs = set(dir(self._model))
        return sorted(proxy_attrs.union(model_attrs))


class ModelRegistry:
    """Model Registry for storing and retrieving trained models.

    This class provides functionality for registering, retrieving, and managing
    trained models in a central registry.

    Attributes:
        registry_dir: Directory where models are stored.
        metadata_file: File where model metadata is stored.
        models: Dictionary of registered models.
        serializer: ModelSerializer instance for serializing/deserializing models.
    """

    def __init__(self, registry_dir: Optional[str] = None, 
                 access_control: Optional[AccessControl] = None,
                 audit_logger: Optional[SecurityAuditLogger] = None,
                 enable_signing: bool = True,
                 enable_encryption: bool = True,
                 keys_path: Optional[str] = None,
                 cache_size: int = 10,
                 lazy_loading: bool = True,
                 compression_level: Optional[int] = None):
        """Initialize the model registry.

        Args:
            registry_dir: Directory where models are stored. If None, defaults to
                'models/registry' in the current working directory.
            access_control: Access control system for model operations. If None, a default one will be created.
            audit_logger: Security audit logger for model operations. If None, a default one will be created.
            enable_signing: Whether to enable model signing for authenticity verification.
            enable_encryption: Whether to enable model encryption for sensitive models.
            keys_path: Path to the directory for storing keys. If None, a default path will be used.
            cache_size: Maximum number of models to keep in memory cache. Default is 10.
            lazy_loading: Whether to enable lazy loading for models. Default is True.
            compression_level: Compression level for model serialization (0-9, None for no compression). Default is None.
        """
        self.registry_dir = registry_dir or os.path.join(os.getcwd(), "models", "registry")
        self.metadata_file = os.path.join(self.registry_dir, "metadata.json")
        self.models = {}
        
        # Performance optimization parameters
        self.cache_size = cache_size
        self.lazy_loading = lazy_loading
        self.compression_level = compression_level
        
        # Initialize model cache with LRU policy
        self.model_cache = functools.lru_cache(maxsize=self.cache_size)(self._load_model_from_disk)
        self.cache_lock = threading.RLock()  # Thread-safe lock for cache operations
        
        # Initialize thread pool for parallel operations
        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        
        # Initialize serializer with compression if specified
        self.serializer = ModelSerializer(compression_level=self.compression_level)
        
        # Set up security components
        self.enable_signing = enable_signing
        self.enable_encryption = enable_encryption
        
        # Set up keys path
        if keys_path is None:
            self.keys_path = os.path.join(self.registry_dir, "keys")
        else:
            self.keys_path = keys_path
        os.makedirs(self.keys_path, exist_ok=True)
        
        # Set up access control
        if access_control is None:
            access_policy_file = os.path.join(self.keys_path, "access_policy.json")
            self.access_control = AccessControl(policy_file=access_policy_file)
        else:
            self.access_control = access_control
        
        # Set up audit logging
        if audit_logger is None:
            audit_log_file = os.path.join(self.registry_dir, "logs", "security_audit.log")
            os.makedirs(os.path.dirname(audit_log_file), exist_ok=True)
            self.audit_logger = SecurityAuditLogger(log_file=audit_log_file)
        else:
            self.audit_logger = audit_logger
        
        # Set up model signer if signing is enabled
        if enable_signing:
            keys_dir = os.path.join(self.keys_path, "signing")
            os.makedirs(keys_dir, exist_ok=True)
            private_key_path = os.path.join(keys_dir, "private_key.pem")
            public_key_path = os.path.join(keys_dir, "public_key.pem")
            self.model_signer = ModelSigner(private_key_path, public_key_path)
        else:
            self.model_signer = None
        
        # Set up model encryption manager if encryption is enabled
        if enable_encryption:
            encryption_key_file = os.path.join(self.keys_path, "encryption", "model_encryption_key.json")
            os.makedirs(os.path.dirname(encryption_key_file), exist_ok=True)
            self.encryption_manager = ModelEncryptionManager(registry_path=self.registry_dir, key_file=encryption_key_file)
        else:
            self.encryption_manager = None
        
        # Create registry directory if it doesn't exist
        os.makedirs(self.registry_dir, exist_ok=True)
        
        # Load metadata if it exists
        self._load_metadata()
        
        logger.info(f"Initialized ModelRegistry at {self.registry_dir}")

    def _load_metadata(self) -> None:
        """Load model metadata from file and database.

        This method loads model metadata from the metadata file and the database.
        """
        # Load from file if it exists
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r") as f:
                    self.models = json.load(f)
                logger.info(f"Loaded metadata for {len(self.models)} models from {self.metadata_file}")
            except Exception as e:
                logger.error(f"Error loading metadata from {self.metadata_file}: {e}")
                self.models = {}
        
        # Load from database
        try:
            db_models = ModelMetadata.get_all_models()
            for db_model in db_models:
                model_dict = db_model.to_registry_dict()
                self.models[db_model.model_id] = model_dict
            logger.info(f"Loaded metadata for {len(db_models)} models from database")
        except Exception as e:
            logger.error(f"Error loading metadata from database: {e}")

    def _save_metadata(self) -> None:
        """Save model metadata to file.

        This method saves model metadata to the metadata file.
        """
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.models, f, indent=2)
            logger.info(f"Saved metadata for {len(self.models)} models to {self.metadata_file}")
        except Exception as e:
            logger.error(f"Error saving metadata to {self.metadata_file}: {e}")

    def register_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        username: Optional[str] = None,
        sign_model: bool = False,
        encrypt_model: bool = False,
        encryption_password: Optional[str] = None,
        source_ip: Optional[str] = None,
    ) -> str:
        """Register a model in the registry.

        Args:
            model: The model to register.
            model_name: Name of the model.
            model_type: Type of the model (e.g., 'random_forest', 'gradient_boosting').
            version: Version of the model. If None, a new version will be generated.
            metadata: Additional metadata for the model.
            metrics: Performance metrics for the model.
            tags: Tags for the model.
            description: Description of the model.
            username: The username of the user registering the model.
            sign_model: Whether to sign the model for authenticity verification.
            encrypt_model: Whether to encrypt the model for security.
            encryption_password: Password for model encryption. If None, the default key will be used.
            source_ip: The source IP address of the request.

        Returns:
            str: The model ID.
            
        Raises:
            PermissionError: If the user does not have permission to register models.
        """
        # Check permission if username is provided
        if username is not None:
            try:
                self.access_control.require_permission(username, Permission.MODEL_CREATE)
            except PermissionError as e:
                # Log the access denied event
                if self.audit_logger is not None:
                    log_model_access_denied(self.audit_logger, username, "N/A", model_name, "create", str(e), source_ip)
                raise
        
        # Generate model ID and version if not provided
        model_id = str(uuid.uuid4())
        if version is None:
            version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model directory
        model_dir = os.path.join(self.registry_dir, model_name, version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"{model_name}_{version}.joblib")
        self.serializer.serialize(model, model_path)
        
        # Create metadata
        model_metadata = {
            "model_id": model_id,
            "model_name": model_name,
            "model_type": model_type,
            "version": version,
            "description": description,
            "location": model_path,
            "status": "active",
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "metadata": metadata or {},
            "metrics": metrics or {},
            "tags": tags or []
        }
        
        # Sign the model if requested and signing is enabled
        if sign_model and self.enable_signing and self.model_signer is not None:
            try:
                signature_path = self.model_signer.sign_model(model_path)
                model_metadata["signed"] = True
                model_metadata["signature_path"] = signature_path
                
                # Log the model signing event
                if self.audit_logger is not None and username is not None:
                    log_model_signed(self.audit_logger, username, model_id, model_name, model_id, source_ip)
            except Exception as e:
                logger.error(f"Failed to sign model {model_name} version {version}: {e}")
                model_metadata["signed"] = False
        else:
            model_metadata["signed"] = False
        
        # Save metadata to registry
        self.models[model_id] = model_metadata
        self._save_metadata()
        
        # Save metadata to database
        try:
            ModelMetadata.create_from_registry_entry(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                version=version,
                location=model_path,
                metadata=metadata or {},
                metrics=metrics or {},
                tags=tags or [],
                description=description,
                status="active"
            )
            
            # Record metrics in metrics history
            if metrics:
                ModelMetric.record_metrics(model_id, metrics)
            
            # Create tags if they don't exist
            if tags:
                for tag in tags:
                    ModelTag.get_or_create(tag_name=tag)
        except Exception as e:
            logger.error(f"Error saving model metadata to database: {e}")
        
        # Encrypt the model if requested and encryption is enabled
        if encrypt_model and self.enable_encryption and self.encryption_manager is not None:
            try:
                self.encryption_manager.encrypt_model(model_name, version, encryption_password)
                
                # Update metadata to reflect encryption
                model_metadata["encrypted"] = True
                self.models[model_id] = model_metadata
                self._save_metadata()
                
                # Log the model encryption event
                if self.audit_logger is not None and username is not None:
                    log_model_encrypted(self.audit_logger, username, model_id, model_name, source_ip)
            except Exception as e:
                logger.error(f"Failed to encrypt model {model_name} version {version}: {e}")
                model_metadata["encrypted"] = False
                self.models[model_id] = model_metadata
                self._save_metadata()
        else:
            model_metadata["encrypted"] = False
            self.models[model_id] = model_metadata
            self._save_metadata()
        
        # Log the model creation event
        if self.audit_logger is not None and username is not None:
            log_model_created(self.audit_logger, username, model_id, model_name, model_type, source_ip)
        
        logger.info(f"Registered model {model_name} with version {version} and ID {model_id}")
        
        return model_id

    def _load_model_from_disk(
        self,
        model_id: str,
        verify_signature: bool = False,
        decryption_password: Optional[str] = None,
        username: Optional[str] = None,
        source_ip: Optional[str] = None
    ) -> Any:
        """Internal method to load a model from disk.
        
        This method handles the actual loading of the model from disk,
        including signature verification and decryption if needed.
        It is used by the LRU cache to load models when needed.
        
        Args:
            model_id: ID of the model to load.
            verify_signature: Whether to verify the model signature before loading.
            decryption_password: Password for model decryption if the model is encrypted.
            username: The username of the user loading the model.
            source_ip: The source IP address of the request.
            
        Returns:
            Any: The loaded model.
            
        Raises:
            ValueError: If the model is not found or signature verification fails.
        """
        if model_id not in self.models:
            raise ValueError(f"Model with ID {model_id} not found in registry")
        
        model_metadata = self.models[model_id]
        model_path = model_metadata["location"]
        model_name = model_metadata["model_name"]
        version = model_metadata["version"]
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model file {model_path} not found")
        
        # Verify signature if requested and the model is signed
        if verify_signature and self.enable_signing and self.model_signer is not None:
            is_signed = model_metadata.get("signed", False)
            if is_signed:
                signature_path = model_metadata.get("signature_path")
                if signature_path and os.path.exists(signature_path):
                    try:
                        is_valid = self.model_signer.verify_model(model_path, signature_path)
                        if not is_valid:
                            error_msg = f"Signature verification failed for model {model_name} version {version}"
                            # Log the signature verification failure
                            if self.audit_logger is not None and username is not None:
                                log_model_signature_verification_failed(
                                    self.audit_logger, username, model_id, model_name, error_msg, source_ip
                                )
                            raise ValueError(error_msg)
                        
                        # Log the signature verification success
                        if self.audit_logger is not None and username is not None:
                            log_model_signature_verified(
                                self.audit_logger, username, model_id, model_name, source_ip
                            )
                    except Exception as e:
                        error_msg = f"Error during signature verification: {str(e)}"
                        # Log the signature verification error
                        if self.audit_logger is not None and username is not None:
                            log_model_signature_verification_failed(
                                self.audit_logger, username, model_id, model_name, error_msg, source_ip
                            )
                        raise ValueError(error_msg)
                else:
                    error_msg = f"Signature file not found for model {model_name} version {version}"
                    logger.warning(error_msg)
            elif verify_signature:
                logger.warning(f"Model {model_name} version {version} is not signed, skipping verification")
        
        # Decrypt the model if it's encrypted
        is_encrypted = model_metadata.get("encrypted", False)
        if is_encrypted and self.enable_encryption and self.encryption_manager is not None:
            try:
                self.encryption_manager.decrypt_model(model_name, version, decryption_password)
                
                # Log the model decryption event
                if self.audit_logger is not None and username is not None:
                    log_model_decrypted(self.audit_logger, username, model_id, model_name, source_ip)
            except Exception as e:
                error_msg = f"Failed to decrypt model {model_name} version {version}: {str(e)}"
                # Log the decryption failure
                if self.audit_logger is not None and username is not None:
                    log_model_decryption_failed(self.audit_logger, username, model_id, model_name, error_msg, source_ip)
                raise ValueError(error_msg)
        
        # Load the model
        model = self.serializer.deserialize(model_path)
        
        logger.debug(f"Loaded model {model_name} with version {version} from disk")
        
        return model
        
    def load_model(
        self, 
        model_id: str, 
        username: Optional[str] = None,
        verify_signature: bool = False,
        decryption_password: Optional[str] = None,
        source_ip: Optional[str] = None,
        use_cache: bool = True
    ) -> Any:
        """Load a model from the registry.

        Args:
            model_id: ID of the model to load.
            username: The username of the user loading the model.
            verify_signature: Whether to verify the model signature before loading.
            decryption_password: Password for model decryption if the model is encrypted.
            source_ip: The source IP address of the request.
            use_cache: Whether to use the model cache. If True, the model will be loaded from
                the cache if available. If False, the model will be loaded from disk.

        Returns:
            Any: The loaded model.

        Raises:
            ValueError: If the model is not found in the registry or signature verification fails.
            PermissionError: If the user does not have permission to load models.
        """
        # Check permission if username is provided
        if username is not None:
            try:
                self.access_control.require_permission(username, Permission.MODEL_READ)
            except PermissionError as e:
                # Log the access denied event
                if self.audit_logger is not None:
                    log_model_access_denied(self.audit_logger, username, model_id, "unknown", "read", str(e), source_ip)
                raise
        
        if model_id not in self.models:
            raise ValueError(f"Model with ID {model_id} not found in registry")
        
        model_metadata = self.models[model_id]
        model_name = model_metadata["model_name"]
        version = model_metadata["version"]
        
        # Create a function to load the model when needed (for lazy loading)
        def load_model_func():
            try:
                if use_cache:
                    with self.cache_lock:  # Thread-safe access to cache
                        model = self.model_cache(model_id, verify_signature, decryption_password, username, source_ip)
                else:
                    # Load directly from disk without caching
                    model = self._load_model_from_disk(model_id, verify_signature, decryption_password, username, source_ip)
                
                # Log the model loading event
                if self.audit_logger is not None and username is not None:
                    log_model_loaded(self.audit_logger, username, model_id, model_name, source_ip)
                
                logger.info(f"Loaded model {model_name} with version {version}")
                return model
            except Exception as e:
                logger.error(f"Error loading model {model_name} version {version}: {str(e)}")
                raise
        
        try:
            # If lazy loading is enabled, return a proxy that will load the model when accessed
            if self.lazy_loading:
                logger.debug(f"Creating lazy proxy for model {model_name} version {version}")
                return LazyModelProxy(load_model_func)
            else:
                # Eager loading - load the model immediately
                return load_model_func()
        except Exception as e:
            logger.error(f"Error setting up model loading for {model_name} version {version}: {str(e)}")
            raise

    def load_models_in_parallel(self, model_ids: List[str], username: Optional[str] = None,
                           verify_signature: bool = False, decryption_password: Optional[str] = None,
                           source_ip: Optional[str] = None, use_cache: bool = True) -> Dict[str, Any]:
        """Load multiple models in parallel.
        
        This method loads multiple models in parallel using a thread pool, which can
        significantly improve performance when loading many models at once.
        
        Args:
            model_ids: List of model IDs to load.
            username: The username of the user loading the models.
            verify_signature: Whether to verify model signatures before loading.
            decryption_password: Password for model decryption if models are encrypted.
            source_ip: The source IP address of the request.
            use_cache: Whether to use the model cache.
            
        Returns:
            Dict[str, Any]: A dictionary mapping model IDs to loaded models.
            
        Raises:
            ValueError: If any model is not found or fails to load.
            PermissionError: If the user does not have permission to load models.
        """
        # Check permission if username is provided
        if username is not None:
            try:
                self.access_control.require_permission(username, Permission.MODEL_READ)
            except PermissionError as e:
                # Log the access denied event
                if self.audit_logger is not None:
                    log_model_access_denied(self.audit_logger, username, "batch_load", "unknown", "read", str(e), source_ip)
                raise
        
        # Filter out invalid model IDs
        valid_model_ids = [model_id for model_id in model_ids if model_id in self.models]
        if len(valid_model_ids) < len(model_ids):
            invalid_ids = set(model_ids) - set(valid_model_ids)
            logger.warning(f"Some model IDs were not found in registry: {invalid_ids}")
        
        if not valid_model_ids:
            return {}
        
        # Define a function to load a single model
        def load_single_model(model_id):
            try:
                return model_id, self.load_model(
                    model_id, username, verify_signature, decryption_password, source_ip, use_cache
                )
            except Exception as e:
                logger.error(f"Error loading model {model_id}: {str(e)}")
                return model_id, None
        
        # Load models in parallel using thread pool
        results = {}
        futures = []
        
        for model_id in valid_model_ids:
            futures.append(self.thread_pool.submit(load_single_model, model_id))
        
        # Collect results
        for future in futures:
            try:
                model_id, model = future.result()
                if model is not None:
                    results[model_id] = model
            except Exception as e:
                logger.error(f"Error in parallel model loading: {str(e)}")
        
        return results
    
    def get_model_by_name_and_version(self, model_name: str, version: str, username: Optional[str] = None,
                                   verify_signature: bool = False, decryption_password: Optional[str] = None,
                                   source_ip: Optional[str] = None, use_cache: bool = True) -> Any:
        """Get a model by name and version.

        Args:
            model_name: Name of the model.
            version: Version of the model.
            username: The username of the user loading the model.
            verify_signature: Whether to verify the model signature before loading.
            decryption_password: Password for model decryption if the model is encrypted.
            source_ip: The source IP address of the request.
            use_cache: Whether to use the model cache.

        Returns:
            Any: The model.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        # Try to find the model in memory
        for model_id, model_metadata in self.models.items():
            if model_metadata["model_name"] == model_name and model_metadata["version"] == version:
                return self.load_model(model_id, username, verify_signature, 
                                      decryption_password, source_ip, use_cache)
        
        # Try to find the model in the database
        db_model = ModelMetadata.get_by_model_name_and_version(model_name, version)
        if db_model is not None:
            model_id = db_model.model_id
            self.models[model_id] = db_model.to_registry_dict()
            return self.load_model(model_id, username, verify_signature, 
                                  decryption_password, source_ip, use_cache)
        
        raise ValueError(f"Model {model_name} with version {version} not found in registry")

    def get_latest_version(self, model_name: str, username: Optional[str] = None,
                          verify_signature: bool = False, decryption_password: Optional[str] = None,
                          source_ip: Optional[str] = None, use_cache: bool = True) -> Any:
        """Get the latest version of a model.

        Args:
            model_name: Name of the model.
            username: The username of the user loading the model.
            verify_signature: Whether to verify the model signature before loading.
            decryption_password: Password for model decryption if the model is encrypted.
            source_ip: The source IP address of the request.
            use_cache: Whether to use the model cache.

        Returns:
            Any: The model.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        # Try to find the latest version in memory
        latest_version = None
        latest_model_id = None
        
        for model_id, model_metadata in self.models.items():
            if model_metadata["model_name"] == model_name:
                if latest_version is None or model_metadata["version"] > latest_version:
                    latest_version = model_metadata["version"]
                    latest_model_id = model_id
        
        if latest_model_id is not None:
            return self.load_model(latest_model_id, username, verify_signature, 
                                  decryption_password, source_ip, use_cache)
        
        # Try to find the latest version in the database
        db_model = ModelMetadata.get_latest_version(model_name)
        if db_model is not None:
            model_id = db_model.model_id
            self.models[model_id] = db_model.to_registry_dict()
            return self.load_model(model_id, username, verify_signature, 
                                  decryption_password, source_ip, use_cache)
        
        raise ValueError(f"Model {model_name} not found in registry")

    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get metadata for a model.

        Args:
            model_id: ID of the model.

        Returns:
            Dict[str, Any]: The model metadata.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        if model_id not in self.models:
            # Try to find the model in the database
            db_model = ModelMetadata.get_by_model_id(model_id)
            if db_model is not None:
                self.models[model_id] = db_model.to_registry_dict()
            else:
                raise ValueError(f"Model with ID {model_id} not found in registry")
        
        return self.models[model_id]

    def update_model_metadata(
        self, 
        model_id: str, 
        metadata: Dict[str, Any],
        username: Optional[str] = None,
        source_ip: Optional[str] = None
    ) -> None:
        """Update metadata for a model.

        Args:
            model_id: ID of the model.
            metadata: New metadata for the model.
            username: The username of the user updating the model metadata.
            source_ip: The source IP address of the request.

        Raises:
            ValueError: If the model is not found in the registry.
            PermissionError: If the user does not have permission to update model metadata.
        """
        # Check permission if username is provided
        if username is not None:
            try:
                self.access_control.require_permission(username, Permission.MODEL_UPDATE)
            except PermissionError as e:
                # Log the access denied event
                if self.audit_logger is not None:
                    log_model_access_denied(self.audit_logger, username, model_id, "unknown", "update", str(e), source_ip)
                raise
        
        if model_id not in self.models:
            # Try to find the model in the database
            db_model = ModelMetadata.get_by_model_id(model_id)
            if db_model is not None:
                self.models[model_id] = db_model.to_registry_dict()
            else:
                raise ValueError(f"Model with ID {model_id} not found in registry")
        
        model_name = self.models[model_id]["model_name"]
        
        # Update metadata in memory
        self.models[model_id]["metadata"] = metadata
        self.models[model_id]["updated_at"] = datetime.datetime.now().isoformat()
        
        # Save metadata to file
        self._save_metadata()
        
        # Update metadata in database
        try:
            db_model = ModelMetadata.get_by_model_id(model_id)
            if db_model is not None:
                db_model.update_metadata(metadata)
        except Exception as e:
            logger.error(f"Error updating model metadata in database: {e}")
        
        # Log the model update event
        if self.audit_logger is not None and username is not None:
            log_model_updated(self.audit_logger, username, model_id, model_name, source_ip)
        
        logger.info(f"Updated metadata for model {model_id}")

    def delete_model(
        self, 
        model_id: str,
        username: Optional[str] = None,
        source_ip: Optional[str] = None
    ) -> None:
        """Delete a model from the registry.

        Args:
            model_id: ID of the model to delete.
            username: The username of the user deleting the model.
            source_ip: The source IP address of the request.

        Raises:
            ValueError: If the model is not found in the registry.
            PermissionError: If the user does not have permission to delete models.
        """
        # Check permission if username is provided
        if username is not None:
            try:
                self.access_control.require_permission(username, Permission.MODEL_DELETE)
            except PermissionError as e:
                # Log the access denied event
                if self.audit_logger is not None:
                    log_model_access_denied(self.audit_logger, username, model_id, "unknown", "delete", str(e), source_ip)
                raise
        
        if model_id not in self.models:
            # Try to find the model in the database
            db_model = ModelMetadata.get_by_model_id(model_id)
            if db_model is not None:
                self.models[model_id] = db_model.to_registry_dict()
            else:
                raise ValueError(f"Model with ID {model_id} not found in registry")
        
        model_metadata = self.models[model_id]
        model_path = model_metadata["location"]
        model_name = model_metadata["model_name"]
        
        # Delete model file
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Delete signature file if it exists
        signature_path = model_metadata.get("signature_path")
        if signature_path and os.path.exists(signature_path):
            os.remove(signature_path)
        
        # Delete model directory if empty
        model_dir = os.path.dirname(model_path)
        if os.path.exists(model_dir) and not os.listdir(model_dir):
            os.rmdir(model_dir)
        
        # Delete model from registry
        del self.models[model_id]
        
        # Save metadata to file
        self._save_metadata()
        
        # Delete model from database
        try:
            db_model = ModelMetadata.get_by_model_id(model_id)
            if db_model is not None:
                db_model.delete()
        except Exception as e:
            logger.error(f"Error deleting model from database: {e}")
        
        # Log the model deletion event
        if self.audit_logger is not None and username is not None:
            log_model_deleted(self.audit_logger, username, model_id, model_name, source_ip)
        
        logger.info(f"Deleted model {model_id}")

    def list_models(self) -> List[Dict[str, Any]]:
        """List all models in the registry.

        Returns:
            List[Dict[str, Any]]: List of model metadata.
        """
        return list(self.models.values())

    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a model.

        Args:
            model_name: Name of the model.

        Returns:
            List[Dict[str, Any]]: List of model metadata for all versions.
        """
        versions = []
        
        # Get versions from memory
        for model_metadata in self.models.values():
            if model_metadata["model_name"] == model_name:
                versions.append(model_metadata)
        
        # Get versions from database
        try:
            db_models = ModelMetadata.get_all_versions(model_name)
            for db_model in db_models:
                model_dict = db_model.to_registry_dict()
                if model_dict["model_id"] not in self.models:
                    versions.append(model_dict)
        except Exception as e:
            logger.error(f"Error getting model versions from database: {e}")
        
        # Sort versions by creation time
        versions.sort(key=lambda x: x["created_at"], reverse=True)
        
        return versions

    def set_model_status(
        self, 
        model_id: str, 
        status: str,
        username: Optional[str] = None,
        source_ip: Optional[str] = None
    ) -> None:
        """Set the status of a model.

        Args:
            model_id: ID of the model.
            status: New status for the model.
            username: The username of the user setting the model status.
            source_ip: The source IP address of the request.

        Raises:
            ValueError: If the model is not found in the registry.
            PermissionError: If the user does not have permission to update model status.
        """
        # Check permission if username is provided
        if username is not None:
            try:
                self.access_control.require_permission(username, Permission.MODEL_UPDATE)
            except PermissionError as e:
                # Log the access denied event
                if self.audit_logger is not None:
                    log_model_access_denied(self.audit_logger, username, model_id, "unknown", "update_status", str(e), source_ip)
                raise
        
        if model_id not in self.models:
            # Try to find the model in the database
            db_model = ModelMetadata.get_by_model_id(model_id)
            if db_model is not None:
                self.models[model_id] = db_model.to_registry_dict()
            else:
                raise ValueError(f"Model with ID {model_id} not found in registry")
        
        # Get the current metadata for updating
        current_metadata = self.models[model_id]["metadata"]
        
        # Update status in memory
        self.models[model_id]["status"] = status
        self.models[model_id]["updated_at"] = datetime.datetime.now().isoformat()
        
        # Save metadata to file
        self._save_metadata()
        
        # Update status in database
        try:
            db_model = ModelMetadata.get_by_model_id(model_id)
            if db_model is not None:
                db_model.update_status(status)
        except Exception as e:
            logger.error(f"Error updating model status in database: {e}")
        
        # Update metadata to include status change
        updated_metadata = current_metadata.copy()
        updated_metadata["status_history"] = updated_metadata.get("status_history", []) + [
            {
                "status": status,
                "timestamp": datetime.datetime.now().isoformat(),
                "updated_by": username
            }
        ]
        
        # Use the update_model_metadata method to ensure proper logging
        self.update_model_metadata(model_id, updated_metadata, username, source_ip)
        
        logger.info(f"Set status of model {model_id} to {status}")

    def export_model(
        self, 
        model_id: str, 
        export_dir: str,
        username: Optional[str] = None,
        sign_model: bool = False,
        encrypt_model: bool = False,
        encryption_password: Optional[str] = None,
        source_ip: Optional[str] = None
    ) -> str:
        """Export a model to a directory.

        Args:
            model_id: ID of the model to export.
            export_dir: Directory to export the model to.
            username: The username of the user exporting the model.
            sign_model: Whether to sign the exported model.
            encrypt_model: Whether to encrypt the exported model.
            encryption_password: Password for model encryption. If None, the default key will be used.
            source_ip: The source IP address of the request.

        Returns:
            str: Path to the exported model.

        Raises:
            ValueError: If the model is not found in the registry.
            PermissionError: If the user does not have permission to export models.
        """
        # Check permission if username is provided
        if username is not None:
            try:
                self.access_control.require_permission(username, Permission.MODEL_READ)
            except PermissionError as e:
                # Log the access denied event
                if self.audit_logger is not None:
                    log_model_access_denied(self.audit_logger, username, model_id, "unknown", "export", str(e), source_ip)
                raise
        
        if model_id not in self.models:
            # Try to find the model in the database
            db_model = ModelMetadata.get_by_model_id(model_id)
            if db_model is not None:
                self.models[model_id] = db_model.to_registry_dict()
            else:
                raise ValueError(f"Model with ID {model_id} not found in registry")
        
        model_metadata = self.models[model_id]
        model_path = model_metadata["location"]
        model_name = model_metadata["model_name"]
        version = model_metadata["version"]
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model file {model_path} not found")
        
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        # If the model is encrypted, decrypt it before export
        is_encrypted = model_metadata.get("encrypted", False)
        if is_encrypted and self.enable_encryption and self.encryption_manager is not None:
            try:
                # Create a temporary decrypted copy for export
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, os.path.basename(model_path))
                shutil.copy2(model_path, temp_path)
                
                # Decrypt the temporary copy
                self.encryption_manager.decrypt_model_file(temp_path, encryption_password)
                
                # Use the decrypted file for export
                export_source_path = temp_path
                
                # Log the model decryption for export event
                if self.audit_logger is not None and username is not None:
                    log_model_decrypted(self.audit_logger, username, model_id, model_name, source_ip, "for export")
            except Exception as e:
                error_msg = f"Failed to decrypt model {model_name} version {version} for export: {str(e)}"
                # Log the decryption failure
                if self.audit_logger is not None and username is not None:
                    log_model_decryption_failed(self.audit_logger, username, model_id, model_name, error_msg, source_ip)
                raise ValueError(error_msg)
        else:
            export_source_path = model_path
        
        # Copy model file to export directory
        export_path = os.path.join(export_dir, os.path.basename(model_path))
        shutil.copy2(export_source_path, export_path)
        
        # Clean up temporary files if created
        if is_encrypted and self.enable_encryption and self.encryption_manager is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Sign the exported model if requested
        if sign_model and self.enable_signing and self.model_signer is not None:
            try:
                signature_path = self.model_signer.sign_model(export_path)
                model_metadata["signed"] = True
                model_metadata["signature_path"] = signature_path
                
                # Log the model signing event
                if self.audit_logger is not None and username is not None:
                    log_model_signed(self.audit_logger, username, model_id, model_name, model_id, source_ip, "during export")
            except Exception as e:
                logger.error(f"Failed to sign exported model {model_name} version {version}: {e}")
        
        # Encrypt the exported model if requested
        if encrypt_model and self.enable_encryption and self.encryption_manager is not None:
            try:
                self.encryption_manager.encrypt_model_file(export_path, encryption_password)
                model_metadata["encrypted"] = True
                
                # Log the model encryption event
                if self.audit_logger is not None and username is not None:
                    log_model_encrypted(self.audit_logger, username, model_id, model_name, source_ip, "during export")
            except Exception as e:
                logger.error(f"Failed to encrypt exported model {model_name} version {version}: {e}")
        
        # Save metadata to export directory
        metadata_path = os.path.join(export_dir, f"{os.path.basename(model_path)}.json")
        with open(metadata_path, "w") as f:
            json.dump(model_metadata, f, indent=2)
        
        # Log the model export event
        if self.audit_logger is not None and username is not None:
            log_model_exported(self.audit_logger, username, model_id, model_name, export_path, source_ip)
        
        logger.info(f"Exported model {model_id} to {export_path}")
        
        return export_path

    def import_model(
        self, 
        model_path: str, 
        metadata_path: Optional[str] = None,
        username: Optional[str] = None,
        verify_signature: bool = False,
        sign_model: bool = False,
        encrypt_model: bool = False,
        encryption_password: Optional[str] = None,
        decryption_password: Optional[str] = None,
        source_ip: Optional[str] = None
    ) -> str:
        """Import a model from a file.

        Args:
            model_path: Path to the model file.
            metadata_path: Path to the model metadata file. If None, the metadata will be
                extracted from the model filename.
            username: The username of the user importing the model.
            verify_signature: Whether to verify the model's signature during import.
            sign_model: Whether to sign the model after import.
            encrypt_model: Whether to encrypt the model after import.
            encryption_password: Password for model encryption. If None, the default key will be used.
            decryption_password: Password for model decryption. If None, the default key will be used.
            source_ip: The source IP address of the request.

        Returns:
            str: ID of the imported model.

        Raises:
            ValueError: If the model file does not exist or signature verification fails.
            PermissionError: If the user does not have permission to import models.
        """
        # Check permission if username is provided
        if username is not None:
            try:
                self.access_control.require_permission(username, Permission.MODEL_CREATE)
            except PermissionError as e:
                # Log the access denied event
                if self.audit_logger is not None:
                    log_model_access_denied(self.audit_logger, username, "unknown", "unknown", "import", str(e), source_ip)
                raise
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model file {model_path} not found")
        
        # Check if the model is encrypted and decrypt if necessary
        is_encrypted = False
        signature_path = f"{model_path}.sig"
        has_signature = os.path.exists(signature_path)
        
        # If the model is encrypted, decrypt it before import
        if self.enable_encryption and self.encryption_manager is not None:
            try:
                # Check if the file is encrypted
                if self.encryption_manager.is_encrypted(model_path):
                    is_encrypted = True
                    # Create a temporary decrypted copy for import
                    temp_dir = tempfile.mkdtemp()
                    temp_path = os.path.join(temp_dir, os.path.basename(model_path))
                    shutil.copy2(model_path, temp_path)
                    
                    # Decrypt the temporary copy
                    self.encryption_manager.decrypt_model_file(temp_path, decryption_password)
                    
                    # Use the decrypted file for import
                    import_source_path = temp_path
                    
                    # Log the model decryption for import event
                    if self.audit_logger is not None and username is not None:
                        log_model_decrypted(self.audit_logger, username, "unknown", os.path.basename(model_path), source_ip, "for import")
                else:
                    import_source_path = model_path
            except Exception as e:
                error_msg = f"Failed to decrypt model {os.path.basename(model_path)} for import: {str(e)}"
                # Log the decryption failure
                if self.audit_logger is not None and username is not None:
                    log_model_decryption_failed(self.audit_logger, username, "unknown", os.path.basename(model_path), error_msg, source_ip)
                raise ValueError(error_msg)
        else:
            import_source_path = model_path
        
        # Verify signature if requested and signature exists
        if verify_signature and has_signature and self.enable_signing and self.model_signer is not None:
            try:
                is_valid = self.model_signer.verify_model(import_source_path, signature_path)
                if not is_valid:
                    error_msg = f"Signature verification failed for model {os.path.basename(model_path)}"
                    # Log the signature verification failure
                    if self.audit_logger is not None and username is not None:
                        log_model_signature_failed(self.audit_logger, username, "unknown", os.path.basename(model_path), error_msg, source_ip)
                    raise ValueError(error_msg)
                
                # Log the successful signature verification
                if self.audit_logger is not None and username is not None:
                    log_model_signature_verified(self.audit_logger, username, "unknown", os.path.basename(model_path), source_ip)
            except Exception as e:
                error_msg = f"Error during signature verification for model {os.path.basename(model_path)}: {str(e)}"
                # Log the signature verification error
                if self.audit_logger is not None and username is not None:
                    log_model_signature_failed(self.audit_logger, username, "unknown", os.path.basename(model_path), error_msg, source_ip)
                raise ValueError(error_msg)
        
        # Load metadata if provided
        if metadata_path is not None and os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                model_metadata = json.load(f)
        else:
            # Extract metadata from model file name
            model_filename = os.path.basename(model_path)
            model_name, version = model_filename.split("_", 1)
            version = version.split(".")[0]
            
            model_metadata = {
                "model_id": str(uuid.uuid4()),
                "model_name": model_name,
                "model_type": "unknown",
                "version": version,
                "description": f"Imported from {model_path}",
                "location": model_path,
                "status": "active",
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "metadata": {},
                "metrics": {},
                "tags": ["imported"],
                "encrypted": is_encrypted,
                "signed": has_signature
            }
        
        # Create model directory
        model_dir = os.path.join(self.registry_dir, model_metadata["model_name"], model_metadata["version"])
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy model file to registry
        registry_path = os.path.join(model_dir, os.path.basename(model_path))
        shutil.copy2(import_source_path, registry_path)
        
        # Clean up temporary files if created
        if is_encrypted and self.enable_encryption and self.encryption_manager is not None:
            shutil.rmtree(os.path.dirname(import_source_path), ignore_errors=True)
        
        # Copy signature file if it exists
        if has_signature:
            registry_signature_path = os.path.join(model_dir, os.path.basename(signature_path))
            shutil.copy2(signature_path, registry_signature_path)
            model_metadata["signature_path"] = registry_signature_path
        
        # Update model location
        model_metadata["location"] = registry_path
        
        # Save metadata to registry
        model_id = model_metadata["model_id"]
        self.models[model_id] = model_metadata
        self._save_metadata()
        
        # Save metadata to database
        try:
            ModelMetadata.create_from_registry_entry(
                model_id=model_id,
                model_name=model_metadata["model_name"],
                model_type=model_metadata["model_type"],
                version=model_metadata["version"],
                location=registry_path,
                metadata=model_metadata["metadata"],
                metrics=model_metadata["metrics"],
                tags=model_metadata["tags"],
                description=model_metadata["description"],
                status=model_metadata["status"]
            )
        except Exception as e:
            logger.error(f"Error saving model metadata to database: {e}")
        
        # Sign the model if requested
        if sign_model and self.enable_signing and self.model_signer is not None:
            try:
                signature_path = self.model_signer.sign_model(registry_path)
                model_metadata["signed"] = True
                model_metadata["signature_path"] = signature_path
                self.models[model_id] = model_metadata
                self._save_metadata()
                
                # Log the model signing event
                if self.audit_logger is not None and username is not None:
                    log_model_signed(self.audit_logger, username, model_id, model_metadata["model_name"], model_id, source_ip)
            except Exception as e:
                logger.error(f"Failed to sign model {model_metadata['model_name']} version {model_metadata['version']}: {e}")
        
        # Encrypt the model if requested
        if encrypt_model and self.enable_encryption and self.encryption_manager is not None:
            try:
                self.encryption_manager.encrypt_model(model_metadata["model_name"], model_metadata["version"], encryption_password)
                model_metadata["encrypted"] = True
                self.models[model_id] = model_metadata
                self._save_metadata()
                
                # Log the model encryption event
                if self.audit_logger is not None and username is not None:
                    log_model_encrypted(self.audit_logger, username, model_id, model_metadata["model_name"], source_ip)
            except Exception as e:
                logger.error(f"Failed to encrypt model {model_metadata['model_name']} version {model_metadata['version']}: {e}")
        
        # Log the model creation event
        if self.audit_logger is not None and username is not None:
            log_model_created(self.audit_logger, username, model_id, model_metadata["model_name"], model_metadata["model_type"], source_ip)
        
        logger.info(f"Imported model {model_metadata['model_name']} with version {model_metadata['version']} and ID {model_id}")
        
        return model_id