#!/usr/bin/env python3
"""
Infrastructure Initialization Script for Phase 1 Tasks 4 and 5
Initializes and configures core infrastructure components and storage directories.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def ensure_storage_directories():
    """
    Phase 1 Task 4: Storage Directory Configuration
    Set up and verify storage directories.
    """
    print("=" * 60)
    print("Phase 1 Task 4: Storage Directory Configuration")
    print("=" * 60)
    
    storage_root = project_root / "storage"
    
    # Define required directories
    directories = {
        "data": storage_root / "data",
        "logs": storage_root / "logs", 
        "models": storage_root / "models",
        "memory": storage_root / "memory",
        "backups": storage_root / "backups",
        "cache": storage_root / "cache" if not (storage_root / "cache").exists() else storage_root / "cache"
    }
    
    # Create directories if they don't exist
    for dir_name, dir_path in directories.items():
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")
        else:
            print(f"✓ Directory exists: {dir_path}")
        
        # Verify directory is writable
        test_file = dir_path / ".test_write"
        try:
            test_file.write_text("test")
            test_file.unlink()
            print(f"✓ Directory {dir_name} is writable")
        except Exception as e:
            print(f"✗ Directory {dir_name} is not writable: {e}")
            return False
    
    print("✓ All storage directories configured successfully")
    return True

def initialize_logging_system():
    """
    Initialize the logging system component of infrastructure.
    """
    print("\n--- Initializing Logging System ---")
    
    try:
        from infrastructure.logging import setup_logging, get_logger
        
        # Setup logging
        setup_logging(log_level=logging.INFO)
        logger = get_logger("infrastructure.init")
        
        logger.info("Logging system initialized successfully")
        print("✓ Logging system initialized")
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize logging system: {e}")
        return False

def initialize_config_system():
    """
    Initialize the configuration management system.
    """
    print("\n--- Initializing Configuration System ---")
    
    try:
        from infrastructure.config import unified_config
        from infrastructure.config.config_manager import ConfigurationManager
        
        # Initialize config manager
        config_manager = ConfigurationManager.get_instance()
        
        # Test configuration loading
        test_config = config_manager.get("database.host", "localhost")
        
        print("✓ Configuration system initialized")
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize configuration system: {e}")
        return False

def initialize_event_system():
    """
    Initialize the event system component.
    """
    print("\n--- Initializing Event System ---")
    
    try:
        from infrastructure.event.event_system import EventSystem, Event
        
        # Initialize event system
        event_system = EventSystem(max_queue_size=1000, enable_persistence=True)
        event_system.start()
        
        # Test event emission
        test_event = event_system.emit("test.event", {"message": "Infrastructure initialization test"})
        
        # Register a test handler
        def test_handler(event):
            print(f"✓ Event received: {event.event_type} - {event.data}")
        
        event_system.register_handler(test_handler, ["test.event"])
        
        # Stop the event system after test
        event_system.stop()
        
        print("✓ Event system initialized")
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize event system: {e}")
        return False

def initialize_security_system():
    """
    Initialize security components.
    """
    print("\n--- Initializing Security System ---")
    
    try:
        from infrastructure.security.access_control import AccessControl
        from infrastructure.security.audit_logging import SecurityAuditLogger
        
        # Initialize access control
        access_control = AccessControl()
        
        # Initialize audit logging
        audit_logger = SecurityAuditLogger()
        
        print("✓ Security system initialized")
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize security system: {e}")
        return False

def initialize_cache_system():
    """
    Initialize caching mechanisms.
    """
    print("\n--- Initializing Cache System ---")
    
    try:
        from infrastructure.cache.redis_cache import RedisCache
        
        # Initialize Redis cache (will use in-memory if Redis not available)
        cache = RedisCache()
        
        # Test cache operations
        cache.set("test_key", "test_value", expiry=60)
        value = cache.get("test_key")
        
        if value == "test_value":
            print("✓ Cache system initialized and tested")
            return True
        else:
            print("✗ Cache test failed")
            return False
            
    except Exception as e:
        print(f"✗ Failed to initialize cache system: {e}")
        return False

def initialize_database_system():
    """
    Initialize database connections and setup.
    """
    print("\n--- Initializing Database System ---")
    
    try:
        from infrastructure.database.setup_databases import setup_all_databases
        from infrastructure.database.verify_db import verify_all_database_connections
        
        # Setup databases
        setup_result = setup_all_databases()
        setup_success = setup_result.get('overall_success', False)
        
        if setup_success:
            # Verify connections
            verify_success = verify_all_database_connections()
            
            if verify_success:
                print("✓ Database system initialized")
                return True
            else:
                print("✗ Database verification failed")
                return False
        else:
            print("✗ Database setup failed")
            return False
            
    except Exception as e:
        print(f"✗ Failed to initialize database system: {e}")
        return False

def run_infrastructure_tests():
    """
    Run comprehensive tests for all infrastructure components.
    """
    print("\n" + "=" * 60)
    print("Running Infrastructure Integration Tests")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Storage directory access
    try:
        storage_path = project_root / "storage" / "data" / "test_file.txt"
        storage_path.write_text("Infrastructure test")
        content = storage_path.read_text()
        storage_path.unlink()
        
        if content == "Infrastructure test":
            print("✓ Storage directory read/write test passed")
            test_results.append(True)
        else:
            print("✗ Storage directory read/write test failed")
            test_results.append(False)
            
    except Exception as e:
        print(f"✗ Storage directory test failed: {e}")
        test_results.append(False)
    
    return all(test_results)

def main():
    """
    Main infrastructure initialization function.
    """
    print("Friday AI Trading System - Infrastructure Initialization")
    print("=" * 60)
    print("Initializing Phase 1 Tasks 4 and 5...")
    
    success = True
    
    # Phase 1 Task 4: Storage Directory Configuration
    if not ensure_storage_directories():
        success = False
    
    print("\n" + "=" * 60)
    print("Phase 1 Task 5: Core Infrastructure Activation")
    print("=" * 60)
    
    # Phase 1 Task 5: Core Infrastructure Activation
    components = [
        ("Logging System", initialize_logging_system),
        ("Configuration System", initialize_config_system),
        ("Event System", initialize_event_system),
        ("Security System", initialize_security_system),
        ("Cache System", initialize_cache_system),
        ("Database System", initialize_database_system)
    ]
    
    for component_name, init_func in components:
        if not init_func():
            success = False
    
    # Run integration tests
    if not run_infrastructure_tests():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ ALL INFRASTRUCTURE COMPONENTS INITIALIZED SUCCESSFULLY")
        print("✓ Phase 1 Tasks 4 and 5 COMPLETED")
    else:
        print("✗ SOME INFRASTRUCTURE COMPONENTS FAILED TO INITIALIZE")
        print("✗ Please check the error messages above")
    
    print("=" * 60)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
