#!/usr/bin/env python3
"""
Test script to verify imports and instantiation of all pipeline components.

This script tests that all the core pipeline classes can be imported and instantiated
without any missing dependencies or import errors.
"""

import sys
import traceback
from typing import List, Tuple

def test_import(module_name: str, class_name: str) -> Tuple[bool, str]:
    """Test importing a class from a module.
    
    Args:
        module_name: Name of the module to import from
        class_name: Name of the class to import
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"✓ Successfully imported {class_name} from {module_name}")
        return True, ""
    except Exception as e:
        error_msg = f"✗ Failed to import {class_name} from {module_name}: {str(e)}"
        print(error_msg)
        return False, error_msg

def test_instantiation(module_name: str, class_name: str, *args, **kwargs) -> Tuple[bool, str]:
    """Test instantiating a class.
    
    Args:
        module_name: Name of the module to import from
        class_name: Name of the class to instantiate
        *args: Arguments to pass to constructor
        **kwargs: Keyword arguments to pass to constructor
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        instance = cls(*args, **kwargs)
        print(f"✓ Successfully instantiated {class_name}")
        return True, ""
    except Exception as e:
        error_msg = f"✗ Failed to instantiate {class_name}: {str(e)}"
        print(error_msg)
        return False, error_msg

def main():
    """Main test function."""
    print("Testing Friday AI Trading System Pipeline Components")
    print("=" * 60)
    
    # List of components to test
    components = [
        ("src.data.integration.data_pipeline", "DataPipeline"),
        ("src.data.processing.data_validator", "DataValidator"),
        ("src.data.processing.data_cleaner", "DataCleaner"),
        ("src.data.processing.feature_engineering", "FeatureEngineer"),
        ("src.data.storage.data_storage", "DataStorage"),
        ("src.data.processing.data_processor", "DataProcessor"),
    ]
    
    print("\n1. Testing Imports:")
    print("-" * 30)
    
    import_results = []
    for module_name, class_name in components:
        success, error = test_import(module_name, class_name)
        import_results.append((module_name, class_name, success, error))
    
    print("\n2. Testing Instantiation:")
    print("-" * 30)
    
    instantiation_results = []
    
    # Test DataPipeline
    try:
        success, error = test_instantiation("src.data.integration.data_pipeline", "DataPipeline", "test_pipeline")
        instantiation_results.append(("DataPipeline", success, error))
    except Exception as e:
        instantiation_results.append(("DataPipeline", False, str(e)))
    
    # Test DataValidator
    try:
        success, error = test_instantiation("src.data.processing.data_validator", "DataValidator")
        instantiation_results.append(("DataValidator", success, error))
    except Exception as e:
        instantiation_results.append(("DataValidator", False, str(e)))
    
    # Test DataCleaner
    try:
        success, error = test_instantiation("src.data.processing.data_cleaner", "DataCleaner")
        instantiation_results.append(("DataCleaner", success, error))
    except Exception as e:
        instantiation_results.append(("DataCleaner", False, str(e)))
    
    # Test FeatureEngineer
    try:
        success, error = test_instantiation("src.data.processing.feature_engineering", "FeatureEngineer")
        instantiation_results.append(("FeatureEngineer", success, error))
    except Exception as e:
        instantiation_results.append(("FeatureEngineer", False, str(e)))
    
    # Test DataProcessor (base class)
    try:
        success, error = test_instantiation("src.data.processing.data_processor", "DataProcessor")
        instantiation_results.append(("DataProcessor", success, error))
    except Exception as e:
        instantiation_results.append(("DataProcessor", False, str(e)))
    
    # Note: DataStorage is abstract, so we'll skip instantiation test
    
    # Print summary
    print("\n3. Test Summary:")
    print("-" * 30)
    
    import_failures = [r for r in import_results if not r[2]]
    instantiation_failures = [r for r in instantiation_results if not r[1]]
    
    print(f"Import Tests: {len(components) - len(import_failures)}/{len(components)} passed")
    print(f"Instantiation Tests: {len(instantiation_results) - len(instantiation_failures)}/{len(instantiation_results)} passed")
    
    if import_failures:
        print("\nImport Failures:")
        for module, class_name, _, error in import_failures:
            print(f"  - {class_name}: {error}")
    
    if instantiation_failures:
        print("\nInstantiation Failures:")
        for class_name, _, error in instantiation_failures:
            print(f"  - {class_name}: {error}")
    
    print("\n4. External Dependencies Check:")
    print("-" * 30)
    
    # Check for key external dependencies
    dependencies = [
        "pandas",
        "numpy", 
        "pytz",
        "json",
        "datetime",
        "traceback",
        "enum",
        "typing",
        "abc",
        "time"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} is available")
        except ImportError:
            print(f"✗ {dep} is missing")
    
    # Test for optional dependencies used in the classes
    optional_deps = [
        ("scikit-learn", "sklearn"),
    ]
    
    print("\nOptional Dependencies:")
    for dep_name, import_name in optional_deps:
        try:
            __import__(import_name)
            print(f"✓ {dep_name} is available")
        except ImportError:
            print(f"? {dep_name} is not installed (used for advanced outlier detection)")
    
    success_count = len(components) - len(import_failures) + len(instantiation_results) - len(instantiation_failures)
    total_tests = len(components) + len(instantiation_results)
    
    if success_count == total_tests:
        print(f"\n🎉 All tests passed! ({success_count}/{total_tests})")
        return 0
    else:
        print(f"\n⚠️  Some tests failed ({success_count}/{total_tests})")
        return 1

if __name__ == "__main__":
    sys.exit(main())
