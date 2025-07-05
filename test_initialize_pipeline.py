#!/usr/bin/env python3
"""Test script to verify the initialize_pipeline function works correctly.

This script tests:
1. Basic pipeline initialization with default settings
2. Pipeline initialization with custom EventSystem
3. Pipeline execution with sample data
4. Verification of stage names for orchestration
"""

import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from pipelines.market_data_pipeline import initialize_pipeline, load_sample_data
from src.infrastructure.logging import get_logger
from src.infrastructure.event import EventSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


def test_basic_initialization():
    """Test basic pipeline initialization."""
    logger.info("=== Testing Basic Pipeline Initialization ===")
    
    try:
        # Initialize pipeline with defaults
        pipeline = initialize_pipeline()
        
        # Verify pipeline structure
        assert pipeline is not None, "Pipeline should not be None"
        assert pipeline.pipeline is not None, "Internal DataPipeline should not be None"
        assert pipeline.event_system is not None, "EventSystem should be created automatically"
        
        # Verify stage names match requirements
        stage_names = [stage['name'] for stage in pipeline.pipeline.stages]
        expected_stages = [
            "enhanced_validation",
            "market_data_cleaning", 
            "feature_engineering",
            "parquet_storage"
        ]
        
        logger.info(f"Pipeline stages: {stage_names}")
        assert stage_names == expected_stages, f"Expected {expected_stages}, got {stage_names}"
        
        logger.info("✅ Basic initialization test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic initialization test failed: {e}")
        return False


def test_custom_event_system():
    """Test pipeline initialization with custom EventSystem."""
    logger.info("=== Testing Custom EventSystem Integration ===")
    
    try:
        # Create and start custom event system
        event_system = EventSystem()
        event_system.start()
        
        # Track events
        events_received = []
        
        def event_handler(event):
            events_received.append(event.event_type)
            logger.info(f"Event: {event.event_type}")
        
        # Register handler
        event_system.register_handler(
            event_handler,
            event_types=["pipeline.start", "pipeline.stage.start", "pipeline.stage.success", "pipeline.success"]
        )
        
        # Initialize pipeline with custom event system
        pipeline = initialize_pipeline(
            event_system=event_system,
            warn_only=True,
            enable_storage=False  # Disable storage for testing
        )
        
        # Verify the event system is properly set
        assert pipeline.event_system is event_system, "Custom EventSystem should be used"
        
        # Process sample data to trigger events
        sample_data = load_sample_data()
        processed_data = pipeline.process_data(sample_data)
        
        # Give a small delay for events to be processed
        import time
        time.sleep(0.1)
        
        # Verify events were fired
        assert len(events_received) > 0, "Events should have been received"
        assert "pipeline.start" in events_received, "Pipeline start event should be fired"
        assert "pipeline.success" in events_received, "Pipeline success event should be fired"
        
        # Stop event system
        event_system.stop()
        
        logger.info(f"✅ Custom EventSystem test passed - {len(events_received)} events received")
        return True
        
    except Exception as e:
        logger.error(f"❌ Custom EventSystem test failed: {e}")
        return False


def test_pipeline_execution():
    """Test pipeline execution with sample data."""
    logger.info("=== Testing Pipeline Execution ===")
    
    try:
        # Initialize pipeline
        pipeline = initialize_pipeline(
            warn_only=True,
            output_dir="test_output",
            enable_storage=False  # Disable storage for testing
        )
        
        # Generate sample data
        sample_data = load_sample_data()
        logger.info(f"Generated sample data: {sample_data.shape}")
        
        # Process data
        processed_data = pipeline.process_data(sample_data)
        logger.info(f"Processed data: {processed_data.shape}")
        
        # Verify processing results
        assert processed_data is not None, "Processed data should not be None"
        assert len(processed_data) > 0, "Processed data should not be empty"
        assert len(processed_data.columns) > len(sample_data.columns), "Should have added features"
        
        # Verify required columns are present
        required_columns = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'timestamp']
        for col in required_columns:
            assert col in processed_data.columns, f"Column {col} should be present"
        
        # Get pipeline summary
        summary = pipeline.get_pipeline_summary()
        assert summary['total_stages'] == 3, "Should have 3 stages (validation, cleaning, feature_engineering)"
        assert summary['pipeline_name'] == "market_data_pipeline", "Pipeline name should match"
        
        logger.info(f"✅ Pipeline execution test passed - {len(processed_data.columns)} columns created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution test failed: {e}")
        return False


def test_stage_names_for_orchestration():
    """Test that stage names match requirements for future orchestration."""
    logger.info("=== Testing Stage Names for Orchestration ===")
    
    try:
        # Test with all stages enabled
        pipeline_full = initialize_pipeline(enable_storage=True)
        stage_names_full = [stage['name'] for stage in pipeline_full.pipeline.stages]
        
        expected_full = [
            "enhanced_validation",
            "market_data_cleaning",
            "feature_engineering", 
            "parquet_storage"
        ]
        
        assert stage_names_full == expected_full, f"Full pipeline stages mismatch: {stage_names_full}"
        
        # Test with storage disabled
        pipeline_no_storage = initialize_pipeline(enable_storage=False)
        stage_names_no_storage = [stage['name'] for stage in pipeline_no_storage.pipeline.stages]
        
        expected_no_storage = [
            "enhanced_validation",
            "market_data_cleaning",
            "feature_engineering"
        ]
        
        assert stage_names_no_storage == expected_no_storage, f"No-storage pipeline stages mismatch: {stage_names_no_storage}"
        
        logger.info("✅ Stage names test passed - all stage names match requirements")
        return True
        
    except Exception as e:
        logger.error(f"❌ Stage names test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting initialize_pipeline tests...")
    
    tests = [
        test_basic_initialization,
        test_custom_event_system,
        test_pipeline_execution,
        test_stage_names_for_orchestration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        logger.info("")  # Add spacing between tests
    
    logger.info(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        logger.info("🎉 All tests passed! The initialize_pipeline function is working correctly.")
        return 0
    else:
        logger.error(f"⚠️  {total - passed} tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
