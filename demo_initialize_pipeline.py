#!/usr/bin/env python3
"""Demonstration of the new initialize_pipeline function.

This script demonstrates:
1. Basic pipeline initialization
2. Pipeline with custom EventSystem integration
3. Processing sample data
4. Monitoring pipeline events
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from pipelines.market_data_pipeline import initialize_pipeline, load_sample_data
from src.infrastructure.logging import get_logger
from src.infrastructure.event import EventSystem

# Setup logger
logger = get_logger(__name__)


def demo_basic_usage():
    """Demonstrate basic pipeline usage."""
    logger.info("=== Demo 1: Basic Pipeline Usage ===")
    
    # Initialize pipeline with default settings
    # This automatically creates and starts an EventSystem
    pipeline = initialize_pipeline(
        warn_only=True,
        output_dir="demo_output",
        enable_all_features=True,
        enable_storage=False  # Disable storage for demo
    )
    
    # Check pipeline configuration
    logger.info(f"Pipeline initialized with {len(pipeline.pipeline.stages)} stages:")
    for stage in pipeline.pipeline.stages:
        logger.info(f"  - {stage['name']} ({stage['type'].name})")
    
    # Generate and process sample data
    sample_data = load_sample_data()
    logger.info(f"Generated sample data: {sample_data.shape}")
    
    processed_data = pipeline.process_data(sample_data)
    logger.info(f"Processed data: {processed_data.shape}")
    logger.info(f"Features added: {len(processed_data.columns) - len(sample_data.columns)}")
    
    return processed_data


def demo_event_system_integration():
    """Demonstrate pipeline with custom EventSystem integration."""
    logger.info("=== Demo 2: EventSystem Integration ===")
    
    # Create and start custom event system
    event_system = EventSystem()
    event_system.start()
    
    # Create event handlers for monitoring
    def stage_monitor(event):
        """Monitor stage events."""
        if event.event_type == "pipeline.stage.start":
            logger.info(f"🚀 Stage starting: {event.data['stage_name']}")
        elif event.event_type == "pipeline.stage.success":
            duration = event.data['duration']
            logger.info(f"✅ Stage completed: {event.data['stage_name']} ({duration:.2f}s)")
    
    def validation_monitor(event):
        """Monitor validation events."""
        if event.event_type == "pipeline.validation.metrics":
            metrics = event.data['validation_metrics']
            rules_passed = metrics['rules_passed']
            rules_tested = metrics['rules_tested']
            logger.info(f"📊 Validation: {rules_passed}/{rules_tested} rules passed")
    
    def pipeline_monitor(event):
        """Monitor overall pipeline events."""
        if event.event_type == "pipeline.start":
            logger.info(f"🏃 Pipeline started: {event.data['pipeline_name']}")
        elif event.event_type == "pipeline.success":
            duration = event.data['duration']
            logger.info(f"🎉 Pipeline completed successfully in {duration:.2f}s")
    
    # Register event handlers
    event_system.register_handler(
        stage_monitor,
        event_types=["pipeline.stage.start", "pipeline.stage.success"]
    )
    
    event_system.register_handler(
        validation_monitor,
        event_types=["pipeline.validation.metrics"]
    )
    
    event_system.register_handler(
        pipeline_monitor,
        event_types=["pipeline.start", "pipeline.success"]
    )
    
    # Initialize pipeline with custom event system
    pipeline = initialize_pipeline(
        event_system=event_system,
        warn_only=True,
        output_dir="demo_output",
        enable_storage=False
    )
    
    # Process data to trigger events
    sample_data = load_sample_data()
    processed_data = pipeline.process_data(sample_data)
    
    # Give a moment for events to be processed
    import time
    time.sleep(0.1)
    
    # Stop event system
    event_system.stop()
    
    return processed_data


def demo_stage_names_verification():
    """Demonstrate that stage names match requirements for orchestration."""
    logger.info("=== Demo 3: Stage Names Verification ===")
    
    # Initialize pipeline with all stages
    pipeline = initialize_pipeline(enable_storage=True)
    
    stage_names = [stage['name'] for stage in pipeline.pipeline.stages]
    logger.info("Pipeline stages for orchestration:")
    for i, name in enumerate(stage_names, 1):
        logger.info(f"  {i}. {name}")
    
    # Verify expected stage names
    expected_stages = [
        "enhanced_validation",
        "market_data_cleaning",
        "feature_engineering",
        "parquet_storage"
    ]
    
    if stage_names == expected_stages:
        logger.info("✅ All stage names match orchestration requirements!")
    else:
        logger.error(f"❌ Stage names mismatch! Expected: {expected_stages}")
    
    return stage_names


def demo_pipeline_configuration_options():
    """Demonstrate different pipeline configuration options."""
    logger.info("=== Demo 4: Configuration Options ===")
    
    # Option 1: Minimal pipeline (no storage)
    pipeline_minimal = initialize_pipeline(
        enable_storage=False,
        enable_all_features=False
    )
    logger.info(f"Minimal pipeline: {len(pipeline_minimal.pipeline.stages)} stages")
    
    # Option 2: Full pipeline with storage
    pipeline_full = initialize_pipeline(
        enable_storage=True,
        enable_all_features=True,
        output_dir="full_pipeline_output"
    )
    logger.info(f"Full pipeline: {len(pipeline_full.pipeline.stages)} stages")
    
    # Option 3: Warn-only mode for production
    pipeline_production = initialize_pipeline(
        warn_only=True,  # Don't fail on validation warnings
        enable_storage=True,
        output_dir="production_output"
    )
    logger.info(f"Production pipeline (warn-only): {len(pipeline_production.pipeline.stages)} stages")
    
    return {
        "minimal": pipeline_minimal,
        "full": pipeline_full,
        "production": pipeline_production
    }


def main():
    """Run all demonstrations."""
    logger.info("Starting initialize_pipeline demonstrations...")
    logger.info("")
    
    try:
        # Demo 1: Basic usage
        demo_basic_usage()
        logger.info("")
        
        # Demo 2: Event system integration
        demo_event_system_integration()
        logger.info("")
        
        # Demo 3: Stage names verification
        demo_stage_names_verification()
        logger.info("")
        
        # Demo 4: Configuration options
        demo_pipeline_configuration_options()
        logger.info("")
        
        logger.info("🎉 All demonstrations completed successfully!")
        logger.info("")
        logger.info("Key Features Demonstrated:")
        logger.info("  ✅ DataPipeline instantiation with EventSystem")
        logger.info("  ✅ Stage order: enhanced_validation → market_data_cleaning → feature_engineering → parquet_storage")
        logger.info("  ✅ Event publishing during pipeline execution")
        logger.info("  ✅ Stage names matching task requirements for orchestration")
        logger.info("  ✅ Flexible configuration options")
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
