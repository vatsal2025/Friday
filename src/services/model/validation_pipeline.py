"""Validation Pipeline for Friday AI Trading System.

This module provides a pipeline-based approach to model validation,
allowing for multi-stage validation with different validation rules.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Type
from enum import Enum

from src.infrastructure.logging import get_logger
from src.orchestration.trading_engine.model_versioning import ValidationResult
from src.services.model.model_validation_rules import ValidationRule, ModelType, ValidationRuleType

# Create logger
logger = get_logger(__name__)


class ValidationStage(Enum):
    """Stages in the validation pipeline."""
    BASIC = "basic"  # Basic validation (interface, etc.)
    DATA_COMPATIBILITY = "data_compatibility"  # Data compatibility validation
    PERFORMANCE = "performance"  # Performance validation
    RESOURCE = "resource"  # Resource usage validation
    SECURITY = "security"  # Security validation
    CUSTOM = "custom"  # Custom validation


class ValidationPipelineResult:
    """Result of running a validation pipeline."""
    
    def __init__(self):
        """Initialize a validation pipeline result."""
        self.start_time = time.time()
        self.end_time = None
        self.duration = None
        self.overall_result = ValidationResult.PASS
        self.stage_results = {}
        self.rule_results = {}
    
    def add_rule_result(self, stage: ValidationStage, rule: ValidationRule, 
                       result: ValidationResult, message: str) -> None:
        """Add a rule validation result.
        
        Args:
            stage: Validation stage
            rule: Validation rule
            result: Validation result
            message: Validation message
        """
        if stage.value not in self.stage_results:
            self.stage_results[stage.value] = {
                "result": ValidationResult.PASS,
                "rules": []
            }
        
        # Add rule result
        rule_result = {
            "rule_name": rule.name,
            "rule_type": rule.rule_type.value,
            "result": result.value,
            "message": message
        }
        self.stage_results[stage.value]["rules"].append(rule_result)
        self.rule_results[rule.name] = rule_result
        
        # Update stage result based on priority: FAIL > ERROR > WARNING > PASS
        stage_result = self.stage_results[stage.value]["result"]
        if result == ValidationResult.FAIL:
            self.stage_results[stage.value]["result"] = ValidationResult.FAIL
        elif result == ValidationResult.ERROR and stage_result != ValidationResult.FAIL:
            self.stage_results[stage.value]["result"] = ValidationResult.ERROR
        elif result == ValidationResult.WARNING and stage_result not in [ValidationResult.FAIL, ValidationResult.ERROR]:
            self.stage_results[stage.value]["result"] = ValidationResult.WARNING
        
        # Update overall result
        if result == ValidationResult.FAIL:
            self.overall_result = ValidationResult.FAIL
        elif result == ValidationResult.ERROR and self.overall_result != ValidationResult.FAIL:
            self.overall_result = ValidationResult.ERROR
        elif result == ValidationResult.WARNING and self.overall_result not in [ValidationResult.FAIL, ValidationResult.ERROR]:
            self.overall_result = ValidationResult.WARNING
    
    def finalize(self) -> None:
        """Finalize the validation result by setting end time and duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation result to a dictionary.
        
        Returns:
            Dictionary representation of the validation result
        """
        return {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration": self.duration,
            "overall_result": self.overall_result.value,
            "stage_results": self.stage_results,
            "rule_results": self.rule_results
        }
    
    def get_failed_rules(self) -> List[Dict[str, Any]]:
        """Get a list of failed validation rules.
        
        Returns:
            List of failed rule results
        """
        return [rule for rule in self.rule_results.values() 
                if rule["result"] == ValidationResult.FAIL.value]
    
    def get_warnings(self) -> List[Dict[str, Any]]:
        """Get a list of validation warnings.
        
        Returns:
            List of warning rule results
        """
        return [rule for rule in self.rule_results.values() 
                if rule["result"] == ValidationResult.WARNING.value]
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """Get a list of validation errors.
        
        Returns:
            List of error rule results
        """
        return [rule for rule in self.rule_results.values() 
                if rule["result"] == ValidationResult.ERROR.value]


class ValidationPipeline:
    """Pipeline for validating models with multiple stages and rules."""
    
    def __init__(self, name: str = "default"):
        """Initialize a validation pipeline.
        
        Args:
            name: Name of the pipeline
        """
        self.name = name
        self.stages = {}
        for stage in ValidationStage:
            self.stages[stage] = []
        logger.info(f"Initialized validation pipeline: {name}")
    
    def add_rule(self, stage: ValidationStage, rule: ValidationRule) -> None:
        """Add a validation rule to a stage.
        
        Args:
            stage: Validation stage to add the rule to
            rule: Validation rule to add
        """
        self.stages[stage].append(rule)
        logger.info(f"Added rule '{rule.name}' to stage '{stage.value}' in pipeline '{self.name}'")
    
    def add_rules(self, stage: ValidationStage, rules: List[ValidationRule]) -> None:
        """Add multiple validation rules to a stage.
        
        Args:
            stage: Validation stage to add the rules to
            rules: List of validation rules to add
        """
        self.stages[stage].extend(rules)
        logger.info(f"Added {len(rules)} rules to stage '{stage.value}' in pipeline '{self.name}'")
    
    def validate(self, model: Any, metadata: Dict[str, Any], 
                stop_on_fail: bool = False) -> ValidationPipelineResult:
        """Validate a model using the pipeline.
        
        Args:
            model: The model to validate
            metadata: Model metadata
            stop_on_fail: Whether to stop validation if a rule fails
            
        Returns:
            ValidationPipelineResult: Result of the validation
        """
        result = ValidationPipelineResult()
        
        # Run validation stages in order
        for stage in ValidationStage:
            stage_rules = self.stages[stage]
            if not stage_rules:
                continue  # Skip empty stages
            
            logger.info(f"Running validation stage '{stage.value}' with {len(stage_rules)} rules")
            
            # Run all rules in this stage
            for rule in stage_rules:
                try:
                    logger.info(f"Running validation rule '{rule.name}'")
                    validation_result, message = rule.validate(model, metadata)
                    result.add_rule_result(stage, rule, validation_result, message)
                    
                    # Log the result
                    log_level = logging.INFO
                    if validation_result == ValidationResult.FAIL:
                        log_level = logging.ERROR
                    elif validation_result == ValidationResult.ERROR:
                        log_level = logging.ERROR
                    elif validation_result == ValidationResult.WARNING:
                        log_level = logging.WARNING
                    
                    logger.log(log_level, f"Validation rule '{rule.name}' result: {validation_result.value} - {message}")
                    
                    # Stop if rule failed and stop_on_fail is True
                    if stop_on_fail and validation_result == ValidationResult.FAIL:
                        logger.warning(f"Stopping validation pipeline due to rule failure: {rule.name}")
                        result.finalize()
                        return result
                
                except Exception as e:
                    logger.error(f"Error running validation rule '{rule.name}': {str(e)}")
                    result.add_rule_result(
                        stage, rule, ValidationResult.ERROR, f"Error running rule: {str(e)}"
                    )
            
            # Stop if stage failed and stop_on_fail is True
            stage_result = result.stage_results.get(stage.value, {}).get("result")
            if stop_on_fail and stage_result == ValidationResult.FAIL:
                logger.warning(f"Stopping validation pipeline due to stage failure: {stage.value}")
                result.finalize()
                return result
        
        # Finalize the result
        result.finalize()
        logger.info(f"Validation pipeline '{self.name}' completed with result: {result.overall_result.value}")
        return result


def create_default_pipeline(model_type: ModelType) -> ValidationPipeline:
    """Create a default validation pipeline for a model type.
    
    Args:
        model_type: Type of model to create a pipeline for
        
    Returns:
        ValidationPipeline: Default validation pipeline for the model type
    """
    from src.services.model.model_validation_rules import get_validation_rules_for_model_type
    
    pipeline = ValidationPipeline(name=f"default_{model_type.value}")
    
    # Get rules for the model type
    rules = get_validation_rules_for_model_type(model_type)
    
    # Add rules to appropriate stages
    for rule in rules:
        if rule.rule_type == ValidationRuleType.INTERFACE:
            pipeline.add_rule(ValidationStage.BASIC, rule)
        elif rule.rule_type == ValidationRuleType.DATA_COMPATIBILITY:
            pipeline.add_rule(ValidationStage.DATA_COMPATIBILITY, rule)
        elif rule.rule_type == ValidationRuleType.PERFORMANCE:
            pipeline.add_rule(ValidationStage.PERFORMANCE, rule)
        elif rule.rule_type == ValidationRuleType.RESOURCE:
            pipeline.add_rule(ValidationStage.RESOURCE, rule)
        elif rule.rule_type == ValidationRuleType.SECURITY:
            pipeline.add_rule(ValidationStage.SECURITY, rule)
        else:  # ValidationRuleType.CUSTOM
            pipeline.add_rule(ValidationStage.CUSTOM, rule)
    
    return pipeline


# Create a registry of validation pipelines
validation_pipeline_registry = {}


def register_pipeline(pipeline: ValidationPipeline) -> None:
    """Register a validation pipeline.
    
    Args:
        pipeline: Validation pipeline to register
    """
    validation_pipeline_registry[pipeline.name] = pipeline
    logger.info(f"Registered validation pipeline: {pipeline.name}")


def get_pipeline(name: str) -> Optional[ValidationPipeline]:
    """Get a registered validation pipeline by name.
    
    Args:
        name: Name of the pipeline to get
        
    Returns:
        ValidationPipeline or None if not found
    """
    return validation_pipeline_registry.get(name)


# Register default pipelines for each model type
for model_type in ModelType:
    pipeline = create_default_pipeline(model_type)
    register_pipeline(pipeline)