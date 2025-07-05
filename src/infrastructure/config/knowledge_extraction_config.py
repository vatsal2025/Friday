# Knowledge Extraction System Configuration

"""
Configuration settings for the Knowledge Extraction System components.

This module imports the unified configuration settings from unified_config.py
and provides access to the Knowledge Extraction configuration.

Components:
- OCR Book Digitizer
- Multimodal Content Processor
- Book Knowledge Extractor
- Knowledge Base Builder
- Strategy Generator
- Knowledge Extraction Integration
"""

from typing import Dict, Any
from infrastructure.config.unified_config import KNOWLEDGE_EXTRACTION_CONFIG

# Map the unified configuration to the component-specific configurations
ocr_config = KNOWLEDGE_EXTRACTION_CONFIG['ocr']
multimodal_processor_config = KNOWLEDGE_EXTRACTION_CONFIG['multimodal_processing']
knowledge_extractor_config = KNOWLEDGE_EXTRACTION_CONFIG['knowledge_extraction']
knowledge_base_config = KNOWLEDGE_EXTRACTION_CONFIG['knowledge_base']
strategy_generator_config = KNOWLEDGE_EXTRACTION_CONFIG['strategy_generation']
integration_config = KNOWLEDGE_EXTRACTION_CONFIG['integration']

# Combined Knowledge Extraction System Configuration
knowledge_extraction_system_config = {
    "ocr_digitizer": ocr_config,
    "multimodal_processor": multimodal_processor_config,
    "knowledge_extractor": knowledge_extractor_config,
    "knowledge_base": knowledge_base_config,
    "strategy_generator": strategy_generator_config,
    "integration": integration_config,
    "system": {
        "debug_mode": False,
        "log_level": "INFO",
        "data_directory": "data/knowledge_extraction",
        "temp_directory": "temp/knowledge_extraction"
    }
}