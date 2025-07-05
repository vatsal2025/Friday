# Knowledge Engine Module

"""
The Knowledge Engine module provides components for extracting, processing,
and utilizing trading knowledge from various sources such as books, articles,
and research papers.

This module includes components for:
- Book digitization using OCR
- Multimodal content processing (text, tables, charts, formulas)
- Knowledge extraction from processed content
- Knowledge base building and management
- Trading strategy generation based on extracted knowledge
"""

from src.orchestration.knowledge_engine.book_knowledge_extractor import BookKnowledgeExtractor
from src.orchestration.knowledge_engine.knowledge_base_builder import KnowledgeBaseBuilder
# Temporarily commenting out problematic import
# from src.orchestration.knowledge_engine.strategy_generator import StrategyGenerator
from src.orchestration.knowledge_engine.multimodal_content_processor import MultimodalContentProcessor
from src.orchestration.knowledge_engine.ocr_book_digitizer import OCRBookDigitizer
from src.orchestration.knowledge_engine.knowledge_extraction_integration import KnowledgeExtractionIntegration

__all__ = [
    'BookKnowledgeExtractor',
    'KnowledgeBaseBuilder',
    # 'StrategyGenerator',
    'MultimodalContentProcessor',
    'OCRBookDigitizer',
    'KnowledgeExtractionIntegration'
]