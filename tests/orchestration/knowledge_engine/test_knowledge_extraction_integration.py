"""Unit tests for the KnowledgeExtractionIntegration class.

This module contains tests for the KnowledgeExtractionIntegration class to ensure
the entire knowledge extraction pipeline works correctly.
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.orchestration.knowledge_engine.knowledge_extraction_integration import KnowledgeExtractionIntegration
from src.infrastructure.config.config_manager import ConfigurationManager
from src.infrastructure.database.mongodb import MongoDB
from src.infrastructure.event.event_system import EventSystem


class TestKnowledgeExtractionIntegration(unittest.TestCase):
    """Tests for the KnowledgeExtractionIntegration class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_config = {
            'enabled': True,
            'ocr': {
                'engine': 'tesseract',
                'language': 'eng',
                'dpi': 300,
                'preprocessing': {
                    'denoise': True,
                    'deskew': True,
                    'contrast_enhancement': True
                },
                'confidence_threshold': 0.7
            },
            'multimodal_processing': {
                'text_processing': {
                    'language': 'en',
                    'spacy_model': 'en_core_web_md'
                },
                'table_processing': {
                    'max_rows': 1000,
                    'max_columns': 100
                }
            },
            'knowledge_extraction': {
                'enabled': True,
                'extraction_methods': {
                    'rule_based': {
                        'enabled': True,
                        'confidence_threshold': 0.7
                    }
                }
            },
            'knowledge_base': {
                'storage': {
                    'type': 'mongodb',
                    'collection_name': 'knowledge_base'
                }
            },
            'strategy_generation': {
                'enabled': True
            }
        }
        
        # Create mocks
        self.mock_config_manager = MagicMock()
        self.mock_config_manager.get_config.return_value = self.mock_config
        
        self.mock_db = MagicMock()
        self.mock_collection = MagicMock()
        self.mock_db.get_collection.return_value = self.mock_collection
        
        self.mock_event_system = MagicMock()
        
        # Mock component classes
        self.mock_ocr_digitizer = MagicMock()
        self.mock_content_processor = MagicMock()
        self.mock_knowledge_extractor = MagicMock()
        self.mock_knowledge_base = MagicMock()
        self.mock_strategy_generator = MagicMock()
        
        # Patch dependencies
        self.config_manager_patch = patch('src.orchestration.knowledge_engine.knowledge_extraction_integration.ConfigManager', 
                                         return_value=self.mock_config_manager)
        self.mongodb_patch = patch('src.orchestration.knowledge_engine.knowledge_extraction_integration.MongoDB', 
                                  return_value=self.mock_db)
        self.event_system_patch = patch('src.orchestration.knowledge_engine.knowledge_extraction_integration.EventSystem', 
                                       return_value=self.mock_event_system)
        self.ocr_digitizer_patch = patch('src.orchestration.knowledge_engine.knowledge_extraction_integration.OCRBookDigitizer', 
                                       return_value=self.mock_ocr_digitizer)
        self.content_processor_patch = patch('src.orchestration.knowledge_engine.knowledge_extraction_integration.MultimodalContentProcessor', 
                                           return_value=self.mock_content_processor)
        self.knowledge_extractor_patch = patch('src.orchestration.knowledge_engine.knowledge_extraction_integration.BookKnowledgeExtractor', 
                                            return_value=self.mock_knowledge_extractor)
        self.knowledge_base_patch = patch('src.orchestration.knowledge_engine.knowledge_extraction_integration.KnowledgeBaseBuilder', 
                                        return_value=self.mock_knowledge_base)
        self.strategy_generator_patch = patch('src.orchestration.knowledge_engine.knowledge_extraction_integration.StrategyGenerator', 
                                           return_value=self.mock_strategy_generator)
        
        # Start patches
        self.mock_config_manager_cls = self.config_manager_patch.start()
        self.mock_db_cls = self.mongodb_patch.start()
        self.mock_event_system_cls = self.event_system_patch.start()
        self.mock_ocr_digitizer_cls = self.ocr_digitizer_patch.start()
        self.mock_content_processor_cls = self.content_processor_patch.start()
        self.mock_knowledge_extractor_cls = self.knowledge_extractor_patch.start()
        self.mock_knowledge_base_cls = self.knowledge_base_patch.start()
        self.mock_strategy_generator_cls = self.strategy_generator_patch.start()
        
        # Create instance
        self.integration = KnowledgeExtractionIntegration()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop patches
        self.config_manager_patch.stop()
        self.mongodb_patch.stop()
        self.event_system_patch.stop()
        self.ocr_digitizer_patch.stop()
        self.content_processor_patch.stop()
        self.knowledge_extractor_patch.stop()
        self.knowledge_base_patch.stop()
        self.strategy_generator_patch.stop()
    
    def test_initialization(self):
        """Test initialization of KnowledgeExtractionIntegration."""
        # Verify config is loaded
        self.mock_config_manager.get_config.assert_called_with('knowledge_extraction')
        
        # Verify database collection is retrieved
        self.mock_db.get_collection.assert_called_with('knowledge_extraction_jobs')
        
        # Verify components are initialized
        self.mock_ocr_digitizer_cls.assert_called_once()
        self.mock_content_processor_cls.assert_called_once()
        self.mock_knowledge_extractor_cls.assert_called_once()
        self.mock_knowledge_base_cls.assert_called_once()
        self.mock_strategy_generator_cls.assert_called_once()
        
        # Verify integration has expected attributes
        self.assertEqual(self.integration.config, self.mock_config)
        self.assertEqual(self.integration.extraction_jobs_collection, self.mock_collection)
        self.assertEqual(self.integration.event_system, self.mock_event_system)
        self.assertEqual(self.integration.ocr_digitizer, self.mock_ocr_digitizer)
        self.assertEqual(self.integration.content_processor, self.mock_content_processor)
        self.assertEqual(self.integration.knowledge_extractor, self.mock_knowledge_extractor)
        self.assertEqual(self.integration.knowledge_base, self.mock_knowledge_base)
        self.assertEqual(self.integration.strategy_generator, self.mock_strategy_generator)
    
    def test_create_extraction_job(self):
        """Test creating an extraction job."""
        # Test data
        book_path = "/path/to/book.pdf"
        metadata = {"title": "Trading Book", "author": "Trader Joe"}
        options = {"ocr": {"language": "eng"}}
        
        # Mock database insert
        self.mock_collection.insert_one.return_value = MagicMock(inserted_id="job_id_1")
        
        # Call method
        job_id = self.integration._create_extraction_job(book_path, metadata, options)
        
        # Verify database was called
        self.mock_collection.insert_one.assert_called_once()
        
        # Verify result
        self.assertEqual(job_id, "job_id_1")
    
    def test_update_job_status(self):
        """Test updating job status."""
        # Test data
        job_id = "job_id_1"
        status = "processing"
        details = {"step": "digitization"}
        
        # Mock database update
        self.mock_collection.update_one.return_value = MagicMock(modified_count=1)
        
        # Call method
        self.integration._update_job_status(job_id, status, details)
        
        # Verify database was called
        self.mock_collection.update_one.assert_called_once()
    
    def test_digitize_book(self):
        """Test digitizing a book."""
        # Test data
        book_path = "/path/to/book.pdf"
        metadata = {"title": "Trading Book", "author": "Trader Joe"}
        job_id = "job_id_1"
        
        # Mock OCR digitizer
        self.mock_ocr_digitizer.digitize_book.return_value = {
            "success": True,
            "book_id": "book_id_1",
            "page_count": 10,
            "content_type": "pdf"
        }
        
        # Call method
        result = self.integration._digitize_book(book_path, metadata, job_id)
        
        # Verify OCR digitizer was called
        self.mock_ocr_digitizer.digitize_book.assert_called_once_with(book_path, metadata)
        
        # Verify job status was updated
        self.mock_collection.update_one.assert_called_once()
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["book_id"], "book_id_1")
    
    def test_process_multimodal_content(self):
        """Test processing multimodal content."""
        # Test data
        book_id = "book_id_1"
        job_id = "job_id_1"
        
        # Mock content processor
        self.mock_content_processor.process_book_content.return_value = {
            "success": True,
            "content_ids": ["content_id_1", "content_id_2"],
            "content_types": {"text": 8, "table": 2}
        }
        
        # Call method
        result = self.integration._process_multimodal_content(book_id, job_id)
        
        # Verify content processor was called
        self.mock_content_processor.process_book_content.assert_called_once_with(book_id)
        
        # Verify job status was updated
        self.mock_collection.update_one.assert_called_once()
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["content_ids"], ["content_id_1", "content_id_2"])
    
    def test_extract_knowledge(self):
        """Test extracting knowledge."""
        # Test data
        book_id = "book_id_1"
        content_ids = ["content_id_1", "content_id_2"]
        job_id = "job_id_1"
        
        # Mock content processor and knowledge extractor
        self.mock_content_processor.get_processed_content.return_value = {
            "success": True,
            "content": [{"type": "text", "content": "Buy when RSI < 30"}]
        }
        
        self.mock_knowledge_extractor.extract_knowledge.return_value = [
            {"rule": "Buy when RSI < 30", "confidence": 0.85}
        ]
        
        # Call method
        result = self.integration._extract_knowledge(book_id, content_ids, job_id)
        
        # Verify content processor and knowledge extractor were called
        self.mock_content_processor.get_processed_content.assert_called_once()
        self.mock_knowledge_extractor.extract_knowledge.assert_called_once()
        
        # Verify job status was updated
        self.mock_collection.update_one.assert_called_once()
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(len(result["knowledge_items"]), 1)
    
    def test_build_knowledge_base(self):
        """Test building knowledge base."""
        # Test data
        knowledge_items = [{"rule": "Buy when RSI < 30", "confidence": 0.85}]
        book_id = "book_id_1"
        job_id = "job_id_1"
        
        # Mock knowledge base
        self.mock_knowledge_base.add_knowledge_items.return_value = {
            "success": True,
            "item_ids": ["item_id_1"],
            "count": 1
        }
        
        # Call method
        result = self.integration._build_knowledge_base(knowledge_items, book_id, job_id)
        
        # Verify knowledge base was called
        self.mock_knowledge_base.add_knowledge_items.assert_called_once_with(knowledge_items)
        
        # Verify job status was updated
        self.mock_collection.update_one.assert_called_once()
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["item_ids"], ["item_id_1"])
    
    def test_generate_strategies(self):
        """Test generating strategies."""
        # Test data
        knowledge_item_ids = ["item_id_1"]
        book_id = "book_id_1"
        job_id = "job_id_1"
        
        # Mock strategy generator
        self.mock_strategy_generator.generate_strategies.return_value = {
            "success": True,
            "strategy_ids": ["strategy_id_1"],
            "count": 1
        }
        
        # Call method
        result = self.integration._generate_strategies(knowledge_item_ids, book_id, job_id)
        
        # Verify strategy generator was called
        self.mock_strategy_generator.generate_strategies.assert_called_once_with(knowledge_item_ids)
        
        # Verify job status was updated
        self.mock_collection.update_one.assert_called_once()
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["strategy_ids"], ["strategy_id_1"])
    
    def test_complete_job(self):
        """Test completing a job."""
        # Test data
        job_id = "job_id_1"
        result = {
            "knowledge_items": 10,
            "knowledge_item_ids": ["item_id_1", "item_id_2"],
            "strategies": 2,
            "strategy_ids": ["strategy_id_1", "strategy_id_2"]
        }
        
        # Call method
        self.integration._complete_job(job_id, result)
        
        # Verify job status was updated
        self.mock_collection.update_one.assert_called_once()
        
        # Verify event was published
        self.mock_event_system.publish.assert_called_once()
    
    def test_process_book_success(self):
        """Test processing a book successfully."""
        # Test data
        book_path = "/path/to/book.pdf"
        metadata = {"title": "Trading Book", "author": "Trader Joe"}
        options = {"ocr": {"language": "eng"}}
        
        # Mock methods
        self.integration._create_extraction_job = MagicMock(return_value="job_id_1")
        self.integration._digitize_book = MagicMock(return_value={
            "success": True,
            "book_id": "book_id_1"
        })
        self.integration._process_multimodal_content = MagicMock(return_value={
            "success": True,
            "content_ids": ["content_id_1"]
        })
        self.integration._extract_knowledge = MagicMock(return_value={
            "success": True,
            "knowledge_items": [{"rule": "Buy when RSI < 30"}],
            "knowledge_item_ids": ["item_id_1"]
        })
        self.integration._build_knowledge_base = MagicMock(return_value={
            "success": True,
            "item_ids": ["item_id_1"]
        })
        self.integration._generate_strategies = MagicMock(return_value={
            "success": True,
            "strategy_ids": ["strategy_id_1"]
        })
        self.integration._complete_job = MagicMock()
        
        # Call method
        result = self.integration.process_book(book_path, metadata, options)
        
        # Verify methods were called
        self.integration._create_extraction_job.assert_called_once_with(book_path, metadata, options)
        self.integration._digitize_book.assert_called_once_with(book_path, metadata, "job_id_1")
        self.integration._process_multimodal_content.assert_called_once_with("book_id_1", "job_id_1")
        self.integration._extract_knowledge.assert_called_once_with("book_id_1", ["content_id_1"], "job_id_1")
        self.integration._build_knowledge_base.assert_called_once()
        self.integration._generate_strategies.assert_called_once()
        self.integration._complete_job.assert_called_once()
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["job_id"], "job_id_1")
    
    def test_process_book_failure(self):
        """Test processing a book with failure."""
        # Test data
        book_path = "/path/to/book.pdf"
        metadata = {"title": "Trading Book", "author": "Trader Joe"}
        options = {"ocr": {"language": "eng"}}
        
        # Mock methods
        self.integration._create_extraction_job = MagicMock(return_value="job_id_1")
        self.integration._digitize_book = MagicMock(return_value={
            "success": False,
            "error": "OCR engine failed"
        })
        self.integration._update_job_status = MagicMock()
        
        # Call method
        result = self.integration.process_book(book_path, metadata, options)
        
        # Verify methods were called
        self.integration._create_extraction_job.assert_called_once_with(book_path, metadata, options)
        self.integration._digitize_book.assert_called_once_with(book_path, metadata, "job_id_1")
        self.integration._update_job_status.assert_called_once_with("job_id_1", "failed", {"error": "OCR engine failed"})
        
        # Verify result
        self.assertFalse(result["success"])
        self.assertEqual(result["job_id"], "job_id_1")
        self.assertEqual(result["error"], "OCR engine failed")
    
    def test_get_job_status(self):
        """Test getting job status."""
        # Test data
        job_id = "job_id_1"
        mock_job = {
            "_id": job_id,
            "status": "completed",
            "book_path": "/path/to/book.pdf",
            "created_at": "2023-01-01T00:00:00Z",
            "completed_at": "2023-01-01T00:10:00Z",
            "result": {
                "knowledge_items": 10,
                "strategies": 2
            }
        }
        
        # Mock database find_one
        self.mock_collection.find_one.return_value = mock_job
        
        # Call method
        result = self.integration.get_job_status(job_id)
        
        # Verify database was called
        self.mock_collection.find_one.assert_called_once_with({"_id": job_id})
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["job"], mock_job)
    
    def test_batch_process_books(self):
        """Test batch processing books."""
        # Test data
        book_paths = ["/path/to/book1.pdf", "/path/to/book2.pdf"]
        metadata = {"source": "Trading Library"}
        options = {"ocr": {"language": "eng"}}
        
        # Mock process_book method
        self.integration.process_book = MagicMock(side_effect=[
            {"success": True, "job_id": "job_id_1"},
            {"success": False, "job_id": "job_id_2", "error": "OCR engine failed"}
        ])
        
        # Call method
        result = self.integration.batch_process_books(book_paths, metadata, options)
        
        # Verify process_book was called for each book
        self.integration.process_book.assert_has_calls([
            call(book_paths[0], metadata, options),
            call(book_paths[1], metadata, options)
        ])
        
        # Verify result
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["successful"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(len(result["job_ids"]), 2)
        self.assertEqual(result["job_ids"], ["job_id_1", "job_id_2"])


if __name__ == '__main__':
    unittest.main()