"""Unit tests for the KnowledgeBaseBuilder class.

This module contains tests for the KnowledgeBaseBuilder class to ensure
it correctly builds and manages the knowledge base.
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.orchestration.knowledge_engine.knowledge_base_builder import KnowledgeBaseBuilder
from src.infrastructure.config.config_manager import ConfigurationManager
from src.infrastructure.database import mongodb
from src.infrastructure.event.event_system import EventSystem


class TestKnowledgeBaseBuilder(unittest.TestCase):
    """Tests for the KnowledgeBaseBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_config = {
            'enabled': True,
            'storage': {
                'type': 'mongodb',
                'connection_string': 'mongodb://localhost:27017',
                'database_name': 'friday_knowledge',
                'collection_name': 'knowledge_base'
            },
            'indexing': {
                'method': 'vector',
                'vector_dimensions': 768,
                'update_frequency': 'realtime'
            },
            'versioning': {
                'enabled': True,
                'max_versions': 5
            },
            'validation': {
                'schema_validation': True,
                'consistency_check': True
            }
        }
        
        # Create mocks
        self.mock_config_manager = MagicMock()
        self.mock_config_manager.get_config.return_value = self.mock_config
        
        self.mock_collection = MagicMock()
        
        # Patch dependencies
        self.config_manager_patch = patch('src.orchestration.knowledge_engine.knowledge_base_builder.ConfigManager', 
                                         return_value=self.mock_config_manager)
        self.mongodb_patch = patch('src.orchestration.knowledge_engine.knowledge_base_builder.mongodb.get_collection', 
                                  return_value=self.mock_collection)
        
        # Start patches
        self.mock_config_manager_cls = self.config_manager_patch.start()
        self.mock_db_cls = self.mongodb_patch.start()
        
        # Create instance
        self.knowledge_base = KnowledgeBaseBuilder()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop patches
        self.config_manager_patch.stop()
        self.mongodb_patch.stop()
    
    def test_initialization(self):
        """Test initialization of KnowledgeBaseBuilder."""
        # Verify config is loaded
        self.mock_config_manager.get_config.assert_called_once_with('knowledge_extraction')
        
        # Verify database collection is retrieved
        mongodb.get_collection.assert_called_with(self.mock_config['storage']['collection_name'])
        
        # Verify knowledge base has expected attributes
        self.assertEqual(self.knowledge_base.config, self.mock_config)
        self.assertEqual(self.knowledge_base.knowledge_collection, self.mock_collection)
    
    def test_add_knowledge_item(self):
        """Test adding a single knowledge item to the knowledge base."""
        # Test data
        knowledge_item = {
            'type': 'rule',
            'content': 'Buy when RSI < 30',
            'source': {'title': 'Trading Book', 'author': 'Trader Joe'},
            'confidence': 0.85,
            'metadata': {'category': 'technical_analysis'}
        }
        
        # Mock database insert
        self.mock_collection.insert_one.return_value = MagicMock(inserted_id='item_id_1')
        
        # Call method
        result = self.knowledge_base.add_knowledge_item(knowledge_item)
        
        # Verify database was called
        self.mock_collection.insert_one.assert_called_once()
        
        # Verify result
        self.assertTrue(result['success'])
        self.assertEqual(result['item_id'], 'item_id_1')
    
    def test_add_knowledge_items(self):
        """Test adding multiple knowledge items to the knowledge base."""
        # Test data
        knowledge_items = [
            {
                'type': 'rule',
                'content': 'Buy when RSI < 30',
                'source': {'title': 'Trading Book', 'author': 'Trader Joe'},
                'confidence': 0.85,
                'metadata': {'category': 'technical_analysis'}
            },
            {
                'type': 'pattern',
                'content': 'Double bottom',
                'source': {'title': 'Trading Book', 'author': 'Trader Joe'},
                'confidence': 0.9,
                'metadata': {'category': 'chart_patterns'}
            }
        ]
        
        # Mock database insert
        self.mock_collection.insert_many.return_value = MagicMock(inserted_ids=['item_id_1', 'item_id_2'])
        
        # Call method
        result = self.knowledge_base.add_knowledge_items(knowledge_items)
        
        # Verify database was called
        self.mock_collection.insert_many.assert_called_once()
        
        # Verify result
        self.assertTrue(result['success'])
        self.assertEqual(len(result['item_ids']), 2)
        self.assertEqual(result['item_ids'], ['item_id_1', 'item_id_2'])
    
    def test_get_knowledge_item(self):
        """Test retrieving a knowledge item by ID."""
        # Test data
        item_id = 'item_id_1'
        mock_item = {
            '_id': item_id,
            'type': 'rule',
            'content': 'Buy when RSI < 30',
            'source': {'title': 'Trading Book', 'author': 'Trader Joe'},
            'confidence': 0.85,
            'metadata': {'category': 'technical_analysis'}
        }
        
        # Mock database find_one
        self.mock_collection.find_one.return_value = mock_item
        
        # Call method
        result = self.knowledge_base.get_knowledge_item(item_id)
        
        # Verify database was called
        self.mock_collection.find_one.assert_called_once_with({'_id': item_id})
        
        # Verify result
        self.assertEqual(result, mock_item)
    
    def test_update_knowledge_item(self):
        """Test updating a knowledge item."""
        # Test data
        item_id = 'item_id_1'
        updates = {
            'confidence': 0.9,
            'metadata.verified': True
        }
        
        # Mock database update_one
        self.mock_collection.update_one.return_value = MagicMock(modified_count=1)
        
        # Call method
        result = self.knowledge_base.update_knowledge_item(item_id, updates)
        
        # Verify database was called
        self.mock_collection.update_one.assert_called_once()
        
        # Verify result
        self.assertTrue(result['success'])
        self.assertEqual(result['modified_count'], 1)
    
    def test_delete_knowledge_item(self):
        """Test deleting a knowledge item."""
        # Test data
        item_id = 'item_id_1'
        
        # Mock database delete_one
        self.mock_collection.delete_one.return_value = MagicMock(deleted_count=1)
        
        # Call method
        result = self.knowledge_base.delete_knowledge_item(item_id)
        
        # Verify database was called
        self.mock_collection.delete_one.assert_called_once_with({'_id': item_id})
        
        # Verify result
        self.assertTrue(result['success'])
        self.assertEqual(result['deleted_count'], 1)
    
    def test_search(self):
        """Test searching the knowledge base."""
        # Test data
        query = "RSI oversold"
        filters = {'type': 'rule'}
        mock_results = [
            {
                '_id': 'item_id_1',
                'type': 'rule',
                'content': 'Buy when RSI < 30',
                'confidence': 0.85
            },
            {
                '_id': 'item_id_2',
                'type': 'rule',
                'content': 'Look for oversold conditions in RSI',
                'confidence': 0.8
            }
        ]
        
        # Mock database find
        self.mock_collection.find.return_value = mock_results
        
        # Call method
        result = self.knowledge_base.search(query, filters)
        
        # Verify database was called
        self.mock_collection.find.assert_called_once()
        
        # Verify result
        self.assertTrue(result['success'])
        self.assertEqual(len(result['results']), 2)
        self.assertEqual(result['results'], mock_results)
    
    def test_get_knowledge_by_source(self):
        """Test retrieving knowledge items by source."""
        # Test data
        source = {'title': 'Trading Book', 'author': 'Trader Joe'}
        mock_results = [
            {
                '_id': 'item_id_1',
                'type': 'rule',
                'content': 'Buy when RSI < 30',
                'source': source,
                'confidence': 0.85
            },
            {
                '_id': 'item_id_2',
                'type': 'pattern',
                'content': 'Double bottom',
                'source': source,
                'confidence': 0.9
            }
        ]
        
        # Mock database find
        self.mock_collection.find.return_value = mock_results
        
        # Call method
        result = self.knowledge_base.get_knowledge_by_source(source)
        
        # Verify database was called
        self.mock_collection.find.assert_called_once()
        
        # Verify result
        self.assertTrue(result['success'])
        self.assertEqual(len(result['items']), 2)
        self.assertEqual(result['items'], mock_results)
    
    def test_get_knowledge_by_type(self):
        """Test retrieving knowledge items by type."""
        # Test data
        item_type = 'rule'
        mock_results = [
            {
                '_id': 'item_id_1',
                'type': 'rule',
                'content': 'Buy when RSI < 30',
                'confidence': 0.85
            },
            {
                '_id': 'item_id_2',
                'type': 'rule',
                'content': 'Sell when RSI > 70',
                'confidence': 0.9
            }
        ]
        
        # Mock database find
        self.mock_collection.find.return_value = mock_results
        
        # Call method
        result = self.knowledge_base.get_knowledge_by_type(item_type)
        
        # Verify database was called
        self.mock_collection.find.assert_called_once_with({'type': item_type})
        
        # Verify result
        self.assertTrue(result['success'])
        self.assertEqual(len(result['items']), 2)
        self.assertEqual(result['items'], mock_results)
    
    def test_get_knowledge_stats(self):
        """Test retrieving knowledge base statistics."""
        # Mock database aggregate
        self.mock_collection.count_documents.side_effect = [
            100,  # Total count
            40,   # Rules count
            30,   # Patterns count
            20,   # Strategies count
            10    # Other count
        ]
        
        # Call method
        result = self.knowledge_base.get_knowledge_stats()
        
        # Verify database was called
        self.assertEqual(self.mock_collection.count_documents.call_count, 5)
        
        # Verify result
        self.assertTrue(result['success'])
        self.assertEqual(result['total_items'], 100)
        self.assertEqual(result['by_type']['rule'], 40)
        self.assertEqual(result['by_type']['pattern'], 30)
        self.assertEqual(result['by_type']['strategy'], 20)
        self.assertEqual(result['by_type']['other'], 10)


if __name__ == '__main__':
    unittest.main()