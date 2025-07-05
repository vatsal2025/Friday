"""Unit tests for the KnowledgeEngineService class."""

import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import json

from src.orchestration.knowledge_engine.knowledge_engine_service import KnowledgeEngineService
from src.infrastructure.event.event_system import EventSystem
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.database.mongodb import MongoDB
from src.orchestration.knowledge_engine.book_knowledge_extractor import BookKnowledgeExtractor


class TestKnowledgeEngineService(unittest.TestCase):
    """Test cases for the KnowledgeEngineService class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock dependencies
        self.mock_config_manager = MagicMock(spec=ConfigManager)
        self.mock_config_manager.get_config.return_value = {
            'storage': {'collection_name': 'test_trading_knowledge'},
            'book_extraction': {'model_path': 'test_model_path'},
            'processing_parameters': {'supported_file_types': ['txt', 'pdf']}
        }
        
        self.mock_event_system = MagicMock(spec=EventSystem)
        self.mock_db = MagicMock(spec=MongoDB)
        self.mock_collection = MagicMock()
        self.mock_db.get_collection.return_value = self.mock_collection
        
        self.mock_book_extractor = MagicMock(spec=BookKnowledgeExtractor)
        
        # Create patches
        self.config_manager_patch = patch('src.orchestration.knowledge_engine.knowledge_engine_service.ConfigManager', 
                                         return_value=self.mock_config_manager)
        self.event_system_patch = patch('src.orchestration.knowledge_engine.knowledge_engine_service.EventSystem', 
                                       return_value=self.mock_event_system)
        self.mongodb_patch = patch('src.orchestration.knowledge_engine.knowledge_engine_service.MongoDB', 
                                  return_value=self.mock_db)
        self.book_extractor_patch = patch('src.orchestration.knowledge_engine.knowledge_engine_service.BookKnowledgeExtractor', 
                                         return_value=self.mock_book_extractor)
        
        # Start patches
        self.config_manager_mock = self.config_manager_patch.start()
        self.event_system_mock = self.event_system_patch.start()
        self.mongodb_mock = self.mongodb_patch.start()
        self.book_extractor_mock = self.book_extractor_patch.start()
        
        # Create service instance
        self.service = KnowledgeEngineService()
    
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Stop patches
        self.config_manager_patch.stop()
        self.event_system_patch.stop()
        self.mongodb_patch.stop()
        self.book_extractor_patch.stop()
    
    def test_initialization(self):
        """Test that the service initializes correctly."""
        # Verify that dependencies were initialized
        self.assertEqual(self.service.config_manager, self.mock_config_manager)
        self.assertEqual(self.service.event_system, self.mock_event_system)
        self.assertEqual(self.service.db, self.mock_db)
        self.assertEqual(self.service.knowledge_collection, self.mock_collection)
        self.assertEqual(self.service.book_extractor, self.mock_book_extractor)
        
        # Verify that event handlers were registered
        self.mock_event_system.subscribe.assert_any_call('knowledge_extraction', self.service._handle_knowledge_extraction)
        self.mock_event_system.subscribe.assert_any_call('knowledge_query', self.service._handle_knowledge_query)
        self.mock_event_system.subscribe.assert_any_call('knowledge_update', self.service._handle_knowledge_update)
    
    def test_extract_knowledge_from_book_file_not_found(self):
        """Test extracting knowledge from a non-existent book file."""
        with patch('os.path.exists', return_value=False):
            result = self.service.extract_knowledge_from_book('non_existent_file.txt')
            
            self.assertFalse(result['success'])
            self.assertEqual(result['error'], 'File not found')
    
    def test_extract_knowledge_from_book_success(self):
        """Test successfully extracting knowledge from a book file."""
        # Mock file existence and content
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data='test book content')), \
             patch.object(self.service, 'save_knowledge_items', return_value={'success': True}):
            
            # Mock the book extractor to return some knowledge items
            extracted_items = [
                {'id': '1', 'type': 'trading_rule', 'content': {'rule': 'Buy low, sell high'}},
                {'id': '2', 'type': 'chart_pattern', 'content': {'pattern': 'Double top'}}
            ]
            self.mock_book_extractor.extract_knowledge.return_value = extracted_items
            
            # Call the method
            result = self.service.extract_knowledge_from_book('test_book.txt')
            
            # Verify the result
            self.assertTrue(result['success'])
            self.assertEqual(result['file_path'], 'test_book.txt')
            self.assertEqual(result['extracted_count'], 2)
            self.assertEqual(result['item_types'], {'trading_rule': 1, 'chart_pattern': 1})
            
            # Verify that the book extractor was called with the correct arguments
            self.mock_book_extractor.extract_knowledge.assert_called_once_with('test book content', {
                'title': 'test_book',
                'file_path': 'test_book.txt',
                'file_type': 'txt'
            })
    
    def test_extract_knowledge_from_book_with_error(self):
        """Test extracting knowledge from a book file with an error."""
        # Mock file existence
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data='test book content')):
            
            # Mock the book extractor to raise an exception
            self.mock_book_extractor.extract_knowledge.side_effect = Exception('Test error')
            
            # Call the method
            result = self.service.extract_knowledge_from_book('test_book.txt')
            
            # Verify the result
            self.assertFalse(result['success'])
            self.assertEqual(result['error'], 'Test error')
            self.assertEqual(result['file_path'], 'test_book.txt')
    
    def test_batch_process_books_directory_not_found(self):
        """Test batch processing books in a non-existent directory."""
        with patch('os.path.exists', return_value=False):
            result = self.service.batch_process_books('non_existent_dir')
            
            self.assertFalse(result['success'])
            self.assertEqual(result['error'], 'Directory not found')
    
    def test_batch_process_books_success(self):
        """Test successfully batch processing books in a directory."""
        # Mock directory existence and content
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True), \
             patch('os.listdir', return_value=['book1.txt', 'book2.pdf', 'image.jpg', 'subdir']):
            
            # Mock file checks
            def is_dir_side_effect(path):
                return 'subdir' in path
            
            with patch('os.path.isdir', side_effect=is_dir_side_effect):
                # Mock extract_knowledge_from_book to return success for book1.txt and failure for book2.pdf
                def extract_side_effect(path):
                    if 'book1.txt' in path:
                        return {
                            'success': True,
                            'file_path': path,
                            'extracted_count': 3,
                            'item_types': {'trading_rule': 2, 'chart_pattern': 1}
                        }
                    else:
                        return {
                            'success': False,
                            'error': 'Test error',
                            'file_path': path
                        }
                
                with patch.object(self.service, 'extract_knowledge_from_book', side_effect=extract_side_effect):
                    # Call the method
                    result = self.service.batch_process_books('test_dir')
                    
                    # Verify the result
                    self.assertEqual(result['total_files'], 2)  # Only .txt and .pdf files
                    self.assertEqual(result['processed_files'], 1)  # Only book1.txt succeeded
                    self.assertEqual(result['failed_files'], 1)  # book2.pdf failed
                    self.assertEqual(result['extracted_items'], 3)  # 3 items from book1.txt
                    self.assertEqual(len(result['file_results']), 2)  # Results for both files
                    
                    # Verify that the event was published
                    self.mock_event_system.publish.assert_called_with('book_batch_processed', {
                        'directory': 'test_dir',
                        'total_files': 2,
                        'processed_files': 1,
                        'extracted_items': 3
                    })
    
    def test_save_knowledge_items_empty_list(self):
        """Test saving an empty list of knowledge items."""
        result = self.service.save_knowledge_items([])
        
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'No items to save')
    
    def test_save_knowledge_items_success(self):
        """Test successfully saving knowledge items."""
        # Mock items to save
        items = [
            {'id': '1', 'type': 'trading_rule', 'content': {'rule': 'Buy low, sell high'}},
            {'id': '2', 'type': 'chart_pattern', 'content': {'pattern': 'Double top'}}
        ]
        
        # Mock the database insert_many method
        mock_result = MagicMock()
        mock_result.inserted_ids = ['1', '2']
        self.mock_collection.insert_many.return_value = mock_result
        
        # Call the method
        result = self.service.save_knowledge_items(items)
        
        # Verify the result
        self.assertTrue(result['success'])
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['item_types'], {'trading_rule': 1, 'chart_pattern': 1})
        
        # Verify that the items were inserted into the database
        self.mock_collection.insert_many.assert_called_once_with(items)
        
        # Verify that the event was published
        self.mock_event_system.publish.assert_called_with('knowledge_stored', {
            'item_count': 2,
            'item_types': {'trading_rule': 1, 'chart_pattern': 1}
        })
    
    def test_save_knowledge_items_with_error(self):
        """Test saving knowledge items with an error."""
        # Mock items to save
        items = [
            {'id': '1', 'type': 'trading_rule', 'content': {'rule': 'Buy low, sell high'}}
        ]
        
        # Mock the database insert_many method to raise an exception
        self.mock_collection.insert_many.side_effect = Exception('Test error')
        
        # Call the method
        result = self.service.save_knowledge_items(items)
        
        # Verify the result
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'Test error')
    
    def test_count_item_types(self):
        """Test counting items by type."""
        # Mock items
        items = [
            {'id': '1', 'type': 'trading_rule'},
            {'id': '2', 'type': 'chart_pattern'},
            {'id': '3', 'type': 'trading_rule'},
            {'id': '4', 'type': 'trading_strategy'}
        ]
        
        # Call the method
        result = self.service._count_item_types(items)
        
        # Verify the result
        self.assertEqual(result, {'trading_rule': 2, 'chart_pattern': 1, 'trading_strategy': 1})
    
    def test_query_knowledge_success(self):
        """Test successfully querying the knowledge base."""
        # Mock query parameters
        query_params = {'type': 'trading_rule', 'text_search': 'buy'}
        
        # Mock the database find method
        mock_results = [
            {'id': '1', 'type': 'trading_rule', 'content': {'rule': 'Buy low, sell high'}},
            {'id': '2', 'type': 'trading_rule', 'content': {'rule': 'Never buy on margin'}}
        ]
        self.mock_collection.find.return_value = mock_results
        
        # Call the method
        results = self.service.query_knowledge(query_params)
        
        # Verify the results
        self.assertEqual(results, mock_results)
        
        # Verify that the database query was built correctly
        expected_mongo_query = {
            'type': 'trading_rule',
            '$or': [
                {'content.rule': {'$regex': 'buy', '$options': 'i'}},
                {'content.pattern': {'$regex': 'buy', '$options': 'i'}},
                {'content.strategy': {'$regex': 'buy', '$options': 'i'}},
                {'content.description': {'$regex': 'buy', '$options': 'i'}}
            ]
        }
        self.mock_collection.find.assert_called_once_with(expected_mongo_query)
    
    def test_query_knowledge_with_error(self):
        """Test querying the knowledge base with an error."""
        # Mock query parameters
        query_params = {'type': 'trading_rule'}
        
        # Mock the database find method to raise an exception
        self.mock_collection.find.side_effect = Exception('Test error')
        
        # Call the method
        results = self.service.query_knowledge(query_params)
        
        # Verify the results
        self.assertEqual(results, [])
    
    def test_build_mongo_query(self):
        """Test building a MongoDB query from query parameters."""
        # Test with all query parameters
        query_params = {
            'type': 'trading_rule',
            'confidence_min': 0.7,
            'text_search': 'buy',
            'source': 'Trading Book'
        }
        
        # Call the method
        mongo_query = self.service._build_mongo_query(query_params)
        
        # Verify the query
        expected_query = {
            'type': 'trading_rule',
            'confidence': {'$gte': 0.7},
            '$or': [
                {'content.rule': {'$regex': 'buy', '$options': 'i'}},
                {'content.pattern': {'$regex': 'buy', '$options': 'i'}},
                {'content.strategy': {'$regex': 'buy', '$options': 'i'}},
                {'content.description': {'$regex': 'buy', '$options': 'i'}}
            ],
            'source.title': {'$regex': 'Trading Book', '$options': 'i'}
        }
        self.assertEqual(mongo_query, expected_query)
        
        # Test with minimal query parameters
        query_params = {'type': 'trading_rule'}
        mongo_query = self.service._build_mongo_query(query_params)
        self.assertEqual(mongo_query, {'type': 'trading_rule'})
    
    def test_update_knowledge_item_success(self):
        """Test successfully updating a knowledge item."""
        # Mock item ID and updates
        item_id = '1'
        updates = {'content': {'rule': 'Updated rule'}}
        
        # Mock the database update_one method
        mock_result = MagicMock()
        mock_result.modified_count = 1
        self.mock_collection.update_one.return_value = mock_result
        
        # Call the method
        result = self.service.update_knowledge_item(item_id, updates)
        
        # Verify the result
        self.assertTrue(result['success'])
        self.assertEqual(result['item_id'], item_id)
        
        # Verify that the database update was called correctly
        self.mock_collection.update_one.assert_called_once_with(
            {'id': item_id},
            {'$set': updates}
        )
        
        # Verify that the event was published
        self.mock_event_system.publish.assert_called_with('knowledge_updated', {
            'item_id': item_id,
            'updates': updates
        })
    
    def test_update_knowledge_item_not_found(self):
        """Test updating a non-existent knowledge item."""
        # Mock item ID and updates
        item_id = '1'
        updates = {'content': {'rule': 'Updated rule'}}
        
        # Mock the database update_one method
        mock_result = MagicMock()
        mock_result.modified_count = 0
        self.mock_collection.update_one.return_value = mock_result
        
        # Call the method
        result = self.service.update_knowledge_item(item_id, updates)
        
        # Verify the result
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'Item not found or not modified')
    
    def test_update_knowledge_item_with_error(self):
        """Test updating a knowledge item with an error."""
        # Mock item ID and updates
        item_id = '1'
        updates = {'content': {'rule': 'Updated rule'}}
        
        # Mock the database update_one method to raise an exception
        self.mock_collection.update_one.side_effect = Exception('Test error')
        
        # Call the method
        result = self.service.update_knowledge_item(item_id, updates)
        
        # Verify the result
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'Test error')
    
    def test_delete_knowledge_item_success(self):
        """Test successfully deleting a knowledge item."""
        # Mock item ID
        item_id = '1'
        
        # Mock the database delete_one method
        mock_result = MagicMock()
        mock_result.deleted_count = 1
        self.mock_collection.delete_one.return_value = mock_result
        
        # Call the method
        result = self.service.delete_knowledge_item(item_id)
        
        # Verify the result
        self.assertTrue(result['success'])
        self.assertEqual(result['item_id'], item_id)
        
        # Verify that the database delete was called correctly
        self.mock_collection.delete_one.assert_called_once_with({'id': item_id})
        
        # Verify that the event was published
        self.mock_event_system.publish.assert_called_with('knowledge_deleted', {
            'item_id': item_id
        })
    
    def test_delete_knowledge_item_not_found(self):
        """Test deleting a non-existent knowledge item."""
        # Mock item ID
        item_id = '1'
        
        # Mock the database delete_one method
        mock_result = MagicMock()
        mock_result.deleted_count = 0
        self.mock_collection.delete_one.return_value = mock_result
        
        # Call the method
        result = self.service.delete_knowledge_item(item_id)
        
        # Verify the result
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'Item not found')
    
    def test_delete_knowledge_item_with_error(self):
        """Test deleting a knowledge item with an error."""
        # Mock item ID
        item_id = '1'
        
        # Mock the database delete_one method to raise an exception
        self.mock_collection.delete_one.side_effect = Exception('Test error')
        
        # Call the method
        result = self.service.delete_knowledge_item(item_id)
        
        # Verify the result
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], 'Test error')
    
    def test_get_knowledge_statistics_success(self):
        """Test successfully getting knowledge base statistics."""
        # Mock the database count_documents method
        self.mock_collection.count_documents.side_effect = [
            100,  # Total count
            50,   # trading_rule count
            30,   # chart_pattern count
            20,   # trading_strategy count
            40,   # high confidence count
            35,   # medium confidence count
            25    # low confidence count
        ]
        
        # Mock the database aggregate method
        mock_top_sources = [
            {'_id': 'Book 1', 'count': 40},
            {'_id': 'Book 2', 'count': 30},
            {'_id': 'Book 3', 'count': 20}
        ]
        self.mock_collection.aggregate.return_value = mock_top_sources
        
        # Call the method
        result = self.service.get_knowledge_statistics()
        
        # Verify the result
        self.assertEqual(result['total_count'], 100)
        self.assertEqual(result['type_counts'], {
            'trading_rule': 50,
            'chart_pattern': 30,
            'trading_strategy': 20
        })
        self.assertEqual(result['confidence_ranges'], {
            'high': 40,
            'medium': 35,
            'low': 25
        })
        self.assertEqual(result['top_sources'], mock_top_sources)
        
        # Verify that the database aggregate was called correctly
        expected_pipeline = [
            {'$group': {'_id': '$source.title', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}},
            {'$limit': 10}
        ]
        self.mock_collection.aggregate.assert_called_once_with(expected_pipeline)
    
    def test_get_knowledge_statistics_with_error(self):
        """Test getting knowledge base statistics with an error."""
        # Mock the database count_documents method to raise an exception
        self.mock_collection.count_documents.side_effect = Exception('Test error')
        
        # Call the method
        result = self.service.get_knowledge_statistics()
        
        # Verify the result
        self.assertEqual(result['total_count'], 0)
        self.assertEqual(result['type_counts'], {})
        self.assertEqual(result['confidence_ranges'], {})
        self.assertEqual(result['top_sources'], [])


if __name__ == '__main__':
    unittest.main()