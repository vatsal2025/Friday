"""Tests for the Knowledge Engine API endpoints."""

import unittest
from unittest.mock import MagicMock, patch
import json
import os
from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.application.api.knowledge_engine_api import router as knowledge_router
from src.orchestration.knowledge_engine.knowledge_engine_service import KnowledgeEngineService


class TestKnowledgeEngineAPI(unittest.TestCase):
    """Test cases for the Knowledge Engine API endpoints."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a FastAPI app and add the knowledge router
        self.app = FastAPI()
        self.app.include_router(knowledge_router)
        
        # Create a test client
        self.client = TestClient(self.app)
        
        # Mock the API key authentication dependency
        self.auth_patch = patch('src.application.api.knowledge_engine_api.api_key_auth', return_value=None)
        self.auth_patch.start()
        
        # Mock the KnowledgeEngineService
        self.service_patch = patch('src.application.api.knowledge_engine_api.KnowledgeEngineService')
        self.mock_service_class = self.service_patch.start()
        self.mock_service = MagicMock(spec=KnowledgeEngineService)
        self.mock_service_class.return_value = self.mock_service
    
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        self.auth_patch.stop()
        self.service_patch.stop()
    
    def test_query_knowledge(self):
        """Test the query_knowledge endpoint."""
        # Mock the service response
        mock_items = [
            {
                'id': '1',
                'type': 'trading_rule',
                'content': {'rule': 'Buy low, sell high'},
                'source': {'title': 'Trading Book'},
                'confidence': 0.9
            },
            {
                'id': '2',
                'type': 'chart_pattern',
                'content': {'pattern': 'Double top'},
                'source': {'title': 'Trading Book'},
                'confidence': 0.8
            }
        ]
        self.mock_service.query_knowledge.return_value = mock_items
        
        # Make the request
        response = self.client.get(
            "/api/knowledge/?type=trading_rule&confidence_min=0.7&text_search=buy&limit=10&offset=0"
        )
        
        # Verify the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_items)
        
        # Verify that the service was called with the correct parameters
        self.mock_service.query_knowledge.assert_called_once_with({
            'type': 'trading_rule',
            'confidence_min': 0.7,
            'text_search': 'buy'
        })
    
    def test_query_knowledge_error(self):
        """Test the query_knowledge endpoint with an error."""
        # Mock the service to raise an exception
        self.mock_service.query_knowledge.side_effect = Exception("Test error")
        
        # Make the request
        response = self.client.get("/api/knowledge/")
        
        # Verify the response
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Test error"})
    
    def test_get_knowledge_statistics(self):
        """Test the get_knowledge_statistics endpoint."""
        # Mock the service response
        mock_stats = {
            'total_count': 100,
            'type_counts': {
                'trading_rule': 50,
                'chart_pattern': 30,
                'trading_strategy': 20
            },
            'confidence_ranges': {
                'high': 40,
                'medium': 35,
                'low': 25
            },
            'top_sources': [
                {'_id': 'Book 1', 'count': 40},
                {'_id': 'Book 2', 'count': 30},
                {'_id': 'Book 3', 'count': 20}
            ]
        }
        self.mock_service.get_knowledge_statistics.return_value = mock_stats
        
        # Make the request
        response = self.client.get("/api/knowledge/statistics")
        
        # Verify the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_stats)
        
        # Verify that the service was called
        self.mock_service.get_knowledge_statistics.assert_called_once()
    
    def test_get_knowledge_statistics_error(self):
        """Test the get_knowledge_statistics endpoint with an error."""
        # Mock the service to raise an exception
        self.mock_service.get_knowledge_statistics.side_effect = Exception("Test error")
        
        # Make the request
        response = self.client.get("/api/knowledge/statistics")
        
        # Verify the response
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Test error"})
    
    def test_get_knowledge_item(self):
        """Test the get_knowledge_item endpoint."""
        # Mock the service response
        mock_item = {
            'id': '1',
            'type': 'trading_rule',
            'content': {'rule': 'Buy low, sell high'},
            'source': {'title': 'Trading Book'},
            'confidence': 0.9
        }
        self.mock_service.query_knowledge.return_value = [mock_item]
        
        # Make the request
        response = self.client.get("/api/knowledge/1")
        
        # Verify the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_item)
        
        # Verify that the service was called with the correct parameters
        self.mock_service.query_knowledge.assert_called_once_with({'id': '1'})
    
    def test_get_knowledge_item_not_found(self):
        """Test the get_knowledge_item endpoint with a non-existent item."""
        # Mock the service to return an empty list
        self.mock_service.query_knowledge.return_value = []
        
        # Make the request
        response = self.client.get("/api/knowledge/1")
        
        # Verify the response
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Knowledge item with ID 1 not found"})
    
    def test_create_knowledge_items(self):
        """Test the create_knowledge_items endpoint."""
        # Mock the service response
        mock_result = {
            'success': True,
            'count': 2,
            'item_types': {'trading_rule': 1, 'chart_pattern': 1}
        }
        self.mock_service.save_knowledge_items.return_value = mock_result
        
        # Create test data
        items = [
            {
                'type': 'trading_rule',
                'content': {'rule': 'Buy low, sell high'},
                'source': {'title': 'Trading Book'},
                'confidence': 0.9
            },
            {
                'type': 'chart_pattern',
                'content': {'pattern': 'Double top'},
                'source': {'title': 'Trading Book'},
                'confidence': 0.8
            }
        ]
        
        # Make the request
        response = self.client.post(
            "/api/knowledge/",
            json=items
        )
        
        # Verify the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_result)
        
        # Verify that the service was called with the correct parameters
        self.mock_service.save_knowledge_items.assert_called_once()
        # Get the actual argument passed to save_knowledge_items
        actual_items = self.mock_service.save_knowledge_items.call_args[0][0]
        self.assertEqual(len(actual_items), 2)
        self.assertEqual(actual_items[0]['type'], 'trading_rule')
        self.assertEqual(actual_items[1]['type'], 'chart_pattern')
    
    def test_create_knowledge_items_error(self):
        """Test the create_knowledge_items endpoint with an error."""
        # Mock the service to return an error
        mock_result = {
            'success': False,
            'error': 'Test error'
        }
        self.mock_service.save_knowledge_items.return_value = mock_result
        
        # Create test data
        items = [
            {
                'type': 'trading_rule',
                'content': {'rule': 'Buy low, sell high'},
                'source': {'title': 'Trading Book'},
                'confidence': 0.9
            }
        ]
        
        # Make the request
        response = self.client.post(
            "/api/knowledge/",
            json=items
        )
        
        # Verify the response
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Test error"})
    
    def test_update_knowledge_item(self):
        """Test the update_knowledge_item endpoint."""
        # Mock the service response
        mock_result = {
            'success': True,
            'item_id': '1'
        }
        self.mock_service.update_knowledge_item.return_value = mock_result
        
        # Create test data
        update_request = {
            'content': {'rule': 'Updated rule'},
            'confidence': 0.95
        }
        
        # Make the request
        response = self.client.put(
            "/api/knowledge/1",
            json=update_request
        )
        
        # Verify the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_result)
        
        # Verify that the service was called with the correct parameters
        self.mock_service.update_knowledge_item.assert_called_once_with('1', update_request)
    
    def test_update_knowledge_item_not_found(self):
        """Test the update_knowledge_item endpoint with a non-existent item."""
        # Mock the service to return an error
        mock_result = {
            'success': False,
            'error': 'Item not found'
        }
        self.mock_service.update_knowledge_item.return_value = mock_result
        
        # Create test data
        update_request = {
            'content': {'rule': 'Updated rule'}
        }
        
        # Make the request
        response = self.client.put(
            "/api/knowledge/1",
            json=update_request
        )
        
        # Verify the response
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Knowledge item with ID 1 not found"})
    
    def test_delete_knowledge_item(self):
        """Test the delete_knowledge_item endpoint."""
        # Mock the service response
        mock_result = {
            'success': True,
            'item_id': '1'
        }
        self.mock_service.delete_knowledge_item.return_value = mock_result
        
        # Make the request
        response = self.client.delete("/api/knowledge/1")
        
        # Verify the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_result)
        
        # Verify that the service was called with the correct parameters
        self.mock_service.delete_knowledge_item.assert_called_once_with('1')
    
    def test_delete_knowledge_item_not_found(self):
        """Test the delete_knowledge_item endpoint with a non-existent item."""
        # Mock the service to return an error
        mock_result = {
            'success': False,
            'error': 'Item not found'
        }
        self.mock_service.delete_knowledge_item.return_value = mock_result
        
        # Make the request
        response = self.client.delete("/api/knowledge/1")
        
        # Verify the response
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Knowledge item with ID 1 not found"})
    
    def test_extract_knowledge_from_book(self):
        """Test the extract_knowledge_from_book endpoint."""
        # Mock the service response
        mock_result = {
            'success': True,
            'file_path': 'temp_test_book.txt',
            'extracted_count': 3,
            'item_types': {'trading_rule': 2, 'chart_pattern': 1}
        }
        self.mock_service.extract_knowledge_from_book.return_value = mock_result
        
        # Mock file operations
        with patch('builtins.open', mock_open()), \
             patch('os.path.exists', return_value=True), \
             patch('os.remove'):
            
            # Create test data
            metadata = {
                'title': 'Test Book',
                'author': 'Test Author',
                'publication_year': 2023
            }
            
            # Make the request
            with open('test_book.txt', 'rb') as f:
                response = self.client.post(
                    "/api/knowledge/extract/book",
                    files={"book_file": ("test_book.txt", b"test content", "text/plain")},
                    data={"metadata": json.dumps(metadata)}
                )
            
            # Verify the response
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), mock_result)
            
            # Verify that the service was called with the correct parameters
            self.mock_service.extract_knowledge_from_book.assert_called_once()
            # The first argument should be the temp file path
            self.assertTrue(self.mock_service.extract_knowledge_from_book.call_args[0][0].startswith('temp_'))
            # The second argument should be the metadata
            self.assertEqual(self.mock_service.extract_knowledge_from_book.call_args[0][1], metadata)
    
    def test_extract_knowledge_from_book_error(self):
        """Test the extract_knowledge_from_book endpoint with an error."""
        # Mock the service to return an error
        mock_result = {
            'success': False,
            'error': 'Test error'
        }
        self.mock_service.extract_knowledge_from_book.return_value = mock_result
        
        # Mock file operations
        with patch('builtins.open', mock_open()), \
             patch('os.path.exists', return_value=True), \
             patch('os.remove'):
            
            # Make the request
            response = self.client.post(
                "/api/knowledge/extract/book",
                files={"book_file": ("test_book.txt", b"test content", "text/plain")}
            )
            
            # Verify the response
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json(), {"detail": "Test error"})
    
    def test_batch_process_books(self):
        """Test the batch_process_books endpoint."""
        # Mock the service response
        mock_result = {
            'total_files': 2,
            'processed_files': 1,
            'failed_files': 1,
            'extracted_items': 3,
            'file_results': [
                {
                    'success': True,
                    'file_path': 'test_dir/book1.txt',
                    'extracted_count': 3,
                    'item_types': {'trading_rule': 2, 'chart_pattern': 1}
                },
                {
                    'success': False,
                    'error': 'Test error',
                    'file_path': 'test_dir/book2.pdf'
                }
            ]
        }
        self.mock_service.batch_process_books.return_value = mock_result
        
        # Make the request
        response = self.client.post(
            "/api/knowledge/extract/batch",
            params={"directory_path": "test_dir"}
        )
        
        # Verify the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_result)
        
        # Verify that the service was called with the correct parameters
        self.mock_service.batch_process_books.assert_called_once_with("test_dir")
    
    def test_batch_process_books_error(self):
        """Test the batch_process_books endpoint with an error."""
        # Mock the service to return an error
        mock_result = {
            'success': False,
            'error': 'Directory not found'
        }
        self.mock_service.batch_process_books.return_value = mock_result
        
        # Make the request
        response = self.client.post(
            "/api/knowledge/extract/batch",
            params={"directory_path": "non_existent_dir"}
        )
        
        # Verify the response
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Directory not found"})


if __name__ == '__main__':
    unittest.main()