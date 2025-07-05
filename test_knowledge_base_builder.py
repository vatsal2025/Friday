import unittest
from unittest.mock import MagicMock, patch

# Import the module directly to test
from src.orchestration.knowledge_engine.knowledge_base_builder import KnowledgeBaseBuilder

class TestKnowledgeBaseBuilder(unittest.TestCase):
    def setUp(self):
        # Mock the MongoDB collection
        self.mock_collection = MagicMock()
        
        # Create a patcher for mongodb.get_collection
        self.patcher = patch('src.infrastructure.database.mongodb.get_collection')
        self.mock_get_collection = self.patcher.start()
        self.mock_get_collection.return_value = self.mock_collection
        
        # Create the KnowledgeBaseBuilder instance
        self.builder = KnowledgeBaseBuilder()
    
    def tearDown(self):
        self.patcher.stop()
    
    def test_initialization(self):
        # Test that the collection was retrieved correctly
        from src.infrastructure.database import mongodb
        mongodb.get_collection.assert_called_with('knowledge_base')
    
    def test_add_knowledge(self):
        # Test data
        knowledge_item = {
            'source': 'test_source',
            'content': 'test_content',
            'metadata': {'type': 'test_type'}
        }
        
        # Call the method
        self.builder.add_knowledge(knowledge_item)
        
        # Assert the collection's insert_one was called with the knowledge item
        self.mock_collection.insert_one.assert_called_once_with(knowledge_item)
    
    def test_query_knowledge(self):
        # Mock return value for find
        mock_cursor = MagicMock()
        mock_cursor.limit.return_value = [{'content': 'test_content'}]
        self.mock_collection.find.return_value = mock_cursor
        
        # Call the method
        result = self.builder.query_knowledge('test_query')
        
        # Assert the collection's find was called with the correct query
        self.mock_collection.find.assert_called_once()
        self.assertEqual(result, [{'content': 'test_content'}])

if __name__ == '__main__':
    unittest.main()