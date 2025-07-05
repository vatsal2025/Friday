"""Unit tests for the BookKnowledgeExtractor class.

This module contains tests for the BookKnowledgeExtractor class to ensure
it correctly extracts knowledge from book content.
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

from src.orchestration.knowledge_engine.book_knowledge_extractor import BookKnowledgeExtractor
from src.infrastructure.config.config_manager import ConfigurationManager
from src.infrastructure.database import mongodb
from src.infrastructure.event.event_system import EventSystem


class TestBookKnowledgeExtractor(unittest.TestCase):
    """Tests for the BookKnowledgeExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock config manager
        self.mock_config = {
            'extraction_categories': {
                'trading_rules': True,
                'chart_patterns': True,
                'trading_strategies': True
            },
            'confidence_thresholds': {
                'trading_rules': 0.7,
                'chart_patterns': 0.7,
                'trading_strategies': 0.7
            },
            'storage': {
                'collection_name': 'trading_knowledge'
            }
        }
        
        # Mock MongoDB
        self.mock_collection = MagicMock()
        
        # Mock event system
        self.mock_event_system = MagicMock()
        
        # Set up patches
        self.mongodb_patch = patch('src.orchestration.knowledge_engine.book_knowledge_extractor.mongodb.get_collection', 
                                  return_value=self.mock_collection)
        
        # Start patches
        self.mock_get_collection = self.mongodb_patch.start()
        
        # Create instance with the mock config and event system
        self.extractor = BookKnowledgeExtractor(config=self.mock_config, event_system=self.mock_event_system)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop patches
        self.mongodb_patch.stop()
    
    def test_initialization(self):
        """Test initialization of BookKnowledgeExtractor."""
        # Verify database collection is retrieved
        self.mock_get_collection.assert_called_once_with(self.mock_config['storage']['collection_name'])
        
        # Verify extractor has expected attributes
        self.assertEqual(self.extractor.config, self.mock_config)
        self.assertEqual(self.extractor.collection, self.mock_collection)
        self.assertEqual(self.extractor.event_system, self.mock_event_system)
    
    def test_extract_knowledge_from_text(self):
        """Test extracting knowledge from text."""
        # Mock internal methods
        self.extractor._extract_trading_rules = MagicMock(return_value=[{'rule': 'Buy when RSI < 30'}])
        self.extractor._extract_patterns = MagicMock(return_value=[{'pattern': 'Double bottom'}])
        self.extractor._extract_strategies = MagicMock(return_value=[{'strategy': 'Trend following'}])
        
        # Test data
        text = "When RSI drops below 30, it's a good time to buy. Look for double bottom patterns."
        source_metadata = {'title': 'Trading Book', 'author': 'Trader Joe'}
        
        # Call method
        result = self.extractor.extract_knowledge_from_text(text, source_metadata)
        
        # Verify internal methods were called
        self.extractor._extract_trading_rules.assert_called_once_with(text)
        self.extractor._extract_patterns.assert_called_once_with(text)
        self.extractor._extract_strategies.assert_called_once_with(text)
        
        # Verify result structure
        self.assertEqual(len(result), 3)  # One rule, one pattern, one strategy
        for item in result:
            self.assertIn('source', item)
            self.assertEqual(item['source'], source_metadata)
            self.assertIn('confidence', item)
            self.assertIn('timestamp', item)
    
    def test_extract_trading_rules(self):
        """Test extracting trading rules from text."""
        # Test data
        text = "Rule 1: Buy when RSI drops below 30. Rule 2: Sell when RSI goes above 70."
        
        # Call method
        result = self.extractor._extract_trading_rules(text)
        
        # Verify result
        self.assertEqual(len(result), 2)  # Two rules
        self.assertIn('rule', result[0])
        self.assertIn('rule', result[1])
        self.assertIn('Buy when RSI drops below 30', result[0]['rule'])
        self.assertIn('Sell when RSI goes above 70', result[1]['rule'])
    
    def test_extract_patterns(self):
        """Test extracting patterns from text."""
        # Test data
        text = "The double bottom pattern is a reversal pattern. Head and shoulders is another reversal pattern."
        
        # Call method
        result = self.extractor._extract_patterns(text)
        
        # Verify result
        self.assertEqual(len(result), 2)  # Two patterns
        self.assertIn('pattern', result[0])
        self.assertIn('pattern', result[1])
        self.assertIn('double bottom', result[0]['pattern'].lower())
        self.assertIn('head and shoulders', result[1]['pattern'].lower())
    
    def test_extract_strategies(self):
        """Test extracting strategies from text."""
        # Test data
        text = "The trend following strategy is simple: buy when price is above MA and sell when below. "
        text += "Mean reversion strategy works by buying oversold conditions."
        
        # Call method
        result = self.extractor._extract_strategies(text)
        
        # Verify result
        self.assertEqual(len(result), 2)  # Two strategies
        self.assertIn('strategy', result[0])
        self.assertIn('strategy', result[1])
        self.assertIn('trend following', result[0]['strategy'].lower())
        self.assertIn('mean reversion', result[1]['strategy'].lower())
    
    def test_extract_entities(self):
        """Test extracting entities from text."""
        # Test data
        text = "RSI is a momentum indicator. Use 4-hour timeframe for Bitcoin trading."
        
        # Call method
        result = self.extractor._extract_entities(text)
        
        # Verify result
        self.assertIn('indicators', result)
        self.assertIn('timeframes', result)
        self.assertIn('assets', result)
        self.assertIn('RSI', result['indicators'])
        self.assertIn('4-hour', result['timeframes'])
        self.assertIn('Bitcoin', result['assets'])
    
    def test_classify_strategy(self):
        """Test classifying strategy types."""
        # Test data
        trend_text = "This strategy follows the trend using moving averages."
        reversion_text = "This strategy looks for mean reversion opportunities."
        breakout_text = "This strategy trades breakouts from consolidation."
        value_text = "This strategy uses fundamental value analysis."
        unknown_text = "This strategy uses custom logic."
        
        # Call method and verify results
        self.assertEqual(self.extractor._classify_strategy(trend_text), "trend_following")
        self.assertEqual(self.extractor._classify_strategy(reversion_text), "mean_reversion")
        self.assertEqual(self.extractor._classify_strategy(breakout_text), "breakout")
        self.assertEqual(self.extractor._classify_strategy(value_text), "value_based")
        self.assertEqual(self.extractor._classify_strategy(unknown_text), "general_strategy")
    
    def test_extract_strategy_steps(self):
        """Test extracting ordered steps from strategy descriptions."""
        # Test data
        text = "Strategy steps:\n1. Identify trend direction\n2. Wait for pullback\n3. Enter on confirmation"
        
        # Call method
        result = self.extractor._extract_strategy_steps(text)
        
        # Verify result
        self.assertEqual(len(result), 3)  # Three steps
        self.assertEqual(result[0], "Identify trend direction")
        self.assertEqual(result[1], "Wait for pullback")
        self.assertEqual(result[2], "Enter on confirmation")
    
    def test_save_extracted_knowledge(self):
        """Test saving extracted knowledge to database."""
        # Test data
        items = [
            {'rule': 'Buy low', 'source': {'title': 'Book'}, 'confidence': 0.8},
            {'pattern': 'Double bottom', 'source': {'title': 'Book'}, 'confidence': 0.9}
        ]
        
        # Mock database insert
        self.mock_collection.insert_many.return_value = MagicMock(inserted_ids=[1, 2])
        
        # Call method
        result = self.extractor.save_extracted_knowledge(items)
        
        # Verify database was called
        self.mock_collection.insert_many.assert_called_once()
        
        # Verify result
        self.assertTrue(result)
    
    def test_extract_knowledge(self):
        """Test the main extract_knowledge method."""
        # Mock internal methods
        self.extractor.extract_knowledge_from_text = MagicMock(return_value=[
            {'rule': 'Buy low', 'confidence': 0.8},
            {'pattern': 'Double bottom', 'confidence': 0.5}  # Below threshold
        ])
        
        # Test data
        text = "Buy low, sell high. Look for double bottoms."
        source_metadata = {'title': 'Trading Book'}
        
        # Call method
        result = self.extractor.extract_knowledge(text, source_metadata)
        
        # Verify internal method was called
        self.extractor.extract_knowledge_from_text.assert_called_once_with(text, source_metadata)
        
        # Verify filtering by confidence threshold
        self.assertEqual(len(result), 1)  # Only one item above threshold
        self.assertIn('rule', result[0])
        self.assertEqual(result[0]['rule'], 'Buy low')


if __name__ == '__main__':
    unittest.main()