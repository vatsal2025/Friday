"""Unit tests for the NLP models implementation in BookKnowledgeExtractor.

This module contains tests for the NLP models used in the BookKnowledgeExtractor class
for entity recognition, relation extraction, sentiment analysis, and text classification.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import the module to be tested
from src.orchestration.knowledge_engine.book_knowledge_extractor import BookKnowledgeExtractor


class TestNLPModels(unittest.TestCase):
    """Tests for the NLP models used in BookKnowledgeExtractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock configuration
        self.mock_config = {
            'enabled': True,
            'extraction_methods': {
                'rule_based': {
                    'enabled': True,
                    'confidence_threshold': 0.7
                },
                'nlp': {
                    'enabled': True,
                    'models': {
                        'entity_recognition': {
                            'type': 'spacy',
                            'model_name': 'en_core_web_md',
                            'confidence_threshold': 0.6
                        },
                        'relation_extraction': {
                            'type': 'transformer',
                            'model_name': 'bert-base-uncased',
                            'confidence_threshold': 0.7
                        },
                        'sentiment_analysis': {
                            'type': 'transformer',
                            'model_name': 'distilbert-base-uncased-finetuned-sst-2-english',
                            'confidence_threshold': 0.8
                        },
                        'text_classification': {
                            'type': 'transformer',
                            'model_name': 'distilbert-base-uncased',
                            'confidence_threshold': 0.75
                        }
                    }
                }
            },
            'extraction_categories': {
                'trading_rules': True,
                'chart_patterns': True,
                'trading_strategies': True
            },
            'confidence_thresholds': {
                'trading_rules': 0.7,
                'chart_patterns': 0.75,
                'trading_strategies': 0.8
            }
        }
        
        # Create mocks for dependencies
        self.mock_event_system = MagicMock()
        
        # Create patches for external libraries
        self.spacy_patch = patch('spacy.load')
        self.transformers_tokenizer_patch = patch('transformers.AutoTokenizer.from_pretrained')
        self.transformers_model_patch = patch('transformers.AutoModelForSequenceClassification.from_pretrained')
        self.transformers_pipeline_patch = patch('transformers.pipeline')
        
        # Start patches
        self.mock_spacy = self.spacy_patch.start()
        self.mock_tokenizer = self.transformers_tokenizer_patch.start()
        self.mock_model = self.transformers_model_patch.start()
        self.mock_pipeline = self.transformers_pipeline_patch.start()
        
        # Configure mocks
        self.mock_nlp = MagicMock()
        self.mock_spacy.return_value = self.mock_nlp
        
        self.mock_tokenizer_instance = MagicMock()
        self.mock_tokenizer.return_value = self.mock_tokenizer_instance
        
        self.mock_model_instance = MagicMock()
        self.mock_model.return_value = self.mock_model_instance
        
        self.mock_pipeline_instance = MagicMock()
        self.mock_pipeline.return_value = self.mock_pipeline_instance
        
        # Create instance of BookKnowledgeExtractor
        self.extractor = BookKnowledgeExtractor(self.mock_config, self.mock_event_system)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop patches
        self.spacy_patch.stop()
        self.transformers_tokenizer_patch.stop()
        self.transformers_model_patch.stop()
        self.transformers_pipeline_patch.stop()
    
    def test_load_nlp_models(self):
        """Test loading NLP models."""
        # Call the method
        self.extractor._load_nlp_models()
        
        # Verify spaCy model was loaded
        self.mock_spacy.assert_called_with('en_core_web_md')
        
        # Verify transformers models were loaded
        self.mock_tokenizer.assert_any_call('bert-base-uncased')
        self.mock_model.assert_any_call('bert-base-uncased')
        self.mock_pipeline.assert_any_call(
            task='sentiment-analysis',
            model='distilbert-base-uncased-finetuned-sst-2-english',
            tokenizer='distilbert-base-uncased-finetuned-sst-2-english'
        )
        self.mock_pipeline.assert_any_call(
            task='text-classification',
            model='distilbert-base-uncased',
            tokenizer='distilbert-base-uncased'
        )
        
        # Verify models are stored in the instance
        self.assertEqual(self.extractor.nlp_models['entity_recognition'], self.mock_nlp)
        self.assertIsNotNone(self.extractor.nlp_models['relation_extraction'])
        self.assertEqual(self.extractor.nlp_models['sentiment_analysis'], self.mock_pipeline_instance)
        self.assertEqual(self.extractor.nlp_models['text_classification'], self.mock_pipeline_instance)
    
    def test_extract_entities(self):
        """Test extracting entities using spaCy."""
        # Test data
        text = "When RSI is below 30, it's a good time to buy. Look for bullish engulfing patterns."
        
        # Mock spaCy entity recognition
        mock_token1 = MagicMock()
        mock_token1.text = "RSI"
        mock_token1.ent_type_ = "INDICATOR"
        mock_token1.ent_iob_ = "B"
        
        mock_token2 = MagicMock()
        mock_token2.text = "below"
        mock_token2.ent_type_ = ""
        mock_token2.ent_iob_ = "O"
        
        mock_token3 = MagicMock()
        mock_token3.text = "30"
        mock_token3.ent_type_ = "VALUE"
        mock_token3.ent_iob_ = "B"
        
        mock_token4 = MagicMock()
        mock_token4.text = "bullish engulfing"
        mock_token4.ent_type_ = "PATTERN"
        mock_token4.ent_iob_ = "B"
        
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = [mock_token1, mock_token2, mock_token3, mock_token4]
        
        mock_ent1 = MagicMock()
        mock_ent1.text = "RSI"
        mock_ent1.label_ = "INDICATOR"
        mock_ent1.start_char = 5
        mock_ent1.end_char = 8
        
        mock_ent2 = MagicMock()
        mock_ent2.text = "30"
        mock_ent2.label_ = "VALUE"
        mock_ent2.start_char = 18
        mock_ent2.end_char = 20
        
        mock_ent3 = MagicMock()
        mock_ent3.text = "bullish engulfing"
        mock_ent3.label_ = "PATTERN"
        mock_ent3.start_char = 60
        mock_ent3.end_char = 77
        
        mock_doc.ents = [mock_ent1, mock_ent2, mock_ent3]
        
        self.mock_nlp.return_value = mock_doc
        
        # Call the method
        entities = self.extractor._extract_entities(text)
        
        # Verify spaCy model was called
        self.mock_nlp.assert_called_with(text)
        
        # Verify entities were extracted correctly
        self.assertEqual(len(entities), 3)
        self.assertEqual(entities[0]['text'], 'RSI')
        self.assertEqual(entities[0]['type'], 'INDICATOR')
        self.assertEqual(entities[1]['text'], '30')
        self.assertEqual(entities[1]['type'], 'VALUE')
        self.assertEqual(entities[2]['text'], 'bullish engulfing')
        self.assertEqual(entities[2]['type'], 'PATTERN')
    
    def test_extract_relations(self):
        """Test extracting relations between entities."""
        # Test data
        text = "When RSI is below 30, it's a good time to buy."
        entities = [
            {'text': 'RSI', 'type': 'INDICATOR', 'start': 5, 'end': 8},
            {'text': '30', 'type': 'VALUE', 'start': 18, 'end': 20},
            {'text': 'buy', 'type': 'ACTION', 'start': 45, 'end': 48}
        ]
        
        # Mock transformer model outputs
        self.mock_tokenizer_instance.encode_plus.return_value = {
            'input_ids': [101, 2043, 2003, 2107, 102],
            'attention_mask': [1, 1, 1, 1, 1],
            'token_type_ids': [0, 0, 0, 0, 0]
        }
        
        mock_output = MagicMock()
        mock_logits = MagicMock()
        mock_logits.detach.return_value.numpy.return_value = np.array([[0.1, 0.8, 0.1]])
        mock_output.logits = mock_logits
        
        self.mock_model_instance.return_value = mock_output
        
        # Call the method
        relations = self.extractor._extract_relations(text, entities)
        
        # Verify transformer model was called
        self.mock_tokenizer_instance.encode_plus.assert_called()
        self.mock_model_instance.assert_called()
        
        # Verify relations were extracted correctly
        self.assertEqual(len(relations), 2)  # RSI-30 and RSI-buy relations
        self.assertEqual(relations[0]['source'], 'RSI')
        self.assertEqual(relations[0]['target'], '30')
        self.assertTrue('relation' in relations[0])
        self.assertTrue('confidence' in relations[0])
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        # Test data
        text = "This trading strategy has been very profitable in bull markets."
        
        # Mock sentiment analysis pipeline
        self.mock_pipeline_instance.return_value = [
            {'label': 'POSITIVE', 'score': 0.92}
        ]
        
        # Call the method
        sentiment = self.extractor._analyze_sentiment(text)
        
        # Verify pipeline was called
        self.mock_pipeline_instance.assert_called_with(text)
        
        # Verify sentiment was analyzed correctly
        self.assertEqual(sentiment['label'], 'POSITIVE')
        self.assertEqual(sentiment['score'], 0.92)
    
    def test_classify_text(self):
        """Test text classification."""
        # Test data
        text = "The moving average convergence divergence (MACD) is a trend-following momentum indicator."
        
        # Mock text classification pipeline
        self.mock_pipeline_instance.return_value = [
            {'label': 'INDICATOR_DESCRIPTION', 'score': 0.85}
        ]
        
        # Call the method
        classification = self.extractor._classify_text(text)
        
        # Verify pipeline was called
        self.mock_pipeline_instance.assert_called_with(text)
        
        # Verify text was classified correctly
        self.assertEqual(classification['label'], 'INDICATOR_DESCRIPTION')
        self.assertEqual(classification['score'], 0.85)
    
    def test_extract_trading_rules_with_nlp(self):
        """Test extracting trading rules using NLP models."""
        # Test data
        text = "Buy when RSI is below 30. Sell when RSI is above 70."
        
        # Mock entity extraction
        self.extractor._extract_entities = MagicMock(return_value=[
            {'text': 'RSI', 'type': 'INDICATOR', 'start': 10, 'end': 13},
            {'text': '30', 'type': 'VALUE', 'start': 23, 'end': 25},
            {'text': 'RSI', 'type': 'INDICATOR', 'start': 37, 'end': 40},
            {'text': '70', 'type': 'VALUE', 'start': 50, 'end': 52}
        ])
        
        # Mock relation extraction
        self.extractor._extract_relations = MagicMock(return_value=[
            {'source': 'RSI', 'target': '30', 'relation': 'BELOW', 'confidence': 0.85},
            {'source': 'RSI', 'target': '70', 'relation': 'ABOVE', 'confidence': 0.87}
        ])
        
        # Mock sentiment analysis
        self.extractor._analyze_sentiment = MagicMock(return_value={
            'label': 'NEUTRAL', 'score': 0.75
        })
        
        # Mock text classification
        self.extractor._classify_text = MagicMock(return_value={
            'label': 'TRADING_RULE', 'score': 0.82
        })
        
        # Call the method
        rules = self.extractor._extract_trading_rules(text)
        
        # Verify NLP methods were called
        self.extractor._extract_entities.assert_called_with(text)
        self.extractor._extract_relations.assert_called()
        self.extractor._analyze_sentiment.assert_called_with(text)
        self.extractor._classify_text.assert_called_with(text)
        
        # Verify rules were extracted correctly
        self.assertEqual(len(rules), 2)
        self.assertTrue(any(rule['rule'].find('Buy when RSI is below 30') != -1 for rule in rules))
        self.assertTrue(any(rule['rule'].find('Sell when RSI is above 70') != -1 for rule in rules))
        self.assertTrue(all('confidence' in rule for rule in rules))
        self.assertTrue(all('sentiment' in rule for rule in rules))
        self.assertTrue(all('classification' in rule for rule in rules))
    
    def test_extract_chart_patterns_with_nlp(self):
        """Test extracting chart patterns using NLP models."""
        # Test data
        text = "The bullish engulfing pattern is a reversal pattern that appears at the bottom of a downtrend."
        
        # Mock entity extraction
        self.extractor._extract_entities = MagicMock(return_value=[
            {'text': 'bullish engulfing', 'type': 'PATTERN', 'start': 4, 'end': 21},
            {'text': 'reversal', 'type': 'PATTERN_TYPE', 'start': 34, 'end': 42},
            {'text': 'downtrend', 'type': 'TREND', 'start': 78, 'end': 87}
        ])
        
        # Mock relation extraction
        self.extractor._extract_relations = MagicMock(return_value=[
            {'source': 'bullish engulfing', 'target': 'reversal', 'relation': 'IS_A', 'confidence': 0.88},
            {'source': 'bullish engulfing', 'target': 'downtrend', 'relation': 'APPEARS_IN', 'confidence': 0.82}
        ])
        
        # Mock sentiment analysis
        self.extractor._analyze_sentiment = MagicMock(return_value={
            'label': 'POSITIVE', 'score': 0.78
        })
        
        # Mock text classification
        self.extractor._classify_text = MagicMock(return_value={
            'label': 'CHART_PATTERN', 'score': 0.90
        })
        
        # Call the method
        patterns = self.extractor._extract_chart_patterns(text)
        
        # Verify NLP methods were called
        self.extractor._extract_entities.assert_called_with(text)
        self.extractor._extract_relations.assert_called()
        self.extractor._analyze_sentiment.assert_called_with(text)
        self.extractor._classify_text.assert_called_with(text)
        
        # Verify patterns were extracted correctly
        self.assertEqual(len(patterns), 1)
        self.assertTrue('pattern' in patterns[0])
        self.assertTrue('description' in patterns[0])
        self.assertTrue('confidence' in patterns[0])
        self.assertTrue('sentiment' in patterns[0])
        self.assertTrue('classification' in patterns[0])
        self.assertEqual(patterns[0]['pattern'], 'bullish engulfing')
    
    def test_extract_trading_strategies_with_nlp(self):
        """Test extracting trading strategies using NLP models."""
        # Test data
        text = "The Moving Average Crossover strategy involves buying when the short-term MA crosses above the long-term MA."
        
        # Mock entity extraction
        self.extractor._extract_entities = MagicMock(return_value=[
            {'text': 'Moving Average Crossover', 'type': 'STRATEGY', 'start': 4, 'end': 28},
            {'text': 'short-term MA', 'type': 'INDICATOR', 'start': 56, 'end': 69},
            {'text': 'long-term MA', 'type': 'INDICATOR', 'start': 91, 'end': 103}
        ])
        
        # Mock relation extraction
        self.extractor._extract_relations = MagicMock(return_value=[
            {'source': 'short-term MA', 'target': 'long-term MA', 'relation': 'CROSSES_ABOVE', 'confidence': 0.86}
        ])
        
        # Mock sentiment analysis
        self.extractor._analyze_sentiment = MagicMock(return_value={
            'label': 'NEUTRAL', 'score': 0.65
        })
        
        # Mock text classification
        self.extractor._classify_text = MagicMock(return_value={
            'label': 'TRADING_STRATEGY', 'score': 0.92
        })
        
        # Call the method
        strategies = self.extractor._extract_trading_strategies(text)
        
        # Verify NLP methods were called
        self.extractor._extract_entities.assert_called_with(text)
        self.extractor._extract_relations.assert_called()
        self.extractor._analyze_sentiment.assert_called_with(text)
        self.extractor._classify_text.assert_called_with(text)
        
        # Verify strategies were extracted correctly
        self.assertEqual(len(strategies), 1)
        self.assertTrue('strategy' in strategies[0])
        self.assertTrue('description' in strategies[0])
        self.assertTrue('confidence' in strategies[0])
        self.assertTrue('sentiment' in strategies[0])
        self.assertTrue('classification' in strategies[0])
        self.assertEqual(strategies[0]['strategy'], 'Moving Average Crossover')
    
    def test_extract_knowledge_with_nlp(self):
        """Test the main extract_knowledge method with NLP models."""
        # Test data
        text = "Buy when RSI is below 30. The bullish engulfing pattern is a reversal pattern."
        metadata = {"source": "Trading Book", "page": 42}
        
        # Mock extraction methods
        self.extractor._extract_trading_rules = MagicMock(return_value=[
            {'rule': 'Buy when RSI is below 30', 'confidence': 0.85}
        ])
        self.extractor._extract_chart_patterns = MagicMock(return_value=[
            {'pattern': 'bullish engulfing', 'description': 'A reversal pattern', 'confidence': 0.82}
        ])
        self.extractor._extract_trading_strategies = MagicMock(return_value=[])
        
        # Call the method
        knowledge_items = self.extractor.extract_knowledge(text, metadata)
        
        # Verify extraction methods were called
        self.extractor._extract_trading_rules.assert_called_with(text)
        self.extractor._extract_chart_patterns.assert_called_with(text)
        self.extractor._extract_trading_strategies.assert_called_with(text)
        
        # Verify knowledge items were created correctly
        self.assertEqual(len(knowledge_items), 2)  # 1 rule + 1 pattern
        self.assertTrue(any(item['type'] == 'trading_rule' for item in knowledge_items))
        self.assertTrue(any(item['type'] == 'chart_pattern' for item in knowledge_items))
        self.assertTrue(all('source' in item for item in knowledge_items))
        self.assertTrue(all('confidence' in item for item in knowledge_items))
        self.assertTrue(all('timestamp' in item for item in knowledge_items))


if __name__ == '__main__':
    unittest.main()