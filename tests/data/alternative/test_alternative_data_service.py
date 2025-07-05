"""Tests for the alternative data service module.

This module contains unit tests for the alternative data service and its components.
"""

import unittest
from unittest.mock import patch, MagicMock, call
from datetime import datetime

# Import the modules to test
from src.data.alternative.alternative_data_service import AlternativeDataService
from src.data.alternative.news_sentiment_analyzer import NewsSentimentAnalyzer
from src.data.alternative.social_media_analyzer import SocialMediaAnalyzer
from src.data.alternative.economic_data_provider import EconomicDataProvider
from src.data.alternative.alternative_data_normalizer import AlternativeDataNormalizer
from src.data.alternative.error_handling import (
    AlternativeDataError, DataSourceUnavailableError, 
    DataProcessingError, DataValidationError
)

class TestAlternativeDataService(unittest.TestCase):
    """Test cases for the AlternativeDataService class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock objects for dependencies
        self.mock_news_analyzer = MagicMock(spec=NewsSentimentAnalyzer)
        self.mock_social_analyzer = MagicMock(spec=SocialMediaAnalyzer)
        self.mock_economic_provider = MagicMock(spec=EconomicDataProvider)
        self.mock_normalizer = MagicMock(spec=AlternativeDataNormalizer)
        
        # Create a service instance with mock dependencies
        with patch('src.data.alternative.alternative_data_service.NewsSentimentAnalyzer', 
                  return_value=self.mock_news_analyzer), \
             patch('src.data.alternative.alternative_data_service.SocialMediaAnalyzer', 
                  return_value=self.mock_social_analyzer), \
             patch('src.data.alternative.alternative_data_service.EconomicDataProvider', 
                  return_value=self.mock_economic_provider), \
             patch('src.data.alternative.alternative_data_service.AlternativeDataNormalizer', 
                  return_value=self.mock_normalizer), \
             patch('src.data.alternative.alternative_data_service.get_collection'), \
             patch('src.data.alternative.alternative_data_service.insert_one'):
            self.service = AlternativeDataService()
    
    def test_update_all_alternative_data(self):
        """Test updating all alternative data for a symbol."""
        # Arrange
        symbols = ['AAPL', 'MSFT']
        
        # Act
        result = self.service.update_all_alternative_data(symbols)
        
        # Assert
        self.mock_news_analyzer.update_news_sentiment.assert_called_with(symbols)
        self.mock_social_analyzer.update_social_metrics.assert_called_with(symbols)
        self.mock_economic_provider.update_economic_indicators.assert_called_once()
        self.mock_normalizer.normalize_all_data.assert_called_with(symbols)
        self.assertTrue(result['success'])
        self.assertEqual(len(result['updated_symbols']), len(symbols))
    
    def test_update_news_sentiment(self):
        """Test updating news sentiment data."""
        # Arrange
        symbols = ['AAPL']
        mock_sentiment_data = {'AAPL': {'sentiment_score': 0.75}}
        self.mock_news_analyzer.update_news_sentiment.return_value = mock_sentiment_data
        
        # Act
        result = self.service.update_news_sentiment(symbols)
        
        # Assert
        self.mock_news_analyzer.update_news_sentiment.assert_called_with(symbols)
        self.assertEqual(result, mock_sentiment_data)
    
    def test_update_social_media_data(self):
        """Test updating social media data."""
        # Arrange
        symbols = ['MSFT']
        mock_social_data = {'MSFT': {'buzz_score': 85}}
        self.mock_social_analyzer.update_social_metrics.return_value = mock_social_data
        
        # Act
        result = self.service.update_social_media_data(symbols)
        
        # Assert
        self.mock_social_analyzer.update_social_metrics.assert_called_with(symbols)
        self.assertEqual(result, mock_social_data)
    
    def test_update_economic_data(self):
        """Test updating economic data."""
        # Arrange
        mock_economic_data = {'GDP': 2.5, 'Unemployment': 3.7}
        self.mock_economic_provider.update_economic_indicators.return_value = mock_economic_data
        
        # Act
        result = self.service.update_economic_data()
        
        # Assert
        self.mock_economic_provider.update_economic_indicators.assert_called_once()
        self.assertEqual(result, mock_economic_data)
    
    def test_get_alternative_data_features(self):
        """Test getting alternative data features."""
        # Arrange
        symbols = ['AAPL', 'MSFT']
        mock_features = {
            'AAPL': [0.75, 85, 0.5],
            'MSFT': [0.6, 90, 0.4]
        }
        self.mock_normalizer.get_feature_matrix.return_value = mock_features
        
        # Act
        result = self.service.get_alternative_data_features(symbols)
        
        # Assert
        self.mock_normalizer.get_feature_matrix.assert_called_with(symbols)
        self.assertEqual(result, mock_features)
    
    def test_get_latest_alternative_data(self):
        """Test getting latest alternative data."""
        # Arrange
        symbol = 'AAPL'
        mock_news_data = {'sentiment_score': 0.75}
        mock_social_data = {'buzz_score': 85}
        mock_economic_data = {'impact_score': 0.5}
        
        self.mock_news_analyzer.get_latest_sentiment.return_value = mock_news_data
        self.mock_social_analyzer.get_latest_metrics.return_value = mock_social_data
        self.mock_economic_provider.get_latest_impact.return_value = mock_economic_data
        
        # Act
        result = self.service.get_latest_alternative_data(symbol)
        
        # Assert
        self.mock_news_analyzer.get_latest_sentiment.assert_called_with(symbol)
        self.mock_social_analyzer.get_latest_metrics.assert_called_with(symbol)
        self.mock_economic_provider.get_latest_impact.assert_called_with(symbol)
        
        self.assertEqual(result['news'], mock_news_data)
        self.assertEqual(result['social'], mock_social_data)
        self.assertEqual(result['economic'], mock_economic_data)
    
    def test_error_handling(self):
        """Test error handling in the service."""
        # Arrange
        symbols = ['AAPL']
        self.mock_news_analyzer.update_news_sentiment.side_effect = DataSourceUnavailableError("API unavailable")
        
        # Mock the insert_one function for error reporting
        with patch('src.data.alternative.alternative_data_service.insert_one') as mock_insert:
            # Act
            result = self.service.update_all_alternative_data(symbols)
            
            # Assert
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            mock_insert.assert_called_once()


class TestNewsSentimentAnalyzer(unittest.TestCase):
    """Test cases for the NewsSentimentAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Patch MongoDB functions
        with patch('src.data.alternative.news_sentiment_analyzer.get_collection'), \
             patch('src.data.alternative.news_sentiment_analyzer.insert_one'), \
             patch('src.data.alternative.news_sentiment_analyzer.find_one'), \
             patch('src.data.alternative.news_sentiment_analyzer.update_one'):
            self.analyzer = NewsSentimentAnalyzer()
    
    @patch('src.data.alternative.news_sentiment_analyzer.requests.get')
    def test_fetch_news(self, mock_get):
        """Test fetching news data."""
        # Arrange
        symbol = 'AAPL'
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'articles': [
                {'title': 'Apple announces new iPhone', 'content': 'Positive news about Apple'},
                {'title': 'Apple stock rises', 'content': 'More positive news'}
            ]
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Act
        news = self.analyzer._fetch_news(symbol)
        
        # Assert
        self.assertEqual(len(news), 2)
        mock_get.assert_called_once()
    
    @patch('src.data.alternative.news_sentiment_analyzer.requests.get')
    def test_fetch_news_error(self, mock_get):
        """Test error handling when fetching news."""
        # Arrange
        symbol = 'AAPL'
        mock_get.side_effect = Exception("API error")
        
        # Act & Assert
        with self.assertRaises(DataSourceUnavailableError):
            self.analyzer._fetch_news(symbol)
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        # Arrange
        news_items = [
            {'title': 'Apple announces new iPhone', 'content': 'Positive news about Apple'},
            {'title': 'Apple stock rises', 'content': 'More positive news'}
        ]
        
        # Act
        sentiment = self.analyzer._analyze_sentiment(news_items)
        
        # Assert
        self.assertIn('sentiment_score', sentiment)
        self.assertIn('sentiment_label', sentiment)
        self.assertIn('confidence', sentiment)
    
    @patch('src.data.alternative.news_sentiment_analyzer.insert_one')
    @patch('src.data.alternative.news_sentiment_analyzer.NewsSentimentAnalyzer._fetch_news')
    @patch('src.data.alternative.news_sentiment_analyzer.NewsSentimentAnalyzer._analyze_sentiment')
    def test_update_news_sentiment(self, mock_analyze, mock_fetch, mock_insert):
        """Test updating news sentiment."""
        # Arrange
        symbols = ['AAPL', 'MSFT']
        mock_news = [
            {'title': 'Apple news', 'content': 'Content about Apple'}
        ]
        mock_sentiment = {
            'sentiment_score': 0.75,
            'sentiment_label': 'positive',
            'confidence': 0.85
        }
        
        mock_fetch.return_value = mock_news
        mock_analyze.return_value = mock_sentiment
        
        # Act
        result = self.analyzer.update_news_sentiment(symbols)
        
        # Assert
        self.assertEqual(mock_fetch.call_count, len(symbols))
        self.assertEqual(mock_analyze.call_count, len(symbols))
        self.assertEqual(mock_insert.call_count, len(symbols) * 2)  # Two collections
        
        for symbol in symbols:
            self.assertIn(symbol, result)
            self.assertEqual(result[symbol], mock_sentiment)


class TestSocialMediaAnalyzer(unittest.TestCase):
    """Test cases for the SocialMediaAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Patch MongoDB functions
        with patch('src.data.alternative.social_media_analyzer.get_collection'), \
             patch('src.data.alternative.social_media_analyzer.insert_one'), \
             patch('src.data.alternative.social_media_analyzer.find_one'), \
             patch('src.data.alternative.social_media_analyzer.update_one'):
            self.analyzer = SocialMediaAnalyzer()
    
    @patch('src.data.alternative.social_media_analyzer.requests.get')
    def test_fetch_social_data(self, mock_get):
        """Test fetching social media data."""
        # Arrange
        symbol = 'AAPL'
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'posts': [
                {'text': 'I love my new iPhone', 'likes': 100, 'shares': 50},
                {'text': 'Apple products are great', 'likes': 200, 'shares': 75}
            ]
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Act
        posts = self.analyzer._fetch_social_data(symbol)
        
        # Assert
        self.assertEqual(len(posts), 2)
        mock_get.assert_called_once()
    
    def test_calculate_social_buzz(self):
        """Test calculating social buzz score."""
        # Arrange
        posts = [
            {'text': 'I love my new iPhone', 'likes': 100, 'shares': 50},
            {'text': 'Apple products are great', 'likes': 200, 'shares': 75}
        ]
        
        # Act
        metrics = self.analyzer._calculate_social_buzz(posts)
        
        # Assert
        self.assertIn('buzz_score', metrics)
        self.assertIn('sentiment_score', metrics)
        self.assertIn('engagement_rate', metrics)
    
    @patch('src.data.alternative.social_media_analyzer.insert_one')
    @patch('src.data.alternative.social_media_analyzer.SocialMediaAnalyzer._fetch_social_data')
    @patch('src.data.alternative.social_media_analyzer.SocialMediaAnalyzer._calculate_social_buzz')
    def test_update_social_metrics(self, mock_calculate, mock_fetch, mock_insert):
        """Test updating social media metrics."""
        # Arrange
        symbols = ['AAPL', 'MSFT']
        mock_posts = [
            {'text': 'I love my new iPhone', 'likes': 100, 'shares': 50}
        ]
        mock_metrics = {
            'buzz_score': 85,
            'sentiment_score': 0.7,
            'engagement_rate': 0.05
        }
        
        mock_fetch.return_value = mock_posts
        mock_calculate.return_value = mock_metrics
        
        # Act
        result = self.analyzer.update_social_metrics(symbols)
        
        # Assert
        self.assertEqual(mock_fetch.call_count, len(symbols))
        self.assertEqual(mock_calculate.call_count, len(symbols))
        self.assertEqual(mock_insert.call_count, len(symbols) * 2)  # Two collections
        
        for symbol in symbols:
            self.assertIn(symbol, result)
            self.assertEqual(result[symbol], mock_metrics)


class TestEconomicDataProvider(unittest.TestCase):
    """Test cases for the EconomicDataProvider class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Patch MongoDB functions
        with patch('src.data.alternative.economic_data_provider.get_collection'), \
             patch('src.data.alternative.economic_data_provider.insert_one'), \
             patch('src.data.alternative.economic_data_provider.find_one'), \
             patch('src.data.alternative.economic_data_provider.update_one'):
            self.provider = EconomicDataProvider()
    
    @patch('src.data.alternative.economic_data_provider.requests.get')
    def test_fetch_economic_indicators(self, mock_get):
        """Test fetching economic indicators."""
        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'indicators': [
                {'name': 'GDP', 'value': 2.5, 'date': '2023-01-01'},
                {'name': 'Unemployment', 'value': 3.7, 'date': '2023-01-01'}
            ]
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Act
        indicators = self.provider._fetch_economic_indicators()
        
        # Assert
        self.assertEqual(len(indicators), 2)
        mock_get.assert_called_once()
    
    @patch('src.data.alternative.economic_data_provider.requests.get')
    def test_fetch_central_bank_data(self, mock_get):
        """Test fetching central bank data."""
        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'central_bank_data': [
                {'bank': 'Federal Reserve', 'rate': 5.25, 'date': '2023-01-01'},
                {'bank': 'ECB', 'rate': 3.75, 'date': '2023-01-01'}
            ]
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Act
        bank_data = self.provider._fetch_central_bank_data()
        
        # Assert
        self.assertEqual(len(bank_data), 2)
        mock_get.assert_called_once()
    
    def test_calculate_economic_impact(self):
        """Test calculating economic impact."""
        # Arrange
        symbol = 'AAPL'
        indicators = [
            {'name': 'GDP', 'value': 2.5, 'date': '2023-01-01'},
            {'name': 'Unemployment', 'value': 3.7, 'date': '2023-01-01'}
        ]
        bank_data = [
            {'bank': 'Federal Reserve', 'rate': 5.25, 'date': '2023-01-01'}
        ]
        
        # Act
        impact = self.provider._calculate_economic_impact(symbol, indicators, bank_data)
        
        # Assert
        self.assertIn('impact_score', impact)
        self.assertIn('rate_sensitivity', impact)
        self.assertIn('economic_exposure', impact)
    
    @patch('src.data.alternative.economic_data_provider.insert_one')
    @patch('src.data.alternative.economic_data_provider.EconomicDataProvider._fetch_economic_indicators')
    @patch('src.data.alternative.economic_data_provider.EconomicDataProvider._fetch_central_bank_data')
    def test_update_economic_indicators(self, mock_fetch_bank, mock_fetch_indicators, mock_insert):
        """Test updating economic indicators."""
        # Arrange
        mock_indicators = [
            {'name': 'GDP', 'value': 2.5, 'date': '2023-01-01'}
        ]
        mock_bank_data = [
            {'bank': 'Federal Reserve', 'rate': 5.25, 'date': '2023-01-01'}
        ]
        
        mock_fetch_indicators.return_value = mock_indicators
        mock_fetch_bank.return_value = mock_bank_data
        
        # Act
        result = self.provider.update_economic_indicators()
        
        # Assert
        mock_fetch_indicators.assert_called_once()
        mock_fetch_bank.assert_called_once()
        self.assertEqual(mock_insert.call_count, 2)  # Two collections
        self.assertIn('indicators', result)
        self.assertIn('central_bank_data', result)


class TestAlternativeDataNormalizer(unittest.TestCase):
    """Test cases for the AlternativeDataNormalizer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Patch MongoDB functions
        with patch('src.data.alternative.alternative_data_normalizer.get_collection'), \
             patch('src.data.alternative.alternative_data_normalizer.insert_one'), \
             patch('src.data.alternative.alternative_data_normalizer.find_one'), \
             patch('src.data.alternative.alternative_data_normalizer.update_one'), \
             patch('src.data.alternative.alternative_data_normalizer.find'):
            self.normalizer = AlternativeDataNormalizer()
    
    @patch('src.data.alternative.alternative_data_normalizer.find_one')
    def test_normalize_news_sentiment(self, mock_find_one):
        """Test normalizing news sentiment data."""
        # Arrange
        symbol = 'AAPL'
        mock_find_one.return_value = {
            'sentiment_score': 0.75,
            'sentiment_label': 'positive',
            'confidence': 0.85
        }
        
        # Act
        normalized = self.normalizer._normalize_news_sentiment(symbol)
        
        # Assert
        self.assertIsInstance(normalized, dict)
        self.assertIn('normalized_sentiment', normalized)
        mock_find_one.assert_called_once()
    
    @patch('src.data.alternative.alternative_data_normalizer.find_one')
    def test_normalize_social_metrics(self, mock_find_one):
        """Test normalizing social media metrics."""
        # Arrange
        symbol = 'AAPL'
        mock_find_one.return_value = {
            'buzz_score': 85,
            'sentiment_score': 0.7,
            'engagement_rate': 0.05
        }
        
        # Act
        normalized = self.normalizer._normalize_social_metrics(symbol)
        
        # Assert
        self.assertIsInstance(normalized, dict)
        self.assertIn('normalized_buzz', normalized)
        mock_find_one.assert_called_once()
    
    @patch('src.data.alternative.alternative_data_normalizer.find_one')
    def test_normalize_economic_data(self, mock_find_one):
        """Test normalizing economic data."""
        # Arrange
        symbol = 'AAPL'
        mock_find_one.return_value = {
            'impact_score': 0.6,
            'rate_sensitivity': 0.8,
            'economic_exposure': 0.4
        }
        
        # Act
        normalized = self.normalizer._normalize_economic_data(symbol)
        
        # Assert
        self.assertIsInstance(normalized, dict)
        self.assertIn('normalized_impact', normalized)
        mock_find_one.assert_called_once()
    
    @patch('src.data.alternative.alternative_data_normalizer.insert_one')
    @patch('src.data.alternative.alternative_data_normalizer.AlternativeDataNormalizer._normalize_news_sentiment')
    @patch('src.data.alternative.alternative_data_normalizer.AlternativeDataNormalizer._normalize_social_metrics')
    @patch('src.data.alternative.alternative_data_normalizer.AlternativeDataNormalizer._normalize_economic_data')
    def test_normalize_all_data(self, mock_norm_econ, mock_norm_social, mock_norm_news, mock_insert):
        """Test normalizing all alternative data."""
        # Arrange
        symbols = ['AAPL', 'MSFT']
        mock_norm_news.return_value = {'normalized_sentiment': 0.8}
        mock_norm_social.return_value = {'normalized_buzz': 0.7}
        mock_norm_econ.return_value = {'normalized_impact': 0.6}
        
        # Act
        result = self.normalizer.normalize_all_data(symbols)
        
        # Assert
        self.assertEqual(mock_norm_news.call_count, len(symbols))
        self.assertEqual(mock_norm_social.call_count, len(symbols))
        self.assertEqual(mock_norm_econ.call_count, len(symbols))
        self.assertEqual(mock_insert.call_count, len(symbols))
        
        for symbol in symbols:
            self.assertIn(symbol, result)
    
    @patch('src.data.alternative.alternative_data_normalizer.find_one')
    def test_get_feature_matrix(self, mock_find_one):
        """Test getting feature matrix."""
        # Arrange
        symbols = ['AAPL', 'MSFT']
        mock_normalized_data = {
            'normalized_sentiment': 0.8,
            'normalized_buzz': 0.7,
            'normalized_impact': 0.6
        }
        mock_find_one.return_value = mock_normalized_data
        
        # Act
        features = self.normalizer.get_feature_matrix(symbols)
        
        # Assert
        self.assertEqual(len(features), len(symbols))
        for symbol in symbols:
            self.assertIn(symbol, features)
            self.assertIsInstance(features[symbol], list)


if __name__ == '__main__':
    unittest.main()