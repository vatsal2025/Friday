"""Example script demonstrating how to integrate the model registry with NLP models.

This script shows how to update the BookKnowledgeExtractor to use the model registry
for loading and managing NLP models.
"""

import os
import spacy
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from src.services.model import (
    ModelRegistry,
    ModelTrainerIntegration,
    ModelLoader,
    load_model
)


class BookKnowledgeExtractorWithRegistry:
    """Extract knowledge from book text using NLP models from the model registry.
    
    This class demonstrates how to update the original BookKnowledgeExtractor
    to use the model registry for loading and managing NLP models.
    """
    
    def __init__(self, config=None):
        """Initialize the extractor with configuration.
        
        Args:
            config: Configuration dictionary with extraction parameters.
        """
        self.config = config or {}
        self.extraction_categories = self.config.get('extraction_categories', [
            'trading_rules', 'chart_patterns', 'trading_strategies'
        ])
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Initialize the model loader
        self.model_loader = ModelLoader()
        
        # Load NLP models from the registry
        self._load_nlp_models()
    
    def _load_nlp_models(self):
        """Load NLP models from the model registry."""
        try:
            # Load spaCy model for entity recognition
            self.nlp = self._load_spacy_model()
            
            # Load transformer models for relation extraction and sentiment analysis
            self.relation_model = self._load_relation_model()
            self.sentiment_model = self._load_sentiment_model()
            
            # Load text classification model for categorizing knowledge
            self.classification_model = self._load_classification_model()
            
            self.models_loaded = True
            print("NLP models loaded successfully from registry")
            
        except Exception as e:
            print(f"Error loading NLP models from registry: {e}")
            print("Falling back to rule-based extraction")
            self.models_loaded = False
    
    def _load_spacy_model(self):
        """Load spaCy model from the registry.
        
        Returns:
            The loaded spaCy model.
        """
        try:
            # Try to load from registry first
            model = self.model_loader.load_model("spacy_financial_ner")
            return model
        except Exception as e:
            print(f"Could not load spaCy model from registry: {e}")
            print("Loading default spaCy model")
            # Fall back to loading directly
            return spacy.load("en_core_web_sm")
    
    def _load_relation_model(self):
        """Load relation extraction model from the registry.
        
        Returns:
            The loaded relation extraction model.
        """
        try:
            # Try to load from registry first
            model = self.model_loader.load_model("financial_relation_extraction")
            return model
        except Exception as e:
            print(f"Could not load relation model from registry: {e}")
            print("Loading default relation model")
            # Fall back to loading directly
            model_name = "bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            return {"tokenizer": tokenizer, "model": model}
    
    def _load_sentiment_model(self):
        """Load sentiment analysis model from the registry.
        
        Returns:
            The loaded sentiment analysis model.
        """
        try:
            # Try to load from registry first
            model = self.model_loader.load_model("financial_sentiment_analysis")
            return model
        except Exception as e:
            print(f"Could not load sentiment model from registry: {e}")
            print("Loading default sentiment model")
            # Fall back to loading directly
            return pipeline("sentiment-analysis")
    
    def _load_classification_model(self):
        """Load text classification model from the registry.
        
        Returns:
            The loaded text classification model.
        """
        try:
            # Try to load from registry first
            model = self.model_loader.load_model("financial_text_classification")
            return model
        except Exception as e:
            print(f"Could not load classification model from registry: {e}")
            print("Loading default classification model")
            # Fall back to loading directly
            return pipeline("text-classification")
    
    def extract_knowledge(self, text):
        """Extract knowledge from the provided text.
        
        Args:
            text: The text to extract knowledge from.
            
        Returns:
            A dictionary containing extracted knowledge items.
        """
        # Implementation would be similar to the original BookKnowledgeExtractor
        # but using the models loaded from the registry
        pass


def register_nlp_models():
    """Register NLP models in the model registry.
    
    This function demonstrates how to register pre-trained NLP models
    in the model registry for later use.
    """
    # Initialize the model registry and trainer integration
    registry = ModelRegistry()
    integration = ModelTrainerIntegration(model_registry=registry)
    
    # Register spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
        model_id = integration.register_model_from_trainer(
            model=nlp,
            model_name="spacy_financial_ner",
            model_type="spacy",
            evaluation_results={
                "metrics": {"accuracy": 0.85, "f1": 0.82},
                "details": {"test_size": 1000}
            },
            training_data_info={
                "source": "spaCy pre-trained model",
                "version": "en_core_web_sm"
            },
            tags=["nlp", "ner", "spacy", "financial"],
            description="spaCy model for named entity recognition in financial texts"
        )
        print(f"Registered spaCy model with ID: {model_id}")
    except Exception as e:
        print(f"Error registering spaCy model: {e}")
    
    # Register BERT model for relation extraction
    try:
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Create a wrapper to store both tokenizer and model
        bert_wrapper = {"tokenizer": tokenizer, "model": model}
        
        model_id = integration.register_model_from_trainer(
            model=bert_wrapper,
            model_name="financial_relation_extraction",
            model_type="transformer",
            evaluation_results={
                "metrics": {"accuracy": 0.88, "f1": 0.85},
                "details": {"test_size": 2000}
            },
            training_data_info={
                "source": "Hugging Face pre-trained model",
                "version": model_name
            },
            tags=["nlp", "relation_extraction", "transformer", "bert", "financial"],
            description="BERT model for relation extraction in financial texts"
        )
        print(f"Registered BERT model with ID: {model_id}")
    except Exception as e:
        print(f"Error registering BERT model: {e}")
    
    # Register sentiment analysis pipeline
    try:
        sentiment_pipeline = pipeline("sentiment-analysis")
        
        model_id = integration.register_model_from_trainer(
            model=sentiment_pipeline,
            model_name="financial_sentiment_analysis",
            model_type="pipeline",
            evaluation_results={
                "metrics": {"accuracy": 0.90, "f1": 0.88},
                "details": {"test_size": 1500}
            },
            training_data_info={
                "source": "Hugging Face pipeline",
                "version": "sentiment-analysis"
            },
            tags=["nlp", "sentiment", "pipeline", "financial"],
            description="Sentiment analysis pipeline for financial texts"
        )
        print(f"Registered sentiment analysis pipeline with ID: {model_id}")
    except Exception as e:
        print(f"Error registering sentiment analysis pipeline: {e}")
    
    # Register text classification pipeline
    try:
        classification_pipeline = pipeline("text-classification")
        
        model_id = integration.register_model_from_trainer(
            model=classification_pipeline,
            model_name="financial_text_classification",
            model_type="pipeline",
            evaluation_results={
                "metrics": {"accuracy": 0.87, "f1": 0.85},
                "details": {"test_size": 1800}
            },
            training_data_info={
                "source": "Hugging Face pipeline",
                "version": "text-classification"
            },
            tags=["nlp", "classification", "pipeline", "financial"],
            description="Text classification pipeline for financial texts"
        )
        print(f"Registered text classification pipeline with ID: {model_id}")
    except Exception as e:
        print(f"Error registering text classification pipeline: {e}")


def main():
    """Main function to demonstrate NLP model registry integration."""
    print("=== NLP Model Registry Integration Example ===")
    
    # Register NLP models in the registry
    print("\n1. Registering NLP models in the registry...")
    register_nlp_models()
    
    # Create an extractor that uses the registry
    print("\n2. Creating BookKnowledgeExtractor with model registry integration...")
    extractor = BookKnowledgeExtractorWithRegistry()
    
    # Example text for extraction
    sample_text = """
    When trading, it's important to follow the trend. A common rule is 'the trend is your friend'.
    Look for chart patterns like head and shoulders or double bottoms to identify potential reversals.
    A simple trading strategy is to buy when the price crosses above a moving average and sell when it crosses below.
    """
    
    print("\n3. Extractor initialized with models from registry")
    print(f"   - Models loaded successfully: {extractor.models_loaded}")
    print(f"   - Available extraction categories: {extractor.extraction_categories}")
    
    print("\n=== Example Completed ===")


if __name__ == "__main__":
    main()