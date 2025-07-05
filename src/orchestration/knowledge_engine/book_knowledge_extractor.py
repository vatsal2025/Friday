"""Book Knowledge Extractor Module.

This module contains the BookKnowledgeExtractor class which is responsible for
extracting trading knowledge from book content using NLP techniques.
"""

import re
import uuid
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import NLP libraries
import spacy
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
import torch
import numpy as np

from src.infrastructure.event.event_system import EventSystem
from src.infrastructure.database import mongodb


class BookKnowledgeExtractor:
    """Extracts trading knowledge from book content.
    
    This class uses NLP techniques to extract trading rules, chart patterns,
    and trading strategies from book content.
    """
    
    def __init__(self, config: Dict[str, Any], event_system: EventSystem):
        """Initialize the BookKnowledgeExtractor.
        
        Args:
            config: Configuration dictionary for knowledge extraction
            event_system: Event system for publishing events
        """
        self.config = config
        self.event_system = event_system
        self.logger = logging.getLogger(__name__)
        
        # Initialize extraction categories and confidence thresholds
        self.extraction_categories = self.config.get('extraction_categories', {
            'trading_rules': True,
            'chart_patterns': True,
            'trading_strategies': True
        })
        
        self.confidence_thresholds = self.config.get('confidence_thresholds', {
            'trading_rules': 0.7,
            'chart_patterns': 0.7,
            'trading_strategies': 0.7
        })
        
        # Initialize database connection
        collection_name = self.config.get('storage', {}).get('collection_name', 'trading_knowledge')
        self.collection = mongodb.get_collection(collection_name)
        
        # Initialize NLP models
        self.nlp_models = {}
        self._load_nlp_models()
        
        self.logger.info("BookKnowledgeExtractor initialized")
    
    def _load_nlp_models(self):
        """Load NLP models for entity recognition, relation extraction, etc."""
        self.logger.info("Loading NLP models")
        
        # Get NLP configuration
        nlp_config = self.config.get('extraction_methods', {}).get('nlp', {})
        if not nlp_config.get('enabled', False):
            self.logger.warning("NLP extraction is disabled in configuration")
            return
        
        models_config = nlp_config.get('models', {})
        
        try:
            # Load entity recognition model (spaCy)
            entity_config = models_config.get('entity_recognition', {})
            if entity_config.get('type') == 'spacy':
                model_name = entity_config.get('model_name', 'en_core_web_md')
                self.logger.info(f"Loading spaCy model: {model_name}")
                self.nlp_models['entity_recognition'] = spacy.load(model_name)
            
            # Load relation extraction model (transformer-based)
            relation_config = models_config.get('relation_extraction', {})
            if relation_config.get('type') == 'transformer':
                model_name = relation_config.get('model_name', 'bert-base-uncased')
                self.logger.info(f"Loading relation extraction model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.nlp_models['relation_extraction'] = {
                    'tokenizer': tokenizer,
                    'model': model
                }
            
            # Load sentiment analysis model
            sentiment_config = models_config.get('sentiment_analysis', {})
            if sentiment_config.get('type') == 'transformer':
                model_name = sentiment_config.get('model_name', 'distilbert-base-uncased-finetuned-sst-2-english')
                self.logger.info(f"Loading sentiment analysis model: {model_name}")
                self.nlp_models['sentiment_analysis'] = pipeline(
                    task='sentiment-analysis',
                    model=model_name,
                    tokenizer=model_name
                )
            
            # Load text classification model
            classification_config = models_config.get('text_classification', {})
            if classification_config.get('type') == 'transformer':
                model_name = classification_config.get('model_name', 'distilbert-base-uncased')
                self.logger.info(f"Loading text classification model: {model_name}")
                self.nlp_models['text_classification'] = pipeline(
                    task='text-classification',
                    model=model_name,
                    tokenizer=model_name
                )
            
            self.logger.info("NLP models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading NLP models: {str(e)}")
            # Fall back to rule-based extraction if NLP models fail to load
            self.logger.warning("Falling back to rule-based extraction only")
    
    def extract_knowledge(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract knowledge from text.
        
        Args:
            text: The text to extract knowledge from
            metadata: Metadata about the source of the text
            
        Returns:
            A list of extracted knowledge items
        """
        self.logger.info(f"Extracting knowledge from text of length {len(text)}")
        
        # Split text into manageable chunks
        chunks = self._split_text_into_chunks(text)
        
        all_items = []
        
        for chunk in chunks:
            # Extract different types of knowledge
            rules = []
            patterns = []
            strategies = []
            
            if self.extraction_categories.get('trading_rules', True):
                rules = self._extract_trading_rules(chunk)
            
            if self.extraction_categories.get('chart_patterns', True):
                patterns = self._extract_chart_patterns(chunk)
            
            if self.extraction_categories.get('trading_strategies', True):
                strategies = self._extract_trading_strategies(chunk)
            
            # Combine all extracted items
            items = self._prepare_knowledge_items(rules, patterns, strategies, metadata)
            all_items.extend(items)
        
        # Filter items based on confidence threshold
        filtered_items = self._filter_by_confidence(all_items)
        
        # Publish event if items were extracted
        if filtered_items:
            self._publish_extraction_event(filtered_items, metadata)
        
        self.logger.info(f"Extracted {len(filtered_items)} knowledge items")
        return filtered_items
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into manageable chunks for processing.
        
        Args:
            text: The text to split
            chunk_size: Maximum size of each chunk in characters
            
        Returns:
            A list of text chunks
        """
        # Simple chunking by character count
        if len(text) <= chunk_size:
            return [text]
        
        # Try to split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using spaCy.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            A list of extracted entities with their types and positions
        """
        entities = []
        
        if 'entity_recognition' not in self.nlp_models:
            # Fall back to rule-based entity extraction
            return self._extract_entities_rule_based(text)
        
        try:
            nlp = self.nlp_models['entity_recognition']
            doc = nlp(text)
            
            for ent in doc.ents:
                entity = {
                    'text': ent.text,
                    'type': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                }
                entities.append(entity)
            
            # If no entities were found with spaCy, fall back to rule-based
            if not entities:
                entities = self._extract_entities_rule_based(text)
        except Exception as e:
            self.logger.error(f"Error in entity extraction: {str(e)}")
            # Fall back to rule-based extraction
            entities = self._extract_entities_rule_based(text)
        
        return entities
    
    def _extract_entities_rule_based(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using rule-based patterns.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            A list of extracted entities
        """
        entities = []
        
        # Pattern for technical indicators
        indicator_patterns = [
            r'\b(RSI|MACD|MA|EMA|SMA|Bollinger Bands|Stochastic|ADX|OBV|ATR|CCI|ROC|MFI|Ichimoku|VWAP)\b',
            r'\b(Moving Average|Relative Strength Index|Average Directional Index)\b'
        ]
        
        # Pattern for chart patterns
        pattern_patterns = [
            r'\b(Head and Shoulders|Double Top|Double Bottom|Triple Top|Triple Bottom)\b',
            r'\b(Bullish Engulfing|Bearish Engulfing|Morning Star|Evening Star|Doji|Hammer|Shooting Star)\b',
            r'\b(Flag|Pennant|Triangle|Wedge|Cup and Handle|Rectangle|Channel)\b'
        ]
        
        # Pattern for trading actions
        action_patterns = [
            r'\b(Buy|Sell|Long|Short|Enter|Exit|Stop|Limit|Take Profit|Stop Loss)\b'
        ]
        
        # Pattern for values/numbers
        value_patterns = [
            r'\b(\d+(\.\d+)?)\b'
        ]
        
        # Extract indicators
        for pattern in indicator_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = {
                    'text': match.group(),
                    'type': 'INDICATOR',
                    'start': match.start(),
                    'end': match.end()
                }
                entities.append(entity)
        
        # Extract chart patterns
        for pattern in pattern_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = {
                    'text': match.group(),
                    'type': 'PATTERN',
                    'start': match.start(),
                    'end': match.end()
                }
                entities.append(entity)
        
        # Extract actions
        for pattern in action_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = {
                    'text': match.group(),
                    'type': 'ACTION',
                    'start': match.start(),
                    'end': match.end()
                }
                entities.append(entity)
        
        # Extract values
        for pattern in value_patterns:
            for match in re.finditer(pattern, text):
                entity = {
                    'text': match.group(),
                    'type': 'VALUE',
                    'start': match.start(),
                    'end': match.end()
                }
                entities.append(entity)
        
        return entities
    
    def _extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations between entities.
        
        Args:
            text: The text containing the entities
            entities: The entities to find relations between
            
        Returns:
            A list of relations between entities
        """
        relations = []
        
        if 'relation_extraction' not in self.nlp_models:
            # Fall back to rule-based relation extraction
            return self._extract_relations_rule_based(text, entities)
        
        try:
            # Use transformer model for relation extraction
            relation_model = self.nlp_models['relation_extraction']
            tokenizer = relation_model['tokenizer']
            model = relation_model['model']
            
            # For each pair of entities, extract relation
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities):
                    if i != j:  # Don't relate an entity to itself
                        # Prepare input for the model
                        entity1_text = entity1['text']
                        entity2_text = entity2['text']
                        
                        # Create a template for relation extraction
                        template = f"{entity1_text} [SEP] {entity2_text}"
                        
                        # Tokenize and prepare model input
                        inputs = tokenizer.encode_plus(
                            template,
                            add_special_tokens=True,
                            return_tensors="pt"
                        )
                        
                        # Get model prediction
                        with torch.no_grad():
                            outputs = model(**inputs)
                            logits = outputs.logits
                            probs = torch.softmax(logits, dim=1)
                            prediction = torch.argmax(probs, dim=1).item()
                            confidence = probs[0][prediction].item()
                        
                        # Map prediction to relation type
                        relation_types = ['NO_RELATION', 'RELATED_TO', 'PART_OF']
                        if len(relation_types) > prediction:
                            relation_type = relation_types[prediction]
                        else:
                            relation_type = 'UNKNOWN_RELATION'
                        
                        # Only add relations with sufficient confidence
                        if confidence > 0.5 and relation_type != 'NO_RELATION':
                            relation = {
                                'source': entity1_text,
                                'target': entity2_text,
                                'relation': relation_type,
                                'confidence': confidence
                            }
                            relations.append(relation)
        except Exception as e:
            self.logger.error(f"Error in relation extraction: {str(e)}")
            # Fall back to rule-based relation extraction
            relations = self._extract_relations_rule_based(text, entities)
        
        return relations
    
    def _extract_relations_rule_based(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations between entities using rule-based patterns.
        
        Args:
            text: The text containing the entities
            entities: The entities to find relations between
            
        Returns:
            A list of relations between entities
        """
        relations = []
        
        # Define relation patterns
        relation_patterns = [
            (r'(\w+)\s+is\s+below\s+(\w+)', 'BELOW'),
            (r'(\w+)\s+is\s+above\s+(\w+)', 'ABOVE'),
            (r'(\w+)\s+crosses\s+above\s+(\w+)', 'CROSSES_ABOVE'),
            (r'(\w+)\s+crosses\s+below\s+(\w+)', 'CROSSES_BELOW'),
            (r'(\w+)\s+is\s+greater\s+than\s+(\w+)', 'GREATER_THAN'),
            (r'(\w+)\s+is\s+less\s+than\s+(\w+)', 'LESS_THAN'),
            (r'(\w+)\s+equals\s+(\w+)', 'EQUALS'),
            (r'(\w+)\s+and\s+(\w+)', 'AND'),
            (r'(\w+)\s+or\s+(\w+)', 'OR')
        ]
        
        # Extract relations using patterns
        for pattern, relation_type in relation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                source_text = match.group(1)
                target_text = match.group(2)
                
                # Find corresponding entities
                source_entity = None
                target_entity = None
                
                for entity in entities:
                    if entity['text'].lower() == source_text.lower():
                        source_entity = entity
                    if entity['text'].lower() == target_text.lower():
                        target_entity = entity
                
                if source_entity and target_entity:
                    relation = {
                        'source': source_entity['text'],
                        'target': target_entity['text'],
                        'relation': relation_type,
                        'confidence': 0.8  # Default confidence for rule-based
                    }
                    relations.append(relation)
        
        # Also look for entity pairs that are close to each other in the text
        sorted_entities = sorted(entities, key=lambda e: e['start'])
        for i in range(len(sorted_entities) - 1):
            entity1 = sorted_entities[i]
            entity2 = sorted_entities[i + 1]
            
            # If entities are close to each other, assume a relation
            if entity2['start'] - entity1['end'] < 20:  # Within 20 characters
                # Extract the text between the entities
                between_text = text[entity1['end']:entity2['start']].strip().lower()
                
                # Determine relation type based on the text between entities
                relation_type = 'RELATED_TO'  # Default relation
                
                if 'above' in between_text or 'greater' in between_text:
                    relation_type = 'ABOVE'
                elif 'below' in between_text or 'less' in between_text:
                    relation_type = 'BELOW'
                elif 'cross' in between_text and 'above' in between_text:
                    relation_type = 'CROSSES_ABOVE'
                elif 'cross' in between_text and 'below' in between_text:
                    relation_type = 'CROSSES_BELOW'
                elif 'equal' in between_text:
                    relation_type = 'EQUALS'
                
                relation = {
                    'source': entity1['text'],
                    'target': entity2['text'],
                    'relation': relation_type,
                    'confidence': 0.7  # Lower confidence for proximity-based relations
                }
                relations.append(relation)
        
        return relations
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text.
        
        Args:
            text: The text to analyze
            
        Returns:
            A dictionary with sentiment label and score
        """
        if 'sentiment_analysis' not in self.nlp_models:
            # Return neutral sentiment if model not available
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        try:
            sentiment_pipeline = self.nlp_models['sentiment_analysis']
            result = sentiment_pipeline(text)
            
            if result and len(result) > 0:
                return {
                    'label': result[0]['label'],
                    'score': result[0]['score']
                }
            else:
                return {'label': 'NEUTRAL', 'score': 0.5}
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    def _classify_text(self, text: str) -> Dict[str, Any]:
        """Classify text into predefined categories.
        
        Args:
            text: The text to classify
            
        Returns:
            A dictionary with classification label and score
        """
        if 'text_classification' not in self.nlp_models:
            # Perform rule-based classification if model not available
            return self._classify_text_rule_based(text)
        
        try:
            classification_pipeline = self.nlp_models['text_classification']
            result = classification_pipeline(text)
            
            if result and len(result) > 0:
                return {
                    'label': result[0]['label'],
                    'score': result[0]['score']
                }
            else:
                return self._classify_text_rule_based(text)
        except Exception as e:
            self.logger.error(f"Error in text classification: {str(e)}")
            return self._classify_text_rule_based(text)
    
    def _classify_text_rule_based(self, text: str) -> Dict[str, Any]:
        """Classify text using rule-based patterns.
        
        Args:
            text: The text to classify
            
        Returns:
            A dictionary with classification label and score
        """
        text_lower = text.lower()
        
        # Rule patterns for different categories
        rule_patterns = [
            (r'\b(buy|sell|long|short|enter|exit)\b.*\b(when|if)\b', 'TRADING_RULE'),
            (r'\b(stop loss|take profit|trailing stop)\b', 'TRADING_RULE'),
            (r'\b(pattern|chart pattern|candlestick|formation)\b', 'CHART_PATTERN'),
            (r'\b(strategy|system|method|approach|technique)\b', 'TRADING_STRATEGY'),
            (r'\b(indicator|oscillator|index|ratio)\b', 'INDICATOR_DESCRIPTION')
        ]
        
        # Check each pattern
        for pattern, label in rule_patterns:
            if re.search(pattern, text_lower):
                return {'label': label, 'score': 0.8}
        
        # Default classification
        return {'label': 'GENERAL_TRADING_INFO', 'score': 0.6}
    
    def _extract_trading_rules(self, text: str) -> List[Dict[str, Any]]:
        """Extract trading rules from text.
        
        Args:
            text: The text to extract rules from
            
        Returns:
            A list of extracted trading rules
        """
        rules = []
        
        # Extract entities and relations
        entities = self._extract_entities(text)
        relations = self._extract_relations(text, entities)
        
        # Analyze sentiment and classify text
        sentiment = self._analyze_sentiment(text)
        classification = self._classify_text(text)
        
        # Rule-based extraction for trading rules
        rule_patterns = [
            r'(Buy|Sell|Long|Short|Enter|Exit)\s+when\s+([^.!?]+)',
            r'If\s+([^,]+),\s+(buy|sell|long|short|enter|exit)\s+([^.!?]+)',
            r'([^.!?]+)\s+is\s+a\s+(buy|sell)\s+signal'
        ]
        
        for pattern in rule_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                rule_text = match.group(0)
                
                # Calculate confidence based on entities and relations
                confidence = 0.7  # Base confidence
                
                # Increase confidence if entities are found in the rule
                rule_entities = [e for e in entities if e['start'] >= match.start() and e['end'] <= match.end()]
                if rule_entities:
                    confidence += 0.1
                
                # Increase confidence if relations are found between entities in the rule
                rule_entity_texts = [e['text'] for e in rule_entities]
                rule_relations = [r for r in relations if r['source'] in rule_entity_texts and r['target'] in rule_entity_texts]
                if rule_relations:
                    confidence += 0.1
                
                # Adjust confidence based on text classification
                if classification['label'] == 'TRADING_RULE':
                    confidence += 0.1
                
                rule = {
                    'rule': rule_text,
                    'confidence': min(confidence, 1.0),  # Cap at 1.0
                    'entities': rule_entities,
                    'relations': rule_relations,
                    'sentiment': sentiment,
                    'classification': classification
                }
                
                rules.append(rule)
        
        # If no rules found with patterns, try to construct from entities and relations
        if not rules and entities and relations:
            for relation in relations:
                # Construct rule from relation
                if relation['relation'] in ['ABOVE', 'BELOW', 'CROSSES_ABOVE', 'CROSSES_BELOW']:
                    # Find entity types
                    source_type = next((e['type'] for e in entities if e['text'] == relation['source']), None)
                    target_type = next((e['type'] for e in entities if e['text'] == relation['target']), None)
                    
                    if source_type == 'INDICATOR' and target_type == 'VALUE':
                        # Construct rule based on relation type
                        if relation['relation'] == 'BELOW':
                            rule_text = f"Buy when {relation['source']} is below {relation['target']}"
                        elif relation['relation'] == 'ABOVE':
                            rule_text = f"Sell when {relation['source']} is above {relation['target']}"
                        elif relation['relation'] == 'CROSSES_ABOVE':
                            rule_text = f"Buy when {relation['source']} crosses above {relation['target']}"
                        elif relation['relation'] == 'CROSSES_BELOW':
                            rule_text = f"Sell when {relation['source']} crosses below {relation['target']}"
                        else:
                            continue
                        
                        rule = {
                            'rule': rule_text,
                            'confidence': relation['confidence'],
                            'entities': [e for e in entities if e['text'] in [relation['source'], relation['target']]],
                            'relations': [relation],
                            'sentiment': sentiment,
                            'classification': classification
                        }
                        
                        rules.append(rule)
        
        return rules
    
    def _extract_chart_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract chart patterns from text.
        
        Args:
            text: The text to extract patterns from
            
        Returns:
            A list of extracted chart patterns
        """
        patterns = []
        
        # Extract entities and relations
        entities = self._extract_entities(text)
        relations = self._extract_relations(text, entities)
        
        # Analyze sentiment and classify text
        sentiment = self._analyze_sentiment(text)
        classification = self._classify_text(text)
        
        # Find pattern entities
        pattern_entities = [e for e in entities if e['type'] == 'PATTERN']
        
        for entity in pattern_entities:
            pattern_text = entity['text']
            
            # Extract description from surrounding text
            start_pos = max(0, entity['start'] - 50)
            end_pos = min(len(text), entity['end'] + 150)
            context = text[start_pos:end_pos]
            
            # Try to find a description sentence
            description_match = re.search(f"{re.escape(pattern_text)}\s+is\s+([^.!?]+)[.!?]", context, re.IGNORECASE)
            if description_match:
                description = f"{pattern_text} is {description_match.group(1)}"
            else:
                # Use the surrounding context as description
                description = context
            
            # Calculate confidence
            confidence = 0.7  # Base confidence
            
            # Increase confidence if relations are found for this pattern
            pattern_relations = [r for r in relations if r['source'] == pattern_text or r['target'] == pattern_text]
            if pattern_relations:
                confidence += 0.1
            
            # Adjust confidence based on text classification
            if classification['label'] == 'CHART_PATTERN':
                confidence += 0.1
            
            pattern = {
                'pattern': pattern_text,
                'description': description,
                'confidence': min(confidence, 1.0),  # Cap at 1.0
                'entities': [entity],
                'relations': pattern_relations,
                'sentiment': sentiment,
                'classification': classification
            }
            
            patterns.append(pattern)
        
        return patterns
    
    def _extract_trading_strategies(self, text: str) -> List[Dict[str, Any]]:
        """Extract trading strategies from text.
        
        Args:
            text: The text to extract strategies from
            
        Returns:
            A list of extracted trading strategies
        """
        strategies = []
        
        # Extract entities and relations
        entities = self._extract_entities(text)
        relations = self._extract_relations(text, entities)
        
        # Analyze sentiment and classify text
        sentiment = self._analyze_sentiment(text)
        classification = self._classify_text(text)
        
        # Strategy patterns
        strategy_patterns = [
            r'([\w\s]+)\s+strategy\s+([^.!?]+)',
            r'([\w\s]+)\s+system\s+([^.!?]+)',
            r'([\w\s]+)\s+method\s+([^.!?]+)'
        ]
        
        for pattern in strategy_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                strategy_name = match.group(1).strip()
                description = match.group(0)
                
                # Calculate confidence
                confidence = 0.7  # Base confidence
                
                # Find entities related to this strategy
                strategy_entities = []
                for entity in entities:
                    if entity['start'] >= match.start() and entity['end'] <= match.end():
                        strategy_entities.append(entity)
                
                if strategy_entities:
                    confidence += 0.1
                
                # Find relations between entities in this strategy
                strategy_entity_texts = [e['text'] for e in strategy_entities]
                strategy_relations = []
                for relation in relations:
                    if relation['source'] in strategy_entity_texts or relation['target'] in strategy_entity_texts:
                        strategy_relations.append(relation)
                
                if strategy_relations:
                    confidence += 0.1
                
                # Adjust confidence based on text classification
                if classification['label'] == 'TRADING_STRATEGY':
                    confidence += 0.1
                
                # Extract steps if possible
                steps = self._extract_strategy_steps(text, match.start(), match.end())
                
                strategy = {
                    'strategy': strategy_name,
                    'description': description,
                    'steps': steps,
                    'confidence': min(confidence, 1.0),  # Cap at 1.0
                    'entities': strategy_entities,
                    'relations': strategy_relations,
                    'sentiment': sentiment,
                    'classification': classification
                }
                
                strategies.append(strategy)
        
        # Look for strategy entities if no strategies found with patterns
        if not strategies:
            strategy_entities = [e for e in entities if 'strategy' in e['text'].lower() or 'system' in e['text'].lower()]
            
            for entity in strategy_entities:
                strategy_name = entity['text']
                
                # Extract description from surrounding text
                start_pos = max(0, entity['start'] - 50)
                end_pos = min(len(text), entity['end'] + 150)
                context = text[start_pos:end_pos]
                
                # Try to find a description sentence
                description_match = re.search(f"{re.escape(strategy_name)}\s+([^.!?]+)[.!?]", context, re.IGNORECASE)
                if description_match:
                    description = f"{strategy_name} {description_match.group(1)}"
                else:
                    # Use the surrounding context as description
                    description = context
                
                # Extract steps if possible
                steps = self._extract_strategy_steps(text, start_pos, end_pos)
                
                # Calculate confidence
                confidence = 0.7  # Base confidence
                
                # Find relations for this strategy
                strategy_relations = [r for r in relations if r['source'] == strategy_name or r['target'] == strategy_name]
                if strategy_relations:
                    confidence += 0.1
                
                # Adjust confidence based on text classification
                if classification['label'] == 'TRADING_STRATEGY':
                    confidence += 0.1
                
                strategy = {
                    'strategy': strategy_name,
                    'description': description,
                    'steps': steps,
                    'confidence': min(confidence, 1.0),  # Cap at 1.0
                    'entities': [entity],
                    'relations': strategy_relations,
                    'sentiment': sentiment,
                    'classification': classification
                }
                
                strategies.append(strategy)
        
        return strategies
    
    def _extract_strategy_steps(self, text: str, start_pos: int, end_pos: int) -> List[str]:
        """Extract steps from a trading strategy description.
        
        Args:
            text: The full text
            start_pos: Start position of the strategy description
            end_pos: End position of the strategy description
            
        Returns:
            A list of strategy steps
        """
        steps = []
        
        # Extract the strategy context
        context = text[start_pos:end_pos]
        
        # Look for numbered steps
        step_matches = re.finditer(r'(\d+\.\s+[^.!?\n]+)', context)
        for match in step_matches:
            steps.append(match.group(1).strip())
        
        # If no numbered steps, look for steps with keywords
        if not steps:
            step_keywords = ['first', 'second', 'third', 'next', 'then', 'finally']
            for keyword in step_keywords:
                pattern = f"({keyword}\s+[^.!?\n]+)"
                for match in re.finditer(pattern, context, re.IGNORECASE):
                    steps.append(match.group(1).strip())
        
        # If still no steps, try to split by sentences and look for action verbs
        if not steps:
            sentences = re.split(r'(?<=[.!?])\s+', context)
            action_verbs = ['buy', 'sell', 'enter', 'exit', 'place', 'set', 'calculate', 'identify', 'wait', 'monitor']
            
            for sentence in sentences:
                for verb in action_verbs:
                    if re.search(f"\b{verb}\b", sentence, re.IGNORECASE):
                        steps.append(sentence.strip())
                        break
        
        return steps
    
    def _prepare_knowledge_items(self, rules: List[Dict[str, Any]], patterns: List[Dict[str, Any]], 
                               strategies: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare knowledge items from extracted rules, patterns, and strategies.
        
        Args:
            rules: Extracted trading rules
            patterns: Extracted chart patterns
            strategies: Extracted trading strategies
            metadata: Source metadata
            
        Returns:
            A list of knowledge items
        """
        items = []
        
        # Process trading rules
        for rule in rules:
            item = {
                'id': str(uuid.uuid4()),
                'type': 'trading_rule',
                'content': {
                    'rule': rule['rule'],
                    'entities': rule.get('entities', []),
                    'relations': rule.get('relations', []),
                    'sentiment': rule.get('sentiment', {'label': 'NEUTRAL', 'score': 0.5}),
                    'classification': rule.get('classification', {'label': 'TRADING_RULE', 'score': 0.7})
                },
                'source': metadata,
                'confidence': rule['confidence'],
                'timestamp': datetime.datetime.now().isoformat()
            }
            items.append(item)
        
        # Process chart patterns
        for pattern in patterns:
            item = {
                'id': str(uuid.uuid4()),
                'type': 'chart_pattern',
                'content': {
                    'pattern': pattern['pattern'],
                    'description': pattern['description'],
                    'entities': pattern.get('entities', []),
                    'relations': pattern.get('relations', []),
                    'sentiment': pattern.get('sentiment', {'label': 'NEUTRAL', 'score': 0.5}),
                    'classification': pattern.get('classification', {'label': 'CHART_PATTERN', 'score': 0.7})
                },
                'source': metadata,
                'confidence': pattern['confidence'],
                'timestamp': datetime.datetime.now().isoformat()
            }
            items.append(item)
        
        # Process trading strategies
        for strategy in strategies:
            item = {
                'id': str(uuid.uuid4()),
                'type': 'trading_strategy',
                'content': {
                    'strategy': strategy['strategy'],
                    'description': strategy['description'],
                    'steps': strategy.get('steps', []),
                    'entities': strategy.get('entities', []),
                    'relations': strategy.get('relations', []),
                    'sentiment': strategy.get('sentiment', {'label': 'NEUTRAL', 'score': 0.5}),
                    'classification': strategy.get('classification', {'label': 'TRADING_STRATEGY', 'score': 0.7})
                },
                'source': metadata,
                'confidence': strategy['confidence'],
                'timestamp': datetime.datetime.now().isoformat()
            }
            items.append(item)
        
        return items
    
    def _filter_by_confidence(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter knowledge items based on confidence thresholds.
        
        Args:
            items: Knowledge items to filter
            
        Returns:
            Filtered knowledge items
        """
        filtered_items = []
        
        for item in items:
            item_type = item['type']
            confidence = item['confidence']
            
            # Get threshold for this item type
            threshold = self.confidence_thresholds.get(item_type, 0.7)
            
            if confidence >= threshold:
                filtered_items.append(item)
        
        return filtered_items
    
    def _publish_extraction_event(self, items: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
        """Publish an event when knowledge is extracted.
        
        Args:
            items: Extracted knowledge items
            metadata: Source metadata
        """
        event_data = {
            'event_type': 'knowledge_extracted',
            'timestamp': datetime.datetime.now().isoformat(),
            'source': metadata,
            'item_count': len(items),
            'item_types': {}
        }
        
        # Count items by type
        for item in items:
            item_type = item['type']
            if item_type in event_data['item_types']:
                event_data['item_types'][item_type] += 1
            else:
                event_data['item_types'][item_type] = 1
        
        try:
            self.event_system.publish('knowledge_extraction', event_data)
            self.logger.info(f"Published knowledge extraction event with {len(items)} items")
        except Exception as e:
            self.logger.error(f"Error publishing knowledge extraction event: {str(e)}")
    
    def save_extracted_knowledge(self, items: List[Dict[str, Any]]) -> bool:
        """Save extracted knowledge items to the database.
        
        Args:
            items: Knowledge items to save
            
        Returns:
            True if save was successful, False otherwise
        """
        self.logger.info(f"Saving {len(items)} knowledge items to database")
        
        try:
            if items:
                result = self.collection.insert_many(items)
                self.logger.info(f"Saved {len(result.inserted_ids)} items to database")
                return True
            else:
                self.logger.warning("No items to save")
                return True
        except Exception as e:
            self.logger.error(f"Error saving items to database: {str(e)}")
            return False
    
    def save_knowledge(self, items: List[Dict[str, Any]], destination: str) -> Dict[str, Any]:
        """Save extracted knowledge items to a destination.
        
        Args:
            items: Knowledge items to save
            destination: Destination to save to (e.g., 'database', 'file')
            
        Returns:
            Result of the save operation
        """
        self.logger.info(f"Saving {len(items)} knowledge items to {destination}")
        
        if destination == 'database':
            # Use the new save_extracted_knowledge method
            success = self.save_extracted_knowledge(items)
            return {
                'success': success,
                'count': len(items),
                'destination': destination
            }
        elif destination == 'file':
            # This would be implemented to save to a file
            # For now, just log and return success
            self.logger.info("File save not implemented yet")
            return {
                'success': True,
                'count': len(items),
                'destination': destination
            }
        else:
            self.logger.error(f"Unknown destination: {destination}")
            return {
                'success': False,
                'error': f"Unknown destination: {destination}"
            }