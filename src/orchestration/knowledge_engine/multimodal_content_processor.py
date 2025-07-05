import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import base64

# Import necessary infrastructure components
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.database import mongodb
from src.infrastructure.event.event_system import EventSystem
from src.infrastructure.logging import get_logger

class MultimodalContentProcessor:
    """
    Processes different types of content (text, tables, charts, mathematical formulas)
    from books and other sources for knowledge extraction.
    
    This component is responsible for handling multi-modal data extraction including:
    - Text content processing
    - Table extraction and structuring
    - Chart and graph analysis
    - Mathematical formula extraction and interpretation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MultimodalContentProcessor with configuration settings.
        
        Args:
            config: Optional configuration dictionary. If None, loads from unified_config.
        """
        self.config_manager = ConfigManager()
        self.config = config if config else self.config_manager.get_config('knowledge_extraction')
        
        # Initialize database connection
        self.content_collection = mongodb.get_collection('extracted_content')
        
        # Initialize event system
        self.event_system = EventSystem()
        
        # Configure logging
        self.logger = get_logger(__name__)
        
        # Load processor configurations
        self._load_processor_config()
    
    def _load_processor_config(self):
        """
        Load processor-specific configurations from the main config.
        """
        # Text processing config
        self.text_config = self.config.get('text_processing', {})
        self.text_min_confidence = self.text_config.get('min_confidence', 0.7)
        
        # Table processing config
        self.table_config = self.config.get('table_processing', {})
        self.table_min_confidence = self.table_config.get('min_confidence', 0.75)
        
        # Chart processing config
        self.chart_config = self.config.get('chart_processing', {})
        self.chart_min_confidence = self.chart_config.get('min_confidence', 0.8)
        
        # Math formula processing config
        self.math_config = self.config.get('math_processing', {})
        self.math_min_confidence = self.math_config.get('min_confidence', 0.85)
        
        # Image processing config
        self.image_config = self.config.get('image_processing', {})
        self.image_min_confidence = self.image_config.get('min_confidence', 0.75)
        
        # Load NLP model settings
        self.nlp_model_config = self.config.get('nlp_model', {})
        self.nlp_model_name = self.nlp_model_config.get('model_name', 'en_core_web_lg')
        
        self.logger.info("Multimodal content processor configuration loaded")
    
    def process_content(self, content_path: str, content_type: str = None, 
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process content from a file based on its type.
        
        Args:
            content_path: Path to the content file
            content_type: Type of content (text, table, chart, math, image)
                          If None, will attempt to detect automatically
            metadata: Optional metadata about the content
            
        Returns:
            Dictionary with processing results
        """
        if not os.path.exists(content_path):
            self.logger.error(f"Content file not found: {content_path}")
            return {'success': False, 'error': 'File not found'}
        
        # Determine content type if not provided
        if content_type is None:
            content_type = self._detect_content_type(content_path)
            self.logger.info(f"Detected content type: {content_type} for {content_path}")
        
        # Process based on content type
        try:
            if content_type == 'text':
                result = self._process_text(content_path, metadata)
            elif content_type == 'table':
                result = self._process_table(content_path, metadata)
            elif content_type == 'chart':
                result = self._process_chart(content_path, metadata)
            elif content_type == 'math':
                result = self._process_math_formula(content_path, metadata)
            elif content_type == 'image':
                result = self._process_image(content_path, metadata)
            else:
                self.logger.warning(f"Unsupported content type: {content_type}")
                return {'success': False, 'error': f"Unsupported content type: {content_type}"}
            
            # Add processing metadata
            result['metadata'] = metadata or {}
            result['metadata'].update({
                'processed_at': datetime.now(),
                'content_type': content_type,
                'source_path': content_path,
                'processor_version': '1.0'
            })
            
            # Save to database if successful
            if result.get('success', False):
                self._save_processed_content(result)
                
                # Publish event
                self.event_system.publish('content_processed', {
                    'content_id': result.get('content_id'),
                    'content_type': content_type,
                    'timestamp': datetime.now()
                })
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _detect_content_type(self, file_path: str) -> str:
        """
        Detect the type of content in the file.
        
        Args:
            file_path: Path to the content file
            
        Returns:
            Detected content type (text, table, chart, math, image)
        """
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Map extensions to content types
        if ext in ['.txt', '.md', '.rst', '.doc', '.docx', '.pdf']:
            return 'text'
        elif ext in ['.csv', '.xlsx', '.xls', '.tsv']:
            return 'table'
        elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            # For images, we need to determine if it's a chart or a regular image
            # This would require more sophisticated image analysis in production
            # For now, we'll default to 'image'
            return 'image'
        elif ext in ['.tex', '.latex']:
            return 'math'
        else:
            # Default to text for unknown types
            return 'text'
    
    def _process_text(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text content for knowledge extraction.
        
        Args:
            file_path: Path to the text file
            metadata: Optional metadata about the content
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing text content: {file_path}")
        
        try:
            # Read the text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # In a production system, this would use NLP models for more sophisticated processing
            # For this implementation, we'll do basic processing
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]
            
            # Extract key sentences (simplified implementation)
            key_sentences = self._extract_key_sentences(text_content)
            
            # Extract entities (simplified implementation)
            entities = self._extract_entities(text_content)
            
            # Create result
            result = {
                'success': True,
                'content_type': 'text',
                'original_content': text_content[:1000] + '...' if len(text_content) > 1000 else text_content,
                'paragraphs_count': len(paragraphs),
                'key_sentences': key_sentences,
                'entities': entities,
                'confidence_score': 0.9,  # Would be calculated based on model confidence in production
                'word_count': len(text_content.split()),
                'processed_content': {
                    'paragraphs': paragraphs[:10],  # Limit for demonstration
                    'summary': self._generate_summary(text_content)
                }
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _process_table(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process table content for knowledge extraction.
        
        Args:
            file_path: Path to the table file
            metadata: Optional metadata about the content
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing table content: {file_path}")
        
        try:
            # In a production system, this would use pandas or similar libraries
            # to read and process tables from various formats
            
            # For this implementation, we'll create a simplified result
            result = {
                'success': True,
                'content_type': 'table',
                'table_format': os.path.splitext(file_path)[1],
                'confidence_score': 0.85,
                'processed_content': {
                    'headers': ['Column 1', 'Column 2', 'Column 3'],  # Example
                    'rows_count': 10,  # Example
                    'columns_count': 3,  # Example
                    'sample_data': [  # Example
                        ['Value 1', 'Value 2', 'Value 3'],
                        ['Value 4', 'Value 5', 'Value 6']
                    ],
                    'numerical_columns': [1, 2],  # Example
                    'categorical_columns': [0]  # Example
                }
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing table: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _process_chart(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process chart/graph content for knowledge extraction.
        
        Args:
            file_path: Path to the chart image file
            metadata: Optional metadata about the content
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing chart content: {file_path}")
        
        try:
            # In a production system, this would use computer vision and chart parsing libraries
            # to extract data from chart images
            
            # For this implementation, we'll create a simplified result
            result = {
                'success': True,
                'content_type': 'chart',
                'chart_type': 'line_chart',  # Example (would be detected in production)
                'confidence_score': 0.8,
                'processed_content': {
                    'title': 'Example Chart',  # Example
                    'x_axis_label': 'Time',  # Example
                    'y_axis_label': 'Value',  # Example
                    'data_series': [  # Example
                        {
                            'name': 'Series 1',
                            'values': [1, 2, 3, 4, 5]
                        },
                        {
                            'name': 'Series 2',
                            'values': [5, 4, 3, 2, 1]
                        }
                    ],
                    'insights': [
                        'Upward trend in Series 1',
                        'Downward trend in Series 2',
                        'Intersection point at x=3'
                    ]
                }
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing chart: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _process_math_formula(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process mathematical formula content for knowledge extraction.
        
        Args:
            file_path: Path to the formula file
            metadata: Optional metadata about the content
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing math formula content: {file_path}")
        
        try:
            # In a production system, this would use LaTeX parsers or OCR with math formula recognition
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                formula_content = f.read()
            
            # For this implementation, we'll create a simplified result
            result = {
                'success': True,
                'content_type': 'math',
                'confidence_score': 0.85,
                'processed_content': {
                    'original_formula': formula_content,
                    'latex_representation': formula_content,  # In production, this might be converted
                    'variables': ['x', 'y', 'z'],  # Example
                    'formula_type': 'equation',  # Example
                    'interpretation': 'This formula represents a relationship between variables x, y, and z',  # Example
                    'complexity_score': 0.7  # Example
                }
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing math formula: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _process_image(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process image content for knowledge extraction.
        
        Args:
            file_path: Path to the image file
            metadata: Optional metadata about the content
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing image content: {file_path}")
        
        try:
            # In a production system, this would use computer vision and image analysis libraries
            
            # For this implementation, we'll create a simplified result
            result = {
                'success': True,
                'content_type': 'image',
                'confidence_score': 0.75,
                'processed_content': {
                    'image_type': 'photograph',  # Example
                    'dimensions': '800x600',  # Example
                    'contains_text': True,  # Example
                    'detected_objects': ['person', 'chart', 'document'],  # Example
                    'extracted_text': 'Sample text from image',  # Example
                    'image_quality': 'high',  # Example
                    'color_profile': 'RGB'  # Example
                }
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _extract_key_sentences(self, text: str) -> List[str]:
        """
        Extract key sentences from text.
        
        Args:
            text: Text to extract key sentences from
            
        Returns:
            List of key sentences
        """
        # In a production system, this would use NLP models for sentence importance scoring
        # For this implementation, we'll use a simplified approach
        
        # Split into sentences (simplified)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Filter for sentences that might contain key information
        key_indicators = [
            'important', 'significant', 'key', 'critical', 'essential',
            'strategy', 'rule', 'principle', 'method', 'technique',
            'trading', 'market', 'stock', 'investment', 'risk',
            'profit', 'loss', 'return', 'analysis', 'indicator'
        ]
        
        key_sentences = []
        for sentence in sentences:
            # Check if sentence contains any key indicators
            if any(indicator in sentence.lower() for indicator in key_indicators):
                key_sentences.append(sentence + '.')
            
            # Limit to 10 key sentences
            if len(key_sentences) >= 10:
                break
        
        return key_sentences
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary with entity types and values
        """
        # In a production system, this would use NLP models for entity recognition
        # For this implementation, we'll use a simplified approach
        
        entities = {
            'trading_strategies': [],
            'market_indicators': [],
            'financial_instruments': [],
            'time_periods': [],
            'risk_factors': []
        }
        
        # Simple pattern matching (would be much more sophisticated in production)
        strategy_patterns = ['trend following', 'mean reversion', 'momentum', 'breakout']
        for pattern in strategy_patterns:
            if pattern in text.lower():
                entities['trading_strategies'].append(pattern)
        
        indicator_patterns = ['moving average', 'rsi', 'macd', 'bollinger bands']
        for pattern in indicator_patterns:
            if pattern in text.lower():
                entities['market_indicators'].append(pattern)
        
        instrument_patterns = ['stock', 'bond', 'option', 'future', 'forex']
        for pattern in instrument_patterns:
            if pattern in text.lower():
                entities['financial_instruments'].append(pattern)
        
        time_patterns = ['daily', 'weekly', 'monthly', 'yearly', 'intraday']
        for pattern in time_patterns:
            if pattern in text.lower():
                entities['time_periods'].append(pattern)
        
        risk_patterns = ['volatility', 'drawdown', 'loss', 'risk-reward', 'stop-loss']
        for pattern in risk_patterns:
            if pattern in text.lower():
                entities['risk_factors'].append(pattern)
        
        return entities
    
    def _generate_summary(self, text: str) -> str:
        """
        Generate a summary of the text.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary text
        """
        # In a production system, this would use NLP models for text summarization
        # For this implementation, we'll use a simplified approach
        
        # Take the first 200 characters as a simple summary
        if len(text) <= 200:
            return text
        
        # Find the end of the first paragraph
        first_para_end = text.find('\n\n')
        if first_para_end > 0 and first_para_end < 500:
            return text[:first_para_end].strip()
        
        # Otherwise return the first 200 characters with ellipsis
        return text[:200].strip() + '...'
    
    def _save_processed_content(self, content_data: Dict[str, Any]) -> str:
        """
        Save processed content to the database.
        
        Args:
            content_data: Processed content data
            
        Returns:
            ID of the saved content
        """
        # Remove large original content to save space
        if 'original_content' in content_data and len(content_data['original_content']) > 1000:
            content_data['original_content'] = content_data['original_content'][:1000] + '...'
        
        # Insert into database
        result = self.content_collection.insert_one(content_data)
        content_id = str(result.inserted_id)
        
        # Update with ID
        self.content_collection.update_one(
            {'_id': result.inserted_id},
            {'$set': {'content_id': content_id}}
        )
        
        content_data['content_id'] = content_id
        
        self.logger.info(f"Saved processed content with ID: {content_id}")
        
        return content_id
    
    def process_batch(self, content_paths: List[str], 
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a batch of content files.
        
        Args:
            content_paths: List of paths to content files
            metadata: Optional metadata to apply to all files
            
        Returns:
            Dictionary with batch processing results
        """
        results = {
            'total': len(content_paths),
            'successful': 0,
            'failed': 0,
            'content_ids': [],
            'errors': []
        }
        
        for path in content_paths:
            result = self.process_content(path, metadata=metadata)
            
            if result.get('success', False):
                results['successful'] += 1
                results['content_ids'].append(result.get('content_id'))
            else:
                results['failed'] += 1
                results['errors'].append({
                    'path': path,
                    'error': result.get('error', 'Unknown error')
                })
        
        # Publish batch event
        self.event_system.publish('content_batch_processed', {
            'total': results['total'],
            'successful': results['successful'],
            'timestamp': datetime.now()
        })
        
        return results
    
    def get_processed_content(self, content_id: str) -> Dict[str, Any]:
        """
        Get processed content by ID.
        
        Args:
            content_id: ID of the content to get
            
        Returns:
            Dictionary with the content or error
        """
        try:
            content = self.content_collection.find_one({'content_id': content_id})
            if not content:
                return {'success': False, 'error': 'Content not found'}
            
            # Convert ObjectId to string for JSON serialization
            content['_id'] = str(content['_id'])
            
            return {'success': True, 'content': content}
        
        except Exception as e:
            self.logger.error(f"Error getting content: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def delete_processed_content(self, content_id: str) -> Dict[str, Any]:
        """
        Delete processed content by ID.
        
        Args:
            content_id: ID of the content to delete
            
        Returns:
            Dictionary with deletion results
        """
        try:
            result = self.content_collection.delete_one({'content_id': content_id})
            success = result.deleted_count > 0
            
            if success:
                self.event_system.publish('content_deleted', {
                    'content_id': content_id,
                    'timestamp': datetime.now()
                })
            
            return {'success': success}
        
        except Exception as e:
            self.logger.error(f"Error deleting content: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def extract_data_from_image(self, image_path: str, extraction_type: str = 'text') -> Dict[str, Any]:
        """
        Extract specific data from an image.
        
        Args:
            image_path: Path to the image file
            extraction_type: Type of data to extract (text, table, chart)
            
        Returns:
            Dictionary with extraction results
        """
        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            return {'success': False, 'error': 'File not found'}
        
        try:
            # In a production system, this would use OCR and image analysis libraries
            # For this implementation, we'll create a simplified result
            
            if extraction_type == 'text':
                result = {
                    'success': True,
                    'extraction_type': 'text',
                    'confidence_score': 0.8,
                    'extracted_data': {
                        'text': 'Sample text extracted from image',
                        'language': 'en',
                        'word_count': 5
                    }
                }
            elif extraction_type == 'table':
                result = {
                    'success': True,
                    'extraction_type': 'table',
                    'confidence_score': 0.75,
                    'extracted_data': {
                        'headers': ['Column 1', 'Column 2'],
                        'rows': [
                            ['Value 1', 'Value 2'],
                            ['Value 3', 'Value 4']
                        ],
                        'rows_count': 2,
                        'columns_count': 2
                    }
                }
            elif extraction_type == 'chart':
                result = {
                    'success': True,
                    'extraction_type': 'chart',
                    'confidence_score': 0.7,
                    'extracted_data': {
                        'chart_type': 'bar_chart',
                        'x_axis_values': ['A', 'B', 'C'],
                        'y_axis_values': [10, 20, 30],
                        'title': 'Sample Chart'
                    }
                }
            else:
                return {'success': False, 'error': f"Unsupported extraction type: {extraction_type}"}
            
            # Add metadata
            result['metadata'] = {
                'source_path': image_path,
                'extracted_at': datetime.now().isoformat(),
                'extraction_type': extraction_type
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error extracting data from image: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def extract_tables_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract tables from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with extraction results
        """
        if not os.path.exists(pdf_path):
            self.logger.error(f"PDF file not found: {pdf_path}")
            return {'success': False, 'error': 'File not found'}
        
        try:
            # In a production system, this would use PDF parsing libraries like PyPDF2, pdfplumber, or tabula-py
            # For this implementation, we'll create a simplified result
            
            result = {
                'success': True,
                'tables_count': 2,  # Example
                'tables': [
                    {
                        'page': 1,
                        'headers': ['Column 1', 'Column 2', 'Column 3'],
                        'rows': [
                            ['Value 1', 'Value 2', 'Value 3'],
                            ['Value 4', 'Value 5', 'Value 6']
                        ],
                        'confidence_score': 0.85
                    },
                    {
                        'page': 2,
                        'headers': ['Name', 'Value'],
                        'rows': [
                            ['Item 1', '100'],
                            ['Item 2', '200']
                        ],
                        'confidence_score': 0.9
                    }
                ],
                'metadata': {
                    'source_path': pdf_path,
                    'extracted_at': datetime.now().isoformat(),
                    'pdf_pages': 2  # Example
                }
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error extracting tables from PDF: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def extract_math_formulas(self, content_path: str) -> Dict[str, Any]:
        """
        Extract mathematical formulas from a document.
        
        Args:
            content_path: Path to the document file
            
        Returns:
            Dictionary with extraction results
        """
        if not os.path.exists(content_path):
            self.logger.error(f"Document file not found: {content_path}")
            return {'success': False, 'error': 'File not found'}
        
        try:
            # In a production system, this would use specialized libraries for math formula extraction
            # For this implementation, we'll create a simplified result
            
            result = {
                'success': True,
                'formulas_count': 2,  # Example
                'formulas': [
                    {
                        'latex': 'E = mc^2',
                        'variables': ['E', 'm', 'c'],
                        'confidence_score': 0.95,
                        'location': 'page 1, paragraph 2'
                    },
                    {
                        'latex': '\\frac{dy}{dx} = 2x',
                        'variables': ['y', 'x'],
                        'confidence_score': 0.9,
                        'location': 'page 3, paragraph 1'
                    }
                ],
                'metadata': {
                    'source_path': content_path,
                    'extracted_at': datetime.now().isoformat(),
                    'document_type': os.path.splitext(content_path)[1]
                }
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error extracting math formulas: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def merge_multimodal_content(self, content_ids: List[str], 
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Merge multiple processed content items into a unified representation.
        
        Args:
            content_ids: List of content IDs to merge
            metadata: Optional metadata for the merged content
            
        Returns:
            Dictionary with merged content
        """
        if not content_ids:
            return {'success': False, 'error': 'No content IDs provided'}
        
        try:
            merged_content = {
                'success': True,
                'source_content_ids': content_ids,
                'content_types': [],
                'merged_at': datetime.now().isoformat(),
                'metadata': metadata or {},
                'text_content': [],
                'tables': [],
                'charts': [],
                'formulas': [],
                'images': []
            }
            
            # Retrieve and merge each content item
            for content_id in content_ids:
                result = self.get_processed_content(content_id)
                if not result.get('success', False):
                    continue
                
                content = result['content']
                content_type = content.get('content_type')
                
                if content_type not in merged_content['content_types']:
                    merged_content['content_types'].append(content_type)
                
                # Merge based on content type
                if content_type == 'text':
                    if 'processed_content' in content and 'paragraphs' in content['processed_content']:
                        merged_content['text_content'].extend(content['processed_content']['paragraphs'])
                elif content_type == 'table':
                    if 'processed_content' in content:
                        merged_content['tables'].append(content['processed_content'])
                elif content_type == 'chart':
                    if 'processed_content' in content:
                        merged_content['charts'].append(content['processed_content'])
                elif content_type == 'math':
                    if 'processed_content' in content:
                        merged_content['formulas'].append(content['processed_content'])
                elif content_type == 'image':
                    if 'processed_content' in content:
                        merged_content['images'].append(content['processed_content'])
            
            # Save the merged content
            merged_id = self._save_processed_content({
                'content_type': 'merged',
                'merged_content': merged_content
            })
            
            merged_content['merged_id'] = merged_id
            
            # Publish merge event
            self.event_system.publish('content_merged', {
                'merged_id': merged_id,
                'source_content_ids': content_ids,
                'timestamp': datetime.now()
            })
            
            return merged_content
        
        except Exception as e:
            self.logger.error(f"Error merging content: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def analyze_content_relationships(self, content_ids: List[str]) -> Dict[str, Any]:
        """
        Analyze relationships between different content items.
        
        Args:
            content_ids: List of content IDs to analyze
            
        Returns:
            Dictionary with relationship analysis
        """
        if not content_ids or len(content_ids) < 2:
            return {'success': False, 'error': 'Need at least two content IDs for relationship analysis'}
        
        try:
            # Get all content items
            contents = []
            for content_id in content_ids:
                result = self.get_processed_content(content_id)
                if result.get('success', False):
                    contents.append(result['content'])
            
            # In a production system, this would use more sophisticated analysis
            # For this implementation, we'll create a simplified result
            
            relationships = {
                'success': True,
                'content_count': len(contents),
                'relationships': []
            }
            
            # Generate simple relationships (would be more sophisticated in production)
            for i in range(len(contents)):
                for j in range(i+1, len(contents)):
                    content1 = contents[i]
                    content2 = contents[j]
                    
                    relationship = {
                        'source_id': content1.get('content_id'),
                        'target_id': content2.get('content_id'),
                        'source_type': content1.get('content_type'),
                        'target_type': content2.get('content_type'),
                        'relationship_type': 'related',  # Would be more specific in production
                        'confidence_score': 0.7,  # Would be calculated in production
                        'description': f"Relationship between {content1.get('content_type')} and {content2.get('content_type')}"
                    }
                    
                    relationships['relationships'].append(relationship)
            
            return relationships
        
        except Exception as e:
            self.logger.error(f"Error analyzing content relationships: {str(e)}")
            return {'success': False, 'error': str(e)}