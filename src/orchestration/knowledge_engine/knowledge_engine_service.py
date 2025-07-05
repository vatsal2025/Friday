"""Knowledge Engine Service Module.

This module contains the KnowledgeEngineService class which orchestrates
the knowledge extraction and management process.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple

from src.infrastructure.event.event_system import EventSystem
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.database.mongodb import MongoDB
from src.orchestration.knowledge_engine.book_knowledge_extractor import BookKnowledgeExtractor


class KnowledgeEngineService:
    """Orchestrates knowledge extraction and management.
    
    This service coordinates the extraction of trading knowledge from various sources,
    manages the knowledge base, and provides access to the extracted knowledge.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the KnowledgeEngineService.
        
        Args:
            config: Optional configuration dictionary. If None, loads from config manager.
        """
        self.config_manager = ConfigManager()
        self.config = config if config else self.config_manager.get_config('knowledge_engine')
        self.logger = logging.getLogger(__name__)
        
        # Initialize event system
        self.event_system = EventSystem()
        
        # Initialize database connection
        self.db = MongoDB()
        self.knowledge_collection = self.db.get_collection(self.config.get('storage', {}).get('collection_name', 'trading_knowledge'))
        
        # Initialize knowledge extractors
        self.book_extractor = BookKnowledgeExtractor(self.config.get('book_extraction', {}), self.event_system)
        
        # Register event handlers
        self._register_event_handlers()
        
        self.logger.info("KnowledgeEngineService initialized")
    
    def _register_event_handlers(self):
        """Register event handlers for knowledge-related events."""
        self.event_system.subscribe('knowledge_extraction', self._handle_knowledge_extraction)
        self.event_system.subscribe('knowledge_query', self._handle_knowledge_query)
        self.event_system.subscribe('knowledge_update', self._handle_knowledge_update)
        
        self.logger.info("Event handlers registered")
    
    def _handle_knowledge_extraction(self, event_data: Dict[str, Any]):
        """Handle knowledge extraction events.
        
        Args:
            event_data: Event data containing extracted knowledge
        """
        self.logger.info(f"Handling knowledge extraction event: {event_data.get('event_type')}")
        
        # Process the extracted knowledge (e.g., store in database)
        # This would be implemented based on specific requirements
        pass
    
    def _handle_knowledge_query(self, event_data: Dict[str, Any]):
        """Handle knowledge query events.
        
        Args:
            event_data: Event data containing query parameters
        """
        self.logger.info(f"Handling knowledge query event: {event_data.get('query_type')}")
        
        # Process the query and return results
        # This would be implemented based on specific requirements
        pass
    
    def _handle_knowledge_update(self, event_data: Dict[str, Any]):
        """Handle knowledge update events.
        
        Args:
            event_data: Event data containing update information
        """
        self.logger.info(f"Handling knowledge update event: {event_data.get('update_type')}")
        
        # Process the update
        # This would be implemented based on specific requirements
        pass
    
    def extract_knowledge_from_book(self, book_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract knowledge from a book file.
        
        Args:
            book_path: Path to the book file
            metadata: Optional metadata about the book
            
        Returns:
            Dictionary with extraction results
        """
        self.logger.info(f"Extracting knowledge from book: {book_path}")
        
        if not os.path.exists(book_path):
            self.logger.error(f"Book file not found: {book_path}")
            return {'success': False, 'error': 'File not found'}
        
        # Prepare metadata if not provided
        if not metadata:
            filename = os.path.basename(book_path)
            metadata = {
                'title': os.path.splitext(filename)[0],
                'file_path': book_path,
                'file_type': os.path.splitext(filename)[1][1:]
            }
        
        try:
            # Read the book content
            with open(book_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract knowledge from the content
            extracted_items = self.book_extractor.extract_knowledge(content, metadata)
            
            # Save the extracted knowledge
            save_result = self.save_knowledge_items(extracted_items)
            
            return {
                'success': save_result.get('success', False),
                'file_path': book_path,
                'extracted_count': len(extracted_items),
                'item_types': self._count_item_types(extracted_items)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting knowledge from book: {str(e)}")
            return {'success': False, 'error': str(e), 'file_path': book_path}
    
    def batch_process_books(self, directory_path: str) -> Dict[str, Any]:
        """Process all book files in a directory.
        
        Args:
            directory_path: Path to directory containing book files
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Batch processing books in directory: {directory_path}")
        
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            self.logger.error(f"Directory not found: {directory_path}")
            return {'success': False, 'error': 'Directory not found'}
        
        results = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'extracted_items': 0,
            'file_results': []
        }
        
        # Get supported file extensions from config
        supported_extensions = self.config.get('processing_parameters', {}).get('supported_file_types', ['txt', 'pdf'])
        
        # Process each file in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Skip directories and unsupported file types
            if os.path.isdir(file_path):
                continue
                
            file_ext = os.path.splitext(filename)[1][1:].lower()
            if file_ext not in supported_extensions:
                continue
            
            results['total_files'] += 1
            
            # Process the file
            file_result = self.extract_knowledge_from_book(file_path)
            results['file_results'].append(file_result)
            
            if file_result.get('success', False):
                results['processed_files'] += 1
                results['extracted_items'] += file_result.get('extracted_count', 0)
            else:
                results['failed_files'] += 1
        
        # Publish event for batch processing completion
        self.event_system.publish('book_batch_processed', {
            'directory': directory_path,
            'total_files': results['total_files'],
            'processed_files': results['processed_files'],
            'extracted_items': results['extracted_items']
        })
        
        return results
    
    def save_knowledge_items(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Save knowledge items to the database.
        
        Args:
            items: List of knowledge items to save
            
        Returns:
            Dictionary with save results
        """
        if not items:
            self.logger.warning("No items to save")
            return {'success': False, 'error': 'No items to save'}
        
        try:
            # Insert items into the database
            result = self.knowledge_collection.insert_many(items)
            
            self.logger.info(f"Saved {len(result.inserted_ids)} knowledge items to database")
            
            # Publish event for knowledge storage
            self.event_system.publish('knowledge_stored', {
                'item_count': len(result.inserted_ids),
                'item_types': self._count_item_types(items)
            })
            
            return {
                'success': True,
                'count': len(result.inserted_ids),
                'item_types': self._count_item_types(items)
            }
        except Exception as e:
            self.logger.error(f"Error saving knowledge items: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _count_item_types(self, items: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count items by type.
        
        Args:
            items: List of knowledge items
            
        Returns:
            Dictionary with counts by item type
        """
        type_counts = {}
        
        for item in items:
            item_type = item.get('type')
            if item_type in type_counts:
                type_counts[item_type] += 1
            else:
                type_counts[item_type] = 1
        
        return type_counts
    
    def query_knowledge(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query the knowledge base.
        
        Args:
            query_params: Query parameters
            
        Returns:
            List of matching knowledge items
        """
        self.logger.info(f"Querying knowledge base with params: {query_params}")
        
        try:
            # Convert query parameters to MongoDB query
            mongo_query = self._build_mongo_query(query_params)
            
            # Execute the query
            results = list(self.knowledge_collection.find(mongo_query))
            
            self.logger.info(f"Query returned {len(results)} results")
            
            return results
        except Exception as e:
            self.logger.error(f"Error querying knowledge base: {str(e)}")
            return []
    
    def _build_mongo_query(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Build a MongoDB query from query parameters.
        
        Args:
            query_params: Query parameters
            
        Returns:
            MongoDB query dictionary
        """
        mongo_query = {}
        
        # Process query parameters
        if 'type' in query_params:
            mongo_query['type'] = query_params['type']
        
        if 'confidence_min' in query_params:
            mongo_query['confidence'] = {'$gte': query_params['confidence_min']}
        
        if 'text_search' in query_params:
            # Text search across multiple fields
            search_fields = ['content.rule', 'content.pattern', 'content.strategy', 'content.description']
            search_conditions = []
            
            for field in search_fields:
                search_conditions.append({field: {'$regex': query_params['text_search'], '$options': 'i'}})
            
            mongo_query['$or'] = search_conditions
        
        if 'source' in query_params:
            mongo_query['source.title'] = {'$regex': query_params['source'], '$options': 'i'}
        
        return mongo_query
    
    def update_knowledge_item(self, item_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a knowledge item.
        
        Args:
            item_id: ID of the item to update
            updates: Dictionary of updates to apply
            
        Returns:
            Dictionary with update results
        """
        self.logger.info(f"Updating knowledge item: {item_id}")
        
        try:
            # Update the item in the database
            result = self.knowledge_collection.update_one(
                {'id': item_id},
                {'$set': updates}
            )
            
            if result.modified_count > 0:
                self.logger.info(f"Updated knowledge item: {item_id}")
                
                # Publish event for knowledge update
                self.event_system.publish('knowledge_updated', {
                    'item_id': item_id,
                    'updates': updates
                })
                
                return {'success': True, 'item_id': item_id}
            else:
                self.logger.warning(f"Knowledge item not found or not modified: {item_id}")
                return {'success': False, 'error': 'Item not found or not modified'}
                
        except Exception as e:
            self.logger.error(f"Error updating knowledge item: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def delete_knowledge_item(self, item_id: str) -> Dict[str, Any]:
        """Delete a knowledge item.
        
        Args:
            item_id: ID of the item to delete
            
        Returns:
            Dictionary with delete results
        """
        self.logger.info(f"Deleting knowledge item: {item_id}")
        
        try:
            # Delete the item from the database
            result = self.knowledge_collection.delete_one({'id': item_id})
            
            if result.deleted_count > 0:
                self.logger.info(f"Deleted knowledge item: {item_id}")
                
                # Publish event for knowledge deletion
                self.event_system.publish('knowledge_deleted', {
                    'item_id': item_id
                })
                
                return {'success': True, 'item_id': item_id}
            else:
                self.logger.warning(f"Knowledge item not found: {item_id}")
                return {'success': False, 'error': 'Item not found'}
                
        except Exception as e:
            self.logger.error(f"Error deleting knowledge item: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base.
        
        Returns:
            Dictionary with knowledge base statistics
        """
        self.logger.info("Getting knowledge base statistics")
        
        try:
            # Get total count
            total_count = self.knowledge_collection.count_documents({})
            
            # Get counts by type
            type_counts = {}
            for type_name in ['trading_rule', 'chart_pattern', 'trading_strategy']:
                type_counts[type_name] = self.knowledge_collection.count_documents({'type': type_name})
            
            # Get counts by confidence range
            confidence_ranges = {
                'high': self.knowledge_collection.count_documents({'confidence': {'$gte': 0.8}}),
                'medium': self.knowledge_collection.count_documents({'confidence': {'$gte': 0.6, '$lt': 0.8}}),
                'low': self.knowledge_collection.count_documents({'confidence': {'$lt': 0.6}})
            }
            
            # Get source statistics
            source_pipeline = [
                {'$group': {'_id': '$source.title', 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}},
                {'$limit': 10}
            ]
            top_sources = list(self.knowledge_collection.aggregate(source_pipeline))
            
            return {
                'total_count': total_count,
                'type_counts': type_counts,
                'confidence_ranges': confidence_ranges,
                'top_sources': top_sources
            }
            
        except Exception as e:
            self.logger.error(f"Error getting knowledge statistics: {str(e)}")
            return {
                'total_count': 0,
                'type_counts': {},
                'confidence_ranges': {},
                'top_sources': []
            }