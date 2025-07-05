import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Import necessary infrastructure components
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.database import mongodb
from src.infrastructure.event.event_system import EventSystem
from src.infrastructure.logging import get_logger

class KnowledgeBaseBuilder:
    """
    Builds and maintains a structured knowledge base from extracted trading knowledge.
    This class is responsible for organizing, indexing, and making searchable the
    knowledge extracted from various sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the KnowledgeBaseBuilder with configuration settings.
        
        Args:
            config: Optional configuration dictionary. If None, loads from unified_config.
        """
        self.config_manager = ConfigManager()
        self.config = config if config else self.config_manager.get_config('knowledge_extraction')
        
        # Initialize database connection
        self.knowledge_collection = mongodb.get_collection(self.config['storage']['collection_name'])
        self.index_collection = mongodb.get_collection(f"{self.config['storage']['collection_name']}_index")
        
        # Initialize event system
        self.event_system = EventSystem()
        
        # Configure logging
        self.logger = get_logger(__name__)
        
        # Initialize knowledge base structure
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """
        Initialize the knowledge base structure if it doesn't exist.
        Creates necessary indexes and metadata documents.
        """
        self.logger.info("Initializing knowledge base structure")
        
        try:
            # Create text indexes for searching
            self.knowledge_collection.create_index([('content', 'text')])
            self.knowledge_collection.create_index('category')
            self.knowledge_collection.create_index('subcategory')
            self.knowledge_collection.create_index('source.title')
            self.knowledge_collection.create_index('confidence_score')
            
            # Check if metadata document exists, create if not
            metadata_doc = self.index_collection.find_one({'document_type': 'metadata'})
            if not metadata_doc:
                metadata = {
                    'document_type': 'metadata',
                    'created_at': datetime.now(),
                    'last_updated': datetime.now(),
                    'version': '1.0',
                    'categories': self.config['categories'],
                    'stats': {
                        'total_items': 0,
                        'by_category': {}
                    }
                }
                self.index_collection.insert_one(metadata)
                self.logger.info("Created knowledge base metadata")
            
            self.logger.info("Knowledge base structure initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing knowledge base: {str(e)}")
            raise RuntimeError(f"Failed to initialize knowledge base: {str(e)}")
    
    def add_knowledge_item(self, item: Dict[str, Any]) -> bool:
        """
        Add a single knowledge item to the knowledge base.
        
        Args:
            item: Knowledge item to add
            
        Returns:
            True if successful, False otherwise
        """
        if not item:
            self.logger.warning("Attempted to add empty knowledge item")
            return False
        
        # Ensure required fields are present
        required_fields = ['content', 'category', 'source']
        for field in required_fields:
            if field not in item:
                self.logger.warning(f"Knowledge item missing required field: {field}")
                return False
        
        try:
            # Add metadata if not present
            if 'added_at' not in item:
                item['added_at'] = datetime.now()
            
            if 'last_updated' not in item:
                item['last_updated'] = datetime.now()
            
            if 'verified' not in item:
                item['verified'] = False
            
            # Insert the item
            result = self.knowledge_collection.insert_one(item)
            
            # Update metadata stats
            self._update_metadata_stats(item['category'])
            
            # Publish event
            self.event_system.publish('knowledge_item_added', {
                'item_id': str(result.inserted_id),
                'category': item['category'],
                'timestamp': datetime.now()
            })
            
            self.logger.info(f"Added knowledge item to category {item['category']}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error adding knowledge item: {str(e)}")
            return False
    
    def add_knowledge_items(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add multiple knowledge items to the knowledge base.
        
        Args:
            items: List of knowledge items to add
            
        Returns:
            Dictionary with results
        """
        if not items:
            self.logger.warning("No knowledge items to add")
            return {'success': False, 'error': 'No items provided', 'added': 0}
        
        results = {
            'total': len(items),
            'added': 0,
            'failed': 0,
            'categories': {}
        }
        
        for item in items:
            success = self.add_knowledge_item(item)
            if success:
                results['added'] += 1
                category = item.get('category', 'unknown')
                results['categories'][category] = results['categories'].get(category, 0) + 1
            else:
                results['failed'] += 1
        
        results['success'] = results['added'] > 0
        
        # Publish event for batch addition
        if results['added'] > 0:
            self.event_system.publish('knowledge_items_batch_added', {
                'added_count': results['added'],
                'categories': results['categories'],
                'timestamp': datetime.now()
            })
        
        return results
    
    def _update_metadata_stats(self, category: str):
        """
        Update metadata statistics when items are added.
        
        Args:
            category: Category of the added item
        """
        try:
            # Update the metadata document with new stats
            self.index_collection.update_one(
                {'document_type': 'metadata'},
                {
                    '$inc': {
                        'stats.total_items': 1,
                        f'stats.by_category.{category}': 1
                    },
                    '$set': {
                        'last_updated': datetime.now()
                    }
                }
            )
        except Exception as e:
            self.logger.error(f"Error updating metadata stats: {str(e)}")
    
    def search_knowledge_base(self, query: str, filters: Optional[Dict[str, Any]] = None, 
                             limit: int = 20, confidence_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for items matching the query and filters.
        
        Args:
            query: Text query to search for
            filters: Optional filters to apply (category, source, etc.)
            limit: Maximum number of results to return
            confidence_threshold: Minimum confidence score for results
            
        Returns:
            List of matching knowledge items
        """
        self.logger.info(f"Searching knowledge base for: {query}")
        
        # Build the search query
        search_query = {}
        
        # Add text search if query is provided
        if query and query.strip():
            search_query['$text'] = {'$search': query}
        
        # Add filters if provided
        if filters:
            for key, value in filters.items():
                search_query[key] = value
        
        # Add confidence threshold
        if confidence_threshold > 0:
            search_query['confidence_score'] = {'$gte': confidence_threshold}
        
        try:
            # Execute the search
            if query and query.strip():
                # With text search, use text score for sorting
                results = list(self.knowledge_collection.find(
                    search_query,
                    {'score': {'$meta': 'textScore'}}
                ).sort([('score', {'$meta': 'textScore'})]).limit(limit))
            else:
                # Without text search, sort by confidence score
                results = list(self.knowledge_collection.find(
                    search_query
                ).sort('confidence_score', -1).limit(limit))
            
            self.logger.info(f"Found {len(results)} results for query: {query}")
            
            # Publish search event
            self.event_system.publish('knowledge_base_searched', {
                'query': query,
                'filters': filters,
                'result_count': len(results),
                'timestamp': datetime.now()
            })
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {str(e)}")
            return []
    
    def get_knowledge_by_category(self, category: str, subcategory: Optional[str] = None, 
                                limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get knowledge items by category and optional subcategory.
        
        Args:
            category: Category to filter by
            subcategory: Optional subcategory to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of matching knowledge items
        """
        query = {'category': category}
        if subcategory:
            query['subcategory'] = subcategory
        
        try:
            results = list(self.knowledge_collection.find(query).limit(limit))
            return results
        except Exception as e:
            self.logger.error(f"Error getting knowledge by category: {str(e)}")
            return []
    
    def get_knowledge_by_source(self, source_title: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get knowledge items by source title.
        
        Args:
            source_title: Title of the source to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of matching knowledge items
        """
        query = {'source.title': source_title}
        
        try:
            results = list(self.knowledge_collection.find(query).limit(limit))
            return results
        except Exception as e:
            self.logger.error(f"Error getting knowledge by source: {str(e)}")
            return []
    
    def update_knowledge_item(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a knowledge item in the knowledge base.
        
        Args:
            item_id: ID of the item to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        if not item_id or not updates:
            return False
        
        try:
            # Ensure last_updated is set
            updates['last_updated'] = datetime.now()
            
            # Update the item
            result = self.knowledge_collection.update_one(
                {'_id': self.db.object_id(item_id)},
                {'$set': updates}
            )
            
            success = result.modified_count > 0
            
            if success:
                # Publish update event
                self.event_system.publish('knowledge_item_updated', {
                    'item_id': item_id,
                    'updated_fields': list(updates.keys()),
                    'timestamp': datetime.now()
                })
            
            return success
        
        except Exception as e:
            self.logger.error(f"Error updating knowledge item: {str(e)}")
            return False
    
    def delete_knowledge_item(self, item_id: str) -> bool:
        """
        Delete a knowledge item from the knowledge base.
        
        Args:
            item_id: ID of the item to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not item_id:
            return False
        
        try:
            # Get the item first to know its category for stats update
            item = self.knowledge_collection.find_one({'_id': self.db.object_id(item_id)})
            if not item:
                return False
            
            # Delete the item
            result = self.knowledge_collection.delete_one({'_id': self.db.object_id(item_id)})
            
            success = result.deleted_count > 0
            
            if success:
                # Update metadata stats
                self.index_collection.update_one(
                    {'document_type': 'metadata'},
                    {
                        '$inc': {
                            'stats.total_items': -1,
                            f'stats.by_category.{item["category"]}': -1
                        },
                        '$set': {
                            'last_updated': datetime.now()
                        }
                    }
                )
                
                # Publish delete event
                self.event_system.publish('knowledge_item_deleted', {
                    'item_id': item_id,
                    'category': item['category'],
                    'timestamp': datetime.now()
                })
            
            return success
        
        except Exception as e:
            self.logger.error(f"Error deleting knowledge item: {str(e)}")
            return False
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with knowledge base statistics
        """
        try:
            # Get metadata document
            metadata = self.index_collection.find_one({'document_type': 'metadata'})
            if not metadata:
                return {'error': 'Metadata not found'}
            
            # Get additional stats that might not be in the metadata
            category_counts = {}
            for category in self.config['categories']:
                count = self.knowledge_collection.count_documents({'category': category})
                category_counts[category] = count
            
            source_counts = {}
            pipeline = [
                {'$group': {'_id': '$source.title', 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}},
                {'$limit': 10}  # Top 10 sources
            ]
            for doc in self.knowledge_collection.aggregate(pipeline):
                if doc['_id']:
                    source_counts[doc['_id']] = doc['count']
            
            # Combine all stats
            stats = {
                'total_items': metadata['stats'].get('total_items', 0),
                'categories': category_counts,
                'top_sources': source_counts,
                'last_updated': metadata.get('last_updated', datetime.now())
            }
            
            return stats
        
        except Exception as e:
            self.logger.error(f"Error getting knowledge base stats: {str(e)}")
            return {'error': str(e)}
    
    def export_knowledge_base(self, file_path: str, filters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export the knowledge base or a filtered subset to a JSON file.
        
        Args:
            file_path: Path to save the exported file
            filters: Optional filters to apply before exporting
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Apply filters if provided
            query = filters if filters else {}
            
            # Get the data to export
            data = list(self.knowledge_collection.find(query))
            
            # Convert ObjectId to string for JSON serialization
            for item in data:
                item['_id'] = str(item['_id'])
                if 'added_at' in item and isinstance(item['added_at'], datetime):
                    item['added_at'] = item['added_at'].isoformat()
                if 'last_updated' in item and isinstance(item['last_updated'], datetime):
                    item['last_updated'] = item['last_updated'].isoformat()
            
            # Create export metadata
            export_data = {
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'item_count': len(data),
                    'filters': filters
                },
                'items': data
            }
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Exported {len(data)} knowledge items to {file_path}")
            
            # Publish export event
            self.event_system.publish('knowledge_base_exported', {
                'file_path': file_path,
                'item_count': len(data),
                'timestamp': datetime.now()
            })
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error exporting knowledge base: {str(e)}")
            return False
    
    def import_knowledge_base(self, file_path: str, overwrite_existing: bool = False) -> Dict[str, Any]:
        """
        Import knowledge items from a JSON file.
        
        Args:
            file_path: Path to the JSON file to import
            overwrite_existing: Whether to overwrite existing items with the same content
            
        Returns:
            Dictionary with import results
        """
        if not os.path.exists(file_path):
            self.logger.error(f"Import file not found: {file_path}")
            return {'success': False, 'error': 'File not found'}
        
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Validate the import data structure
            if 'items' not in import_data or not isinstance(import_data['items'], list):
                return {'success': False, 'error': 'Invalid import file format'}
            
            items = import_data['items']
            
            # Process each item
            results = {
                'total': len(items),
                'imported': 0,
                'skipped': 0,
                'failed': 0
            }
            
            for item in items:
                # Remove _id field if present to avoid conflicts
                if '_id' in item:
                    del item['_id']
                
                # Check if item already exists
                existing = None
                if 'content' in item:
                    existing = self.knowledge_collection.find_one({'content': item['content']})
                
                if existing and not overwrite_existing:
                    results['skipped'] += 1
                    continue
                
                # Add or update the item
                if existing and overwrite_existing:
                    success = self.update_knowledge_item(str(existing['_id']), item)
                else:
                    success = self.add_knowledge_item(item)
                
                if success:
                    results['imported'] += 1
                else:
                    results['failed'] += 1
            
            results['success'] = results['imported'] > 0
            
            # Publish import event
            self.event_system.publish('knowledge_base_imported', {
                'file_path': file_path,
                'imported_count': results['imported'],
                'timestamp': datetime.now()
            })
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error importing knowledge base: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def verify_knowledge_item(self, item_id: str, verified: bool = True, 
                            verification_notes: Optional[str] = None) -> bool:
        """
        Mark a knowledge item as verified or unverified.
        
        Args:
            item_id: ID of the item to verify
            verified: Whether the item is verified
            verification_notes: Optional notes about the verification
            
        Returns:
            True if successful, False otherwise
        """
        updates = {
            'verified': verified,
            'verification_timestamp': datetime.now()
        }
        
        if verification_notes:
            updates['verification_notes'] = verification_notes
        
        return self.update_knowledge_item(item_id, updates)
    
    def build_knowledge_graph(self) -> Dict[str, Any]:
        """
        Build a knowledge graph from the items in the knowledge base.
        This creates relationships between related knowledge items.
        
        Returns:
            Dictionary with graph building results
        """
        self.logger.info("Building knowledge graph from knowledge base items")
        
        try:
            # Create a graph collection if it doesn't exist
            graph_collection = self.db.get_collection(f"{self.config['storage']['collection_name']}_graph")
            
            # Clear existing graph data
            graph_collection.delete_many({})  
            
            # Get all knowledge items
            items = list(self.knowledge_collection.find({}))
            
            # Create nodes for each item
            nodes = []
            for item in items:
                node = {
                    'node_id': str(item['_id']),
                    'node_type': 'knowledge_item',
                    'category': item['category'],
                    'subcategory': item.get('subcategory', ''),
                    'content': item['content'],
                    'source': item.get('source', {}).get('title', 'Unknown'),
                    'confidence': item.get('confidence_score', 0.0)
                }
                nodes.append(node)
            
            # Insert nodes
            if nodes:
                graph_collection.insert_many(nodes)
            
            # Create edges between related nodes
            edges = []
            for i, item1 in enumerate(items):
                for j, item2 in enumerate(items):
                    if i != j:  # Don't connect node to itself
                        # Calculate similarity or relationship strength
                        # In a production system, this would use more sophisticated methods
                        relationship = self._calculate_relationship(item1, item2)
                        
                        if relationship['strength'] > 0.5:  # Only create edges for strong relationships
                            edge = {
                                'edge_type': 'related',
                                'source_id': str(item1['_id']),
                                'target_id': str(item2['_id']),
                                'strength': relationship['strength'],
                                'relationship_type': relationship['type']
                            }
                            edges.append(edge)
            
            # Insert edges in batches to avoid overwhelming the database
            batch_size = 1000
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i+batch_size]
                if batch:
                    graph_collection.insert_many(batch)
            
            # Create indexes for the graph
            graph_collection.create_index('node_id')
            graph_collection.create_index('node_type')
            graph_collection.create_index(['source_id', 'target_id'])
            
            results = {
                'success': True,
                'node_count': len(nodes),
                'edge_count': len(edges),
                'timestamp': datetime.now()
            }
            
            # Publish graph building event
            self.event_system.publish('knowledge_graph_built', results)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error building knowledge graph: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_relationship(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the relationship between two knowledge items.
        
        Args:
            item1: First knowledge item
            item2: Second knowledge item
            
        Returns:
            Dictionary with relationship information
        """
        # In a production system, this would use NLP techniques like semantic similarity
        # For this implementation, we'll use a simplified approach
        
        relationship = {
            'strength': 0.0,
            'type': 'unknown'
        }
        
        # Check if items are in the same category
        if item1.get('category') == item2.get('category'):
            relationship['strength'] += 0.3
            relationship['type'] = 'same_category'
        
        # Check if items are from the same source
        if item1.get('source', {}).get('title') == item2.get('source', {}).get('title'):
            relationship['strength'] += 0.2
            relationship['type'] = 'same_source'
        
        # Check for shared entities
        entities1 = item1.get('extracted_entities', {})
        entities2 = item2.get('extracted_entities', {})
        
        shared_entities = 0
        for entity_type in entities1:
            if entity_type in entities2:
                for entity in entities1[entity_type]:
                    if entity in entities2[entity_type]:
                        shared_entities += 1
        
        if shared_entities > 0:
            relationship['strength'] += min(0.5, shared_entities * 0.1)  # Cap at 0.5
            relationship['type'] = 'shared_entities'
        
        # Simple text similarity (would be more sophisticated in production)
        if 'content' in item1 and 'content' in item2:
            words1 = set(item1['content'].lower().split())
            words2 = set(item2['content'].lower().split())
            
            if words1 and words2:  # Avoid division by zero
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                
                jaccard_similarity = len(intersection) / len(union)
                relationship['strength'] += jaccard_similarity * 0.3
                
                if jaccard_similarity > 0.3:
                    relationship['type'] = 'content_similarity'
        
        return relationship