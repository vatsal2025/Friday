"""Knowledge base storage using MongoDB for the Friday AI Trading System.

This module provides specialized functions for storing and retrieving knowledge base data.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.collection import Collection
from pymongo.database import Database

from src.infrastructure.config import get_config
from src.infrastructure.database.mongodb import get_database, get_collection
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)

# Collection names
KNOWLEDGE_COLLECTION = "knowledge"
RULES_COLLECTION = "rules"
STRATEGIES_COLLECTION = "strategies"
CONCEPTS_COLLECTION = "concepts"
RELATIONSHIPS_COLLECTION = "relationships"


def initialize_knowledge_base() -> None:
    """Initialize the knowledge base collections and indexes.

    This function creates the necessary collections and indexes for the knowledge base.
    """
    db = get_database()
    
    # Create collections if they don't exist
    if KNOWLEDGE_COLLECTION not in db.list_collection_names():
        db.create_collection(KNOWLEDGE_COLLECTION)
        logger.info("Created knowledge collection")
    
    if RULES_COLLECTION not in db.list_collection_names():
        db.create_collection(RULES_COLLECTION)
        logger.info("Created rules collection")
    
    if STRATEGIES_COLLECTION not in db.list_collection_names():
        db.create_collection(STRATEGIES_COLLECTION)
        logger.info("Created strategies collection")
    
    if CONCEPTS_COLLECTION not in db.list_collection_names():
        db.create_collection(CONCEPTS_COLLECTION)
        logger.info("Created concepts collection")
    
    if RELATIONSHIPS_COLLECTION not in db.list_collection_names():
        db.create_collection(RELATIONSHIPS_COLLECTION)
        logger.info("Created relationships collection")
    
    # Create indexes
    knowledge_collection = get_collection(KNOWLEDGE_COLLECTION)
    knowledge_collection.create_index([("title", ASCENDING)], unique=True)
    knowledge_collection.create_index([("tags", ASCENDING)])
    knowledge_collection.create_index([("source", ASCENDING)])
    knowledge_collection.create_index([("created_at", DESCENDING)])
    logger.info("Created indexes for knowledge collection")
    
    rules_collection = get_collection(RULES_COLLECTION)
    rules_collection.create_index([("name", ASCENDING)], unique=True)
    rules_collection.create_index([("category", ASCENDING)])
    rules_collection.create_index([("tags", ASCENDING)])
    logger.info("Created indexes for rules collection")
    
    strategies_collection = get_collection(STRATEGIES_COLLECTION)
    strategies_collection.create_index([("name", ASCENDING)], unique=True)
    strategies_collection.create_index([("category", ASCENDING)])
    strategies_collection.create_index([("tags", ASCENDING)])
    strategies_collection.create_index([("performance.sharpe_ratio", DESCENDING)])
    logger.info("Created indexes for strategies collection")
    
    concepts_collection = get_collection(CONCEPTS_COLLECTION)
    concepts_collection.create_index([("name", ASCENDING)], unique=True)
    concepts_collection.create_index([("category", ASCENDING)])
    logger.info("Created indexes for concepts collection")
    
    relationships_collection = get_collection(RELATIONSHIPS_COLLECTION)
    relationships_collection.create_index([("source", ASCENDING), ("target", ASCENDING), ("type", ASCENDING)], unique=True)
    logger.info("Created indexes for relationships collection")


def store_knowledge_item(title: str, content: str, source: str, tags: List[str] = None, 
                        metadata: Dict[str, Any] = None) -> str:
    """Store a knowledge item in the knowledge base.

    Args:
        title: The title of the knowledge item.
        content: The content of the knowledge item.
        source: The source of the knowledge item (e.g., book title, website).
        tags: Tags for categorizing the knowledge item.
        metadata: Additional metadata for the knowledge item.

    Returns:
        str: The ID of the stored knowledge item.
    """
    if tags is None:
        tags = []
    
    if metadata is None:
        metadata = {}
    
    knowledge_item = {
        "title": title,
        "content": content,
        "source": source,
        "tags": tags,
        "metadata": metadata,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    
    collection = get_collection(KNOWLEDGE_COLLECTION)
    
    # Check if the item already exists
    existing_item = collection.find_one({"title": title})
    if existing_item:
        # Update the existing item
        knowledge_item["updated_at"] = datetime.utcnow()
        collection.update_one({"_id": existing_item["_id"]}, {"$set": knowledge_item})
        logger.info("Updated knowledge item: %s", title)
        return str(existing_item["_id"])
    else:
        # Insert a new item
        result = collection.insert_one(knowledge_item)
        logger.info("Stored knowledge item: %s", title)
        return str(result.inserted_id)


def store_rule(name: str, description: str, condition: str, action: str, category: str = None,
               tags: List[str] = None, metadata: Dict[str, Any] = None) -> str:
    """Store a rule in the knowledge base.

    Args:
        name: The name of the rule.
        description: The description of the rule.
        condition: The condition part of the rule.
        action: The action part of the rule.
        category: The category of the rule.
        tags: Tags for categorizing the rule.
        metadata: Additional metadata for the rule.

    Returns:
        str: The ID of the stored rule.
    """
    if tags is None:
        tags = []
    
    if metadata is None:
        metadata = {}
    
    rule = {
        "name": name,
        "description": description,
        "condition": condition,
        "action": action,
        "category": category,
        "tags": tags,
        "metadata": metadata,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    
    collection = get_collection(RULES_COLLECTION)
    
    # Check if the rule already exists
    existing_rule = collection.find_one({"name": name})
    if existing_rule:
        # Update the existing rule
        rule["updated_at"] = datetime.utcnow()
        collection.update_one({"_id": existing_rule["_id"]}, {"$set": rule})
        logger.info("Updated rule: %s", name)
        return str(existing_rule["_id"])
    else:
        # Insert a new rule
        result = collection.insert_one(rule)
        logger.info("Stored rule: %s", name)
        return str(result.inserted_id)


def store_strategy(name: str, description: str, rules: List[str], implementation: str = None,
                  category: str = None, tags: List[str] = None, performance: Dict[str, float] = None,
                  metadata: Dict[str, Any] = None) -> str:
    """Store a strategy in the knowledge base.

    Args:
        name: The name of the strategy.
        description: The description of the strategy.
        rules: List of rule IDs that make up the strategy.
        implementation: The implementation code of the strategy.
        category: The category of the strategy.
        tags: Tags for categorizing the strategy.
        performance: Performance metrics for the strategy.
        metadata: Additional metadata for the strategy.

    Returns:
        str: The ID of the stored strategy.
    """
    if tags is None:
        tags = []
    
    if performance is None:
        performance = {}
    
    if metadata is None:
        metadata = {}
    
    strategy = {
        "name": name,
        "description": description,
        "rules": rules,
        "implementation": implementation,
        "category": category,
        "tags": tags,
        "performance": performance,
        "metadata": metadata,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    
    collection = get_collection(STRATEGIES_COLLECTION)
    
    # Check if the strategy already exists
    existing_strategy = collection.find_one({"name": name})
    if existing_strategy:
        # Update the existing strategy
        strategy["updated_at"] = datetime.utcnow()
        collection.update_one({"_id": existing_strategy["_id"]}, {"$set": strategy})
        logger.info("Updated strategy: %s", name)
        return str(existing_strategy["_id"])
    else:
        # Insert a new strategy
        result = collection.insert_one(strategy)
        logger.info("Stored strategy: %s", name)
        return str(result.inserted_id)


def store_concept(name: str, description: str, category: str = None,
                 related_concepts: List[str] = None, metadata: Dict[str, Any] = None) -> str:
    """Store a concept in the knowledge base.

    Args:
        name: The name of the concept.
        description: The description of the concept.
        category: The category of the concept.
        related_concepts: List of related concept IDs.
        metadata: Additional metadata for the concept.

    Returns:
        str: The ID of the stored concept.
    """
    if related_concepts is None:
        related_concepts = []
    
    if metadata is None:
        metadata = {}
    
    concept = {
        "name": name,
        "description": description,
        "category": category,
        "related_concepts": related_concepts,
        "metadata": metadata,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    
    collection = get_collection(CONCEPTS_COLLECTION)
    
    # Check if the concept already exists
    existing_concept = collection.find_one({"name": name})
    if existing_concept:
        # Update the existing concept
        concept["updated_at"] = datetime.utcnow()
        collection.update_one({"_id": existing_concept["_id"]}, {"$set": concept})
        logger.info("Updated concept: %s", name)
        return str(existing_concept["_id"])
    else:
        # Insert a new concept
        result = collection.insert_one(concept)
        logger.info("Stored concept: %s", name)
        return str(result.inserted_id)


def store_relationship(source_id: str, target_id: str, relationship_type: str,
                      metadata: Dict[str, Any] = None) -> str:
    """Store a relationship between two entities in the knowledge base.

    Args:
        source_id: The ID of the source entity.
        target_id: The ID of the target entity.
        relationship_type: The type of relationship.
        metadata: Additional metadata for the relationship.

    Returns:
        str: The ID of the stored relationship.
    """
    if metadata is None:
        metadata = {}
    
    relationship = {
        "source": source_id,
        "target": target_id,
        "type": relationship_type,
        "metadata": metadata,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    
    collection = get_collection(RELATIONSHIPS_COLLECTION)
    
    # Check if the relationship already exists
    existing_relationship = collection.find_one({
        "source": source_id,
        "target": target_id,
        "type": relationship_type,
    })
    
    if existing_relationship:
        # Update the existing relationship
        relationship["updated_at"] = datetime.utcnow()
        collection.update_one({"_id": existing_relationship["_id"]}, {"$set": relationship})
        logger.info("Updated relationship: %s -> %s (%s)", source_id, target_id, relationship_type)
        return str(existing_relationship["_id"])
    else:
        # Insert a new relationship
        result = collection.insert_one(relationship)
        logger.info("Stored relationship: %s -> %s (%s)", source_id, target_id, relationship_type)
        return str(result.inserted_id)


def search_knowledge_base(query: str, collection_name: str = KNOWLEDGE_COLLECTION,
                         limit: int = 10) -> List[Dict[str, Any]]:
    """Search the knowledge base for items matching the query.

    Args:
        query: The search query.
        collection_name: The name of the collection to search.
        limit: The maximum number of results to return.

    Returns:
        List[Dict[str, Any]]: The search results.
    """
    collection = get_collection(collection_name)
    
    # Create a text search query
    results = collection.find(
        {"$text": {"$search": query}},
        {"score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})]).limit(limit)
    
    # Convert ObjectId to string for JSON serialization
    serialized_results = []
    for result in results:
        result["_id"] = str(result["_id"])
        serialized_results.append(result)
    
    return serialized_results


def get_related_items(item_id: str, relationship_type: str = None) -> List[Dict[str, Any]]:
    """Get items related to the given item.

    Args:
        item_id: The ID of the item.
        relationship_type: The type of relationship to filter by.

    Returns:
        List[Dict[str, Any]]: The related items.
    """
    relationships_collection = get_collection(RELATIONSHIPS_COLLECTION)
    
    # Create a query to find relationships where the item is the source
    query = {"source": item_id}
    if relationship_type:
        query["type"] = relationship_type
    
    relationships = relationships_collection.find(query)
    
    # Get the target items
    related_items = []
    for relationship in relationships:
        target_id = relationship["target"]
        target_type = relationship.get("metadata", {}).get("target_type")
        
        if target_type:
            target_collection = get_collection(target_type)
            target_item = target_collection.find_one({"_id": ObjectId(target_id)})
            if target_item:
                target_item["_id"] = str(target_item["_id"])
                target_item["relationship"] = relationship["type"]
                related_items.append(target_item)
    
    return related_items