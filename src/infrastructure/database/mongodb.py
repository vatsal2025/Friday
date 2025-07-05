"""MongoDB integration for the Friday AI Trading System.

This module provides MongoDB connection and operations for storing unstructured data.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from bson import ObjectId
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database

from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)

# MongoDB connection instance
_mongo_client = None
_mongo_db = None


def get_mongo_client() -> MongoClient:
    """Get a MongoDB client instance.

    Returns:
        MongoClient: A MongoDB client instance.
    """
    global _mongo_client

    if _mongo_client is None:
        # Get MongoDB configuration
        mongo_config = get_config("mongodb")
        if not mongo_config:
            # If MongoDB config is not found, use default values
            mongo_config = {
                "host": "localhost",
                "port": 27017,
                "username": None,
                "password": None,
                "auth_source": "admin",
                "connect_timeout_ms": 5000,
                "server_selection_timeout_ms": 5000,
            }

        # Build connection string
        if mongo_config.get("username") and mongo_config.get("password"):
            connection_string = f"mongodb://{mongo_config['username']}:{mongo_config['password']}@{mongo_config['host']}:{mongo_config['port']}/{mongo_config.get('auth_source', 'admin')}"
        else:
            connection_string = f"mongodb://{mongo_config['host']}:{mongo_config['port']}/"

        # Create MongoDB client
        try:
            _mongo_client = MongoClient(
                connection_string,
                connectTimeoutMS=mongo_config.get("connect_timeout_ms", 5000),
                serverSelectionTimeoutMS=mongo_config.get("server_selection_timeout_ms", 5000),
            )
            # Test connection
            _mongo_client.admin.command("ping")
            logger.info("MongoDB connection established successfully")
        except errors.ConnectionFailure as e:
            logger.error("Failed to connect to MongoDB: %s", str(e))
            raise

    return _mongo_client


def get_database(db_name: Optional[str] = None) -> Database:
    """Get a MongoDB database instance.

    Args:
        db_name: The name of the database. If None, the default database from config will be used.

    Returns:
        Database: A MongoDB database instance.
    """
    global _mongo_db

    if _mongo_db is None or db_name:
        client = get_mongo_client()
        
        # If db_name is not provided, get it from config
        if not db_name:
            db_name = get_config("mongodb", "db_name") or "friday"
        
        _mongo_db = client[db_name]
        logger.debug("Using MongoDB database: %s", db_name)

    return _mongo_db if not db_name else get_mongo_client()[db_name]


def get_collection(collection_name: str, db_name: str = None, validator: Dict = None) -> Collection:
    """Get a MongoDB collection.

    Args:
        collection_name: The name of the collection.
        db_name: The name of the database. If not provided, the default database will be used.
        validator: Optional JSON Schema validator for the collection.

    Returns:
        Collection: The MongoDB collection.
    """
    db = get_database(db_name)
    
    # Check if the collection exists
    if collection_name not in db.list_collection_names():
        # Create the collection with the validator if provided
        if validator:
            db.create_collection(
                collection_name,
                validator=validator,
                validationAction="warn"  # Only warn on validation failures, don't reject
            )
            logger.info(f"Created collection {collection_name} with schema validation")
        else:
            db.create_collection(collection_name)
            logger.info(f"Created collection {collection_name}")
    elif validator:
        # If the collection exists and a validator is provided, update the validator
        db.command({
            "collMod": collection_name,
            "validator": validator,
            "validationAction": "warn"  # Only warn on validation failures, don't reject
        })
        logger.info(f"Updated schema validation for collection {collection_name}")
    
    return db[collection_name]


def insert_one(collection_name: str, document: Dict[str, Any], db_name: Optional[str] = None) -> str:
    """Insert a document into a collection.

    Args:
        collection_name: The name of the collection.
        document: The document to insert.
        db_name: The name of the database. If None, the default database will be used.

    Returns:
        str: The ID of the inserted document.
    """
    try:
        # Add created_at and updated_at timestamps if not present
        if "created_at" not in document:
            document["created_at"] = datetime.utcnow()
        if "updated_at" not in document:
            document["updated_at"] = document["created_at"]

        collection = get_collection(collection_name, db_name)
        result = collection.insert_one(document)
        logger.debug("Document inserted with ID: %s", result.inserted_id)
        return str(result.inserted_id)
    except Exception as e:
        logger.error("Failed to insert document: %s", str(e))
        raise


def insert_many(collection_name: str, documents: List[Dict[str, Any]], db_name: Optional[str] = None) -> List[str]:
    """Insert multiple documents into a collection.

    Args:
        collection_name: The name of the collection.
        documents: The documents to insert.
        db_name: The name of the database. If None, the default database will be used.

    Returns:
        List[str]: The IDs of the inserted documents.
    """
    try:
        # Add created_at and updated_at timestamps if not present
        now = datetime.utcnow()
        for doc in documents:
            if "created_at" not in doc:
                doc["created_at"] = now
            if "updated_at" not in doc:
                doc["updated_at"] = doc["created_at"]

        collection = get_collection(collection_name, db_name)
        result = collection.insert_many(documents)
        inserted_ids = [str(id) for id in result.inserted_ids]
        logger.debug("%d documents inserted", len(inserted_ids))
        return inserted_ids
    except Exception as e:
        logger.error("Failed to insert documents: %s", str(e))
        raise


def find(collection_name: str, query: Dict[str, Any], projection: Optional[Dict[str, Any]] = None, 
        sort: Optional[List[tuple]] = None, limit: Optional[int] = None, 
        skip: Optional[int] = None, db_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find documents in a collection.

    Args:
        collection_name: The name of the collection.
        query: The query to find the documents.
        projection: The projection to apply to the results.
        sort: The sort order to apply to the results.
        limit: The maximum number of documents to return.
        skip: The number of documents to skip.
        db_name: The name of the database. If None, the default database will be used.

    Returns:
        List[Dict[str, Any]]: The found documents.
    """
    try:
        collection = get_collection(collection_name, db_name)
        cursor = collection.find(query, projection)
        
        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        
        # Convert ObjectId to string for JSON serialization
        result = []
        for doc in cursor:
            if '_id' in doc and isinstance(doc['_id'], ObjectId):
                doc['_id'] = str(doc['_id'])
            result.append(doc)
        
        logger.debug("Found %d documents matching query", len(result))
        return result
    except Exception as e:
        logger.error("Failed to find documents: %s", str(e))
        raise


def find_one(collection_name: str, query: Dict[str, Any], db_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Find a document in a collection.

    Args:
        collection_name: The name of the collection.
        query: The query to find the document.
        db_name: The name of the database. If None, the default database will be used.

    Returns:
        Optional[Dict[str, Any]]: The found document, or None if not found.
    """
    try:
        collection = get_collection(collection_name, db_name)
        result = collection.find_one(query)
        return result
    except Exception as e:
        logger.error("Failed to find document: %s", str(e))
        raise


def find_many(collection_name: str, query: Dict[str, Any], sort: Optional[List[tuple]] = None, 
              limit: Optional[int] = None, skip: Optional[int] = None, 
              db_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find documents in a collection.

    Args:
        collection_name: The name of the collection.
        query: The query to find the documents.
        sort: The sort criteria. A list of (key, direction) pairs.
        limit: The maximum number of documents to return.
        skip: The number of documents to skip.
        db_name: The name of the database. If None, the default database will be used.

    Returns:
        List[Dict[str, Any]]: The found documents.
    """
    try:
        collection = get_collection(collection_name, db_name)
        cursor = collection.find(query)

        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)

        return list(cursor)
    except Exception as e:
        logger.error("Failed to find documents: %s", str(e))
        raise


def update_one(collection_name: str, query: Dict[str, Any], update: Dict[str, Any], 
               upsert: bool = False, db_name: Optional[str] = None) -> int:
    """Update a document in a collection.

    Args:
        collection_name: The name of the collection.
        query: The query to find the document to update.
        update: The update to apply.
        upsert: Whether to insert a new document if no document matches the query.
        db_name: The name of the database. If None, the default database will be used.

    Returns:
        int: The number of documents modified.
    """
    try:
        # Add updated_at timestamp if not in $set
        if "$set" in update:
            if "updated_at" not in update["$set"]:
                update["$set"]["updated_at"] = datetime.utcnow()
        else:
            update["$set"] = {"updated_at": datetime.utcnow()}

        collection = get_collection(collection_name, db_name)
        result = collection.update_one(query, update, upsert=upsert)
        logger.debug("%d document(s) modified", result.modified_count)
        return result.modified_count
    except Exception as e:
        logger.error("Failed to update document: %s", str(e))
        raise


def update_many(collection_name: str, query: Dict[str, Any], update: Dict[str, Any], 
                upsert: bool = False, db_name: Optional[str] = None) -> int:
    """Update multiple documents in a collection.

    Args:
        collection_name: The name of the collection.
        query: The query to find the documents to update.
        update: The update to apply.
        upsert: Whether to insert a new document if no document matches the query.
        db_name: The name of the database. If None, the default database will be used.

    Returns:
        int: The number of documents modified.
    """
    try:
        # Add updated_at timestamp if not in $set
        if "$set" in update:
            if "updated_at" not in update["$set"]:
                update["$set"]["updated_at"] = datetime.utcnow()
        else:
            update["$set"] = {"updated_at": datetime.utcnow()}

        collection = get_collection(collection_name, db_name)
        result = collection.update_many(query, update, upsert=upsert)
        logger.debug("%d document(s) modified", result.modified_count)
        return result.modified_count
    except Exception as e:
        logger.error("Failed to update documents: %s", str(e))
        raise


def delete_one(collection_name: str, query: Dict[str, Any], db_name: Optional[str] = None) -> int:
    """Delete a document from a collection.

    Args:
        collection_name: The name of the collection.
        query: The query to find the document to delete.
        db_name: The name of the database. If None, the default database will be used.

    Returns:
        int: The number of documents deleted.
    """
    try:
        collection = get_collection(collection_name, db_name)
        result = collection.delete_one(query)
        logger.debug("%d document(s) deleted", result.deleted_count)
        return result.deleted_count
    except Exception as e:
        logger.error("Failed to delete document: %s", str(e))
        raise


def delete_many(collection_name: str, query: Dict[str, Any], db_name: Optional[str] = None) -> int:
    """Delete multiple documents from a collection.

    Args:
        collection_name: The name of the collection.
        query: The query to find the documents to delete.
        db_name: The name of the database. If None, the default database will be used.

    Returns:
        int: The number of documents deleted.
    """
    try:
        collection = get_collection(collection_name, db_name)
        result = collection.delete_many(query)
        logger.debug("%d document(s) deleted", result.deleted_count)
        return result.deleted_count
    except Exception as e:
        logger.error("Failed to delete documents: %s", str(e))
        raise


def count_documents(collection_name: str, query: Dict[str, Any], db_name: Optional[str] = None) -> int:
    """Count documents in a collection.

    Args:
        collection_name: The name of the collection.
        query: The query to count documents.
        db_name: The name of the database. If None, the default database will be used.

    Returns:
        int: The number of documents matching the query.
    """
    try:
        collection = get_collection(collection_name, db_name)
        return collection.count_documents(query)
    except Exception as e:
        logger.error("Failed to count documents: %s", str(e))
        raise


def create_index(collection_name: str, keys: List[tuple], unique: bool = False, 
                 db_name: Optional[str] = None) -> str:
    """Create an index on a collection.

    Args:
        collection_name: The name of the collection.
        keys: The keys to index. A list of (key, direction) pairs.
        unique: Whether the index should enforce uniqueness.
        db_name: The name of the database. If None, the default database will be used.

    Returns:
        str: The name of the created index.
    """
    try:
        collection = get_collection(collection_name, db_name)
        result = collection.create_index(keys, unique=unique)
        logger.debug("Index created: %s", result)
        return result
    except Exception as e:
        logger.error("Failed to create index: %s", str(e))
        raise


def drop_collection(collection_name: str, db_name: Optional[str] = None) -> None:
    """Drop a collection.

    Args:
        collection_name: The name of the collection to drop.
        db_name: The name of the database. If None, the default database will be used.
    """
    try:
        db = get_database(db_name)
        db.drop_collection(collection_name)
        logger.debug("Collection dropped: %s", collection_name)
    except Exception as e:
        logger.error("Failed to drop collection: %s", str(e))
        raise


def backup_database(backup_dir: str, db_name: Optional[str] = None) -> str:
    """Backup a MongoDB database using mongodump.

    Args:
        backup_dir: The directory to store the backup.
        db_name: The name of the database to backup. If None, the default database will be used.

    Returns:
        str: The path to the backup file.
    """
    if not db_name:
        db_name = get_config("mongodb", "db_name") or "friday"

    # Create backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)

    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"{db_name}_{timestamp}")

    # Get MongoDB connection details
    mongo_config = get_config("mongodb")
    host = mongo_config.get("host", "localhost")
    port = mongo_config.get("port", 27017)
    username = mongo_config.get("username")
    password = mongo_config.get("password")
    auth_source = mongo_config.get("auth_source", "admin")

    # Build mongodump command
    cmd = f"mongodump --host {host} --port {port} --db {db_name} --out {backup_path}"
    if username and password:
        cmd += f" --username {username} --password {password} --authenticationDatabase {auth_source}"

    # Execute mongodump command
    try:
        import subprocess
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info("Database backup completed successfully: %s", backup_path)
        return backup_path
    except subprocess.CalledProcessError as e:
        logger.error("Database backup failed: %s", e.stderr)
        raise


def restore_database(backup_path: str, db_name: Optional[str] = None) -> None:
    """Restore a MongoDB database using mongorestore.

    Args:
        backup_path: The path to the backup directory.
        db_name: The name of the database to restore to. If None, the original database name will be used.
    """
    # Get MongoDB connection details
    mongo_config = get_config("mongodb")
    host = mongo_config.get("host", "localhost")
    port = mongo_config.get("port", 27017)
    username = mongo_config.get("username")
    password = mongo_config.get("password")
    auth_source = mongo_config.get("auth_source", "admin")

    # Build mongorestore command
    cmd = f"mongorestore --host {host} --port {port} --dir {backup_path}"
    if username and password:
        cmd += f" --username {username} --password {password} --authenticationDatabase {auth_source}"
    if db_name:
        cmd += f" --nsFrom '{os.path.basename(backup_path)}.*' --nsTo '{db_name}.*'"

    # Execute mongorestore command
    try:
        import subprocess
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info("Database restore completed successfully from: %s", backup_path)
    except subprocess.CalledProcessError as e:
        logger.error("Database restore failed: %s", e.stderr)
        raise


def close_connection() -> None:
    """Close the MongoDB connection."""
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None
        logger.debug("MongoDB connection closed")