"""MongoDB storage module for the Friday AI Trading System.

This module provides the MongoDBStorage class for storing and retrieving data
from MongoDB databases.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import json
from bson import ObjectId
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager
from src.data.storage.data_storage import DataStorage, StorageError

# Create logger
logger = get_logger(__name__)


class MongoDBJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MongoDB objects.

    This encoder handles MongoDB-specific types like ObjectId.
    """

    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)


class MongoDBStorage(DataStorage):
    """Class for storing and retrieving data from MongoDB databases.

    This class provides methods for storing and retrieving data from MongoDB databases.
    It supports various MongoDB operations like inserting, updating, deleting, and
    querying documents.

    Attributes:
        config: Configuration manager.
        connection_string: Connection string for the database.
        client: MongoDB client.
        db: MongoDB database.
        metadata: Dictionary for storing metadata about the storage operations.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        database_name: Optional[str] = None,
        config: Optional[ConfigManager] = None,
    ):
        """Initialize a MongoDB storage instance.

        Args:
            connection_string: Connection string for the database. If None, it will be
                retrieved from the configuration.
            database_name: Name of the database to use. If None, it will be retrieved
                from the configuration.
            config: Configuration manager. If None, a new one will be created.
        """
        super().__init__(config)

        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None

        # Connect if connection string and database name are provided
        if self.connection_string and self.database_name:
            self.connect()

    def connect(self) -> None:
        """Connect to the MongoDB database.

        If no connection string or database name was provided in the constructor,
        they will be retrieved from the configuration.

        Raises:
            StorageError: If connection fails.
        """
        try:
            # If no connection string was provided, get it from the configuration
            if not self.connection_string:
                self.connection_string = self.config.get("mongodb.connection_string")
                if not self.connection_string:
                    raise ValueError("No MongoDB connection string provided")

            # If no database name was provided, get it from the configuration
            if not self.database_name:
                self.database_name = self.config.get("mongodb.database_name")
                if not self.database_name:
                    raise ValueError("No MongoDB database name provided")

            # Create client and connect to the database
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.database_name]

            # Test connection
            self.client.admin.command("ping")

            logger.info(f"Connected to MongoDB database: {self.database_name}")
            self._record_operation_metadata(
                "connect",
                connection_string=self.connection_string,
                database_name=self.database_name
            )

        except Exception as e:
            self._handle_error("connecting to MongoDB database", e)

    def disconnect(self) -> None:
        """Disconnect from the MongoDB database.

        Raises:
            StorageError: If disconnection fails.
        """
        try:
            if self.client:
                self.client.close()
                self.client = None
                self.db = None
                logger.info("Disconnected from MongoDB database")
                self._record_operation_metadata("disconnect")

        except Exception as e:
            self._handle_error("disconnecting from MongoDB database", e)

    def is_connected(self) -> bool:
        """Check if connected to the MongoDB database.

        Returns:
            bool: True if connected, False otherwise.
        """
        if not self.client or not self.db:
            return False

        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    def store_data(
        self,
        data: pd.DataFrame,
        collection_name: str,
        if_exists: str = "append",
        index: bool = True,
        chunk_size: Optional[int] = None,
        **kwargs
    ) -> bool:
        """Store data in the MongoDB database.

        Args:
            data: The data to store.
            collection_name: The name of the collection to store the data in.
            if_exists: What to do if the collection exists ('fail', 'replace', or 'append').
            index: Whether to store the index as a field named '_index'.
            chunk_size: Number of documents to insert at once.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            StorageError: If storing fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Get the collection
            collection = self.db[collection_name]

            # Check if the collection exists and handle accordingly
            if collection.estimated_document_count() > 0:
                if if_exists == "fail":
                    raise ValueError(f"Collection '{collection_name}' already exists")
                elif if_exists == "replace":
                    collection.delete_many({})

            # Prepare data for storage
            prepared_data = self._prepare_data_for_storage(data)

            # Convert DataFrame to list of dictionaries
            records = prepared_data.to_dict("records")

            # Add index as a field if requested
            if index and not prepared_data.index.name and not prepared_data.index.equals(pd.RangeIndex.from_range(range(len(prepared_data)))):
                for i, record in enumerate(records):
                    record["_index"] = prepared_data.index[i]

            # Insert documents in chunks if requested
            if chunk_size and len(records) > chunk_size:
                for i in range(0, len(records), chunk_size):
                    chunk = records[i:i + chunk_size]
                    collection.insert_many(chunk)
            else:
                if records:  # Only insert if there are records
                    collection.insert_many(records)

            logger.info(f"Stored {len(records)} documents in collection '{collection_name}'")
            self._record_operation_metadata(
                "store_data",
                collection_name=collection_name,
                document_count=len(records),
                if_exists=if_exists
            )

            return True

        except Exception as e:
            self._handle_error(f"storing data in collection '{collection_name}'", e)
            return False

    def retrieve_data(
        self,
        collection_name: str,
        query: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort: Optional[List[Tuple[str, int]]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Retrieve data from the MongoDB database.

        Args:
            collection_name: The name of the collection to retrieve data from.
            query: Query for filtering the data.
            projection: Fields to include or exclude.
            limit: Maximum number of documents to retrieve.
            skip: Number of documents to skip.
            sort: List of (field, direction) pairs for sorting.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: The retrieved data.

        Raises:
            StorageError: If retrieval fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Get the collection
            collection = self.db[collection_name]

            # Build the query
            cursor = collection.find(
                filter=query or {},
                projection=projection,
            )

            # Apply sorting if requested
            if sort:
                cursor = cursor.sort(sort)

            # Apply skip if requested
            if skip:
                cursor = cursor.skip(skip)

            # Apply limit if requested
            if limit:
                cursor = cursor.limit(limit)

            # Convert cursor to list and then to DataFrame
            documents = list(cursor)

            # Handle empty result
            if not documents:
                logger.info(f"No documents found in collection '{collection_name}'")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(documents)

            # Set index if '_index' field exists and is not needed as a column
            if "_index" in df.columns and kwargs.get("set_index", True):
                df.set_index("_index", inplace=True)

            # Remove MongoDB's _id field if not needed
            if "_id" in df.columns and not kwargs.get("keep_id", False):
                df.drop("_id", axis=1, inplace=True)

            logger.info(f"Retrieved {len(df)} documents from collection '{collection_name}'")
            self._record_operation_metadata(
                "retrieve_data",
                collection_name=collection_name,
                document_count=len(df),
                query=query
            )

            return df

        except Exception as e:
            self._handle_error(f"retrieving data from collection '{collection_name}'", e)
            return pd.DataFrame()

    def delete_data(
        self,
        collection_name: str,
        query: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """Delete data from the MongoDB database.

        Args:
            collection_name: The name of the collection to delete data from.
            query: Query for filtering the data to delete.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            StorageError: If deletion fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Get the collection
            collection = self.db[collection_name]

            # Delete documents
            result = collection.delete_many(query or {})

            logger.info(f"Deleted {result.deleted_count} documents from collection '{collection_name}'")
            self._record_operation_metadata(
                "delete_data",
                collection_name=collection_name,
                deleted_count=result.deleted_count,
                query=query
            )

            return True

        except Exception as e:
            self._handle_error(f"deleting data from collection '{collection_name}'", e)
            return False

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in the MongoDB database.

        Args:
            collection_name: The name of the collection to check.

        Returns:
            bool: True if the collection exists, False otherwise.
        """
        try:
            if not self.is_connected():
                self.connect()

            return collection_name in self.db.list_collection_names()

        except Exception as e:
            self._handle_error(f"checking if collection '{collection_name}' exists", e)
            return False

    def list_collections(self) -> List[str]:
        """List all collections in the MongoDB database.

        Returns:
            List[str]: List of collection names.

        Raises:
            StorageError: If listing fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            collections = self.db.list_collection_names()

            logger.info(f"Listed {len(collections)} collections")
            self._record_operation_metadata("list_collections", collection_count=len(collections))

            return collections

        except Exception as e:
            self._handle_error("listing collections", e)
            return []

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection in the MongoDB database.

        Args:
            collection_name: The name of the collection.

        Returns:
            Dict[str, Any]: Dictionary with collection information.

        Raises:
            StorageError: If getting information fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Check if the collection exists
            if not self.collection_exists(collection_name):
                raise ValueError(f"Collection '{collection_name}' does not exist")

            # Get the collection
            collection = self.db[collection_name]

            # Get collection stats
            stats = self.db.command("collStats", collection_name)

            # Get a sample document to infer schema
            sample = collection.find_one()
            schema = self._infer_schema(sample) if sample else {}

            # Build the result
            info = {
                "name": collection_name,
                "count": collection.estimated_document_count(),
                "size": stats.get("size"),
                "avg_document_size": stats.get("avgObjSize"),
                "storage_size": stats.get("storageSize"),
                "indexes": stats.get("nindexes"),
                "index_size": stats.get("totalIndexSize"),
                "schema": schema,
            }

            logger.info(f"Got information for collection '{collection_name}'")
            self._record_operation_metadata("get_collection_info", collection_name=collection_name)

            return info

        except Exception as e:
            self._handle_error(f"getting information for collection '{collection_name}'", e)
            return {}

    def create_index(
        self,
        collection_name: str,
        keys: List[Tuple[str, int]],
        unique: bool = False,
        **kwargs
    ) -> bool:
        """Create an index on a collection in the MongoDB database.

        Args:
            collection_name: The name of the collection.
            keys: List of (field, direction) pairs for the index.
            unique: Whether the index should enforce uniqueness.
            **kwargs: Additional keyword arguments for pymongo.Collection.create_index().

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            StorageError: If index creation fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Get the collection
            collection = self.db[collection_name]

            # Create the index
            index_name = collection.create_index(keys, unique=unique, **kwargs)

            logger.info(f"Created index '{index_name}' on collection '{collection_name}'")
            self._record_operation_metadata(
                "create_index",
                collection_name=collection_name,
                index_name=index_name,
                keys=keys,
                unique=unique
            )

            return True

        except Exception as e:
            self._handle_error(f"creating index on collection '{collection_name}'", e)
            return False

    def drop_index(
        self,
        collection_name: str,
        index_name: str,
        **kwargs
    ) -> bool:
        """Drop an index from a collection in the MongoDB database.

        Args:
            collection_name: The name of the collection.
            index_name: The name of the index to drop.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            StorageError: If index dropping fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Get the collection
            collection = self.db[collection_name]

            # Drop the index
            collection.drop_index(index_name)

            logger.info(f"Dropped index '{index_name}' from collection '{collection_name}'")
            self._record_operation_metadata(
                "drop_index",
                collection_name=collection_name,
                index_name=index_name
            )

            return True

        except Exception as e:
            self._handle_error(f"dropping index '{index_name}' from collection '{collection_name}'", e)
            return False

    def list_indexes(
        self,
        collection_name: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """List all indexes on a collection in the MongoDB database.

        Args:
            collection_name: The name of the collection.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            List[Dict[str, Any]]: List of index information dictionaries.

        Raises:
            StorageError: If listing fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Get the collection
            collection = self.db[collection_name]

            # List the indexes
            indexes = list(collection.list_indexes())

            logger.info(f"Listed {len(indexes)} indexes on collection '{collection_name}'")
            self._record_operation_metadata(
                "list_indexes",
                collection_name=collection_name,
                index_count=len(indexes)
            )

            return indexes

        except Exception as e:
            self._handle_error(f"listing indexes on collection '{collection_name}'", e)
            return []

    def update_data(
        self,
        collection_name: str,
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False,
        multi: bool = True,
        **kwargs
    ) -> bool:
        """Update data in the MongoDB database.

        Args:
            collection_name: The name of the collection.
            query: Query for filtering the data to update.
            update: Update operations to apply.
            upsert: Whether to insert a new document if no document matches the query.
            multi: Whether to update multiple documents.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            StorageError: If update fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Get the collection
            collection = self.db[collection_name]

            # Update documents
            if multi:
                result = collection.update_many(query, update, upsert=upsert)
                modified_count = result.modified_count
                upserted_id = result.upserted_id
            else:
                result = collection.update_one(query, update, upsert=upsert)
                modified_count = result.modified_count
                upserted_id = result.upserted_id

            logger.info(f"Updated {modified_count} documents in collection '{collection_name}'")
            self._record_operation_metadata(
                "update_data",
                collection_name=collection_name,
                modified_count=modified_count,
                upserted_id=str(upserted_id) if upserted_id else None,
                query=query
            )

            return True

        except Exception as e:
            self._handle_error(f"updating data in collection '{collection_name}'", e)
            return False

    def aggregate(
        self,
        collection_name: str,
        pipeline: List[Dict[str, Any]],
        **kwargs
    ) -> pd.DataFrame:
        """Perform an aggregation on a collection in the MongoDB database.

        Args:
            collection_name: The name of the collection.
            pipeline: Aggregation pipeline.
            **kwargs: Additional keyword arguments for pymongo.Collection.aggregate().

        Returns:
            pd.DataFrame: The result of the aggregation.

        Raises:
            StorageError: If aggregation fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Get the collection
            collection = self.db[collection_name]

            # Perform the aggregation
            cursor = collection.aggregate(pipeline, **kwargs)

            # Convert cursor to list and then to DataFrame
            documents = list(cursor)

            # Handle empty result
            if not documents:
                logger.info(f"No documents found in aggregation on collection '{collection_name}'")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(documents)

            # Remove MongoDB's _id field if not needed
            if "_id" in df.columns and not kwargs.get("keep_id", False):
                df.drop("_id", axis=1, inplace=True)

            logger.info(f"Performed aggregation on collection '{collection_name}', got {len(df)} results")
            self._record_operation_metadata(
                "aggregate",
                collection_name=collection_name,
                result_count=len(df),
                pipeline=pipeline[:2]  # Only log the first two stages for brevity
            )

            return df

        except Exception as e:
            self._handle_error(f"performing aggregation on collection '{collection_name}'", e)
            return pd.DataFrame()

    def drop_collection(
        self,
        collection_name: str,
        **kwargs
    ) -> bool:
        """Drop a collection from the MongoDB database.

        Args:
            collection_name: The name of the collection to drop.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            StorageError: If dropping fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Drop the collection
            self.db.drop_collection(collection_name)

            logger.info(f"Dropped collection '{collection_name}'")
            self._record_operation_metadata("drop_collection", collection_name=collection_name)

            return True

        except Exception as e:
            self._handle_error(f"dropping collection '{collection_name}'", e)
            return False

    def backup_collection(
        self,
        collection_name: str,
        backup_collection_name: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Backup a collection in the MongoDB database.

        Args:
            collection_name: The name of the collection to backup.
            backup_collection_name: The name of the backup collection. If None, a name
                will be generated based on the original collection name and the current
                timestamp.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            StorageError: If backup fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Generate a backup collection name if not provided
            if not backup_collection_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_collection_name = f"{collection_name}_backup_{timestamp}"

            # Check if the source collection exists
            if not self.collection_exists(collection_name):
                raise ValueError(f"Collection '{collection_name}' does not exist")

            # Get the source collection
            source_collection = self.db[collection_name]

            # Create the backup collection
            backup_collection = self.db[backup_collection_name]

            # Copy all documents from source to backup
            documents = list(source_collection.find({}))
            if documents:
                backup_collection.insert_many(documents)

            # Copy all indexes from source to backup
            for index in source_collection.list_indexes():
                # Skip the default _id index
                if index["name"] == "_id_":
                    continue

                # Extract index information
                keys = [(k, v) for k, v in index["key"].items()]
                options = {k: v for k, v in index.items() if k not in ["key", "v", "ns"]}

                # Create the index on the backup collection
                backup_collection.create_index(keys, **options)

            logger.info(f"Backed up collection '{collection_name}' to '{backup_collection_name}'")
            self._record_operation_metadata(
                "backup_collection",
                collection_name=collection_name,
                backup_collection_name=backup_collection_name,
                document_count=len(documents)
            )

            return True

        except Exception as e:
            self._handle_error(f"backing up collection '{collection_name}'", e)
            return False

    def _infer_schema(self, document: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
        """Infer the schema of a MongoDB document.

        Args:
            document: The document to infer the schema from.
            prefix: Prefix for nested fields.

        Returns:
            Dict[str, str]: Dictionary mapping field names to their types.
        """
        schema = {}

        for key, value in document.items():
            field_name = f"{prefix}{key}" if prefix else key

            if isinstance(value, dict):
                # Recursively infer schema for nested documents
                nested_schema = self._infer_schema(value, f"{field_name}.")
                schema.update(nested_schema)
            elif isinstance(value, list):
                # For lists, infer the type of the first element if the list is not empty
                if value:
                    if isinstance(value[0], dict):
                        nested_schema = self._infer_schema(value[0], f"{field_name}[].")
                        schema.update(nested_schema)
                    else:
                        schema[field_name] = f"Array<{type(value[0]).__name__}>"
                else:
                    schema[field_name] = "Array"
            else:
                # For simple types, just use the type name
                schema[field_name] = type(value).__name__

        return schema