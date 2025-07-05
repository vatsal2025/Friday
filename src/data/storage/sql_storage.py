"""SQL storage module for the Friday AI Trading System.

This module provides the SQLStorage class for storing and retrieving data
from SQL databases using SQLAlchemy.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import sqlalchemy
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from src.infrastructure.logging import get_logger
from src.infrastructure.config import ConfigManager
from src.infrastructure.database import get_engine, get_session_factory
from src.data.storage.data_storage import DataStorage, StorageError

# Create logger
logger = get_logger(__name__)


class SQLStorage(DataStorage):
    """Class for storing and retrieving data from SQL databases.

    This class provides methods for storing and retrieving data from SQL databases
    using SQLAlchemy. It supports various SQL dialects including SQLite, MySQL,
    PostgreSQL, and others supported by SQLAlchemy.

    Attributes:
        config: Configuration manager.
        connection_string: Connection string for the database.
        engine: SQLAlchemy engine.
        session_factory: SQLAlchemy session factory.
        metadata: Dictionary for storing metadata about the storage operations.
        schema: Database schema to use.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        schema: Optional[str] = None,
        config: Optional[ConfigManager] = None,
    ):
        """Initialize a SQL storage instance.

        Args:
            connection_string: Connection string for the database. If None, it will be
                retrieved from the configuration.
            schema: Database schema to use.
            config: Configuration manager. If None, a new one will be created.
        """
        super().__init__(config)

        self.connection_string = connection_string
        self.schema = schema
        self.engine = None
        self.session_factory = None

        # Connect if connection string is provided
        if self.connection_string:
            self.connect()

    def connect(self) -> None:
        """Connect to the SQL database.

        If no connection string was provided in the constructor, it will be
        retrieved from the configuration.

        Raises:
            StorageError: If connection fails.
        """
        try:
            # If no connection string was provided, get it from the configuration
            if not self.connection_string:
                self.connection_string = self.config.get("database.connection_string")
                if not self.connection_string:
                    # Try to get the connection string from the infrastructure module
                    self.engine = get_engine()
                    self.session_factory = get_session_factory()
                    return

            # Create engine and session factory
            self.engine = sqlalchemy.create_engine(self.connection_string)
            self.session_factory = sqlalchemy.orm.sessionmaker(bind=self.engine)

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info(f"Connected to SQL database: {self.connection_string}")
            self._record_operation_metadata("connect", connection_string=self.connection_string)

        except Exception as e:
            self._handle_error("connecting to SQL database", e)

    def disconnect(self) -> None:
        """Disconnect from the SQL database.

        Raises:
            StorageError: If disconnection fails.
        """
        try:
            if self.engine:
                self.engine.dispose()
                self.engine = None
                self.session_factory = None
                logger.info("Disconnected from SQL database")
                self._record_operation_metadata("disconnect")

        except Exception as e:
            self._handle_error("disconnecting from SQL database", e)

    def is_connected(self) -> bool:
        """Check if connected to the SQL database.

        Returns:
            bool: True if connected, False otherwise.
        """
        if not self.engine:
            return False

        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def store_data(
        self,
        data: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
        index: bool = True,
        chunk_size: Optional[int] = None,
        dtype: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """Store data in the SQL database.

        Args:
            data: The data to store.
            table_name: The name of the table to store the data in.
            if_exists: What to do if the table exists ('fail', 'replace', or 'append').
            index: Whether to store the index.
            chunk_size: Number of rows to insert at once.
            dtype: Dictionary of column name to SQL type.
            **kwargs: Additional keyword arguments for pandas.to_sql().

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            StorageError: If storing fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Prepare data for storage
            prepared_data = self._prepare_data_for_storage(data)

            # Store data using pandas.to_sql()
            prepared_data.to_sql(
                name=table_name,
                con=self.engine,
                if_exists=if_exists,
                index=index,
                schema=self.schema,
                chunksize=chunk_size,
                dtype=dtype,
                **kwargs
            )

            logger.info(f"Stored {len(prepared_data)} rows in table '{table_name}'")
            self._record_operation_metadata(
                "store_data",
                table_name=table_name,
                row_count=len(prepared_data),
                if_exists=if_exists
            )

            return True

        except Exception as e:
            self._handle_error(f"storing data in table '{table_name}'", e)
            return False

    def retrieve_data(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        condition: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Retrieve data from the SQL database.

        Args:
            table_name: The name of the table to retrieve data from.
            columns: List of columns to retrieve. If None, all columns are retrieved.
            condition: Condition for filtering the data (e.g., "date > '2021-01-01'").
            limit: Maximum number of rows to retrieve.
            offset: Number of rows to skip.
            order_by: Column(s) to order by (e.g., "date DESC").
            **kwargs: Additional keyword arguments for pandas.read_sql().

        Returns:
            pd.DataFrame: The retrieved data.

        Raises:
            StorageError: If retrieval fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Build the SQL query
            columns_str = "*" if not columns else ", ".join(columns)
            query = f"SELECT {columns_str} FROM {table_name}"

            if condition:
                query += f" WHERE {condition}"

            if order_by:
                query += f" ORDER BY {order_by}"

            if limit:
                query += f" LIMIT {limit}"

            if offset:
                query += f" OFFSET {offset}"

            # Execute the query and return the results as a DataFrame
            result = pd.read_sql(query, self.engine, **kwargs)

            logger.info(f"Retrieved {len(result)} rows from table '{table_name}'")
            self._record_operation_metadata(
                "retrieve_data",
                table_name=table_name,
                row_count=len(result),
                condition=condition
            )

            return result

        except Exception as e:
            self._handle_error(f"retrieving data from table '{table_name}'", e)
            return pd.DataFrame()

    def delete_data(
        self,
        table_name: str,
        condition: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Delete data from the SQL database.

        Args:
            table_name: The name of the table to delete data from.
            condition: Condition for filtering the data to delete.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            StorageError: If deletion fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Build the SQL query
            query = f"DELETE FROM {table_name}"
            if condition:
                query += f" WHERE {condition}"

            # Execute the query
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                row_count = result.rowcount

            logger.info(f"Deleted {row_count} rows from table '{table_name}'")
            self._record_operation_metadata(
                "delete_data",
                table_name=table_name,
                row_count=row_count,
                condition=condition
            )

            return True

        except Exception as e:
            self._handle_error(f"deleting data from table '{table_name}'", e)
            return False

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the SQL database.

        Args:
            table_name: The name of the table to check.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        try:
            if not self.is_connected():
                self.connect()

            inspector = inspect(self.engine)
            return inspector.has_table(table_name, schema=self.schema)

        except Exception as e:
            self._handle_error(f"checking if table '{table_name}' exists", e)
            return False

    def list_tables(self) -> List[str]:
        """List all tables in the SQL database.

        Returns:
            List[str]: List of table names.

        Raises:
            StorageError: If listing fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            inspector = inspect(self.engine)
            tables = inspector.get_table_names(schema=self.schema)

            logger.info(f"Listed {len(tables)} tables")
            self._record_operation_metadata("list_tables", table_count=len(tables))

            return tables

        except Exception as e:
            self._handle_error("listing tables", e)
            return []

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table in the SQL database.

        Args:
            table_name: The name of the table.

        Returns:
            Dict[str, Any]: Dictionary with table information.

        Raises:
            StorageError: If getting information fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            inspector = inspect(self.engine)

            # Check if the table exists
            if not inspector.has_table(table_name, schema=self.schema):
                raise ValueError(f"Table '{table_name}' does not exist")

            # Get table information
            columns = inspector.get_columns(table_name, schema=self.schema)
            primary_keys = inspector.get_primary_keys(table_name, schema=self.schema)
            foreign_keys = inspector.get_foreign_keys(table_name, schema=self.schema)
            indexes = inspector.get_indexes(table_name, schema=self.schema)

            # Count rows
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = result.scalar()

            # Build the result
            info = {
                "name": table_name,
                "schema": self.schema,
                "columns": [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True),
                        "default": col.get("default"),
                    }
                    for col in columns
                ],
                "primary_keys": primary_keys,
                "foreign_keys": [
                    {
                        "constrained_columns": fk["constrained_columns"],
                        "referred_table": fk["referred_table"],
                        "referred_columns": fk["referred_columns"],
                    }
                    for fk in foreign_keys
                ],
                "indexes": [
                    {
                        "name": idx["name"],
                        "columns": idx["column_names"],
                        "unique": idx["unique"],
                    }
                    for idx in indexes
                ],
                "row_count": row_count,
            }

            logger.info(f"Got information for table '{table_name}'")
            self._record_operation_metadata("get_table_info", table_name=table_name)

            return info

        except Exception as e:
            self._handle_error(f"getting information for table '{table_name}'", e)
            return {}

    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        return_dataframe: bool = False,
        **kwargs
    ) -> Any:
        """Execute a custom query on the SQL database.

        Args:
            query: The query to execute.
            params: Parameters for the query.
            return_dataframe: Whether to return the result as a DataFrame.
            **kwargs: Additional keyword arguments for pandas.read_sql().

        Returns:
            Any: The result of the query. If return_dataframe is True, a DataFrame.
                Otherwise, the raw result from SQLAlchemy.

        Raises:
            StorageError: If query execution fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Execute the query
            if return_dataframe:
                result = pd.read_sql(query, self.engine, params=params, **kwargs)
            else:
                with self.engine.connect() as conn:
                    result = conn.execute(text(query), params or {})
                    if result.returns_rows:
                        result = result.fetchall()

            logger.info(f"Executed custom query: {query[:100]}...")
            self._record_operation_metadata(
                "execute_query",
                query=query[:100],
                return_dataframe=return_dataframe
            )

            return result

        except Exception as e:
            self._handle_error(f"executing query: {query[:100]}...", e)
            return None

    def create_table(
        self,
        table_name: str,
        columns: Dict[str, Any],
        primary_key: Optional[Union[str, List[str]]] = None,
        if_not_exists: bool = True,
        **kwargs
    ) -> bool:
        """Create a table in the SQL database.

        Args:
            table_name: The name of the table to create.
            columns: Dictionary mapping column names to SQLAlchemy types.
            primary_key: Column name(s) to use as primary key.
            if_not_exists: Whether to add IF NOT EXISTS to the query.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            StorageError: If table creation fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Create a metadata object
            metadata = sqlalchemy.MetaData()

            # Define the table
            table_args = {}
            if self.schema:
                table_args["schema"] = self.schema

            # Convert primary_key to a list if it's a string
            if isinstance(primary_key, str):
                primary_key = [primary_key]

            # Create the table object
            table = sqlalchemy.Table(
                table_name,
                metadata,
                *[
                    sqlalchemy.Column(
                        name,
                        type_,
                        primary_key=(name in primary_key) if primary_key else False,
                    )
                    for name, type_ in columns.items()
                ],
                **table_args
            )

            # Create the table
            if if_not_exists:
                table.create(self.engine, checkfirst=True)
            else:
                table.create(self.engine)

            logger.info(f"Created table '{table_name}'")
            self._record_operation_metadata(
                "create_table",
                table_name=table_name,
                column_count=len(columns),
                if_not_exists=if_not_exists
            )

            return True

        except Exception as e:
            self._handle_error(f"creating table '{table_name}'", e)
            return False

    def drop_table(
        self,
        table_name: str,
        if_exists: bool = True,
        **kwargs
    ) -> bool:
        """Drop a table from the SQL database.

        Args:
            table_name: The name of the table to drop.
            if_exists: Whether to add IF EXISTS to the query.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            StorageError: If table dropping fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Create a metadata object
            metadata = sqlalchemy.MetaData()

            # Define the table
            table_args = {}
            if self.schema:
                table_args["schema"] = self.schema

            # Create the table object
            table = sqlalchemy.Table(
                table_name,
                metadata,
                **table_args
            )

            # Drop the table
            if if_exists:
                table.drop(self.engine, checkfirst=True)
            else:
                table.drop(self.engine)

            logger.info(f"Dropped table '{table_name}'")
            self._record_operation_metadata(
                "drop_table",
                table_name=table_name,
                if_exists=if_exists
            )

            return True

        except Exception as e:
            self._handle_error(f"dropping table '{table_name}'", e)
            return False

    def backup_table(
        self,
        table_name: str,
        backup_table_name: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Backup a table in the SQL database.

        Args:
            table_name: The name of the table to backup.
            backup_table_name: The name of the backup table. If None, a name will be
                generated based on the original table name and the current timestamp.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            bool: True if successful, False otherwise.

        Raises:
            StorageError: If backup fails.
        """
        try:
            if not self.is_connected():
                self.connect()

            # Generate a backup table name if not provided
            if not backup_table_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_table_name = f"{table_name}_backup_{timestamp}"

            # Check if the source table exists
            if not self.table_exists(table_name):
                raise ValueError(f"Table '{table_name}' does not exist")

            # Create the backup table
            query = f"CREATE TABLE {backup_table_name} AS SELECT * FROM {table_name}"
            self.execute_query(query)

            logger.info(f"Backed up table '{table_name}' to '{backup_table_name}'")
            self._record_operation_metadata(
                "backup_table",
                table_name=table_name,
                backup_table_name=backup_table_name
            )

            return True

        except Exception as e:
            self._handle_error(f"backing up table '{table_name}'", e)
            return False