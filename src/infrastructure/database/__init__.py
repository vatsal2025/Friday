"""Database module for the Friday AI Trading System.

This module provides functions for database connection and operations.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from src.infrastructure.config import get_config
from src.infrastructure.logging import get_logger

# Import MongoDB functions
from .mongodb import (
    get_mongo_client, get_database, get_collection,
    insert_one, insert_many, find, find_one, find_many,
    update_one, update_many, delete_one, delete_many,
    count_documents
)

# Import Redis initialization functions
from .initialize_db import initialize_redis_structures

# Create logger
logger = get_logger(__name__)

# Create declarative base for models
Base = declarative_base()

# Global engine instance
_engine: Optional[Engine] = None

# Global session factory
_SessionFactory: Optional[sessionmaker] = None


def get_connection_string() -> str:
    """Get the database connection string based on configuration.

    Returns:
        str: The database connection string.
    """
    db_config = get_config("database")
    db_type = db_config["type"]

    if db_type == "sqlite":
        db_path = db_config["path"]
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return f"sqlite:///{db_path}"
    elif db_type == "mysql":
        host = db_config["host"]
        port = db_config["port"]
        username = db_config["username"]
        password = db_config["password"]
        database = db_config["database"]
        return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    elif db_type == "postgresql":
        host = db_config["host"]
        port = db_config["port"]
        username = db_config["username"]
        password = db_config["password"]
        database = db_config["database"]
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def get_engine() -> Engine:
    """Get the SQLAlchemy engine instance.

    Returns:
        Engine: The SQLAlchemy engine instance.
    """
    global _engine

    if _engine is None:
        connection_string = get_connection_string()
        db_config = get_config("database")
        _engine = create_engine(
            connection_string,
            pool_size=db_config["pool_size"],
            max_overflow=db_config["max_overflow"],
            pool_timeout=db_config["timeout"],
            pool_pre_ping=True,
            echo=False,
        )
        logger.info("Database engine created with connection string: %s", connection_string)

    return _engine


def get_session_factory() -> sessionmaker:
    """Get the SQLAlchemy session factory.

    Returns:
        sessionmaker: The SQLAlchemy session factory.
    """
    global _SessionFactory

    if _SessionFactory is None:
        engine = get_engine()
        _SessionFactory = sessionmaker(bind=engine)
        logger.info("Database session factory created")

    return _SessionFactory


def get_session() -> Session:
    """Get a new SQLAlchemy session.

    Returns:
        Session: A new SQLAlchemy session.
    """
    session_factory = get_session_factory()
    return session_factory()


def init_db() -> None:
    """Initialize the database by creating all tables.

    Returns:
        None
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables created")


def drop_db() -> None:
    """Drop all database tables.

    Returns:
        None
    """
    engine = get_engine()
    Base.metadata.drop_all(engine)
    logger.info("Database tables dropped")


def execute_query(
    query: str, params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Execute a raw SQL query and return the results.

    Args:
        query: The SQL query to execute.
        params: The parameters to bind to the query.

    Returns:
        List[Dict[str, Any]]: The query results as a list of dictionaries.
    """
    if params is None:
        params = {}

    with get_session() as session:
        result = session.execute(text(query), params)
        return [dict(row) for row in result]


def execute_transaction(
    queries: List[Tuple[str, Optional[Dict[str, Any]]]]
) -> None:
    """Execute multiple SQL queries in a transaction.

    Args:
        queries: A list of tuples containing the SQL query and parameters.

    Returns:
        None
    """
    with get_session() as session:
        try:
            for query, params in queries:
                session.execute(text(query), params or {})
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Transaction failed: %s", str(e))
            raise


def get_table_names() -> List[str]:
    """Get a list of all table names in the database.

    Returns:
        List[str]: A list of table names.
    """
    engine = get_engine()
    inspector = engine.dialect.inspector
    return inspector.get_table_names()


def get_table_columns(table_name: str) -> List[Dict[str, Any]]:
    """Get a list of columns for a table.

    Args:
        table_name: The name of the table.

    Returns:
        List[Dict[str, Any]]: A list of column information dictionaries.
    """
    engine = get_engine()
    inspector = engine.dialect.inspector
    return inspector.get_columns(table_name)


def get_table_primary_keys(table_name: str) -> List[str]:
    """Get a list of primary key column names for a table.

    Args:
        table_name: The name of the table.

    Returns:
        List[str]: A list of primary key column names.
    """
    engine = get_engine()
    inspector = engine.dialect.inspector
    return inspector.get_primary_keys(table_name)


def get_table_foreign_keys(table_name: str) -> List[Dict[str, Any]]:
    """Get a list of foreign keys for a table.

    Args:
        table_name: The name of the table.

    Returns:
        List[Dict[str, Any]]: A list of foreign key information dictionaries.
    """
    engine = get_engine()
    inspector = engine.dialect.inspector
    return inspector.get_foreign_keys(table_name)


def get_table_indexes(table_name: str) -> List[Dict[str, Any]]:
    """Get a list of indexes for a table.

    Args:
        table_name: The name of the table.

    Returns:
        List[Dict[str, Any]]: A list of index information dictionaries.
    """
    engine = get_engine()
    inspector = engine.dialect.inspector
    return inspector.get_indexes(table_name)


def get_table_row_count(table_name: str) -> int:
    """Get the number of rows in a table.

    Args:
        table_name: The name of the table.

    Returns:
        int: The number of rows in the table.
    """
    with get_session() as session:
        result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        return result.scalar()


def backup_database(backup_path: Optional[str] = None) -> str:
    """Backup the database to a file.

    Args:
        backup_path: The path to save the backup file. If None, a default path will be used.

    Returns:
        str: The path to the backup file.

    Raises:
        NotImplementedError: If the database type is not supported for backup.
    """
    db_config = get_config("database")
    db_type = db_config["type"]

    if backup_path is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(os.path.dirname(db_config["path"]), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"friday_{timestamp}.db")

    if db_type == "sqlite":
        import shutil
        shutil.copy2(db_config["path"], backup_path)
        logger.info("Database backed up to %s", backup_path)
        return backup_path
    else:
        raise NotImplementedError(f"Backup not implemented for database type: {db_type}")


def restore_database(backup_path: str) -> None:
    """Restore the database from a backup file.

    Args:
        backup_path: The path to the backup file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the backup file does not exist.
        NotImplementedError: If the database type is not supported for restore.
    """
    if not os.path.exists(backup_path):
        raise FileNotFoundError(f"Backup file not found: {backup_path}")

    db_config = get_config("database")
    db_type = db_config["type"]

    if db_type == "sqlite":
        import shutil
        # Close any existing connections
        global _engine, _SessionFactory
        if _engine is not None:
            _engine.dispose()
            _engine = None
        _SessionFactory = None

        # Restore the database
        shutil.copy2(backup_path, db_config["path"])
        logger.info("Database restored from %s", backup_path)
    else:
        raise NotImplementedError(f"Restore not implemented for database type: {db_type}")