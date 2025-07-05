"""Base model for database entities in the Friday AI Trading System.

This module provides a base model class for all database entities.
"""

import datetime
from typing import Any, Dict, List, Optional, Union, Type, Tuple, TypeVar

from sqlalchemy import Column, DateTime, Integer, String, inspect, text
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import Session

from src.infrastructure.database import Base, get_session
from src.infrastructure.logging import get_logger
from src.infrastructure.utils import generate_uuid

# Create logger
logger = get_logger(__name__)

# Type variable for the model class
T = TypeVar("T", bound="BaseModel")


class BaseModel(Base):
    """Base model for all database entities.

    This class provides common fields and methods for all database entities.
    """

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False, default=generate_uuid)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    @declared_attr
    def __tablename__(cls) -> str:
        """Get the table name for the model.

        Returns:
            str: The table name.
        """
        return cls.__name__.lower()

    @classmethod
    def create(cls: Type[T], **kwargs: Any) -> T:
        """Create a new instance of the model and save it to the database.

        Args:
            **kwargs: The model attributes.

        Returns:
            T: The created model instance.
        """
        instance = cls(**kwargs)
        with get_session() as session:
            session.add(instance)
            session.commit()
            session.refresh(instance)
            logger.debug("Created %s with ID %s", cls.__name__, instance.id)
        return instance

    @classmethod
    def get_by_id(cls: Type[T], id: int) -> Optional[T]:
        """Get a model instance by ID.

        Args:
            id: The ID of the model instance.

        Returns:
            Optional[T]: The model instance, or None if not found.
        """
        with get_session() as session:
            instance = session.query(cls).filter(cls.id == id).first()
            if instance is None:
                logger.debug("%s with ID %s not found", cls.__name__, id)
            return instance

    @classmethod
    def get_by_uuid(cls: Type[T], uuid: str) -> Optional[T]:
        """Get a model instance by UUID.

        Args:
            uuid: The UUID of the model instance.

        Returns:
            Optional[T]: The model instance, or None if not found.
        """
        with get_session() as session:
            instance = session.query(cls).filter(cls.uuid == uuid).first()
            if instance is None:
                logger.debug("%s with UUID %s not found", cls.__name__, uuid)
            return instance

    @classmethod
    def get_all(cls: Type[T]) -> List[T]:
        """Get all model instances.

        Returns:
            List[T]: A list of all model instances.
        """
        with get_session() as session:
            instances = session.query(cls).all()
            logger.debug("Retrieved %s %s instances", len(instances), cls.__name__)
            return instances

    @classmethod
    def count(cls: Type[T]) -> int:
        """Get the count of model instances.

        Returns:
            int: The count of model instances.
        """
        with get_session() as session:
            count = session.query(cls).count()
            logger.debug("Counted %s %s instances", count, cls.__name__)
            return count

    def update(self: T, **kwargs: Any) -> T:
        """Update the model instance with the given attributes.

        Args:
            **kwargs: The model attributes to update.

        Returns:
            T: The updated model instance.
        """
        with get_session() as session:
            session.add(self)
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            session.commit()
            session.refresh(self)
            logger.debug("Updated %s with ID %s", self.__class__.__name__, self.id)
        return self

    def delete(self: T) -> None:
        """Delete the model instance from the database.

        Returns:
            None
        """
        with get_session() as session:
            session.delete(self)
            session.commit()
            logger.debug("Deleted %s with ID %s", self.__class__.__name__, self.id)

    def to_dict(self: T) -> Dict[str, Any]:
        """Convert the model instance to a dictionary.

        Returns:
            Dict[str, Any]: The model instance as a dictionary.
        """
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create a model instance from a dictionary.

        Args:
            data: The dictionary containing the model attributes.

        Returns:
            T: The created model instance.
        """
        return cls(**data)

    def save(self: T) -> T:
        """Save the model instance to the database.

        Returns:
            T: The saved model instance.
        """
        with get_session() as session:
            session.add(self)
            session.commit()
            session.refresh(self)
            logger.debug("Saved %s with ID %s", self.__class__.__name__, self.id)
        return self

    @classmethod
    def bulk_create(cls: Type[T], items: List[Dict[str, Any]]) -> List[T]:
        """Create multiple model instances in bulk.

        Args:
            items: A list of dictionaries containing the model attributes.

        Returns:
            List[T]: The created model instances.
        """
        instances = [cls(**item) for item in items]
        with get_session() as session:
            session.add_all(instances)
            session.commit()
            for instance in instances:
                session.refresh(instance)
            logger.debug(
                "Bulk created %s %s instances", len(instances), cls.__name__
            )
        return instances

    @classmethod
    def bulk_update(cls: Type[T], items: List[Dict[str, Any]]) -> List[T]:
        """Update multiple model instances in bulk.

        Args:
            items: A list of dictionaries containing the model attributes.
                Each dictionary must contain an 'id' or 'uuid' key.

        Returns:
            List[T]: The updated model instances.

        Raises:
            ValueError: If an item does not contain an 'id' or 'uuid' key.
        """
        instances: List[T] = []
        with get_session() as session:
            for item in items:
                if "id" in item:
                    instance = session.query(cls).filter(cls.id == item["id"]).first()
                elif "uuid" in item:
                    instance = (
                        session.query(cls).filter(cls.uuid == item["uuid"]).first()
                    )
                else:
                    raise ValueError(
                        f"Item does not contain an 'id' or 'uuid' key: {item}"
                    )

                if instance is not None:
                    for key, value in item.items():
                        if hasattr(instance, key):
                            setattr(instance, key, value)
                    instances.append(instance)

            session.commit()
            for instance in instances:
                session.refresh(instance)
            logger.debug(
                "Bulk updated %s %s instances", len(instances), cls.__name__
            )
        return instances

    @classmethod
    def bulk_delete(cls: Type[T], ids: List[int]) -> int:
        """Delete multiple model instances in bulk.

        Args:
            ids: A list of model instance IDs.

        Returns:
            int: The number of deleted model instances.
        """
        with get_session() as session:
            count = session.query(cls).filter(cls.id.in_(ids)).delete()
            session.commit()
            logger.debug("Bulk deleted %s %s instances", count, cls.__name__)
        return count

    @classmethod
    def bulk_delete_by_uuid(cls: Type[T], uuids: List[str]) -> int:
        """Delete multiple model instances in bulk by UUID.

        Args:
            uuids: A list of model instance UUIDs.

        Returns:
            int: The number of deleted model instances.
        """
        with get_session() as session:
            count = session.query(cls).filter(cls.uuid.in_(uuids)).delete()
            session.commit()
            logger.debug("Bulk deleted %s %s instances by UUID", count, cls.__name__)
        return count

    @classmethod
    def get_or_create(cls: Type[T], defaults: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Tuple[T, bool]:
        """Get or create a model instance.

        Args:
            defaults: Default values to use when creating the model instance.
            **kwargs: The model attributes to filter by.

        Returns:
            Tuple[T, bool]: A tuple containing the model instance and a boolean
                indicating whether the instance was created.
        """
        if defaults is None:
            defaults = {}

        with get_session() as session:
            instance = session.query(cls).filter_by(**kwargs).first()
            if instance is not None:
                created = False
                logger.debug(
                    "Retrieved existing %s with ID %s", cls.__name__, instance.id
                )
            else:
                instance = cls(**{**kwargs, **defaults})
                session.add(instance)
                session.commit()
                session.refresh(instance)
                created = True
                logger.debug("Created %s with ID %s", cls.__name__, instance.id)
        return instance, created

    @classmethod
    def update_or_create(cls: Type[T], defaults: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Tuple[T, bool]:
        """Update or create a model instance.

        Args:
            defaults: Default values to use when creating the model instance.
            **kwargs: The model attributes to filter by.

        Returns:
            Tuple[T, bool]: A tuple containing the model instance and a boolean
                indicating whether the instance was created.
        """
        if defaults is None:
            defaults = {}

        with get_session() as session:
            instance = session.query(cls).filter_by(**kwargs).first()
            if instance is not None:
                created = False
                for key, value in defaults.items():
                    if hasattr(instance, key):
                        setattr(instance, key, value)
                logger.debug(
                    "Updated existing %s with ID %s", cls.__name__, instance.id
                )
            else:
                instance = cls(**{**kwargs, **defaults})
                session.add(instance)
                created = True
                logger.debug("Created %s with ID %s", cls.__name__, instance.id)
            session.commit()
            session.refresh(instance)
        return instance, created

    @classmethod
    def filter(cls: Type[T], **kwargs: Any) -> List[T]:
        """Filter model instances by the given attributes.

        Args:
            **kwargs: The model attributes to filter by.

        Returns:
            List[T]: The filtered model instances.
        """
        with get_session() as session:
            instances = session.query(cls).filter_by(**kwargs).all()
            logger.debug(
                "Filtered %s %s instances by %s",
                len(instances),
                cls.__name__,
                kwargs,
            )
            return instances

    @classmethod
    def filter_first(cls: Type[T], **kwargs: Any) -> Optional[T]:
        """Filter model instances by the given attributes and return the first one.

        Args:
            **kwargs: The model attributes to filter by.

        Returns:
            Optional[T]: The first filtered model instance, or None if not found.
        """
        with get_session() as session:
            instance = session.query(cls).filter_by(**kwargs).first()
            if instance is None:
                logger.debug(
                    "%s with attributes %s not found", cls.__name__, kwargs
                )
            return instance

    @classmethod
    def exists(cls: Type[T], **kwargs: Any) -> bool:
        """Check if a model instance with the given attributes exists.

        Args:
            **kwargs: The model attributes to filter by.

        Returns:
            bool: True if a model instance with the given attributes exists,
                False otherwise.
        """
        with get_session() as session:
            exists = session.query(session.query(cls).filter_by(**kwargs).exists()).scalar()
            logger.debug(
                "%s with attributes %s %s",
                cls.__name__,
                kwargs,
                "exists" if exists else "does not exist",
            )
            return exists

    @classmethod
    def get_or_none(cls: Type[T], **kwargs: Any) -> Optional[T]:
        """Get a model instance by the given attributes, or None if not found.

        Args:
            **kwargs: The model attributes to filter by.

        Returns:
            Optional[T]: The model instance, or None if not found.
        """
        with get_session() as session:
            instance = session.query(cls).filter_by(**kwargs).first()
            if instance is None:
                logger.debug(
                    "%s with attributes %s not found", cls.__name__, kwargs
                )
            return instance

    @classmethod
    def get_in_bulk(cls: Type[T], ids: List[int]) -> Dict[int, T]:
        """Get multiple model instances by ID.

        Args:
            ids: A list of model instance IDs.

        Returns:
            Dict[int, T]: A dictionary mapping IDs to model instances.
        """
        with get_session() as session:
            instances = session.query(cls).filter(cls.id.in_(ids)).all()
            result = {instance.id: instance for instance in instances}
            logger.debug(
                "Retrieved %s %s instances by ID", len(result), cls.__name__
            )
            return result

    @classmethod
    def get_in_bulk_by_uuid(cls: Type[T], uuids: List[str]) -> Dict[str, T]:
        """Get multiple model instances by UUID.

        Args:
            uuids: A list of model instance UUIDs.

        Returns:
            Dict[str, T]: A dictionary mapping UUIDs to model instances.
        """
        with get_session() as session:
            instances = session.query(cls).filter(cls.uuid.in_(uuids)).all()
            result = {instance.uuid: instance for instance in instances}
            logger.debug(
                "Retrieved %s %s instances by UUID", len(result), cls.__name__
            )
            return result

    @classmethod
    def get_latest(cls: Type[T], limit: int = 10) -> List[T]:
        """Get the latest model instances.

        Args:
            limit: The maximum number of instances to return.

        Returns:
            List[T]: The latest model instances.
        """
        with get_session() as session:
            instances = (
                session.query(cls).order_by(cls.created_at.desc()).limit(limit).all()
            )
            logger.debug("Retrieved %s latest %s instances", len(instances), cls.__name__)
            return instances

    @classmethod
    def get_oldest(cls: Type[T], limit: int = 10) -> List[T]:
        """Get the oldest model instances.

        Args:
            limit: The maximum number of instances to return.

        Returns:
            List[T]: The oldest model instances.
        """
        with get_session() as session:
            instances = (
                session.query(cls).order_by(cls.created_at.asc()).limit(limit).all()
            )
            logger.debug("Retrieved %s oldest %s instances", len(instances), cls.__name__)
            return instances

    @classmethod
    def get_random(cls: Type[T], limit: int = 10) -> List[T]:
        """Get random model instances.

        Args:
            limit: The maximum number of instances to return.

        Returns:
            List[T]: Random model instances.
        """
        with get_session() as session:
            # SQLAlchemy doesn't have a built-in random function that works across all databases
            # This implementation uses the random function of the database
            instances = session.query(cls).order_by(text("RANDOM()")).limit(limit).all()
            logger.debug("Retrieved %s random %s instances", len(instances), cls.__name__)
            return instances

    @classmethod
    def truncate(cls: Type[T]) -> None:
        """Truncate the model table.

        Returns:
            None
        """
        with get_session() as session:
            session.query(cls).delete()
            session.commit()
            logger.debug("Truncated %s table", cls.__name__)

    @classmethod
    def get_by_created_at_range(
        cls: Type[T], start_date: datetime.datetime, end_date: datetime.datetime
    ) -> List[T]:
        """Get model instances created within a date range.

        Args:
            start_date: The start date.
            end_date: The end date.

        Returns:
            List[T]: The model instances created within the date range.
        """
        with get_session() as session:
            instances = (
                session.query(cls)
                .filter(cls.created_at >= start_date, cls.created_at <= end_date)
                .all()
            )
            logger.debug(
                "Retrieved %s %s instances created between %s and %s",
                len(instances),
                cls.__name__,
                start_date,
                end_date,
            )
            return instances

    @classmethod
    def get_by_updated_at_range(
        cls: Type[T], start_date: datetime.datetime, end_date: datetime.datetime
    ) -> List[T]:
        """Get model instances updated within a date range.

        Args:
            start_date: The start date.
            end_date: The end date.

        Returns:
            List[T]: The model instances updated within the date range.
        """
        with get_session() as session:
            instances = (
                session.query(cls)
                .filter(cls.updated_at >= start_date, cls.updated_at <= end_date)
                .all()
            )
            logger.debug(
                "Retrieved %s %s instances updated between %s and %s",
                len(instances),
                cls.__name__,
                start_date,
                end_date,
            )
            return instances