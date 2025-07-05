"""Database models for model registry.

This module provides database models for storing model metadata in the database.
"""

import json
import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

from sqlalchemy import Column, String, DateTime, Text, Boolean, ForeignKey, Integer, Float
from sqlalchemy.orm import relationship

from src.infrastructure.database.base_model import BaseModel
from src.infrastructure.logging import get_logger

# Create logger
logger = get_logger(__name__)


class ModelMetadata(BaseModel):
    """Database model for storing model metadata.

    This class provides a database model for storing model metadata in the database.

    Attributes:
        model_id: The model ID.
        model_name: The model name.
        model_type: The model type.
        version: The model version.
        description: The model description.
        location: The model file location.
        status: The model status.
        created_at: The creation timestamp.
        updated_at: The last update timestamp.
        metadata_json: The model metadata as JSON.
        metrics_json: The model metrics as JSON.
        tags_json: The model tags as JSON.
    """

    __tablename__ = "model_metadata"

    model_id = Column(String(255), nullable=False, unique=True, index=True)
    model_name = Column(String(255), nullable=False, index=True)
    model_type = Column(String(255), nullable=False, index=True)
    version = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    location = Column(String(1024), nullable=False)
    status = Column(String(50), nullable=False, default="active")
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, 
                       onupdate=datetime.datetime.utcnow)
    metadata_json = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    tags_json = Column(Text, nullable=True)

    @classmethod
    def create_from_registry_entry(cls, model_id: str, model_name: str, model_type: str, 
                                 version: str, location: str, metadata: Dict[str, Any], 
                                 metrics: Dict[str, float], tags: List[str] = None, 
                                 description: str = None, status: str = "active") -> "ModelMetadata":
        """Create a model metadata entry from registry data.

        Args:
            model_id: The model ID.
            model_name: The model name.
            model_type: The model type.
            version: The model version.
            location: The model file location.
            metadata: The model metadata.
            metrics: The model metrics.
            tags: The model tags.
            description: The model description.
            status: The model status.

        Returns:
            ModelMetadata: The created model metadata entry.
        """
        return cls.create(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            version=version,
            description=description,
            location=location,
            status=status,
            metadata_json=json.dumps(metadata) if metadata else None,
            metrics_json=json.dumps(metrics) if metrics else None,
            tags_json=json.dumps(tags) if tags else None
        )

    def to_registry_dict(self) -> Dict[str, Any]:
        """Convert the model metadata to a registry dictionary.

        Returns:
            Dict[str, Any]: The registry dictionary.
        """
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "description": self.description,
            "location": self.location,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": json.loads(self.metadata_json) if self.metadata_json else {},
            "metrics": json.loads(self.metrics_json) if self.metrics_json else {},
            "tags": json.loads(self.tags_json) if self.tags_json else []
        }

    @classmethod
    def get_by_model_id(cls, model_id: str) -> Optional["ModelMetadata"]:
        """Get a model metadata entry by model ID.

        Args:
            model_id: The model ID.

        Returns:
            Optional[ModelMetadata]: The model metadata entry, or None if not found.
        """
        return cls.query.filter_by(model_id=model_id).first()

    @classmethod
    def get_by_model_name_and_version(cls, model_name: str, version: str) -> Optional["ModelMetadata"]:
        """Get a model metadata entry by model name and version.

        Args:
            model_name: The model name.
            version: The model version.

        Returns:
            Optional[ModelMetadata]: The model metadata entry, or None if not found.
        """
        return cls.query.filter_by(model_name=model_name, version=version).first()

    @classmethod
    def get_latest_version(cls, model_name: str) -> Optional["ModelMetadata"]:
        """Get the latest version of a model.

        Args:
            model_name: The model name.

        Returns:
            Optional[ModelMetadata]: The model metadata entry, or None if not found.
        """
        return cls.query.filter_by(model_name=model_name).order_by(cls.created_at.desc()).first()

    @classmethod
    def get_all_versions(cls, model_name: str) -> List["ModelMetadata"]:
        """Get all versions of a model.

        Args:
            model_name: The model name.

        Returns:
            List[ModelMetadata]: The model metadata entries.
        """
        return cls.query.filter_by(model_name=model_name).order_by(cls.created_at.desc()).all()

    @classmethod
    def get_all_models(cls) -> List["ModelMetadata"]:
        """Get all models.

        Returns:
            List[ModelMetadata]: The model metadata entries.
        """
        return cls.query.order_by(cls.model_name, cls.created_at.desc()).all()

    @classmethod
    def get_models_by_type(cls, model_type: str) -> List["ModelMetadata"]:
        """Get all models of a specific type.

        Args:
            model_type: The model type.

        Returns:
            List[ModelMetadata]: The model metadata entries.
        """
        return cls.query.filter_by(model_type=model_type).order_by(cls.model_name, cls.created_at.desc()).all()

    @classmethod
    def get_models_by_tag(cls, tag: str) -> List["ModelMetadata"]:
        """Get all models with a specific tag.

        Args:
            tag: The tag.

        Returns:
            List[ModelMetadata]: The model metadata entries.
        """
        # This is a simple implementation that loads all models and filters in Python
        # A more efficient implementation would use a database-specific JSON query
        all_models = cls.get_all_models()
        return [model for model in all_models 
                if model.tags_json and tag in json.loads(model.tags_json)]

    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update the model metadata.

        Args:
            metadata: The new metadata.
        """
        self.metadata_json = json.dumps(metadata)
        self.updated_at = datetime.datetime.utcnow()
        self.save()

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update the model metrics.

        Args:
            metrics: The new metrics.
        """
        self.metrics_json = json.dumps(metrics)
        self.updated_at = datetime.datetime.utcnow()
        self.save()

    def update_tags(self, tags: List[str]) -> None:
        """Update the model tags.

        Args:
            tags: The new tags.
        """
        self.tags_json = json.dumps(tags)
        self.updated_at = datetime.datetime.utcnow()
        self.save()

    def update_status(self, status: str) -> None:
        """Update the model status.

        Args:
            status: The new status.
        """
        self.status = status
        self.updated_at = datetime.datetime.utcnow()
        self.save()


class ModelTag(BaseModel):
    """Database model for storing model tags.

    This class provides a database model for storing model tags in the database.

    Attributes:
        tag_name: The tag name.
        description: The tag description.
    """

    __tablename__ = "model_tags"

    tag_name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)

    @classmethod
    def get_or_create(cls, tag_name: str, description: str = None) -> "ModelTag":
        """Get or create a tag.

        Args:
            tag_name: The tag name.
            description: The tag description.

        Returns:
            ModelTag: The tag.
        """
        tag = cls.query.filter_by(tag_name=tag_name).first()
        if tag is None:
            tag = cls.create(tag_name=tag_name, description=description)
        return tag


class ModelMetric(BaseModel):
    """Database model for storing model metrics history.

    This class provides a database model for storing model metrics history in the database.

    Attributes:
        model_id: The model ID.
        metric_name: The metric name.
        metric_value: The metric value.
        timestamp: The timestamp.
    """

    __tablename__ = "model_metrics"

    model_id = Column(String(255), nullable=False, index=True)
    metric_name = Column(String(255), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)

    @classmethod
    def record_metrics(cls, model_id: str, metrics: Dict[str, float]) -> List["ModelMetric"]:
        """Record metrics for a model.

        Args:
            model_id: The model ID.
            metrics: The metrics.

        Returns:
            List[ModelMetric]: The created metric entries.
        """
        timestamp = datetime.datetime.utcnow()
        metric_entries = []
        
        for metric_name, metric_value in metrics.items():
            metric_entry = cls.create(
                model_id=model_id,
                metric_name=metric_name,
                metric_value=metric_value,
                timestamp=timestamp
            )
            metric_entries.append(metric_entry)
        
        return metric_entries

    @classmethod
    def get_metric_history(cls, model_id: str, metric_name: str) -> List["ModelMetric"]:
        """Get the history of a metric for a model.

        Args:
            model_id: The model ID.
            metric_name: The metric name.

        Returns:
            List[ModelMetric]: The metric entries.
        """
        return cls.query.filter_by(model_id=model_id, metric_name=metric_name)\
            .order_by(cls.timestamp).all()

    @classmethod
    def get_latest_metrics(cls, model_id: str) -> Dict[str, float]:
        """Get the latest metrics for a model.

        Args:
            model_id: The model ID.

        Returns:
            Dict[str, float]: The metrics.
        """
        # This is a simple implementation that might not be the most efficient
        # A more efficient implementation would use a database-specific query
        metrics = {}
        for metric in cls.query.filter_by(model_id=model_id).all():
            if metric.metric_name not in metrics or \
               metrics[metric.metric_name]["timestamp"] < metric.timestamp:
                metrics[metric.metric_name] = {
                    "value": metric.metric_value,
                    "timestamp": metric.timestamp
                }
        
        return {k: v["value"] for k, v in metrics.items()}