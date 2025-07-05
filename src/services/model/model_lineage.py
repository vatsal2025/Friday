"""Model Lineage for Friday AI Trading System.

This module provides functionality for tracking model dependencies and visualizing model lineage.
"""

import os
import json
import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from dataclasses import dataclass

from src.infrastructure.logging import get_logger
from src.services.model.model_registry import ModelRegistry
from src.services.model.model_versioning import ModelVersioning
from src.orchestration.trading_engine.model_versioning import ModelVersionStatus

# Create logger
logger = get_logger(__name__)


@dataclass
class ModelDependency:
    """Represents a dependency between models.
    
    Attributes:
        source_model_id: ID of the source model.
        source_model_name: Name of the source model.
        source_model_version: Version of the source model.
        target_model_id: ID of the target model.
        target_model_name: Name of the target model.
        target_model_version: Version of the target model.
        dependency_type: Type of dependency (e.g., 'feature', 'ensemble', 'pipeline').
        description: Description of the dependency.
    """
    source_model_id: str
    source_model_name: str
    source_model_version: str
    target_model_id: str
    target_model_name: str
    target_model_version: str
    dependency_type: str
    description: Optional[str] = None


class ModelLineage:
    """Model Lineage for tracking and visualizing model dependencies.

    This class provides functionality for tracking model dependencies and visualizing model lineage.

    Attributes:
        registry: The model registry instance.
        versioning: The model versioning instance.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None, versioning: Optional[ModelVersioning] = None):
        """Initialize the model lineage.

        Args:
            registry: The model registry. If None, a new one will be created.
            versioning: The model versioning. If None, a new one will be created.
        """
        self.registry = registry or ModelRegistry()
        self.versioning = versioning or ModelVersioning(registry=self.registry)
        logger.info("Initialized ModelLineage")

    def register_model_dependency(
        self,
        source_model_id: str,
        target_model_id: str,
        dependency_type: str,
        description: Optional[str] = None,
    ) -> None:
        """Register a dependency between two models.

        Args:
            source_model_id: ID of the source model.
            target_model_id: ID of the target model.
            dependency_type: Type of dependency (e.g., 'feature', 'ensemble', 'pipeline').
            description: Description of the dependency.

        Raises:
            ValueError: If either model is not found in the registry.
        """
        # Get model metadata
        try:
            source_metadata = self.registry.get_model_metadata(source_model_id)
            target_metadata = self.registry.get_model_metadata(target_model_id)
        except ValueError as e:
            logger.error(f"Error getting model metadata: {e}")
            raise

        # Create dependency object
        dependency = ModelDependency(
            source_model_id=source_model_id,
            source_model_name=source_metadata["model_name"],
            source_model_version=source_metadata["version"],
            target_model_id=target_model_id,
            target_model_name=target_metadata["model_name"],
            target_model_version=target_metadata["version"],
            dependency_type=dependency_type,
            description=description,
        )

        # Update source model metadata with dependency
        if "dependencies" not in source_metadata:
            source_metadata["dependencies"] = {}
        if "dependent_models" not in source_metadata["dependencies"]:
            source_metadata["dependencies"]["dependent_models"] = []

        # Add dependency to source model
        source_metadata["dependencies"]["dependent_models"].append({
            "model_id": target_model_id,
            "model_name": target_metadata["model_name"],
            "model_version": target_metadata["version"],
            "dependency_type": dependency_type,
            "description": description,
        })

        # Update target model metadata with dependency
        if "dependencies" not in target_metadata:
            target_metadata["dependencies"] = {}
        if "dependency_models" not in target_metadata["dependencies"]:
            target_metadata["dependencies"]["dependency_models"] = []

        # Add dependency to target model
        target_metadata["dependencies"]["dependency_models"].append({
            "model_id": source_model_id,
            "model_name": source_metadata["model_name"],
            "model_version": source_metadata["version"],
            "dependency_type": dependency_type,
            "description": description,
        })

        # Update model metadata in registry
        self.registry.update_model_metadata(source_model_id, source_metadata)
        self.registry.update_model_metadata(target_model_id, target_metadata)

        logger.info(f"Registered dependency between models {source_model_id} and {target_model_id}")

    def get_model_dependencies(
        self,
        model_id: str,
        include_indirect: bool = False,
        max_depth: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get dependencies for a model.

        Args:
            model_id: ID of the model.
            include_indirect: Whether to include indirect dependencies.
            max_depth: Maximum depth for indirect dependencies.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary with upstream and downstream dependencies.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        # Get model metadata
        try:
            model_metadata = self.registry.get_model_metadata(model_id)
        except ValueError as e:
            logger.error(f"Error getting model metadata: {e}")
            raise

        # Initialize result
        result = {
            "upstream": [],  # Models this model depends on
            "downstream": [],  # Models that depend on this model
        }

        # Get direct dependencies
        if "dependencies" in model_metadata:
            # Upstream dependencies (models this model depends on)
            if "dependency_models" in model_metadata["dependencies"]:
                result["upstream"] = model_metadata["dependencies"]["dependency_models"]

            # Downstream dependencies (models that depend on this model)
            if "dependent_models" in model_metadata["dependencies"]:
                result["downstream"] = model_metadata["dependencies"]["dependent_models"]

        # Include indirect dependencies if requested
        if include_indirect and max_depth > 1:
            # Get indirect upstream dependencies
            visited_upstream = set([model_id])
            upstream_queue = [(dep["model_id"], 1) for dep in result["upstream"]]
            
            while upstream_queue:
                current_id, depth = upstream_queue.pop(0)
                if current_id in visited_upstream or depth >= max_depth:
                    continue
                
                visited_upstream.add(current_id)
                
                try:
                    current_metadata = self.registry.get_model_metadata(current_id)
                    if "dependencies" in current_metadata and "dependency_models" in current_metadata["dependencies"]:
                        for dep in current_metadata["dependencies"]["dependency_models"]:
                            if dep["model_id"] not in visited_upstream:
                                result["upstream"].append({
                                    **dep,
                                    "indirect": True,
                                    "depth": depth + 1,
                                })
                                upstream_queue.append((dep["model_id"], depth + 1))
                except ValueError:
                    continue
            
            # Get indirect downstream dependencies
            visited_downstream = set([model_id])
            downstream_queue = [(dep["model_id"], 1) for dep in result["downstream"]]
            
            while downstream_queue:
                current_id, depth = downstream_queue.pop(0)
                if current_id in visited_downstream or depth >= max_depth:
                    continue
                
                visited_downstream.add(current_id)
                
                try:
                    current_metadata = self.registry.get_model_metadata(current_id)
                    if "dependencies" in current_metadata and "dependent_models" in current_metadata["dependencies"]:
                        for dep in current_metadata["dependencies"]["dependent_models"]:
                            if dep["model_id"] not in visited_downstream:
                                result["downstream"].append({
                                    **dep,
                                    "indirect": True,
                                    "depth": depth + 1,
                                })
                                downstream_queue.append((dep["model_id"], depth + 1))
                except ValueError:
                    continue

        return result

    def visualize_model_lineage(
        self,
        model_id: Optional[str] = None,
        model_name: Optional[str] = None,
        include_indirect: bool = True,
        max_depth: int = 3,
        output_format: str = "base64",
    ) -> str:
        """Visualize model lineage as a directed graph.

        Args:
            model_id: ID of the model. If None, model_name must be provided.
            model_name: Name of the model. If None, model_id must be provided.
            include_indirect: Whether to include indirect dependencies.
            max_depth: Maximum depth for indirect dependencies.
            output_format: Output format for the visualization ('base64', 'png', 'svg').

        Returns:
            str: Visualization in the specified format.

        Raises:
            ValueError: If neither model_id nor model_name is provided, or if the model is not found.
        """
        if model_id is None and model_name is None:
            raise ValueError("Either model_id or model_name must be provided")

        # Get model ID if only name is provided
        if model_id is None:
            models = self.registry.get_models()
            for m in models:
                if m["model_name"] == model_name:
                    model_id = m["model_id"]
                    break
            if model_id is None:
                raise ValueError(f"Model with name {model_name} not found")

        # Get model metadata
        model_metadata = self.registry.get_model_metadata(model_id)
        model_name = model_metadata["model_name"]
        model_version = model_metadata["version"]

        # Get dependencies
        dependencies = self.get_model_dependencies(model_id, include_indirect, max_depth)

        # Create directed graph
        G = nx.DiGraph()

        # Add central node
        central_node = f"{model_name}\n{model_version}"
        G.add_node(central_node, color="lightblue", style="filled")

        # Add upstream dependencies (models this model depends on)
        for dep in dependencies["upstream"]:
            node_name = f"{dep['model_name']}\n{dep['model_version']}"
            G.add_node(node_name, color="lightgreen", style="filled")
            G.add_edge(node_name, central_node, label=dep["dependency_type"])

        # Add downstream dependencies (models that depend on this model)
        for dep in dependencies["downstream"]:
            node_name = f"{dep['model_name']}\n{dep['model_version']}"
            G.add_node(node_name, color="salmon", style="filled")
            G.add_edge(central_node, node_name, label=dep["dependency_type"])

        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        node_colors = [G.nodes[n].get("color", "lightgray") for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
        
        # Draw edge labels
        edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f"Model Lineage for {model_name} (version {model_version})")
        plt.axis("off")
        
        # Return visualization in the specified format
        if output_format == "base64":
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            plt.close()
            return img_str
        elif output_format == "png":
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            plt.close()
            return buf.getvalue()
        elif output_format == "svg":
            buf = io.BytesIO()
            plt.savefig(buf, format="svg", bbox_inches="tight")
            buf.seek(0)
            plt.close()
            return buf.getvalue().decode("utf-8")
        else:
            plt.close()
            raise ValueError(f"Unsupported output format: {output_format}")

    def get_model_lineage_graph(self, model_id: str) -> Dict[str, Any]:
        """Get model lineage as a graph structure for custom visualization.

        Args:
            model_id: ID of the model.

        Returns:
            Dict[str, Any]: Graph structure with nodes and edges.

        Raises:
            ValueError: If the model is not found in the registry.
        """
        # Get model metadata
        model_metadata = self.registry.get_model_metadata(model_id)
        model_name = model_metadata["model_name"]
        model_version = model_metadata["version"]

        # Get dependencies
        dependencies = self.get_model_dependencies(model_id, include_indirect=True, max_depth=3)

        # Create graph structure
        nodes = []
        edges = []

        # Add central node
        central_node_id = model_id
        nodes.append({
            "id": central_node_id,
            "label": f"{model_name}\n{model_version}",
            "type": "central",
            "metadata": model_metadata,
        })

        # Add upstream dependencies (models this model depends on)
        for dep in dependencies["upstream"]:
            node_id = dep["model_id"]
            nodes.append({
                "id": node_id,
                "label": f"{dep['model_name']}\n{dep['model_version']}",
                "type": "upstream",
                "metadata": dep,
            })
            edges.append({
                "source": node_id,
                "target": central_node_id,
                "label": dep["dependency_type"],
                "metadata": dep,
            })

        # Add downstream dependencies (models that depend on this model)
        for dep in dependencies["downstream"]:
            node_id = dep["model_id"]
            nodes.append({
                "id": node_id,
                "label": f"{dep['model_name']}\n{dep['model_version']}",
                "type": "downstream",
                "metadata": dep,
            })
            edges.append({
                "source": central_node_id,
                "target": node_id,
                "label": dep["dependency_type"],
                "metadata": dep,
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "central_node": central_node_id,
        }