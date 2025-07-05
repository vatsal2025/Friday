"""API documentation generation for external system integration.

This module provides utilities for generating API documentation for external system integrations.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type
import json
import yaml
import logging
import os
from pathlib import Path

from src.infrastructure.logging import get_logger
from src.infrastructure.error import FridayError

# Create logger
logger = get_logger(__name__)


class ApiDocError(FridayError):
    """Exception raised for errors in API documentation generation."""
    pass


class ApiEndpoint:
    """Class representing an API endpoint."""
    
    def __init__(self, name: str, path: str, method: str, description: str, 
                 parameters: Optional[List[Dict[str, Any]]] = None,
                 request_body: Optional[Dict[str, Any]] = None,
                 responses: Optional[Dict[str, Dict[str, Any]]] = None,
                 auth_required: bool = True,
                 tags: Optional[List[str]] = None):
        """Initialize an API endpoint.
        
        Args:
            name: The name of the endpoint.
            path: The path of the endpoint.
            method: The HTTP method of the endpoint.
            description: The description of the endpoint.
            parameters: The parameters of the endpoint.
            request_body: The request body of the endpoint.
            responses: The responses of the endpoint.
            auth_required: Whether authentication is required for the endpoint.
            tags: Tags for the endpoint.
        """
        self.name = name
        self.path = path
        self.method = method.upper()
        self.description = description
        self.parameters = parameters or []
        self.request_body = request_body
        self.responses = responses or {}
        self.auth_required = auth_required
        self.tags = tags or []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the endpoint to a dictionary.
        
        Returns:
            Dict[str, Any]: The endpoint as a dictionary.
        """
        return {
            'name': self.name,
            'path': self.path,
            'method': self.method,
            'description': self.description,
            'parameters': self.parameters,
            'request_body': self.request_body,
            'responses': self.responses,
            'auth_required': self.auth_required,
            'tags': self.tags
        }
        
    @classmethod
    def from_dict(cls, endpoint_dict: Dict[str, Any]) -> 'ApiEndpoint':
        """Create an endpoint from a dictionary.
        
        Args:
            endpoint_dict: The endpoint dictionary.
            
        Returns:
            ApiEndpoint: The created endpoint.
        """
        return cls(
            name=endpoint_dict.get('name', ''),
            path=endpoint_dict.get('path', ''),
            method=endpoint_dict.get('method', 'GET'),
            description=endpoint_dict.get('description', ''),
            parameters=endpoint_dict.get('parameters'),
            request_body=endpoint_dict.get('request_body'),
            responses=endpoint_dict.get('responses'),
            auth_required=endpoint_dict.get('auth_required', True),
            tags=endpoint_dict.get('tags')
        )


class ApiDocumentation:
    """Class for generating API documentation."""
    
    def __init__(self, title: str, description: str, version: str):
        """Initialize API documentation.
        
        Args:
            title: The title of the API.
            description: The description of the API.
            version: The version of the API.
        """
        self.title = title
        self.description = description
        self.version = version
        self.endpoints: List[ApiEndpoint] = []
        self.tags: List[Dict[str, str]] = []
        
    def add_endpoint(self, endpoint: ApiEndpoint):
        """Add an endpoint to the documentation.
        
        Args:
            endpoint: The endpoint to add.
        """
        self.endpoints.append(endpoint)
        
    def add_tag(self, name: str, description: str):
        """Add a tag to the documentation.
        
        Args:
            name: The name of the tag.
            description: The description of the tag.
        """
        self.tags.append({
            'name': name,
            'description': description
        })
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the documentation to a dictionary.
        
        Returns:
            Dict[str, Any]: The documentation as a dictionary.
        """
        return {
            'title': self.title,
            'description': self.description,
            'version': self.version,
            'tags': self.tags,
            'endpoints': [endpoint.to_dict() for endpoint in self.endpoints]
        }
        
    def to_openapi(self) -> Dict[str, Any]:
        """Convert the documentation to OpenAPI format.
        
        Returns:
            Dict[str, Any]: The documentation in OpenAPI format.
        """
        paths = {}
        for endpoint in self.endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}
                
            method = endpoint.method.lower()
            paths[endpoint.path][method] = {
                'summary': endpoint.name,
                'description': endpoint.description,
                'tags': endpoint.tags,
                'parameters': endpoint.parameters,
                'responses': {
                    code: {
                        'description': response.get('description', ''),
                        'content': response.get('content', {})
                    } for code, response in endpoint.responses.items()
                }
            }
            
            if endpoint.request_body:
                paths[endpoint.path][method]['requestBody'] = {
                    'description': endpoint.request_body.get('description', ''),
                    'content': endpoint.request_body.get('content', {}),
                    'required': endpoint.request_body.get('required', True)
                }
                
            if endpoint.auth_required:
                if 'security' not in paths[endpoint.path][method]:
                    paths[endpoint.path][method]['security'] = []
                paths[endpoint.path][method]['security'].append({'BearerAuth': []})
                
        return {
            'openapi': '3.0.0',
            'info': {
                'title': self.title,
                'description': self.description,
                'version': self.version
            },
            'tags': self.tags,
            'paths': paths,
            'components': {
                'securitySchemes': {
                    'BearerAuth': {
                        'type': 'http',
                        'scheme': 'bearer',
                        'bearerFormat': 'JWT'
                    }
                }
            }
        }
        
    def to_json(self) -> str:
        """Convert the documentation to JSON.
        
        Returns:
            str: The documentation as JSON.
        """
        return json.dumps(self.to_dict(), indent=2)
        
    def to_yaml(self) -> str:
        """Convert the documentation to YAML.
        
        Returns:
            str: The documentation as YAML.
        """
        return yaml.dump(self.to_dict(), sort_keys=False)
        
    def to_openapi_json(self) -> str:
        """Convert the documentation to OpenAPI JSON.
        
        Returns:
            str: The documentation as OpenAPI JSON.
        """
        return json.dumps(self.to_openapi(), indent=2)
        
    def to_openapi_yaml(self) -> str:
        """Convert the documentation to OpenAPI YAML.
        
        Returns:
            str: The documentation as OpenAPI YAML.
        """
        return yaml.dump(self.to_openapi(), sort_keys=False)
        
    def save(self, file_path: str, format_type: str = 'json'):
        """Save the documentation to a file.
        
        Args:
            file_path: The path to save the documentation to.
            format_type: The format to save the documentation in ('json', 'yaml', 'openapi_json', 'openapi_yaml').
            
        Raises:
            ApiDocError: If the format type is invalid.
        """
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Get the content based on the format type
            if format_type == 'json':
                content = self.to_json()
            elif format_type == 'yaml':
                content = self.to_yaml()
            elif format_type == 'openapi_json':
                content = self.to_openapi_json()
            elif format_type == 'openapi_yaml':
                content = self.to_openapi_yaml()
            else:
                raise ApiDocError(f"Invalid format type: {format_type}")
                
            # Write the content to the file
            with open(file_path, 'w') as f:
                f.write(content)
                
            logger.info(f"Saved API documentation to {file_path}")
        except Exception as e:
            raise ApiDocError(f"Failed to save API documentation: {str(e)}") from e
        
    @classmethod
    def from_dict(cls, doc_dict: Dict[str, Any]) -> 'ApiDocumentation':
        """Create documentation from a dictionary.
        
        Args:
            doc_dict: The documentation dictionary.
            
        Returns:
            ApiDocumentation: The created documentation.
        """
        doc = cls(
            title=doc_dict.get('title', ''),
            description=doc_dict.get('description', ''),
            version=doc_dict.get('version', '1.0.0')
        )
        
        # Add tags
        for tag in doc_dict.get('tags', []):
            doc.add_tag(tag.get('name', ''), tag.get('description', ''))
            
        # Add endpoints
        for endpoint_dict in doc_dict.get('endpoints', []):
            endpoint = ApiEndpoint.from_dict(endpoint_dict)
            doc.add_endpoint(endpoint)
            
        return doc
        
    @classmethod
    def from_json(cls, json_str: str) -> 'ApiDocumentation':
        """Create documentation from JSON.
        
        Args:
            json_str: The JSON string.
            
        Returns:
            ApiDocumentation: The created documentation.
            
        Raises:
            ApiDocError: If the JSON is invalid.
        """
        try:
            doc_dict = json.loads(json_str)
            return cls.from_dict(doc_dict)
        except Exception as e:
            raise ApiDocError(f"Failed to parse JSON: {str(e)}") from e
        
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'ApiDocumentation':
        """Create documentation from YAML.
        
        Args:
            yaml_str: The YAML string.
            
        Returns:
            ApiDocumentation: The created documentation.
            
        Raises:
            ApiDocError: If the YAML is invalid.
        """
        try:
            doc_dict = yaml.safe_load(yaml_str)
            return cls.from_dict(doc_dict)
        except Exception as e:
            raise ApiDocError(f"Failed to parse YAML: {str(e)}") from e
        
    @classmethod
    def from_file(cls, file_path: str) -> 'ApiDocumentation':
        """Create documentation from a file.
        
        Args:
            file_path: The path to the file.
            
        Returns:
            ApiDocumentation: The created documentation.
            
        Raises:
            ApiDocError: If the file is invalid.
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Determine the format based on the file extension
            if file_path.endswith('.json'):
                return cls.from_json(content)
            elif file_path.endswith(('.yaml', '.yml')):
                return cls.from_yaml(content)
            else:
                raise ApiDocError(f"Unsupported file format: {file_path}")
        except Exception as e:
            raise ApiDocError(f"Failed to load API documentation from file: {str(e)}") from e


def generate_api_documentation(system_id: str, system_name: str, endpoints: List[Dict[str, Any]], 
                             output_dir: str, format_type: str = 'openapi_json'):
    """Generate API documentation for an external system.
    
    Args:
        system_id: The ID of the external system.
        system_name: The name of the external system.
        endpoints: The endpoints of the external system.
        output_dir: The directory to save the documentation to.
        format_type: The format to save the documentation in.
        
    Returns:
        str: The path to the generated documentation.
        
    Raises:
        ApiDocError: If the documentation generation fails.
    """
    try:
        # Create the API documentation
        doc = ApiDocumentation(
            title=f"{system_name} API",
            description=f"API documentation for {system_name}",
            version="1.0.0"
        )
        
        # Add tags
        doc.add_tag('Authentication', 'Authentication endpoints')
        doc.add_tag('Data', 'Data endpoints')
        doc.add_tag('System', 'System endpoints')
        
        # Add endpoints
        for endpoint_dict in endpoints:
            endpoint = ApiEndpoint.from_dict(endpoint_dict)
            doc.add_endpoint(endpoint)
            
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the documentation
        file_name = f"{system_id}_api"
        if format_type == 'json':
            file_path = os.path.join(output_dir, f"{file_name}.json")
        elif format_type == 'yaml':
            file_path = os.path.join(output_dir, f"{file_name}.yaml")
        elif format_type == 'openapi_json':
            file_path = os.path.join(output_dir, f"{file_name}_openapi.json")
        elif format_type == 'openapi_yaml':
            file_path = os.path.join(output_dir, f"{file_name}_openapi.yaml")
        else:
            raise ApiDocError(f"Invalid format type: {format_type}")
            
        doc.save(file_path, format_type)
        
        return file_path
    except Exception as e:
        raise ApiDocError(f"Failed to generate API documentation: {str(e)}") from e


def generate_api_documentation_from_config(config: Dict[str, Any], output_dir: str, 
                                         format_type: str = 'openapi_json'):
    """Generate API documentation from a system configuration.
    
    Args:
        config: The system configuration.
        output_dir: The directory to save the documentation to.
        format_type: The format to save the documentation in.
        
    Returns:
        str: The path to the generated documentation.
        
    Raises:
        ApiDocError: If the documentation generation fails.
    """
    try:
        system_id = config.get('system_id')
        system_name = config.get('name')
        
        if not system_id or not system_name:
            raise ApiDocError("System ID and name are required")
            
        # Extract endpoints from the configuration
        endpoints = []
        
        # Add authentication endpoints
        auth_type = config.get('auth_type')
        if auth_type:
            endpoints.append({
                'name': 'Authenticate',
                'path': '/authenticate',
                'method': 'POST',
                'description': f"Authenticate with the {system_name} API using {auth_type} authentication",
                'request_body': {
                    'description': 'Authentication credentials',
                    'required': True,
                    'content': {
                        'application/json': {
                            'schema': {
                                'type': 'object',
                                'properties': {
                                    'credentials': {
                                        'type': 'object',
                                        'description': 'Authentication credentials'
                                    }
                                },
                                'required': ['credentials']
                            }
                        }
                    }
                },
                'responses': {
                    '200': {
                        'description': 'Authentication successful',
                        'content': {
                            'application/json': {
                                'schema': {
                                    'type': 'object',
                                    'properties': {
                                        'token': {
                                            'type': 'string',
                                            'description': 'Authentication token'
                                        }
                                    }
                                }
                            }
                        }
                    },
                    '401': {
                        'description': 'Authentication failed'
                    }
                },
                'auth_required': False,
                'tags': ['Authentication']
            })
            
        # Add data endpoints
        for endpoint_type in ['order_endpoints', 'account_endpoints', 'data_endpoints', 'subscription_endpoints']:
            endpoints_config = config.get(endpoint_type, {})
            for endpoint_name, endpoint_config in endpoints_config.items():
                endpoint_path = endpoint_config.get('path', f"/{endpoint_name}")
                endpoint_method = endpoint_config.get('method', 'GET')
                endpoint_description = endpoint_config.get('description', f"{endpoint_name} endpoint")
                endpoint_params = endpoint_config.get('parameters', [])
                endpoint_request_body = endpoint_config.get('request_body')
                endpoint_responses = endpoint_config.get('responses', {
                    '200': {
                        'description': 'Success'
                    },
                    '400': {
                        'description': 'Bad request'
                    },
                    '401': {
                        'description': 'Unauthorized'
                    },
                    '500': {
                        'description': 'Internal server error'
                    }
                })
                
                endpoints.append({
                    'name': endpoint_name,
                    'path': endpoint_path,
                    'method': endpoint_method,
                    'description': endpoint_description,
                    'parameters': endpoint_params,
                    'request_body': endpoint_request_body,
                    'responses': endpoint_responses,
                    'auth_required': True,
                    'tags': ['Data']
                })
                
        # Generate the documentation
        return generate_api_documentation(system_id, system_name, endpoints, output_dir, format_type)
    except Exception as e:
        raise ApiDocError(f"Failed to generate API documentation from config: {str(e)}") from e


def generate_api_documentation_for_all_systems(configs: List[Dict[str, Any]], output_dir: str, 
                                             format_type: str = 'openapi_json'):
    """Generate API documentation for all external systems.
    
    Args:
        configs: The system configurations.
        output_dir: The directory to save the documentation to.
        format_type: The format to save the documentation in.
        
    Returns:
        List[str]: The paths to the generated documentation.
        
    Raises:
        ApiDocError: If the documentation generation fails.
    """
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate documentation for each system
        doc_paths = []
        for config in configs:
            try:
                doc_path = generate_api_documentation_from_config(config, output_dir, format_type)
                doc_paths.append(doc_path)
            except Exception as e:
                logger.error(f"Failed to generate API documentation for system {config.get('system_id')}: {str(e)}")
                
        return doc_paths
    except Exception as e:
        raise ApiDocError(f"Failed to generate API documentation for all systems: {str(e)}") from e