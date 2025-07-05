"""Troubleshooting module for the Friday AI Trading System.

This module provides troubleshooting guides and utilities to help diagnose and fix common issues.
"""

import logging
import os
import sys
import platform
import json
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

from src.infrastructure.logging import get_logger
from src.infrastructure.error import ErrorSeverity, ErrorCategory, FridayError

# Create logger
logger = get_logger(__name__)


class IssueCategory(Enum):
    """Categories of issues that can occur in the system."""
    CONNECTIVITY = auto()      # Network connectivity issues
    AUTHENTICATION = auto()    # Authentication and authorization issues
    DATA = auto()              # Data retrieval, processing, or validation issues
    PERFORMANCE = auto()       # Performance-related issues
    CONFIGURATION = auto()     # Configuration-related issues
    SYSTEM = auto()            # System-level issues
    DATABASE = auto()          # Database-related issues
    API = auto()               # API-related issues
    MODEL = auto()             # Model-related issues
    TRADING = auto()           # Trading-related issues
    UNKNOWN = auto()           # Unknown issues


class IssueSeverity(Enum):
    """Severity levels for issues."""
    CRITICAL = auto()    # Critical issues that prevent the system from functioning
    HIGH = auto()        # High-severity issues that significantly impact functionality
    MEDIUM = auto()      # Medium-severity issues that partially impact functionality
    LOW = auto()         # Low-severity issues that have minimal impact
    INFO = auto()        # Informational issues


class TroubleshootingStep:
    """A step in a troubleshooting guide."""
    
    def __init__(self, description: str, action: str, verification: str = None):
        """Initialize a troubleshooting step.
        
        Args:
            description: Description of the step
            action: Action to take
            verification: How to verify the step was successful
        """
        self.description = description
        self.action = action
        self.verification = verification
    
    def to_dict(self) -> Dict[str, str]:
        """Convert the step to a dictionary.
        
        Returns:
            Dictionary representation of the step
        """
        result = {
            'description': self.description,
            'action': self.action
        }
        
        if self.verification:
            result['verification'] = self.verification
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'TroubleshootingStep':
        """Create a step from a dictionary.
        
        Args:
            data: Dictionary representation of the step
            
        Returns:
            TroubleshootingStep instance
        """
        return cls(
            description=data['description'],
            action=data['action'],
            verification=data.get('verification')
        )
    
    def __str__(self) -> str:
        """Get a string representation of the step.
        
        Returns:
            String representation
        """
        result = f"Description: {self.description}\nAction: {self.action}"
        
        if self.verification:
            result += f"\nVerification: {self.verification}"
        
        return result


class TroubleshootingGuide:
    """A guide for troubleshooting an issue."""
    
    def __init__(self, 
                 title: str, 
                 issue_id: str,
                 category: IssueCategory,
                 severity: IssueSeverity,
                 symptoms: List[str],
                 possible_causes: List[str],
                 steps: List[TroubleshootingStep],
                 prevention: List[str] = None,
                 related_issues: List[str] = None,
                 references: List[str] = None):
        """Initialize a troubleshooting guide.
        
        Args:
            title: Title of the guide
            issue_id: Unique identifier for the issue
            category: Category of the issue
            severity: Severity of the issue
            symptoms: List of symptoms that indicate the issue
            possible_causes: List of possible causes of the issue
            steps: List of troubleshooting steps
            prevention: List of prevention measures
            related_issues: List of related issue IDs
            references: List of references for further information
        """
        self.title = title
        self.issue_id = issue_id
        self.category = category
        self.severity = severity
        self.symptoms = symptoms
        self.possible_causes = possible_causes
        self.steps = steps
        self.prevention = prevention or []
        self.related_issues = related_issues or []
        self.references = references or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the guide to a dictionary.
        
        Returns:
            Dictionary representation of the guide
        """
        return {
            'title': self.title,
            'issue_id': self.issue_id,
            'category': self.category.name,
            'severity': self.severity.name,
            'symptoms': self.symptoms,
            'possible_causes': self.possible_causes,
            'steps': [step.to_dict() for step in self.steps],
            'prevention': self.prevention,
            'related_issues': self.related_issues,
            'references': self.references
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TroubleshootingGuide':
        """Create a guide from a dictionary.
        
        Args:
            data: Dictionary representation of the guide
            
        Returns:
            TroubleshootingGuide instance
        """
        return cls(
            title=data['title'],
            issue_id=data['issue_id'],
            category=IssueCategory[data['category']],
            severity=IssueSeverity[data['severity']],
            symptoms=data['symptoms'],
            possible_causes=data['possible_causes'],
            steps=[TroubleshootingStep.from_dict(step) for step in data['steps']],
            prevention=data.get('prevention', []),
            related_issues=data.get('related_issues', []),
            references=data.get('references', [])
        )
    
    def __str__(self) -> str:
        """Get a string representation of the guide.
        
        Returns:
            String representation
        """
        result = f"# {self.title} ({self.issue_id})\n"
        result += f"Category: {self.category.name}\n"
        result += f"Severity: {self.severity.name}\n\n"
        
        result += "## Symptoms\n"
        for symptom in self.symptoms:
            result += f"- {symptom}\n"
        result += "\n"
        
        result += "## Possible Causes\n"
        for cause in self.possible_causes:
            result += f"- {cause}\n"
        result += "\n"
        
        result += "## Troubleshooting Steps\n"
        for i, step in enumerate(self.steps, 1):
            result += f"### Step {i}: {step.description}\n"
            result += f"Action: {step.action}\n"
            if step.verification:
                result += f"Verification: {step.verification}\n"
            result += "\n"
        
        if self.prevention:
            result += "## Prevention\n"
            for prevention in self.prevention:
                result += f"- {prevention}\n"
            result += "\n"
        
        if self.related_issues:
            result += "## Related Issues\n"
            for issue in self.related_issues:
                result += f"- {issue}\n"
            result += "\n"
        
        if self.references:
            result += "## References\n"
            for reference in self.references:
                result += f"- {reference}\n"
        
        return result


class TroubleshootingRegistry:
    """Registry for troubleshooting guides."""
    
    _instance = None
    
    def __new__(cls):
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super(TroubleshootingRegistry, cls).__new__(cls)
            cls._instance._guides = {}
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the registry."""
        if not self._initialized:
            self._guides = {}
            self._initialized = True
    
    def register_guide(self, guide: TroubleshootingGuide) -> None:
        """Register a troubleshooting guide.
        
        Args:
            guide: Troubleshooting guide to register
        """
        self._guides[guide.issue_id] = guide
        logger.debug(f"Registered troubleshooting guide: {guide.issue_id} - {guide.title}")
    
    def get_guide(self, issue_id: str) -> Optional[TroubleshootingGuide]:
        """Get a troubleshooting guide by ID.
        
        Args:
            issue_id: ID of the guide to get
            
        Returns:
            Troubleshooting guide or None if not found
        """
        return self._guides.get(issue_id)
    
    def get_guides_by_category(self, category: IssueCategory) -> List[TroubleshootingGuide]:
        """Get troubleshooting guides by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of troubleshooting guides
        """
        return [guide for guide in self._guides.values() if guide.category == category]
    
    def get_guides_by_severity(self, severity: IssueSeverity) -> List[TroubleshootingGuide]:
        """Get troubleshooting guides by severity.
        
        Args:
            severity: Severity to filter by
            
        Returns:
            List of troubleshooting guides
        """
        return [guide for guide in self._guides.values() if guide.severity == severity]
    
    def search_guides(self, query: str) -> List[TroubleshootingGuide]:
        """Search for troubleshooting guides.
        
        Args:
            query: Search query
            
        Returns:
            List of matching troubleshooting guides
        """
        query = query.lower()
        results = []
        
        for guide in self._guides.values():
            # Check if the query is in the title, symptoms, or possible causes
            if (query in guide.title.lower() or
                any(query in symptom.lower() for symptom in guide.symptoms) or
                any(query in cause.lower() for cause in guide.possible_causes)):
                results.append(guide)
        
        return results
    
    def get_all_guides(self) -> List[TroubleshootingGuide]:
        """Get all troubleshooting guides.
        
        Returns:
            List of all troubleshooting guides
        """
        return list(self._guides.values())
    
    def load_guides_from_file(self, file_path: str) -> None:
        """Load troubleshooting guides from a JSON file.
        
        Args:
            file_path: Path to the JSON file
        """
        try:
            with open(file_path, 'r') as f:
                guides_data = json.load(f)
            
            for guide_data in guides_data:
                guide = TroubleshootingGuide.from_dict(guide_data)
                self.register_guide(guide)
            
            logger.info(f"Loaded {len(guides_data)} troubleshooting guides from {file_path}")
        except Exception as e:
            logger.error(f"Error loading troubleshooting guides from {file_path}: {e}")
    
    def save_guides_to_file(self, file_path: str) -> None:
        """Save troubleshooting guides to a JSON file.
        
        Args:
            file_path: Path to the JSON file
        """
        try:
            guides_data = [guide.to_dict() for guide in self._guides.values()]
            
            # Create the directory if it doesn't exist
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            with open(file_path, 'w') as f:
                json.dump(guides_data, f, indent=2)
            
            logger.info(f"Saved {len(guides_data)} troubleshooting guides to {file_path}")
        except Exception as e:
            logger.error(f"Error saving troubleshooting guides to {file_path}: {e}")


class ErrorToGuideMapper:
    """Maps error types to troubleshooting guides."""
    
    _instance = None
    
    def __new__(cls):
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super(ErrorToGuideMapper, cls).__new__(cls)
            cls._instance._mappings = {}
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the mapper."""
        if not self._initialized:
            self._mappings = {}
            self._initialized = True
    
    def register_mapping(self, error_type: type, issue_id: str) -> None:
        """Register a mapping from an error type to a troubleshooting guide.
        
        Args:
            error_type: Error type to map
            issue_id: ID of the troubleshooting guide
        """
        self._mappings[error_type] = issue_id
        logger.debug(f"Registered error mapping: {error_type.__name__} -> {issue_id}")
    
    def get_guide_for_error(self, error: Exception) -> Optional[TroubleshootingGuide]:
        """Get a troubleshooting guide for an error.
        
        Args:
            error: Error to get a guide for
            
        Returns:
            Troubleshooting guide or None if not found
        """
        # Check for exact type match
        issue_id = self._mappings.get(type(error))
        
        # If no exact match, check for parent types
        if issue_id is None:
            for error_type, mapped_issue_id in self._mappings.items():
                if isinstance(error, error_type):
                    issue_id = mapped_issue_id
                    break
        
        # If a mapping was found, get the guide
        if issue_id is not None:
            registry = TroubleshootingRegistry()
            return registry.get_guide(issue_id)
        
        return None


class DiagnosticCheck:
    """A diagnostic check for troubleshooting."""
    
    def __init__(self, name: str, description: str):
        """Initialize a diagnostic check.
        
        Args:
            name: Name of the check
            description: Description of the check
        """
        self.name = name
        self.description = description
    
    def run(self) -> Tuple[bool, str]:
        """Run the diagnostic check.
        
        Returns:
            Tuple of (success, message)
        """
        raise NotImplementedError("Subclasses must implement run()")


class NetworkConnectivityCheck(DiagnosticCheck):
    """Check network connectivity to a host."""
    
    def __init__(self, host: str, port: int = None, timeout: float = 5.0):
        """Initialize a network connectivity check.
        
        Args:
            host: Host to check connectivity to
            port: Port to check connectivity to, or None to just check host
            timeout: Timeout in seconds
        """
        super().__init__(
            name=f"Network connectivity to {host}:{port if port else 'N/A'}",
            description=f"Check network connectivity to {host}:{port if port else 'N/A'}"
        )
        self.host = host
        self.port = port
        self.timeout = timeout
    
    def run(self) -> Tuple[bool, str]:
        """Run the diagnostic check.
        
        Returns:
            Tuple of (success, message)
        """
        import socket
        
        try:
            if self.port is not None:
                # Check connectivity to host and port
                socket.create_connection((self.host, self.port), self.timeout)
                return True, f"Successfully connected to {self.host}:{self.port}"
            else:
                # Just check if the host is resolvable
                socket.gethostbyname(self.host)
                return True, f"Successfully resolved {self.host}"
        except socket.timeout:
            return False, f"Timeout connecting to {self.host}:{self.port if self.port else 'N/A'}"
        except socket.gaierror:
            return False, f"Could not resolve {self.host}"
        except ConnectionRefusedError:
            return False, f"Connection refused to {self.host}:{self.port}"
        except Exception as e:
            return False, f"Error connecting to {self.host}:{self.port if self.port else 'N/A'}: {e}"


class DatabaseConnectivityCheck(DiagnosticCheck):
    """Check database connectivity."""
    
    def __init__(self, connection_string: str, query: str = "SELECT 1"):
        """Initialize a database connectivity check.
        
        Args:
            connection_string: Database connection string
            query: Query to execute to check connectivity
        """
        super().__init__(
            name="Database connectivity",
            description="Check database connectivity"
        )
        self.connection_string = connection_string
        self.query = query
    
    def run(self) -> Tuple[bool, str]:
        """Run the diagnostic check.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Determine the database type from the connection string
            if self.connection_string.startswith('mongodb'):
                return self._check_mongodb()
            elif 'postgresql' in self.connection_string or 'postgres' in self.connection_string:
                return self._check_postgresql()
            elif 'mysql' in self.connection_string:
                return self._check_mysql()
            elif 'sqlite' in self.connection_string:
                return self._check_sqlite()
            else:
                return False, f"Unsupported database type in connection string: {self.connection_string}"
        except Exception as e:
            return False, f"Error checking database connectivity: {e}"
    
    def _check_mongodb(self) -> Tuple[bool, str]:
        """Check MongoDB connectivity.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            import pymongo
            client = pymongo.MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            return True, "Successfully connected to MongoDB"
        except Exception as e:
            return False, f"Error connecting to MongoDB: {e}"
    
    def _check_postgresql(self) -> Tuple[bool, str]:
        """Check PostgreSQL connectivity.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            import psycopg2
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            cursor.execute(self.query)
            cursor.close()
            conn.close()
            return True, "Successfully connected to PostgreSQL"
        except Exception as e:
            return False, f"Error connecting to PostgreSQL: {e}"
    
    def _check_mysql(self) -> Tuple[bool, str]:
        """Check MySQL connectivity.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            import mysql.connector
            conn = mysql.connector.connect(self.connection_string)
            cursor = conn.cursor()
            cursor.execute(self.query)
            cursor.close()
            conn.close()
            return True, "Successfully connected to MySQL"
        except Exception as e:
            return False, f"Error connecting to MySQL: {e}"
    
    def _check_sqlite(self) -> Tuple[bool, str]:
        """Check SQLite connectivity.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            import sqlite3
            conn = sqlite3.connect(self.connection_string.replace('sqlite:///', ''))
            cursor = conn.cursor()
            cursor.execute(self.query)
            cursor.close()
            conn.close()
            return True, "Successfully connected to SQLite"
        except Exception as e:
            return False, f"Error connecting to SQLite: {e}"


class FileSystemCheck(DiagnosticCheck):
    """Check file system access and permissions."""
    
    def __init__(self, path: str, check_write: bool = False):
        """Initialize a file system check.
        
        Args:
            path: Path to check
            check_write: Whether to check write access
        """
        super().__init__(
            name=f"File system access to {path}",
            description=f"Check file system access to {path}"
        )
        self.path = path
        self.check_write = check_write
    
    def run(self) -> Tuple[bool, str]:
        """Run the diagnostic check.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if the path exists
            if not os.path.exists(self.path):
                return False, f"Path does not exist: {self.path}"
            
            # Check read access
            if not os.access(self.path, os.R_OK):
                return False, f"No read access to {self.path}"
            
            # Check write access if requested
            if self.check_write and not os.access(self.path, os.W_OK):
                return False, f"No write access to {self.path}"
            
            # Check if it's a directory or file
            if os.path.isdir(self.path):
                return True, f"Successfully accessed directory {self.path}"
            else:
                return True, f"Successfully accessed file {self.path}"
        except Exception as e:
            return False, f"Error checking file system access to {self.path}: {e}"


class ApiEndpointCheck(DiagnosticCheck):
    """Check API endpoint availability."""
    
    def __init__(self, url: str, method: str = 'GET', headers: Dict[str, str] = None, data: Any = None, timeout: float = 5.0):
        """Initialize an API endpoint check.
        
        Args:
            url: URL to check
            method: HTTP method to use
            headers: HTTP headers to include
            data: Data to send with the request
            timeout: Timeout in seconds
        """
        super().__init__(
            name=f"API endpoint {method} {url}",
            description=f"Check API endpoint {method} {url}"
        )
        self.url = url
        self.method = method
        self.headers = headers or {}
        self.data = data
        self.timeout = timeout
    
    def run(self) -> Tuple[bool, str]:
        """Run the diagnostic check.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            import requests
            
            response = requests.request(
                method=self.method,
                url=self.url,
                headers=self.headers,
                json=self.data if self.data else None,
                timeout=self.timeout
            )
            
            if response.status_code < 400:
                return True, f"Successfully accessed API endpoint {self.method} {self.url} (status code: {response.status_code})"
            else:
                return False, f"API endpoint {self.method} {self.url} returned error status code: {response.status_code}"
        except requests.exceptions.Timeout:
            return False, f"Timeout accessing API endpoint {self.method} {self.url}"
        except requests.exceptions.ConnectionError:
            return False, f"Connection error accessing API endpoint {self.method} {self.url}"
        except Exception as e:
            return False, f"Error accessing API endpoint {self.method} {self.url}: {e}"


class DiagnosticRunner:
    """Runs diagnostic checks for troubleshooting."""
    
    def __init__(self):
        """Initialize a diagnostic runner."""
        self.checks = []
    
    def add_check(self, check: DiagnosticCheck) -> None:
        """Add a diagnostic check.
        
        Args:
            check: Diagnostic check to add
        """
        self.checks.append(check)
    
    def run_checks(self) -> List[Dict[str, Any]]:
        """Run all diagnostic checks.
        
        Returns:
            List of check results
        """
        results = []
        
        for check in self.checks:
            try:
                logger.info(f"Running diagnostic check: {check.name}")
                success, message = check.run()
                
                results.append({
                    'name': check.name,
                    'description': check.description,
                    'success': success,
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                })
                
                logger.info(f"Diagnostic check result: {check.name} - {'Success' if success else 'Failure'}: {message}")
            except Exception as e:
                logger.error(f"Error running diagnostic check {check.name}: {e}")
                
                results.append({
                    'name': check.name,
                    'description': check.description,
                    'success': False,
                    'message': f"Error running check: {e}",
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def run_check(self, check_name: str) -> Optional[Dict[str, Any]]:
        """Run a specific diagnostic check.
        
        Args:
            check_name: Name of the check to run
            
        Returns:
            Check result or None if not found
        """
        for check in self.checks:
            if check.name == check_name:
                try:
                    logger.info(f"Running diagnostic check: {check.name}")
                    success, message = check.run()
                    
                    result = {
                        'name': check.name,
                        'description': check.description,
                        'success': success,
                        'message': message,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    logger.info(f"Diagnostic check result: {check.name} - {'Success' if success else 'Failure'}: {message}")
                    
                    return result
                except Exception as e:
                    logger.error(f"Error running diagnostic check {check.name}: {e}")
                    
                    return {
                        'name': check.name,
                        'description': check.description,
                        'success': False,
                        'message': f"Error running check: {e}",
                        'timestamp': datetime.now().isoformat()
                    }
        
        return None
    
    def generate_report(self, results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a diagnostic report.
        
        Args:
            results: Check results, or None to run all checks
            
        Returns:
            Diagnostic report
        """
        if results is None:
            results = self.run_checks()
        
        # Count successes and failures
        success_count = sum(1 for result in results if result['success'])
        failure_count = len(results) - success_count
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'summary': {
                'total_checks': len(results),
                'success_count': success_count,
                'failure_count': failure_count,
                'success_rate': success_count / len(results) if results else 0
            },
            'results': results
        }
        
        return report
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for the report.
        
        Returns:
            Dictionary with system information
        """
        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'hostname': platform.node(),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_report(self, report: Dict[str, Any], file_path: str = None) -> str:
        """Save a diagnostic report to a file.
        
        Args:
            report: Diagnostic report
            file_path: Path to save the report to, or None to generate a default path
            
        Returns:
            Path to the saved report
        """
        # Generate a default file path if not provided
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"diagnostic_report_{timestamp}.json"
        
        # Create the directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the report
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved diagnostic report to {file_path}")
        
        return file_path


class TroubleshootingAssistant:
    """Assistant for troubleshooting issues."""
    
    def __init__(self):
        """Initialize a troubleshooting assistant."""
        self.registry = TroubleshootingRegistry()
        self.mapper = ErrorToGuideMapper()
        self.runner = DiagnosticRunner()
    
    def get_guide_for_error(self, error: Exception) -> Optional[TroubleshootingGuide]:
        """Get a troubleshooting guide for an error.
        
        Args:
            error: Error to get a guide for
            
        Returns:
            Troubleshooting guide or None if not found
        """
        return self.mapper.get_guide_for_error(error)
    
    def search_guides(self, query: str) -> List[TroubleshootingGuide]:
        """Search for troubleshooting guides.
        
        Args:
            query: Search query
            
        Returns:
            List of matching troubleshooting guides
        """
        return self.registry.search_guides(query)
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run diagnostics.
        
        Returns:
            Diagnostic report
        """
        return self.runner.generate_report()
    
    def get_recommendations(self, error: Exception = None, symptoms: List[str] = None) -> List[TroubleshootingGuide]:
        """Get troubleshooting recommendations.
        
        Args:
            error: Error to get recommendations for
            symptoms: Symptoms to get recommendations for
            
        Returns:
            List of recommended troubleshooting guides
        """
        recommendations = []
        
        # Get recommendations based on error
        if error is not None:
            guide = self.get_guide_for_error(error)
            if guide:
                recommendations.append(guide)
        
        # Get recommendations based on symptoms
        if symptoms:
            for symptom in symptoms:
                guides = self.search_guides(symptom)
                for guide in guides:
                    if guide not in recommendations:
                        recommendations.append(guide)
        
        return recommendations
    
    def generate_custom_guide(self, error: Exception) -> TroubleshootingGuide:
        """Generate a custom troubleshooting guide for an error.
        
        Args:
            error: Error to generate a guide for
            
        Returns:
            Custom troubleshooting guide
        """
        # Determine the category and severity based on the error type
        if isinstance(error, FridayError):
            category = self._map_error_category_to_issue_category(error.category)
            severity = self._map_error_severity_to_issue_severity(error.severity)
            guidance = error.troubleshooting_guidance
        else:
            category = IssueCategory.UNKNOWN
            severity = IssueSeverity.MEDIUM
            guidance = "No specific troubleshooting guidance available."
        
        # Generate a unique ID for the guide
        issue_id = f"CUSTOM-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create the guide
        guide = TroubleshootingGuide(
            title=f"Troubleshooting {type(error).__name__}",
            issue_id=issue_id,
            category=category,
            severity=severity,
            symptoms=[str(error)],
            possible_causes=["Unknown"],
            steps=[TroubleshootingStep(
                description="Review error details",
                action=f"Review the error message: {error}",
                verification="Understand the error message"
            )]
        )
        
        # Add troubleshooting guidance if available
        if guidance:
            guide.steps.append(TroubleshootingStep(
                description="Follow troubleshooting guidance",
                action=guidance,
                verification="Issue is resolved"
            ))
        
        # Add diagnostic checks
        guide.steps.append(TroubleshootingStep(
            description="Run diagnostics",
            action="Run the diagnostic checks to identify potential issues",
            verification="All diagnostic checks pass"
        ))
        
        return guide
    
    def _map_error_category_to_issue_category(self, error_category: ErrorCategory) -> IssueCategory:
        """Map an error category to an issue category.
        
        Args:
            error_category: Error category to map
            
        Returns:
            Corresponding issue category
        """
        mapping = {
            ErrorCategory.SYSTEM: IssueCategory.SYSTEM,
            ErrorCategory.NETWORK: IssueCategory.CONNECTIVITY,
            ErrorCategory.DATABASE: IssueCategory.DATABASE,
            ErrorCategory.API: IssueCategory.API,
            ErrorCategory.DATA: IssueCategory.DATA,
            ErrorCategory.RETRY: IssueCategory.PERFORMANCE,
            ErrorCategory.AUTHENTICATION: IssueCategory.AUTHENTICATION,
            ErrorCategory.CONFIGURATION: IssueCategory.CONFIGURATION,
            ErrorCategory.MODEL: IssueCategory.MODEL,
            ErrorCategory.TRADING: IssueCategory.TRADING
        }
        
        return mapping.get(error_category, IssueCategory.UNKNOWN)
    
    def _map_error_severity_to_issue_severity(self, error_severity: ErrorSeverity) -> IssueSeverity:
        """Map an error severity to an issue severity.
        
        Args:
            error_severity: Error severity to map
            
        Returns:
            Corresponding issue severity
        """
        mapping = {
            ErrorSeverity.CRITICAL: IssueSeverity.CRITICAL,
            ErrorSeverity.HIGH: IssueSeverity.HIGH,
            ErrorSeverity.MEDIUM: IssueSeverity.MEDIUM,
            ErrorSeverity.LOW: IssueSeverity.LOW,
            ErrorSeverity.INFO: IssueSeverity.INFO
        }
        
        return mapping.get(error_severity, IssueSeverity.MEDIUM)


# Initialize the troubleshooting system
def initialize_troubleshooting(guides_file: str = None) -> TroubleshootingAssistant:
    """Initialize the troubleshooting system.
    
    Args:
        guides_file: Path to a JSON file with troubleshooting guides
        
    Returns:
        TroubleshootingAssistant instance
    """
    logger.info("Initializing troubleshooting system")
    
    # Create a troubleshooting assistant
    assistant = TroubleshootingAssistant()
    
    # Load guides from file if provided
    if guides_file and os.path.exists(guides_file):
        assistant.registry.load_guides_from_file(guides_file)
    
    # Register common diagnostic checks
    assistant.runner.add_check(NetworkConnectivityCheck("google.com", 443))
    assistant.runner.add_check(NetworkConnectivityCheck("api.example.com", 443))
    assistant.runner.add_check(FileSystemCheck("storage/logs", check_write=True))
    
    return assistant


# Register common troubleshooting guides
def register_common_guides():
    """Register common troubleshooting guides."""
    registry = TroubleshootingRegistry()
    
    # Network connectivity issues
    registry.register_guide(TroubleshootingGuide(
        title="Network Connectivity Issues",
        issue_id="NETWORK-001",
        category=IssueCategory.CONNECTIVITY,
        severity=IssueSeverity.HIGH,
        symptoms=[
            "Unable to connect to external APIs",
            "Timeout errors when making API calls",
            "Connection refused errors"
        ],
        possible_causes=[
            "Network firewall blocking connections",
            "DNS resolution issues",
            "API endpoint is down",
            "Proxy configuration issues"
        ],
        steps=[
            TroubleshootingStep(
                description="Check basic connectivity",
                action="Ping the API endpoint to check basic connectivity",
                verification="Ping should succeed"
            ),
            TroubleshootingStep(
                description="Check DNS resolution",
                action="Use nslookup or dig to check DNS resolution",
                verification="DNS resolution should succeed"
            ),
            TroubleshootingStep(
                description="Check firewall settings",
                action="Check if the firewall is blocking outbound connections",
                verification="Firewall should allow outbound connections to the API endpoint"
            ),
            TroubleshootingStep(
                description="Check proxy settings",
                action="Check if proxy settings are correctly configured",
                verification="Proxy settings should be correct"
            ),
            TroubleshootingStep(
                description="Check API status",
                action="Check the API status page or contact the API provider",
                verification="API should be operational"
            )
        ],
        prevention=[
            "Implement circuit breakers to handle API outages",
            "Set up monitoring for API connectivity",
            "Implement fallback mechanisms for critical operations"
        ],
        references=[
            "https://docs.example.com/api/troubleshooting"
        ]
    ))
    
    # Database connectivity issues
    registry.register_guide(TroubleshootingGuide(
        title="Database Connectivity Issues",
        issue_id="DATABASE-001",
        category=IssueCategory.DATABASE,
        severity=IssueSeverity.HIGH,
        symptoms=[
            "Unable to connect to the database",
            "Database query timeouts",
            "Connection pool exhaustion"
        ],
        possible_causes=[
            "Database server is down",
            "Network connectivity issues",
            "Authentication issues",
            "Connection pool configuration issues"
        ],
        steps=[
            TroubleshootingStep(
                description="Check database server status",
                action="Check if the database server is running",
                verification="Database server should be running"
            ),
            TroubleshootingStep(
                description="Check network connectivity",
                action="Check network connectivity to the database server",
                verification="Network connectivity should be available"
            ),
            TroubleshootingStep(
                description="Check authentication",
                action="Verify database credentials",
                verification="Credentials should be correct"
            ),
            TroubleshootingStep(
                description="Check connection pool settings",
                action="Review connection pool configuration",
                verification="Connection pool should be properly configured"
            )
        ],
        prevention=[
            "Implement database connection retry logic",
            "Set up monitoring for database connectivity",
            "Implement circuit breakers for database operations"
        ],
        references=[
            "https://docs.example.com/database/troubleshooting"
        ]
    ))
    
    # Data processing issues
    registry.register_guide(TroubleshootingGuide(
        title="Data Processing Issues",
        issue_id="DATA-001",
        category=IssueCategory.DATA,
        severity=IssueSeverity.MEDIUM,
        symptoms=[
            "Data validation errors",
            "Missing or incomplete data",
            "Data format errors"
        ],
        possible_causes=[
            "Data source providing invalid data",
            "Data transformation errors",
            "Schema changes in data source"
        ],
        steps=[
            TroubleshootingStep(
                description="Check data source",
                action="Verify that the data source is providing valid data",
                verification="Data source should provide valid data"
            ),
            TroubleshootingStep(
                description="Check data transformation logic",
                action="Review data transformation logic for errors",
                verification="Data transformation logic should be correct"
            ),
            TroubleshootingStep(
                description="Check for schema changes",
                action="Check if the data source schema has changed",
                verification="Data source schema should match expected schema"
            )
        ],
        prevention=[
            "Implement robust data validation",
            "Set up monitoring for data quality",
            "Implement schema versioning"
        ],
        references=[
            "https://docs.example.com/data/troubleshooting"
        ]
    ))
    
    # Authentication issues
    registry.register_guide(TroubleshootingGuide(
        title="Authentication Issues",
        issue_id="AUTH-001",
        category=IssueCategory.AUTHENTICATION,
        severity=IssueSeverity.HIGH,
        symptoms=[
            "Authentication failures",
            "Unauthorized access errors",
            "Token expiration errors"
        ],
        possible_causes=[
            "Invalid credentials",
            "Expired tokens",
            "Authentication service issues",
            "Permission configuration issues"
        ],
        steps=[
            TroubleshootingStep(
                description="Check credentials",
                action="Verify that the credentials are correct",
                verification="Credentials should be correct"
            ),
            TroubleshootingStep(
                description="Check token expiration",
                action="Check if the authentication token has expired",
                verification="Token should be valid"
            ),
            TroubleshootingStep(
                description="Check authentication service",
                action="Verify that the authentication service is operational",
                verification="Authentication service should be operational"
            ),
            TroubleshootingStep(
                description="Check permissions",
                action="Review permission configuration",
                verification="Permissions should be correctly configured"
            )
        ],
        prevention=[
            "Implement token refresh logic",
            "Set up monitoring for authentication failures",
            "Implement proper error handling for authentication issues"
        ],
        references=[
            "https://docs.example.com/auth/troubleshooting"
        ]
    ))
    
    # Performance issues
    registry.register_guide(TroubleshootingGuide(
        title="Performance Issues",
        issue_id="PERF-001",
        category=IssueCategory.PERFORMANCE,
        severity=IssueSeverity.MEDIUM,
        symptoms=[
            "Slow response times",
            "High CPU usage",
            "High memory usage",
            "Timeouts"
        ],
        possible_causes=[
            "Inefficient code",
            "Resource constraints",
            "Database query performance issues",
            "External service performance issues"
        ],
        steps=[
            TroubleshootingStep(
                description="Check system resources",
                action="Monitor CPU, memory, and disk usage",
                verification="Resource usage should be within acceptable limits"
            ),
            TroubleshootingStep(
                description="Check database performance",
                action="Review database query performance",
                verification="Database queries should be efficient"
            ),
            TroubleshootingStep(
                description="Check external service performance",
                action="Monitor external service response times",
                verification="External services should respond within acceptable timeframes"
            ),
            TroubleshootingStep(
                description="Profile code",
                action="Use profiling tools to identify performance bottlenecks",
                verification="Code should be efficient"
            )
        ],
        prevention=[
            "Implement performance monitoring",
            "Optimize database queries",
            "Implement caching",
            "Use asynchronous processing for long-running tasks"
        ],
        references=[
            "https://docs.example.com/performance/troubleshooting"
        ]
    ))
    
    # Configuration issues
    registry.register_guide(TroubleshootingGuide(
        title="Configuration Issues",
        issue_id="CONFIG-001",
        category=IssueCategory.CONFIGURATION,
        severity=IssueSeverity.MEDIUM,
        symptoms=[
            "Missing configuration values",
            "Invalid configuration values",
            "Configuration conflicts"
        ],
        possible_causes=[
            "Configuration files not found",
            "Environment variables not set",
            "Configuration values not valid"
        ],
        steps=[
            TroubleshootingStep(
                description="Check configuration files",
                action="Verify that configuration files exist and are accessible",
                verification="Configuration files should exist and be accessible"
            ),
            TroubleshootingStep(
                description="Check environment variables",
                action="Verify that required environment variables are set",
                verification="Environment variables should be set"
            ),
            TroubleshootingStep(
                description="Validate configuration values",
                action="Check if configuration values are valid",
                verification="Configuration values should be valid"
            )
        ],
        prevention=[
            "Implement configuration validation",
            "Use default values for non-critical configuration",
            "Document required configuration"
        ],
        references=[
            "https://docs.example.com/config/troubleshooting"
        ]
    ))
    
    # Register error mappings
    mapper = ErrorToGuideMapper()
    
    # Map error types to troubleshooting guides
    from src.infrastructure.error import NetworkError, DatabaseError, DataError, AuthenticationError
    
    mapper.register_mapping(NetworkError, "NETWORK-001")
    mapper.register_mapping(DatabaseError, "DATABASE-001")
    mapper.register_mapping(DataError, "DATA-001")
    mapper.register_mapping(AuthenticationError, "AUTH-001")