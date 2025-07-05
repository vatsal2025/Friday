"""Tests for the troubleshooting module.

This module contains tests for the troubleshooting functionality, including:
- Troubleshooting guides
- Diagnostic checks
- Error to guide mapping
- Troubleshooting assistant
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import json

from src.infrastructure.troubleshooting import (
    IssueCategory, IssueSeverity, TroubleshootingStep, TroubleshootingGuide,
    TroubleshootingRegistry, ErrorToGuideMapper, DiagnosticCheck,
    NetworkConnectivityCheck, DatabaseConnectivityCheck, FileSystemCheck,
    ApiEndpointCheck, DiagnosticRunner, TroubleshootingAssistant,
    initialize_troubleshooting
)
from src.infrastructure.error import (
    FridayError, NetworkError, DatabaseError, DataError, AuthenticationError
)


class TestTroubleshootingGuide(unittest.TestCase):
    """Tests for the TroubleshootingGuide class."""

    def test_create_guide(self):
        """Test creating a troubleshooting guide."""
        # Create steps
        steps = [
            TroubleshootingStep(
                "Check network connectivity",
                "Verify that the system can connect to the required network resources.",
                ["ping example.com", "traceroute example.com"]
            ),
            TroubleshootingStep(
                "Check firewall settings",
                "Ensure that the firewall is not blocking the required connections.",
                ["Check firewall rules", "Temporarily disable firewall to test"]
            )
        ]

        # Create guide
        guide = TroubleshootingGuide(
            title="Network Connectivity Issues",
            guide_id="NET001",
            category=IssueCategory.NETWORK,
            severity=IssueSeverity.HIGH,
            symptoms=["Cannot connect to remote servers", "Timeouts when making API calls"],
            causes=["Network is down", "Firewall is blocking connections"],
            steps=steps,
            prevention=["Implement network monitoring", "Configure proper firewall rules"],
            related_issues=["NET002", "API001"],
            references=["https://example.com/network-troubleshooting"]
        )

        # Check guide properties
        self.assertEqual(guide.title, "Network Connectivity Issues")
        self.assertEqual(guide.guide_id, "NET001")
        self.assertEqual(guide.category, IssueCategory.NETWORK)
        self.assertEqual(guide.severity, IssueSeverity.HIGH)
        self.assertEqual(len(guide.steps), 2)
        self.assertEqual(guide.steps[0].title, "Check network connectivity")
        self.assertEqual(guide.steps[1].title, "Check firewall settings")

    def test_to_dict(self):
        """Test converting a guide to a dictionary."""
        # Create a simple guide
        guide = TroubleshootingGuide(
            title="Test Guide",
            guide_id="TEST001",
            category=IssueCategory.NETWORK,
            severity=IssueSeverity.MEDIUM,
            symptoms=["Symptom 1"],
            causes=["Cause 1"],
            steps=[
                TroubleshootingStep("Step 1", "Description 1", ["Action 1"])
            ]
        )

        # Convert to dictionary
        guide_dict = guide.to_dict()

        # Check dictionary values
        self.assertEqual(guide_dict["title"], "Test Guide")
        self.assertEqual(guide_dict["guide_id"], "TEST001")
        self.assertEqual(guide_dict["category"], "NETWORK")
        self.assertEqual(guide_dict["severity"], "MEDIUM")
        self.assertEqual(guide_dict["symptoms"], ["Symptom 1"])
        self.assertEqual(guide_dict["causes"], ["Cause 1"])
        self.assertEqual(len(guide_dict["steps"]), 1)
        self.assertEqual(guide_dict["steps"][0]["title"], "Step 1")

    def test_from_dict(self):
        """Test creating a guide from a dictionary."""
        # Create a dictionary representation of a guide
        guide_dict = {
            "title": "Test Guide",
            "guide_id": "TEST001",
            "category": "NETWORK",
            "severity": "MEDIUM",
            "symptoms": ["Symptom 1"],
            "causes": ["Cause 1"],
            "steps": [
                {
                    "title": "Step 1",
                    "description": "Description 1",
                    "actions": ["Action 1"]
                }
            ],
            "prevention": ["Prevention 1"],
            "related_issues": ["REL001"],
            "references": ["https://example.com"]
        }

        # Create guide from dictionary
        guide = TroubleshootingGuide.from_dict(guide_dict)

        # Check guide properties
        self.assertEqual(guide.title, "Test Guide")
        self.assertEqual(guide.guide_id, "TEST001")
        self.assertEqual(guide.category, IssueCategory.NETWORK)
        self.assertEqual(guide.severity, IssueSeverity.MEDIUM)
        self.assertEqual(guide.symptoms, ["Symptom 1"])
        self.assertEqual(guide.causes, ["Cause 1"])
        self.assertEqual(len(guide.steps), 1)
        self.assertEqual(guide.steps[0].title, "Step 1")
        self.assertEqual(guide.prevention, ["Prevention 1"])
        self.assertEqual(guide.related_issues, ["REL001"])
        self.assertEqual(guide.references, ["https://example.com"])


class TestTroubleshootingRegistry(unittest.TestCase):
    """Tests for the TroubleshootingRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the singleton instance for each test
        TroubleshootingRegistry._instance = None
        self.registry = TroubleshootingRegistry()

        # Create a test guide
        steps = [
            TroubleshootingStep("Step 1", "Description 1", ["Action 1"])
        ]
        self.test_guide = TroubleshootingGuide(
            title="Test Guide",
            guide_id="TEST001",
            category=IssueCategory.NETWORK,
            severity=IssueSeverity.MEDIUM,
            symptoms=["Cannot connect", "Network timeout"],
            causes=["Network is down"],
            steps=steps
        )

    def test_register_guide(self):
        """Test registering a guide."""
        # Register the guide
        self.registry.register_guide(self.test_guide)

        # Check that the guide was registered
        self.assertIn("TEST001", self.registry.guides)
        self.assertEqual(self.registry.guides["TEST001"], self.test_guide)

    def test_get_guide(self):
        """Test getting a guide by ID."""
        # Register the guide
        self.registry.register_guide(self.test_guide)

        # Get the guide
        guide = self.registry.get_guide("TEST001")

        # Check that the correct guide was returned
        self.assertEqual(guide, self.test_guide)

        # Try to get a non-existent guide
        guide = self.registry.get_guide("NONEXISTENT")
        self.assertIsNone(guide)

    def test_search_guides(self):
        """Test searching for guides."""
        # Register the guide
        self.registry.register_guide(self.test_guide)

        # Create another guide
        steps = [
            TroubleshootingStep("Step 1", "Description 1", ["Action 1"])
        ]
        another_guide = TroubleshootingGuide(
            title="Database Guide",
            guide_id="DB001",
            category=IssueCategory.DATABASE,
            severity=IssueSeverity.HIGH,
            symptoms=["Cannot connect to database", "Query timeout"],
            causes=["Database is down"],
            steps=steps
        )
        self.registry.register_guide(another_guide)

        # Search for guides by keyword
        results = self.registry.search_guides("network")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], self.test_guide)

        # Search for guides by symptom
        results = self.registry.search_guides("timeout")
        self.assertEqual(len(results), 2)  # Both guides have "timeout" in symptoms

        # Search for guides by category
        results = self.registry.search_guides_by_category(IssueCategory.DATABASE)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], another_guide)

        # Search for guides by severity
        results = self.registry.search_guides_by_severity(IssueSeverity.HIGH)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], another_guide)

    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("json.dump")
    def test_save_guides(self, mock_json_dump, mock_open):
        """Test saving guides to a file."""
        # Register the guide
        self.registry.register_guide(self.test_guide)

        # Save guides to a file
        self.registry.save_guides("guides.json")

        # Check that the file was opened and written to
        mock_open.assert_called_once_with("guides.json", "w")
        mock_json_dump.assert_called_once()

    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data=json.dumps([
        {
            "title": "Test Guide",
            "guide_id": "TEST001",
            "category": "NETWORK",
            "severity": "MEDIUM",
            "symptoms": ["Symptom 1"],
            "causes": ["Cause 1"],
            "steps": [
                {
                    "title": "Step 1",
                    "description": "Description 1",
                    "actions": ["Action 1"]
                }
            ]
        }
    ]))
    def test_load_guides(self, mock_open):
        """Test loading guides from a file."""
        # Load guides from a file
        self.registry.load_guides("guides.json")

        # Check that the guide was loaded
        self.assertIn("TEST001", self.registry.guides)
        self.assertEqual(self.registry.guides["TEST001"].title, "Test Guide")


class TestErrorToGuideMapper(unittest.TestCase):
    """Tests for the ErrorToGuideMapper class."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the singleton instances for each test
        ErrorToGuideMapper._instance = None
        TroubleshootingRegistry._instance = None

        self.mapper = ErrorToGuideMapper()
        self.registry = TroubleshootingRegistry()

        # Create a test guide
        steps = [
            TroubleshootingStep("Step 1", "Description 1", ["Action 1"])
        ]
        self.network_guide = TroubleshootingGuide(
            title="Network Guide",
            guide_id="NET001",
            category=IssueCategory.NETWORK,
            severity=IssueSeverity.MEDIUM,
            symptoms=["Cannot connect"],
            causes=["Network is down"],
            steps=steps
        )
        self.registry.register_guide(self.network_guide)

        # Create another guide
        self.database_guide = TroubleshootingGuide(
            title="Database Guide",
            guide_id="DB001",
            category=IssueCategory.DATABASE,
            severity=IssueSeverity.HIGH,
            symptoms=["Cannot connect to database"],
            causes=["Database is down"],
            steps=steps
        )
        self.registry.register_guide(self.database_guide)

    def test_map_error_to_guide(self):
        """Test mapping an error type to a guide."""
        # Map error types to guides
        self.mapper.map_error_to_guide(NetworkError, "NET001")
        self.mapper.map_error_to_guide(DatabaseError, "DB001")

        # Check the mappings
        self.assertIn(NetworkError, self.mapper.error_to_guide_map)
        self.assertEqual(self.mapper.error_to_guide_map[NetworkError], "NET001")
        self.assertIn(DatabaseError, self.mapper.error_to_guide_map)
        self.assertEqual(self.mapper.error_to_guide_map[DatabaseError], "DB001")

    def test_get_guide_for_error(self):
        """Test getting a guide for an error."""
        # Map error types to guides
        self.mapper.map_error_to_guide(NetworkError, "NET001")
        self.mapper.map_error_to_guide(DatabaseError, "DB001")

        # Get guide for a network error
        error = NetworkError("Network connection failed")
        guide = self.mapper.get_guide_for_error(error)

        # Check that the correct guide was returned
        self.assertEqual(guide, self.network_guide)

        # Get guide for a database error
        error = DatabaseError("Database connection failed")
        guide = self.mapper.get_guide_for_error(error)

        # Check that the correct guide was returned
        self.assertEqual(guide, self.database_guide)

        # Get guide for an unmapped error
        error = DataError("Data validation failed")
        guide = self.mapper.get_guide_for_error(error)

        # Check that no guide was returned
        self.assertIsNone(guide)

    def test_get_guide_for_error_with_inheritance(self):
        """Test getting a guide for an error using inheritance."""
        # Map the base error type to a guide
        self.mapper.map_error_to_guide(FridayError, "NET001")

        # Get guide for a specific error type that inherits from the base
        error = NetworkError("Network connection failed")
        guide = self.mapper.get_guide_for_error(error)

        # Check that the guide for the base error was returned
        self.assertEqual(guide, self.network_guide)


class TestDiagnosticCheck(unittest.TestCase):
    """Tests for the DiagnosticCheck class and its subclasses."""

    def test_network_connectivity_check(self):
        """Test the NetworkConnectivityCheck class."""
        # Create a check with a mock function
        mock_function = MagicMock(return_value=True)
        check = NetworkConnectivityCheck(
            "Test Network Check",
            "Check network connectivity",
            "example.com",
            check_function=mock_function
        )

        # Run the check
        result = check.run()

        # Check that the function was called with the correct arguments
        mock_function.assert_called_once_with("example.com")

        # Check the result
        self.assertTrue(result.success)
        self.assertEqual(result.message, "Network connectivity check passed: example.com")

        # Test with a failing check
        mock_function.return_value = False
        result = check.run()

        # Check the result
        self.assertFalse(result.success)
        self.assertEqual(result.message, "Network connectivity check failed: example.com")

    def test_database_connectivity_check(self):
        """Test the DatabaseConnectivityCheck class."""
        # Create a check with a mock function
        mock_function = MagicMock(return_value=True)
        check = DatabaseConnectivityCheck(
            "Test Database Check",
            "Check database connectivity",
            "test_db",
            check_function=mock_function
        )

        # Run the check
        result = check.run()

        # Check that the function was called with the correct arguments
        mock_function.assert_called_once_with("test_db")

        # Check the result
        self.assertTrue(result.success)
        self.assertEqual(result.message, "Database connectivity check passed: test_db")

    def test_file_system_check(self):
        """Test the FileSystemCheck class."""
        # Create a check with a mock function
        mock_function = MagicMock(return_value=True)
        check = FileSystemCheck(
            "Test File System Check",
            "Check file system access",
            "/test/path",
            check_function=mock_function
        )

        # Run the check
        result = check.run()

        # Check that the function was called with the correct arguments
        mock_function.assert_called_once_with("/test/path")

        # Check the result
        self.assertTrue(result.success)
        self.assertEqual(result.message, "File system check passed: /test/path")

    def test_api_endpoint_check(self):
        """Test the ApiEndpointCheck class."""
        # Create a check with a mock function
        mock_function = MagicMock(return_value=True)
        check = ApiEndpointCheck(
            "Test API Check",
            "Check API endpoint",
            "https://api.example.com",
            check_function=mock_function
        )

        # Run the check
        result = check.run()

        # Check that the function was called with the correct arguments
        mock_function.assert_called_once_with("https://api.example.com")

        # Check the result
        self.assertTrue(result.success)
        self.assertEqual(result.message, "API endpoint check passed: https://api.example.com")


class TestDiagnosticRunner(unittest.TestCase):
    """Tests for the DiagnosticRunner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = DiagnosticRunner()

        # Create mock checks
        self.network_check = MagicMock()
        self.network_check.name = "Network Check"
        self.network_check.run.return_value = MagicMock(success=True, message="Network check passed")

        self.database_check = MagicMock()
        self.database_check.name = "Database Check"
        self.database_check.run.return_value = MagicMock(success=False, message="Database check failed")

    def test_add_check(self):
        """Test adding a check to the runner."""
        # Add checks
        self.runner.add_check(self.network_check)
        self.runner.add_check(self.database_check)

        # Check that the checks were added
        self.assertEqual(len(self.runner.checks), 2)
        self.assertIn(self.network_check, self.runner.checks)
        self.assertIn(self.database_check, self.runner.checks)

    def test_run_checks(self):
        """Test running all checks."""
        # Add checks
        self.runner.add_check(self.network_check)
        self.runner.add_check(self.database_check)

        # Run checks
        results = self.runner.run_checks()

        # Check that all checks were run
        self.network_check.run.assert_called_once()
        self.database_check.run.assert_called_once()

        # Check the results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].success, True)
        self.assertEqual(results[0].message, "Network check passed")
        self.assertEqual(results[1].success, False)
        self.assertEqual(results[1].message, "Database check failed")

    def test_run_specific_checks(self):
        """Test running specific checks."""
        # Add checks
        self.runner.add_check(self.network_check)
        self.runner.add_check(self.database_check)

        # Run only the network check
        results = self.runner.run_checks(check_names=["Network Check"])

        # Check that only the network check was run
        self.network_check.run.assert_called_once()
        self.database_check.run.assert_not_called()

        # Check the results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].success, True)
        self.assertEqual(results[0].message, "Network check passed")

    def test_generate_report(self):
        """Test generating a diagnostic report."""
        # Add checks
        self.runner.add_check(self.network_check)
        self.runner.add_check(self.database_check)

        # Run checks
        self.runner.run_checks()

        # Generate report
        report = self.runner.generate_report()

        # Check the report
        self.assertIn("timestamp", report)
        self.assertIn("results", report)
        self.assertEqual(len(report["results"]), 2)
        self.assertEqual(report["results"][0]["name"], "Network Check")
        self.assertEqual(report["results"][0]["success"], True)
        self.assertEqual(report["results"][0]["message"], "Network check passed")
        self.assertEqual(report["results"][1]["name"], "Database Check")
        self.assertEqual(report["results"][1]["success"], False)
        self.assertEqual(report["results"][1]["message"], "Database check failed")
        self.assertEqual(report["summary"]["total"], 2)
        self.assertEqual(report["summary"]["passed"], 1)
        self.assertEqual(report["summary"]["failed"], 1)


class TestTroubleshootingAssistant(unittest.TestCase):
    """Tests for the TroubleshootingAssistant class."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the singleton instances for each test
        TroubleshootingRegistry._instance = None
        ErrorToGuideMapper._instance = None

        self.registry = TroubleshootingRegistry()
        self.mapper = ErrorToGuideMapper()
        self.assistant = TroubleshootingAssistant()

        # Create test guides
        steps = [
            TroubleshootingStep("Step 1", "Description 1", ["Action 1"])
        ]
        self.network_guide = TroubleshootingGuide(
            title="Network Guide",
            guide_id="NET001",
            category=IssueCategory.NETWORK,
            severity=IssueSeverity.MEDIUM,
            symptoms=["Cannot connect", "Network timeout"],
            causes=["Network is down"],
            steps=steps
        )
        self.registry.register_guide(self.network_guide)

        self.database_guide = TroubleshootingGuide(
            title="Database Guide",
            guide_id="DB001",
            category=IssueCategory.DATABASE,
            severity=IssueSeverity.HIGH,
            symptoms=["Cannot connect to database", "Query timeout"],
            causes=["Database is down"],
            steps=steps
        )
        self.registry.register_guide(self.database_guide)

        # Map error types to guides
        self.mapper.map_error_to_guide(NetworkError, "NET001")
        self.mapper.map_error_to_guide(DatabaseError, "DB001")

    def test_get_guide_for_error(self):
        """Test getting a guide for an error."""
        # Get guide for a network error
        error = NetworkError("Network connection failed")
        guide = self.assistant.get_guide_for_error(error)

        # Check that the correct guide was returned
        self.assertEqual(guide, self.network_guide)

    def test_search_guides(self):
        """Test searching for guides."""
        # Search for guides by keyword
        results = self.assistant.search_guides("network")

        # Check the results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], self.network_guide)

    def test_get_guides_for_symptoms(self):
        """Test getting guides for symptoms."""
        # Get guides for symptoms
        results = self.assistant.get_guides_for_symptoms(["Cannot connect", "timeout"])

        # Check the results
        self.assertEqual(len(results), 2)  # Both guides match these symptoms

    def test_run_diagnostics(self):
        """Test running diagnostics."""
        # Create mock checks
        network_check = MagicMock()
        network_check.name = "Network Check"
        network_check.run.return_value = MagicMock(success=True, message="Network check passed")

        database_check = MagicMock()
        database_check.name = "Database Check"
        database_check.run.return_value = MagicMock(success=False, message="Database check failed")

        # Add checks to the assistant's diagnostic runner
        self.assistant.diagnostic_runner.add_check(network_check)
        self.assistant.diagnostic_runner.add_check(database_check)

        # Run diagnostics
        report = self.assistant.run_diagnostics()

        # Check the report
        self.assertIn("timestamp", report)
        self.assertIn("results", report)
        self.assertEqual(len(report["results"]), 2)
        self.assertEqual(report["summary"]["total"], 2)
        self.assertEqual(report["summary"]["passed"], 1)
        self.assertEqual(report["summary"]["failed"], 1)

    def test_generate_custom_guide_for_error(self):
        """Test generating a custom guide for an error."""
        # Create an error with no mapped guide
        error = AuthenticationError("Authentication failed")

        # Generate a custom guide
        guide = self.assistant.generate_custom_guide_for_error(error)

        # Check the guide
        self.assertIsNotNone(guide)
        self.assertEqual(guide.category, IssueCategory.AUTHENTICATION)
        self.assertIn("Authentication failed", guide.symptoms)


class TestInitializeTroubleshooting(unittest.TestCase):
    """Tests for the initialize_troubleshooting function."""

    @patch("src.infrastructure.troubleshooting.TroubleshootingRegistry")
    @patch("src.infrastructure.troubleshooting.ErrorToGuideMapper")
    @patch("src.infrastructure.troubleshooting.register_common_guides")
    def test_initialize_troubleshooting(self, mock_register_common_guides, mock_mapper, mock_registry):
        """Test that initialize_troubleshooting sets up troubleshooting correctly."""
        # Create mock instances
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_mapper_instance = MagicMock()
        mock_mapper.return_value = mock_mapper_instance

        # Call initialize_troubleshooting
        initialize_troubleshooting()

        # Check that the registry and mapper were initialized
        mock_registry.assert_called_once()
        mock_mapper.assert_called_once()

        # Check that common guides were registered
        mock_register_common_guides.assert_called_once()


if __name__ == "__main__":
    unittest.main()