"""Predefined troubleshooting guides for common error scenarios.

This module contains predefined troubleshooting guides for common error scenarios
in the Friday AI Trading System.
"""

from src.infrastructure.troubleshooting import (
    TroubleshootingGuide, TroubleshootingStep, IssueCategory, IssueSeverity,
    TroubleshootingRegistry, ErrorToGuideMapper
)
from src.infrastructure.error import (
    NetworkError, DatabaseError, DataError, AuthenticationError,
    ConfigurationError, SystemError, RetryExhaustedError, APIError
)


def register_network_guides():
    """Register network-related troubleshooting guides."""
    registry = TroubleshootingRegistry()
    mapper = ErrorToGuideMapper()
    
    # API connectivity issues
    api_connectivity_guide = TroubleshootingGuide(
        title="API Connectivity Issues",
        issue_id="NETWORK-001",
        category=IssueCategory.CONNECTIVITY,
        severity=IssueSeverity.HIGH,
        symptoms=[
            "Unable to connect to external APIs",
            "Timeout errors when making API calls",
            "Connection refused errors",
            "SSL/TLS certificate validation errors"
        ],
        possible_causes=[
            "Network firewall blocking connections",
            "DNS resolution issues",
            "API endpoint is down or unreachable",
            "Proxy configuration issues",
            "SSL/TLS certificate issues",
            "Rate limiting or IP blocking by the API provider"
        ],
        steps=[
            TroubleshootingStep(
                description="Check basic connectivity",
                action="Ping the API endpoint to check basic connectivity: ping api.example.com",
                verification="Ping should succeed with responses from the server"
            ),
            TroubleshootingStep(
                description="Check DNS resolution",
                action="Use nslookup or dig to check DNS resolution: nslookup api.example.com",
                verification="DNS resolution should return valid IP addresses"
            ),
            TroubleshootingStep(
                description="Check HTTPS connectivity",
                action="Use curl or a similar tool to test HTTPS connectivity: curl -v https://api.example.com/health",
                verification="Connection should succeed with HTTP 200 OK response"
            ),
            TroubleshootingStep(
                description="Check firewall settings",
                action="Check if the firewall is blocking outbound connections to the API endpoint",
                verification="Firewall should allow outbound connections to the API endpoint on port 443"
            ),
            TroubleshootingStep(
                description="Check proxy settings",
                action="Verify proxy settings in the application configuration",
                verification="Proxy settings should be correct if a proxy is required"
            ),
            TroubleshootingStep(
                description="Check API status",
                action="Check the API status page or contact the API provider",
                verification="API should be operational according to the status page"
            ),
            TroubleshootingStep(
                description="Check API credentials",
                action="Verify that API keys or authentication tokens are valid and not expired",
                verification="Authentication should succeed with the provided credentials"
            ),
            TroubleshootingStep(
                description="Check for rate limiting",
                action="Review API usage and check if you're hitting rate limits",
                verification="API usage should be within allowed limits"
            )
        ],
        prevention=[
            "Implement circuit breakers to handle API outages",
            "Set up monitoring for API connectivity",
            "Implement fallback mechanisms for critical operations",
            "Use exponential backoff for retries",
            "Monitor API usage to avoid rate limiting"
        ],
        references=[
            "https://docs.example.com/api/troubleshooting",
            "https://status.example.com"
        ]
    )
    registry.register_guide(api_connectivity_guide)
    mapper.register_mapping(NetworkError, "NETWORK-001")
    
    # Market data connectivity issues
    market_data_guide = TroubleshootingGuide(
        title="Market Data Connectivity Issues",
        issue_id="NETWORK-002",
        category=IssueCategory.CONNECTIVITY,
        severity=IssueSeverity.CRITICAL,
        symptoms=[
            "Unable to retrieve market data",
            "Stale or outdated market data",
            "Missing price updates",
            "Websocket disconnections"
        ],
        possible_causes=[
            "Market data provider outage",
            "Network connectivity issues",
            "Authentication or subscription issues",
            "Websocket connection instability",
            "Rate limiting by the data provider"
        ],
        steps=[
            TroubleshootingStep(
                description="Check market data provider status",
                action="Check the status page of the market data provider",
                verification="Market data provider should be operational"
            ),
            TroubleshootingStep(
                description="Check network connectivity",
                action="Verify network connectivity to the market data provider",
                verification="Network connectivity should be stable"
            ),
            TroubleshootingStep(
                description="Check authentication",
                action="Verify that API keys or authentication tokens for the market data provider are valid",
                verification="Authentication should succeed with the provided credentials"
            ),
            TroubleshootingStep(
                description="Check subscription status",
                action="Verify that you have an active subscription for the required market data",
                verification="Subscription should be active and include the required data"
            ),
            TroubleshootingStep(
                description="Check websocket connection",
                action="Monitor websocket connection status and reconnection attempts",
                verification="Websocket connection should be stable with minimal disconnections"
            ),
            TroubleshootingStep(
                description="Check for rate limiting",
                action="Review data usage and check if you're hitting rate limits",
                verification="Data usage should be within allowed limits"
            )
        ],
        prevention=[
            "Implement automatic reconnection for websocket connections",
            "Set up monitoring for market data connectivity",
            "Implement fallback data sources",
            "Cache recent market data for temporary outages",
            "Implement circuit breakers to handle provider outages"
        ],
        references=[
            "https://docs.example.com/market-data/troubleshooting",
            "https://status.marketdataprovider.com"
        ]
    )
    registry.register_guide(market_data_guide)
    
    # Websocket connectivity issues
    websocket_guide = TroubleshootingGuide(
        title="Websocket Connectivity Issues",
        issue_id="NETWORK-003",
        category=IssueCategory.CONNECTIVITY,
        severity=IssueSeverity.HIGH,
        symptoms=[
            "Frequent websocket disconnections",
            "Unable to establish websocket connection",
            "Missing real-time updates",
            "Websocket connection timeouts"
        ],
        possible_causes=[
            "Network instability",
            "Firewall or proxy blocking websocket connections",
            "Server-side connection limits",
            "Authentication issues",
            "Client-side resource constraints"
        ],
        steps=[
            TroubleshootingStep(
                description="Check network stability",
                action="Monitor network stability and packet loss",
                verification="Network should be stable with minimal packet loss"
            ),
            TroubleshootingStep(
                description="Check firewall settings",
                action="Verify that firewalls allow websocket connections (typically on ports 80/443)",
                verification="Firewall should allow websocket connections"
            ),
            TroubleshootingStep(
                description="Check authentication",
                action="Verify that authentication tokens for websocket connections are valid",
                verification="Authentication should succeed with the provided tokens"
            ),
            TroubleshootingStep(
                description="Check connection limits",
                action="Verify that you're not exceeding server-side connection limits",
                verification="Connection count should be within allowed limits"
            ),
            TroubleshootingStep(
                description="Check client resources",
                action="Monitor client-side resource usage (memory, CPU, file descriptors)",
                verification="Resource usage should be within acceptable limits"
            ),
            TroubleshootingStep(
                description="Check reconnection logic",
                action="Review websocket reconnection logic for proper backoff and retry behavior",
                verification="Reconnection logic should use exponential backoff"
            )
        ],
        prevention=[
            "Implement robust reconnection logic with exponential backoff",
            "Set up monitoring for websocket connection status",
            "Implement heartbeats to detect connection issues early",
            "Optimize resource usage for websocket connections",
            "Use connection pooling if multiple connections are needed"
        ],
        references=[
            "https://docs.example.com/websocket/troubleshooting",
            "https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API/Writing_WebSocket_client_applications"
        ]
    )
    registry.register_guide(websocket_guide)


def register_database_guides():
    """Register database-related troubleshooting guides."""
    registry = TroubleshootingRegistry()
    mapper = ErrorToGuideMapper()
    
    # Database connectivity issues
    db_connectivity_guide = TroubleshootingGuide(
        title="Database Connectivity Issues",
        issue_id="DATABASE-001",
        category=IssueCategory.DATABASE,
        severity=IssueSeverity.HIGH,
        symptoms=[
            "Unable to connect to the database",
            "Database query timeouts",
            "Connection pool exhaustion",
            "'Too many connections' errors"
        ],
        possible_causes=[
            "Database server is down or unreachable",
            "Network connectivity issues",
            "Authentication issues",
            "Connection pool configuration issues",
            "Database server resource constraints",
            "Firewall blocking database connections"
        ],
        steps=[
            TroubleshootingStep(
                description="Check database server status",
                action="Verify that the database server is running and accessible",
                verification="Database server should be running and responding to basic queries"
            ),
            TroubleshootingStep(
                description="Check network connectivity",
                action="Test network connectivity to the database server: ping db.example.com",
                verification="Network connectivity should be available with stable latency"
            ),
            TroubleshootingStep(
                description="Check authentication",
                action="Verify database credentials and permissions",
                verification="Authentication should succeed with the provided credentials"
            ),
            TroubleshootingStep(
                description="Check connection pool settings",
                action="Review connection pool configuration (max connections, timeout, etc.)",
                verification="Connection pool should be properly configured for the workload"
            ),
            TroubleshootingStep(
                description="Check database server resources",
                action="Monitor database server resource usage (CPU, memory, disk I/O)",
                verification="Resource usage should be within acceptable limits"
            ),
            TroubleshootingStep(
                description="Check for connection leaks",
                action="Review application code for potential connection leaks",
                verification="Connections should be properly closed after use"
            ),
            TroubleshootingStep(
                description="Check firewall settings",
                action="Verify that firewalls allow connections to the database port",
                verification="Firewall should allow connections to the database port"
            )
        ],
        prevention=[
            "Implement database connection retry logic",
            "Set up monitoring for database connectivity and performance",
            "Implement circuit breakers for database operations",
            "Use connection pooling with appropriate settings",
            "Implement proper connection cleanup in application code",
            "Set up database server monitoring and alerting"
        ],
        references=[
            "https://docs.example.com/database/troubleshooting",
            "https://www.postgresql.org/docs/current/runtime-config-connection.html"
        ]
    )
    registry.register_guide(db_connectivity_guide)
    mapper.register_mapping(DatabaseError, "DATABASE-001")
    
    # Database query performance issues
    query_performance_guide = TroubleshootingGuide(
        title="Database Query Performance Issues",
        issue_id="DATABASE-002",
        category=IssueCategory.DATABASE,
        severity=IssueSeverity.MEDIUM,
        symptoms=[
            "Slow database queries",
            "Query timeouts",
            "High database CPU usage",
            "Increasing query execution times"
        ],
        possible_causes=[
            "Missing or inefficient indexes",
            "Complex queries with joins on large tables",
            "Table fragmentation",
            "Outdated statistics",
            "Resource contention",
            "Large result sets"
        ],
        steps=[
            TroubleshootingStep(
                description="Identify slow queries",
                action="Use database monitoring tools to identify slow queries",
                verification="Slow queries should be identified with execution plans"
            ),
            TroubleshootingStep(
                description="Analyze query execution plans",
                action="Use EXPLAIN or similar tools to analyze query execution plans",
                verification="Execution plans should be analyzed for inefficiencies"
            ),
            TroubleshootingStep(
                description="Check indexes",
                action="Verify that appropriate indexes exist for the queries",
                verification="Indexes should exist for columns used in WHERE, JOIN, and ORDER BY clauses"
            ),
            TroubleshootingStep(
                description="Update statistics",
                action="Update database statistics for accurate query planning",
                verification="Statistics should be up-to-date"
            ),
            TroubleshootingStep(
                description="Optimize queries",
                action="Rewrite inefficient queries to use indexes and reduce complexity",
                verification="Optimized queries should have improved execution times"
            ),
            TroubleshootingStep(
                description="Check for resource contention",
                action="Monitor for locks, blocking, and resource contention",
                verification="Minimal contention should be observed"
            ),
            TroubleshootingStep(
                description="Consider query caching",
                action="Implement query caching for frequently executed queries",
                verification="Cached queries should have significantly reduced execution times"
            )
        ],
        prevention=[
            "Regularly review and optimize database queries",
            "Implement query performance monitoring",
            "Use appropriate indexes for common query patterns",
            "Regularly update database statistics",
            "Implement query timeouts to prevent long-running queries",
            "Consider read replicas for read-heavy workloads"
        ],
        references=[
            "https://docs.example.com/database/performance",
            "https://use-the-index-luke.com/"
        ]
    )
    registry.register_guide(query_performance_guide)
    
    # Database connection pool issues
    connection_pool_guide = TroubleshootingGuide(
        title="Database Connection Pool Issues",
        issue_id="DATABASE-003",
        category=IssueCategory.DATABASE,
        severity=IssueSeverity.HIGH,
        symptoms=[
            "Connection pool exhaustion",
            "Long wait times for database connections",
            "'Too many connections' errors",
            "Connection timeouts"
        ],
        possible_causes=[
            "Insufficient pool size for workload",
            "Connection leaks in application code",
            "Long-running transactions holding connections",
            "Inefficient connection usage patterns",
            "Database server connection limits"
        ],
        steps=[
            TroubleshootingStep(
                description="Check connection pool metrics",
                action="Monitor connection pool metrics (active, idle, waiting connections)",
                verification="Connection pool usage should be within expected ranges"
            ),
            TroubleshootingStep(
                description="Check for connection leaks",
                action="Review application code for potential connection leaks",
                verification="Connections should be properly closed after use"
            ),
            TroubleshootingStep(
                description="Check transaction duration",
                action="Monitor transaction durations to identify long-running transactions",
                verification="Transactions should complete within expected timeframes"
            ),
            TroubleshootingStep(
                description="Review connection pool configuration",
                action="Verify connection pool settings (max size, timeout, idle timeout)",
                verification="Connection pool should be configured appropriately for the workload"
            ),
            TroubleshootingStep(
                description="Check database server connection limits",
                action="Verify the maximum connections setting on the database server",
                verification="Database server should allow sufficient connections"
            ),
            TroubleshootingStep(
                description="Optimize connection usage",
                action="Review application code for inefficient connection usage patterns",
                verification="Connections should be used efficiently and released promptly"
            )
        ],
        prevention=[
            "Implement proper connection handling with try-finally blocks",
            "Set appropriate connection pool sizes based on workload",
            "Monitor connection pool metrics",
            "Implement connection timeouts to prevent hanging connections",
            "Use connection pooling middleware",
            "Consider using a connection proxy for advanced connection management"
        ],
        references=[
            "https://docs.example.com/database/connection-pooling",
            "https://github.com/brettwooldridge/HikariCP"
        ]
    )
    registry.register_guide(connection_pool_guide)


def register_data_guides():
    """Register data-related troubleshooting guides."""
    registry = TroubleshootingRegistry()
    mapper = ErrorToGuideMapper()
    
    # Data validation issues
    data_validation_guide = TroubleshootingGuide(
        title="Data Validation Issues",
        issue_id="DATA-001",
        category=IssueCategory.DATA,
        severity=IssueSeverity.MEDIUM,
        symptoms=[
            "Data validation errors",
            "Schema validation failures",
            "Type conversion errors",
            "Missing required fields"
        ],
        possible_causes=[
            "Data source providing invalid data",
            "Schema changes in data source",
            "Incorrect data transformation logic",
            "Inconsistent data formats",
            "Bugs in validation logic"
        ],
        steps=[
            TroubleshootingStep(
                description="Examine validation errors",
                action="Review validation error messages to understand specific issues",
                verification="Error messages should provide clear information about validation failures"
            ),
            TroubleshootingStep(
                description="Check data source",
                action="Verify that the data source is providing data in the expected format",
                verification="Data source should provide data matching the expected schema"
            ),
            TroubleshootingStep(
                description="Check for schema changes",
                action="Compare current data schema with expected schema",
                verification="Data schema should match the expected schema"
            ),
            TroubleshootingStep(
                description="Review transformation logic",
                action="Examine data transformation logic for errors",
                verification="Transformation logic should correctly handle the input data"
            ),
            TroubleshootingStep(
                description="Test validation logic",
                action="Test validation logic with sample data to verify correctness",
                verification="Validation logic should correctly identify valid and invalid data"
            ),
            TroubleshootingStep(
                description="Check for edge cases",
                action="Test validation with edge cases (null values, empty strings, etc.)",
                verification="Validation should handle edge cases correctly"
            )
        ],
        prevention=[
            "Implement comprehensive data validation",
            "Use schema validation libraries",
            "Monitor data quality metrics",
            "Implement data quality checks in the pipeline",
            "Document expected data formats and schemas",
            "Set up alerts for data validation failures"
        ],
        references=[
            "https://docs.example.com/data/validation",
            "https://json-schema.org/"
        ]
    )
    registry.register_guide(data_validation_guide)
    mapper.register_mapping(DataError, "DATA-001")
    
    # Missing or incomplete data issues
    missing_data_guide = TroubleshootingGuide(
        title="Missing or Incomplete Data Issues",
        issue_id="DATA-002",
        category=IssueCategory.DATA,
        severity=IssueSeverity.HIGH,
        symptoms=[
            "Missing data points",
            "Incomplete data sets",
            "Data gaps in time series",
            "Partial responses from data sources"
        ],
        possible_causes=[
            "Data source outages",
            "API rate limiting",
            "Network connectivity issues",
            "Data processing errors",
            "Filtering or aggregation issues",
            "Timing issues in data collection"
        ],
        steps=[
            TroubleshootingStep(
                description="Identify missing data",
                action="Analyze data to identify specific missing elements or patterns",
                verification="Missing data patterns should be identified"
            ),
            TroubleshootingStep(
                description="Check data source availability",
                action="Verify that data sources were available during the collection period",
                verification="Data sources should have been available"
            ),
            TroubleshootingStep(
                description="Check for rate limiting",
                action="Review API usage logs for rate limiting or throttling",
                verification="API usage should be within allowed limits"
            ),
            TroubleshootingStep(
                description="Check data processing logs",
                action="Review data processing logs for errors or warnings",
                verification="Data processing should complete without errors"
            ),
            TroubleshootingStep(
                description="Verify data collection timing",
                action="Check if data collection timing aligns with data availability",
                verification="Data collection should occur when data is available"
            ),
            TroubleshootingStep(
                description="Test data retrieval",
                action="Manually retrieve data from sources to verify availability",
                verification="Data should be retrievable from sources"
            )
        ],
        prevention=[
            "Implement data completeness checks",
            "Set up monitoring for data collection processes",
            "Use retry mechanisms for transient failures",
            "Implement fallback data sources",
            "Cache data to handle source outages",
            "Set up alerts for data gaps"
        ],
        references=[
            "https://docs.example.com/data/completeness",
            "https://docs.example.com/data/collection"
        ]
    )
    registry.register_guide(missing_data_guide)
    
    # Data processing pipeline issues
    pipeline_guide = TroubleshootingGuide(
        title="Data Processing Pipeline Issues",
        issue_id="DATA-003",
        category=IssueCategory.DATA,
        severity=IssueSeverity.HIGH,
        symptoms=[
            "Pipeline failures",
            "Stuck or hanging pipeline stages",
            "Inconsistent processing results",
            "Data quality issues in processed data"
        ],
        possible_causes=[
            "Errors in pipeline stages",
            "Resource constraints",
            "Dependency failures",
            "Data format incompatibilities between stages",
            "Concurrency issues",
            "Configuration errors"
        ],
        steps=[
            TroubleshootingStep(
                description="Check pipeline logs",
                action="Review pipeline execution logs for errors or warnings",
                verification="Logs should provide information about pipeline execution"
            ),
            TroubleshootingStep(
                description="Identify failing stages",
                action="Determine which pipeline stages are failing or problematic",
                verification="Failing stages should be identified"
            ),
            TroubleshootingStep(
                description="Check resource usage",
                action="Monitor resource usage during pipeline execution",
                verification="Resource usage should be within acceptable limits"
            ),
            TroubleshootingStep(
                description="Verify dependencies",
                action="Check that all pipeline dependencies are available and functioning",
                verification="Dependencies should be available and functioning"
            ),
            TroubleshootingStep(
                description="Test individual stages",
                action="Test problematic pipeline stages in isolation",
                verification="Individual stages should function correctly"
            ),
            TroubleshootingStep(
                description="Check data formats",
                action="Verify that data formats are compatible between pipeline stages",
                verification="Data formats should be compatible"
            ),
            TroubleshootingStep(
                description="Review pipeline configuration",
                action="Check pipeline configuration for errors or misconfigurations",
                verification="Pipeline should be correctly configured"
            )
        ],
        prevention=[
            "Implement comprehensive pipeline monitoring",
            "Use pipeline orchestration tools",
            "Implement data quality checks between stages",
            "Design pipelines with fault tolerance",
            "Implement retry mechanisms for transient failures",
            "Set up alerts for pipeline failures"
        ],
        references=[
            "https://docs.example.com/data/pipeline",
            "https://airflow.apache.org/docs/"
        ]
    )
    registry.register_guide(pipeline_guide)


def register_authentication_guides():
    """Register authentication-related troubleshooting guides."""
    registry = TroubleshootingRegistry()
    mapper = ErrorToGuideMapper()
    
    # Authentication issues
    auth_guide = TroubleshootingGuide(
        title="Authentication Issues",
        issue_id="AUTH-001",
        category=IssueCategory.AUTHENTICATION,
        severity=IssueSeverity.HIGH,
        symptoms=[
            "Authentication failures",
            "Unauthorized access errors",
            "Token expiration errors",
            "Invalid credentials errors"
        ],
        possible_causes=[
            "Invalid or expired credentials",
            "Incorrect authentication method",
            "Authentication service issues",
            "Clock synchronization issues",
            "Permission configuration issues",
            "Network connectivity problems"
        ],
        steps=[
            TroubleshootingStep(
                description="Check credentials",
                action="Verify that authentication credentials are correct and not expired",
                verification="Credentials should be valid and current"
            ),
            TroubleshootingStep(
                description="Check authentication method",
                action="Confirm that the correct authentication method is being used",
                verification="Authentication method should match the service requirements"
            ),
            TroubleshootingStep(
                description="Check authentication service",
                action="Verify that the authentication service is operational",
                verification="Authentication service should be responding normally"
            ),
            TroubleshootingStep(
                description="Check clock synchronization",
                action="Verify that system clocks are synchronized (important for token-based auth)",
                verification="System clock should be accurately synchronized"
            ),
            TroubleshootingStep(
                description="Check permissions",
                action="Review permission settings for the authenticated user or service",
                verification="User or service should have the required permissions"
            ),
            TroubleshootingStep(
                description="Check network connectivity",
                action="Verify network connectivity to authentication services",
                verification="Network connectivity should be stable"
            ),
            TroubleshootingStep(
                description="Review authentication logs",
                action="Examine authentication logs for specific error messages",
                verification="Logs should provide detailed information about authentication failures"
            )
        ],
        prevention=[
            "Implement token refresh logic",
            "Set up monitoring for authentication failures",
            "Use secure credential storage",
            "Implement proper error handling for authentication issues",
            "Use authentication libraries rather than custom implementations",
            "Set up alerts for unusual authentication patterns"
        ],
        references=[
            "https://docs.example.com/auth/troubleshooting",
            "https://auth0.com/docs/troubleshoot"
        ]
    )
    registry.register_guide(auth_guide)
    mapper.register_mapping(AuthenticationError, "AUTH-001")
    
    # Authorization issues
    authorization_guide = TroubleshootingGuide(
        title="Authorization Issues",
        issue_id="AUTH-002",
        category=IssueCategory.AUTHENTICATION,
        severity=IssueSeverity.HIGH,
        symptoms=[
            "Access denied errors",
            "Insufficient permissions errors",
            "Forbidden resource access",
            "Role-based access control failures"
        ],
        possible_causes=[
            "Insufficient user permissions",
            "Incorrect role assignments",
            "Policy configuration issues",
            "Resource ownership problems",
            "Missing or incorrect scopes in tokens",
            "Authorization service issues"
        ],
        steps=[
            TroubleshootingStep(
                description="Check user permissions",
                action="Verify that the user has the required permissions for the requested action",
                verification="User should have the necessary permissions"
            ),
            TroubleshootingStep(
                description="Check role assignments",
                action="Review role assignments for the user or service",
                verification="User or service should be assigned appropriate roles"
            ),
            TroubleshootingStep(
                description="Check policy configuration",
                action="Examine authorization policy configuration",
                verification="Policies should be correctly configured"
            ),
            TroubleshootingStep(
                description="Check resource ownership",
                action="Verify resource ownership and access control settings",
                verification="Resource access controls should be properly set"
            ),
            TroubleshootingStep(
                description="Check token scopes",
                action="Examine token scopes for OAuth or similar token-based authorization",
                verification="Tokens should include the required scopes"
            ),
            TroubleshootingStep(
                description="Check authorization service",
                action="Verify that the authorization service is operational",
                verification="Authorization service should be responding normally"
            ),
            TroubleshootingStep(
                description="Review authorization logs",
                action="Examine authorization logs for specific error messages",
                verification="Logs should provide detailed information about authorization failures"
            )
        ],
        prevention=[
            "Implement principle of least privilege",
            "Document required permissions for each operation",
            "Set up monitoring for authorization failures",
            "Implement proper error handling for authorization issues",
            "Use authorization frameworks rather than custom implementations",
            "Regularly audit permissions and access controls"
        ],
        references=[
            "https://docs.example.com/auth/authorization",
            "https://www.oauth.com/oauth2-servers/scope/"
        ]
    )
    registry.register_guide(authorization_guide)


def register_api_guides():
    """Register API-related troubleshooting guides."""
    registry = TroubleshootingRegistry()
    mapper = ErrorToGuideMapper()
    
    # API error response issues
    api_error_guide = TroubleshootingGuide(
        title="API Error Response Issues",
        issue_id="API-001",
        category=IssueCategory.API,
        severity=IssueSeverity.HIGH,
        symptoms=[
            "HTTP error status codes (4xx, 5xx)",
            "API request failures",
            "Unexpected API response formats",
            "API validation errors"
        ],
        possible_causes=[
            "Invalid request parameters",
            "Authentication or authorization issues",
            "API rate limiting or throttling",
            "API service issues",
            "API version incompatibilities",
            "Network connectivity problems"
        ],
        steps=[
            TroubleshootingStep(
                description="Check error response",
                action="Examine the API error response for specific error codes and messages",
                verification="Error response should provide detailed information about the failure"
            ),
            TroubleshootingStep(
                description="Verify request parameters",
                action="Confirm that request parameters are valid and properly formatted",
                verification="Request parameters should meet API requirements"
            ),
            TroubleshootingStep(
                description="Check authentication",
                action="Verify that authentication credentials are valid and properly included",
                verification="Authentication should be correct"
            ),
            TroubleshootingStep(
                description="Check for rate limiting",
                action="Review API usage and check for rate limiting or throttling headers",
                verification="API usage should be within allowed limits"
            ),
            TroubleshootingStep(
                description="Check API status",
                action="Verify that the API service is operational",
                verification="API service should be available"
            ),
            TroubleshootingStep(
                description="Check API version",
                action="Confirm that you're using a supported API version",
                verification="API version should be supported"
            ),
            TroubleshootingStep(
                description="Test with API tools",
                action="Use tools like Postman or curl to test API requests directly",
                verification="Direct API requests should help isolate the issue"
            )
        ],
        prevention=[
            "Implement comprehensive error handling for API requests",
            "Validate request parameters before sending",
            "Use API client libraries when available",
            "Implement retry logic with exponential backoff",
            "Monitor API usage to avoid rate limiting",
            "Set up alerts for API failures"
        ],
        references=[
            "https://docs.example.com/api/errors",
            "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status"
        ]
    )
    registry.register_guide(api_error_guide)
    mapper.register_mapping(APIError, "API-001")
    
    # API rate limiting issues
    rate_limiting_guide = TroubleshootingGuide(
        title="API Rate Limiting Issues",
        issue_id="API-002",
        category=IssueCategory.API,
        severity=IssueSeverity.MEDIUM,
        symptoms=[
            "HTTP 429 Too Many Requests errors",
            "Rate limit exceeded messages",
            "Throttling notifications",
            "Increasing API request failures"
        ],
        possible_causes=[
            "Exceeding API rate limits",
            "Inefficient API usage patterns",
            "Concurrent requests exceeding limits",
            "Insufficient rate limit allocation",
            "Missing rate limit headers in responses"
        ],
        steps=[
            TroubleshootingStep(
                description="Check rate limit status",
                action="Examine API responses for rate limit headers (X-RateLimit-*)",
                verification="Rate limit headers should provide information about limits and usage"
            ),
            TroubleshootingStep(
                description="Review API usage patterns",
                action="Analyze API request patterns and frequency",
                verification="Request patterns should be identified"
            ),
            TroubleshootingStep(
                description="Check for request batching",
                action="Determine if requests can be batched to reduce API calls",
                verification="Opportunities for request batching should be identified"
            ),
            TroubleshootingStep(
                description="Implement caching",
                action="Evaluate if response caching can reduce API calls",
                verification="Cacheable responses should be identified"
            ),
            TroubleshootingStep(
                description="Review concurrency settings",
                action="Check concurrent request settings in API clients",
                verification="Concurrency should be appropriate for rate limits"
            ),
            TroubleshootingStep(
                description="Contact API provider",
                action="If limits are too restrictive, contact the API provider about increased limits",
                verification="API provider should be contacted if necessary"
            )
        ],
        prevention=[
            "Implement rate limiting awareness in API clients",
            "Use response caching to reduce API calls",
            "Batch API requests when possible",
            "Implement exponential backoff for retry attempts",
            "Monitor API usage against limits",
            "Distribute requests evenly over time"
        ],
        references=[
            "https://docs.example.com/api/rate-limits",
            "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429"
        ]
    )
    registry.register_guide(rate_limiting_guide)
    
    # API response parsing issues
    parsing_guide = TroubleshootingGuide(
        title="API Response Parsing Issues",
        issue_id="API-003",
        category=IssueCategory.API,
        severity=IssueSeverity.MEDIUM,
        symptoms=[
            "JSON parsing errors",
            "Unexpected response formats",
            "Missing fields in API responses",
            "Type conversion errors"
        ],
        possible_causes=[
            "API response format changes",
            "Incorrect content type handling",
            "API version incompatibilities",
            "Unexpected null or empty values",
            "Character encoding issues",
            "Malformed responses from API"
        ],
        steps=[
            TroubleshootingStep(
                description="Examine raw response",
                action="Capture and examine the raw API response",
                verification="Raw response should be available for analysis"
            ),
            TroubleshootingStep(
                description="Check response format",
                action="Verify that the response format matches expectations",
                verification="Response format should match expected schema"
            ),
            TroubleshootingStep(
                description="Check content type",
                action="Confirm that content type headers are correctly interpreted",
                verification="Content type handling should be correct"
            ),
            TroubleshootingStep(
                description="Check API version",
                action="Verify that you're using the correct API version",
                verification="API version should be compatible with client code"
            ),
            TroubleshootingStep(
                description="Test with API tools",
                action="Use tools like Postman or curl to examine responses directly",
                verification="Direct API requests should help isolate the issue"
            ),
            TroubleshootingStep(
                description="Update parsing code",
                action="Update response parsing code to handle the current response format",
                verification="Parsing code should handle the response correctly"
            )
        ],
        prevention=[
            "Use schema validation for API responses",
            "Implement defensive parsing with proper error handling",
            "Stay updated on API changes and versioning",
            "Test with a variety of response scenarios",
            "Use API client libraries when available",
            "Log unexpected response formats for analysis"
        ],
        references=[
            "https://docs.example.com/api/responses",
            "https://json-schema.org/"
        ]
    )
    registry.register_guide(parsing_guide)


def register_system_guides():
    """Register system-related troubleshooting guides."""
    registry = TroubleshootingRegistry()
    mapper = ErrorToGuideMapper()
    
    # System resource issues
    resource_guide = TroubleshootingGuide(
        title="System Resource Issues",
        issue_id="SYSTEM-001",
        category=IssueCategory.SYSTEM,
        severity=IssueSeverity.HIGH,
        symptoms=[
            "High CPU usage",
            "Memory exhaustion",
            "Disk space shortages",
            "Slow system performance",
            "Out of memory errors"
        ],
        possible_causes=[
            "Resource-intensive processes",
            "Memory leaks",
            "Insufficient system resources",
            "Inefficient resource usage",
            "Background processes consuming resources",
            "Temporary file accumulation"
        ],
        steps=[
            TroubleshootingStep(
                description="Monitor resource usage",
                action="Use system monitoring tools to track CPU, memory, and disk usage",
                verification="Resource usage patterns should be identified"
            ),
            TroubleshootingStep(
                description="Identify resource-intensive processes",
                action="Determine which processes are consuming the most resources",
                verification="Resource-intensive processes should be identified"
            ),
            TroubleshootingStep(
                description="Check for memory leaks",
                action="Monitor memory usage over time to identify potential leaks",
                verification="Memory usage patterns should be analyzed"
            ),
            TroubleshootingStep(
                description="Check disk space",
                action="Verify available disk space and identify large files or directories",
                verification="Disk space usage should be analyzed"
            ),
            TroubleshootingStep(
                description="Review system configuration",
                action="Check system configuration for resource limits and allocations",
                verification="System configuration should be appropriate for workload"
            ),
            TroubleshootingStep(
                description="Optimize resource usage",
                action="Implement resource usage optimizations in application code",
                verification="Resource usage should be optimized"
            ),
            TroubleshootingStep(
                description="Consider scaling resources",
                action="Evaluate if additional system resources are needed",
                verification="Resource scaling needs should be assessed"
            )
        ],
        prevention=[
            "Implement resource usage monitoring",
            "Set up alerts for resource thresholds",
            "Regularly clean up temporary files",
            "Implement efficient resource management in application code",
            "Use resource pooling where appropriate",
            "Consider containerization for resource isolation"
        ],
        references=[
            "https://docs.example.com/system/resources",
            "https://www.brendangregg.com/linuxperf.html"
        ]
    )
    registry.register_guide(resource_guide)
    mapper.register_mapping(SystemError, "SYSTEM-001")
    
    # Configuration issues
    config_guide = TroubleshootingGuide(
        title="Configuration Issues",
        issue_id="SYSTEM-002",
        category=IssueCategory.CONFIGURATION,
        severity=IssueSeverity.MEDIUM,
        symptoms=[
            "Missing configuration values",
            "Invalid configuration settings",
            "Configuration-related errors",
            "Unexpected system behavior"
        ],
        possible_causes=[
            "Missing configuration files",
            "Environment variables not set",
            "Configuration format errors",
            "Incorrect configuration values",
            "Configuration version mismatches",
            "Permission issues with configuration files"
        ],
        steps=[
            TroubleshootingStep(
                description="Check configuration files",
                action="Verify that configuration files exist and are accessible",
                verification="Configuration files should exist and be readable"
            ),
            TroubleshootingStep(
                description="Validate configuration format",
                action="Check configuration files for syntax or format errors",
                verification="Configuration format should be valid"
            ),
            TroubleshootingStep(
                description="Check environment variables",
                action="Verify that required environment variables are set",
                verification="Environment variables should be set correctly"
            ),
            TroubleshootingStep(
                description="Review configuration values",
                action="Check if configuration values are appropriate and valid",
                verification="Configuration values should be valid"
            ),
            TroubleshootingStep(
                description="Check configuration version",
                action="Verify that configuration version matches system requirements",
                verification="Configuration version should be compatible"
            ),
            TroubleshootingStep(
                description="Check file permissions",
                action="Verify that configuration files have appropriate permissions",
                verification="File permissions should allow access by the application"
            )
        ],
        prevention=[
            "Implement configuration validation",
            "Use configuration management tools",
            "Document required configuration settings",
            "Provide default values where appropriate",
            "Implement configuration change logging",
            "Use version control for configuration files"
        ],
        references=[
            "https://docs.example.com/system/configuration",
            "https://12factor.net/config"
        ]
    )
    registry.register_guide(config_guide)
    mapper.register_mapping(ConfigurationError, "SYSTEM-002")


def register_retry_guides():
    """Register retry-related troubleshooting guides."""
    registry = TroubleshootingRegistry()
    mapper = ErrorToGuideMapper()
    
    # Retry exhaustion issues
    retry_guide = TroubleshootingGuide(
        title="Retry Exhaustion Issues",
        issue_id="RETRY-001",
        category=IssueCategory.PERFORMANCE,
        severity=IssueSeverity.HIGH,
        symptoms=[
            "Retry attempts exhausted errors",
            "Persistent operation failures despite retries",
            "Increasing retry latency",
            "Timeout errors after multiple retries"
        ],
        possible_causes=[
            "Persistent underlying issue not resolved by retries",
            "Inadequate retry strategy",
            "Insufficient retry attempts",
            "Inappropriate retry delay",
            "Non-retryable errors being retried",
            "Timeout settings too short for operation"
        ],
        steps=[
            TroubleshootingStep(
                description="Identify the underlying issue",
                action="Examine error logs to determine the root cause of failures",
                verification="Root cause should be identified"
            ),
            TroubleshootingStep(
                description="Review retry strategy",
                action="Check if the retry strategy is appropriate for the operation",
                verification="Retry strategy should match operation characteristics"
            ),
            TroubleshootingStep(
                description="Check retry attempts",
                action="Verify if the maximum retry attempts are sufficient",
                verification="Retry attempts should be appropriate for the operation"
            ),
            TroubleshootingStep(
                description="Review retry delay",
                action="Check if retry delay and backoff strategy are appropriate",
                verification="Retry delay should allow for issue resolution"
            ),
            TroubleshootingStep(
                description="Check error classification",
                action="Verify that errors are correctly classified as retryable or non-retryable",
                verification="Error classification should be correct"
            ),
            TroubleshootingStep(
                description="Check timeout settings",
                action="Review timeout settings for operations and retries",
                verification="Timeout settings should be appropriate"
            )
        ],
        prevention=[
            "Implement appropriate retry strategies for different operations",
            "Use exponential backoff with jitter for retries",
            "Correctly classify errors as retryable or non-retryable",
            "Set appropriate timeout values",
            "Implement circuit breakers for persistent failures",
            "Monitor retry patterns and adjust strategies accordingly"
        ],
        references=[
            "https://docs.example.com/error/retry",
            "https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/"
        ]
    )
    registry.register_guide(retry_guide)
    mapper.register_mapping(RetryExhaustedError, "RETRY-001")


def register_all_guides():
    """Register all troubleshooting guides."""
    register_network_guides()
    register_database_guides()
    register_data_guides()
    register_authentication_guides()
    register_api_guides()
    register_system_guides()
    register_retry_guides()