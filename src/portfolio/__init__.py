"""Portfolio management module for tracking positions and performance."""

# Import main classes for easier access
from .portfolio_manager import PortfolioManager
from .performance_calculator import PerformanceCalculator
from .tax_manager import TaxManager
from .allocation_manager import AllocationManager
from .portfolio_factory import PortfolioFactory

# Import integration module
try:
    from .portfolio_integration import PortfolioIntegration, create_portfolio_integration
except ImportError:
    # Integration module might not be available in some environments
    pass
