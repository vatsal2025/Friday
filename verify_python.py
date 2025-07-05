#!/usr/bin/env python
"""
Friday AI Trading System - Python Environment Verification

This script verifies that the Python environment meets the requirements
for running the Friday AI Trading System, including Python version and
required packages.
"""

import sys
import os
import platform
import subprocess
import importlib.util
from typing import Dict, List, Tuple, Optional, Any

# Minimum required Python version
MIN_PYTHON_VERSION = (3, 10)

# Required packages for the Friday AI Trading System
REQUIRED_PACKAGES = [
    "pymongo",      # MongoDB client
    "redis",        # Redis client
    "fastapi",      # API framework
    "uvicorn",      # ASGI server
    "numpy",        # Numerical computing
    "pandas",       # Data analysis
    "scikit-learn", # Machine learning
    "requests",     # HTTP requests
    "pydantic",     # Data validation
    "python-dotenv", # Environment variables
    "websockets",   # WebSocket support
    "aiohttp",      # Async HTTP client
]

# Optional packages that enhance functionality
OPTIONAL_PACKAGES = [
    "matplotlib",   # Plotting
    "seaborn",      # Statistical data visualization
    "plotly",       # Interactive plots
    "dash",         # Dashboard framework
    "pytest",       # Testing framework
    "black",        # Code formatter
    "flake8",       # Linter
    "mypy",         # Type checking
]


def check_python_version() -> Tuple[bool, str]:
    """Check if the Python version meets the minimum requirements.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating whether the
            Python version meets the requirements, and a message describing
            the result.
    """
    current_version = sys.version_info
    
    if current_version.major < MIN_PYTHON_VERSION[0] or \
       (current_version.major == MIN_PYTHON_VERSION[0] and 
        current_version.minor < MIN_PYTHON_VERSION[1]):
        return False, f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ is required, but you have {sys.version.split()[0]}"
    
    return True, f"Python version {sys.version.split()[0]} meets the requirements"


def check_package_installed(package_name: str) -> Tuple[bool, Optional[str]]:
    """Check if a package is installed.

    Args:
        package_name: The name of the package to check.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating whether
            the package is installed, and the version of the package if it is installed,
            or None if it is not installed.
    """
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return False, None
    
    try:
        # Try to get the version
        package = __import__(package_name)
        version = getattr(package, "__version__", "unknown")
        return True, version
    except (ImportError, AttributeError):
        return True, "unknown"


def check_required_packages() -> Dict[str, Dict[str, Any]]:
    """Check if all required packages are installed.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing the status of each required package.
    """
    results = {}
    
    for package in REQUIRED_PACKAGES:
        installed, version = check_package_installed(package)
        results[package] = {
            "installed": installed,
            "version": version,
            "required": True
        }
    
    for package in OPTIONAL_PACKAGES:
        installed, version = check_package_installed(package)
        results[package] = {
            "installed": installed,
            "version": version,
            "required": False
        }
    
    return results


def check_virtual_env() -> Tuple[bool, str]:
    """Check if running in a virtual environment.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating whether
            the script is running in a virtual environment, and a message
            describing the result.
    """
    in_venv = hasattr(sys, 'real_prefix') or \
              (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        return True, f"Running in virtual environment: {sys.prefix}"
    else:
        return False, "Not running in a virtual environment"


def check_pip_version() -> Tuple[bool, str]:
    """Check the pip version.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating whether
            pip is installed, and a message describing the result.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return False, "pip is not installed or not in the PATH"


def print_system_info() -> None:
    """Print system information."""
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Platform: {platform.platform()}")
    
    # Check if running in a virtual environment
    venv_active, venv_message = check_virtual_env()
    print(f"Virtual Environment: {venv_message}")
    
    # Check pip version
    pip_installed, pip_message = check_pip_version()
    print(f"Pip: {pip_message}")
    
    print()


def print_package_table(package_results: Dict[str, Dict[str, Any]]) -> None:
    """Print a table of package statuses.

    Args:
        package_results: A dictionary containing the status of each package.
    """
    # Calculate column widths
    name_width = max(len("Package"), max(len(name) for name in package_results.keys()))
    version_width = max(len("Version"), max(len(str(info["version"])) for info in package_results.values() if info["version"] is not None))
    status_width = max(len("Status"), len("Not Installed"))
    required_width = max(len("Required"), len("Optional"))
    
    # Print header
    header = f"| {'Package':{name_width}} | {'Version':{version_width}} | {'Status':{status_width}} | {'Required':{required_width}} |"
    separator = f"+-{'-' * name_width}-+-{'-' * version_width}-+-{'-' * status_width}-+-{'-' * required_width}-+"
    
    print(separator)
    print(header)
    print(separator)
    
    # Print rows
    for name, info in sorted(package_results.items()):
        status = "Installed" if info["installed"] else "Not Installed"
        required = "Required" if info["required"] else "Optional"
        version = info["version"] if info["installed"] and info["version"] is not None else ""
        
        row = f"| {name:{name_width}} | {version:{version_width}} | {status:{status_width}} | {required:{required_width}} |"
        print(row)
    
    print(separator)


def print_installation_instructions(missing_packages: List[str]) -> None:
    """Print installation instructions for missing packages.

    Args:
        missing_packages: A list of missing package names.
    """
    if not missing_packages:
        return
    
    print("\nInstallation Instructions:")
    print("To install missing required packages, run:")
    print(f"pip install {' '.join(missing_packages)}")
    print()
    print("Or use the provided install_dependencies.py script:")
    print("python install_dependencies.py")
    print()


def main() -> int:
    """Main function to verify the Python environment.

    Returns:
        int: Exit code. 0 for success, 1 for failure.
    """
    print("====================================================")
    print("Friday AI Trading System - Python Environment Check")
    print("====================================================")
    print()
    
    # Print system information
    print_system_info()
    
    # Check Python version
    python_version_ok, python_version_message = check_python_version()
    print(f"Python Version Check: {python_version_message}")
    
    if not python_version_ok:
        print("\nError: Python version requirement not met.")
        print(f"Please install Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} or later.")
        return 1
    
    # Check required packages
    print("\nChecking required packages...")
    package_results = check_required_packages()
    
    # Print package table
    print_package_table(package_results)
    
    # Check if any required packages are missing
    missing_required = [name for name, info in package_results.items() 
                       if info["required"] and not info["installed"]]
    
    if missing_required:
        print("\nError: Some required packages are missing.")
        print_installation_instructions(missing_required)
        return 1
    
    # Check if any optional packages are missing
    missing_optional = [name for name, info in package_results.items() 
                       if not info["required"] and not info["installed"]]
    
    if missing_optional:
        print("\nNote: Some optional packages are missing.")
        print("These packages are not required but may enhance functionality.")
        print_installation_instructions(missing_optional)
    
    print("\nSuccess! Your Python environment meets all requirements.")
    print("You can now run the Friday AI Trading System.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())