#!/usr/bin/env python

import os
import sys
import subprocess
import platform
import importlib
from pathlib import Path

# Try to import pkg_resources, install if not available
try:
    import pkg_resources
except ImportError:
    print("pkg_resources not found, installing setuptools...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"])
    import pkg_resources

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(message, status_type='info'):
    """Print a formatted status message"""
    if status_type == 'info':
        print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} {message}")
    elif status_type == 'success':
        print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {message}")
    elif status_type == 'warning':
        print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {message}")
    elif status_type == 'error':
        print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {message}")
    elif status_type == 'header':
        print(f"\n{Colors.HEADER}{Colors.BOLD}{message}{Colors.ENDC}\n")

def check_python_version():
    """Check Python version"""
    print_status("Checking Python version...", "header")
    
    python_version = sys.version_info
    version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
    
    print_status(f"Python version: {version_str}")
    
    if python_version.major == 3 and python_version.minor >= 10:
        print_status("Python version is 3.10 or higher.", "success")
        return True
    else:
        print_status(f"Python 3.10+ is required, but {version_str} is installed.", "error")
        return False

def check_virtual_environment():
    """Check if running in a virtual environment"""
    print_status("Checking virtual environment...", "header")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_status(f"Running in virtual environment: {sys.prefix}", "success")
        return True
    else:
        print_status("Not running in a virtual environment.", "warning")
        print_status("It is recommended to run in a virtual environment.")
        return False

def check_required_packages():
    """Check if required packages are installed"""
    print_status("Checking required packages...", "header")
    
    required_packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "pytest",
        "tqdm",
        "pyyaml",
        "requests",
        "python-dotenv",
        "fastapi",
        "uvicorn",
        "websocket-client",
        "sqlalchemy",
        "redis",
        "pymongo",
        "pydantic",
        "aiohttp",
        "asyncio",
        "httpx",
        "python-multipart",
        "mcp-server",  # Required for MCP Server Configuration
        "mcp-client"   # Required for MCP Server Configuration
    ]
    
    optional_packages = [
        "alpaca-trade-api",
        "ccxt",
        "ta-lib"  # Optional for Phase 1 Task 2
    ]
    
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    # Check required packages
    missing_packages = []
    for package in required_packages:
        package_key = package.lower()
        if package_key in installed_packages:
            print_status(f"{package} {installed_packages[package_key]} is installed.", "success")
        else:
            print_status(f"{package} is not installed.", "error")
            missing_packages.append(package)
    
    # Check optional packages
    print_status("\nChecking optional packages:", "header")
    for package in optional_packages:
        package_key = package.lower()
        if package_key in installed_packages:
            print_status(f"{package} {installed_packages[package_key]} is installed.", "success")
        else:
            print_status(f"{package} is not installed.", "warning")
    
    if missing_packages:
        print_status("\nMissing required packages:", "error")
        for package in missing_packages:
            print_status(f"  - {package}", "error")
        print_status("\nInstall missing packages with:\n  pip install " + " ".join(missing_packages))
        return False
    else:
        print_status("\nAll required packages are installed.", "success")
        return True

def check_import_functionality():
    """Check if packages can be imported and used"""
    print_status("Testing package imports...", "header")
    
    packages_to_test = [
        "numpy",
        "pandas",
        "sklearn",
        "matplotlib",
        "yaml",
        "requests",
        "dotenv",
        "fastapi",
        "uvicorn",
        "websocket",
        "sqlalchemy",
        "redis",
        "pymongo",
        "pydantic",
        "aiohttp",
        "asyncio",
        "httpx",
        "mcp",  # Added MCP for testing
        "mcp_server"   # Added MCP Server for testing
    ]
    
    import_failures = []
    for package in packages_to_test:
        try:
            importlib.import_module(package)
            print_status(f"Successfully imported {package}", "success")
        except ImportError as e:
            print_status(f"Failed to import {package}: {str(e)}", "error")
            import_failures.append(package)
    
    if import_failures:
        print_status("\nSome packages could not be imported.", "error")
        return False
    else:
        print_status("\nAll packages can be imported successfully.", "success")
        return True

def check_env_file():
    """Check if .env file exists and has required variables"""
    print_status("Checking environment variables...", "header")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print_status(".env file not found, but .env.example exists.", "warning")
            print_status("Please create a .env file based on .env.example.")
        else:
            print_status("Neither .env nor .env.example file found.", "error")
        return False
    
    print_status(".env file exists.", "success")
    
    # Check if .env file has content
    env_content = env_file.read_text()
    if not env_content.strip():
        print_status(".env file is empty.", "error")
        return False
    
    # API keys are not required for this phase
    print_status("API keys are not required for Phase 1 Task 2.", "info")
    print_status("You can configure API keys later when needed.", "info")
    
    return True

def main():
    """Main function to verify the environment"""
    print_status("Friday AI Trading System - Environment Verification", "header")
    
    # Store check results individually to better diagnose issues
    python_check = check_python_version()
    venv_check = check_virtual_environment()
    packages_check = check_required_packages()
    import_check = check_import_functionality()
    env_check = check_env_file()
    
    # Consider virtual environment as a warning, not a failure
    critical_checks = [python_check, packages_check, import_check, env_check]
    
    # Summary
    print_status("\nVerification Summary:", "header")
    if all(critical_checks):
        print_status("All critical checks passed! The environment is properly configured.", "success")
        if not venv_check:
            print_status("Note: Running in a virtual environment is recommended but not required.", "info")
        return 0
    else:
        print_status("Some checks failed. Please address the issues above.", "warning")
        return 1

if __name__ == "__main__":
    sys.exit(main())