#!/usr/bin/env python

import os
import sys
import subprocess
import platform
import re
from pathlib import Path

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

def run_command(command, shell=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(
            command,
            shell=shell,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()

def verify_python_version():
    """Verify that Python 3.10+ is installed"""
    print_status("Verifying Python version...", "header")
    
    # Get Python version
    python_version = sys.version_info
    version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
    
    print_status(f"Detected Python version: {version_str}")
    
    # Check if version is 3.10 or higher
    if python_version.major == 3 and python_version.minor >= 10:
        print_status("Python 3.10+ is installed.", "success")
        return True
    else:
        print_status(f"Python 3.10+ is required, but {version_str} is installed.", "error")
        print_status("Please install Python 3.10 or higher and try again.")
        return False

def create_virtual_environment():
    """Create and activate a virtual environment"""
    print_status("Setting up virtual environment...", "header")
    
    venv_dir = "venv"
    
    # Check if virtual environment already exists
    if os.path.exists(venv_dir):
        print_status(f"Virtual environment already exists at {venv_dir}", "warning")
        recreate = input("Do you want to recreate it? (y/n): ").lower() == 'y'
        if recreate:
            import shutil
            shutil.rmtree(venv_dir)
            print_status("Removed existing virtual environment.")
        else:
            print_status("Using existing virtual environment.")
            return True
    
    # Create virtual environment
    print_status(f"Creating virtual environment in {venv_dir}...")
    success, output = run_command([sys.executable, "-m", "venv", venv_dir], shell=False)
    
    if success:
        print_status("Virtual environment created successfully.", "success")
        
        # Print activation instructions
        if platform.system() == "Windows":
            activate_cmd = f"{venv_dir}\\Scripts\\activate"
        else:
            activate_cmd = f"source {venv_dir}/bin/activate"
            
        print_status(f"To activate the virtual environment, run:\n   {activate_cmd}")
        return True
    else:
        print_status(f"Failed to create virtual environment: {output}", "error")
        return False

def install_dependencies():
    """Install dependencies from requirements.txt"""
    print_status("Installing dependencies...", "header")
    
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print_status(f"Requirements file '{req_file}' not found.", "error")
        return False
    
    # Determine pip command based on platform
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip first
    print_status("Upgrading pip to latest version...")
    success, output = run_command(f"{pip_cmd} install --upgrade pip")
    if not success:
        print_status(f"Failed to upgrade pip: {output}", "warning")
    
    # Install dependencies
    print_status(f"Installing dependencies from {req_file}...")
    success, output = run_command(f"{pip_cmd} install -r {req_file}")
    
    if success:
        print_status("Dependencies installed successfully.", "success")
        return True
    else:
        print_status(f"Failed to install dependencies: {output}", "error")
        return False

def check_dependency_conflicts():
    """Check for dependency conflicts"""
    print_status("Checking for dependency conflicts...", "header")
    
    # Determine pip command based on platform
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    success, output = run_command(f"{pip_cmd} check")
    
    if success:
        print_status("No dependency conflicts found.", "success")
        return True
    else:
        print_status("Dependency conflicts found:", "warning")
        print(output)
        return False

def check_for_updates():
    """Check for dependency updates"""
    print_status("Checking for dependency updates...", "header")
    
    # Determine pip command based on platform
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    success, output = run_command(f"{pip_cmd} list --outdated")
    
    if success:
        if "Package" in output and "Version" in output:
            print_status("The following packages have updates available:", "warning")
            print(output)
        else:
            print_status("All packages are up to date.", "success")
        return True
    else:
        print_status(f"Failed to check for updates: {output}", "error")
        return False

def create_env_file():
    """Create .env file from .env.example if it doesn't exist"""
    print_status("Setting up environment variables...", "header")
    
    env_example = ".env.example"
    env_file = ".env"
    
    if not os.path.exists(env_example):
        print_status(f"{env_example} file not found. Skipping environment setup.", "warning")
        return False
    
    if os.path.exists(env_file):
        print_status(f"{env_file} already exists. Skipping creation.", "info")
        return True
    
    # Copy .env.example to .env
    with open(env_example, 'r') as example_file:
        example_content = example_file.read()
    
    with open(env_file, 'w') as env_file_obj:
        env_file_obj.write(example_content)
    
    print_status(f"{env_file} created from {env_example}.", "success")
    print_status("API keys are not required for Phase 1 Task 1.", "info")
    print_status("You can configure API keys later when needed.", "info")
    return True

def verify_environment():
    """Verify that the environment is properly configured"""
    print_status("Verifying environment configuration...", "header")
    
    # Determine python command based on platform
    if platform.system() == "Windows":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    # Try to import key packages
    test_script = """import numpy
import pandas
import sklearn
import matplotlib
import yaml
import requests
import dotenv
import fastapi
import uvicorn
import websocket
import sqlalchemy
import redis
import pymongo
import pydantic
import aiohttp
import asyncio
import httpx
import talib  # TA-Lib is now required
print("All imports successful!")"""
    
    # Write test script to temporary file
    with open("test_imports.py", "w") as f:
        f.write(test_script)
    
    # Run test script
    success, output = run_command(f"{python_cmd} test_imports.py")
    
    # Clean up
    os.remove("test_imports.py")
    
    if success and "All imports successful!" in output:
        print_status("Environment is properly configured.", "success")
        return True
    else:
        print_status("Environment verification failed.", "error")
        print_status(f"Error: {output}", "error")
        return False

def main():
    """Main function to set up the environment"""
    print_status("Friday AI Trading System - Environment Setup", "header")
    
    # Verify Python version
    if not verify_python_version():
        return False
    
    # Create virtual environment
    if not create_virtual_environment():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check for dependency conflicts
    check_dependency_conflicts()
    
    # Check for updates
    check_for_updates()
    
    # Create .env file
    create_env_file()
    
    # Verify environment
    if verify_environment():
        print_status("\nEnvironment setup completed successfully!", "header")
        print_status("To activate the virtual environment, run:")
        if platform.system() == "Windows":
            print_status("   venv\\Scripts\\activate", "info")
        else:
            print_status("   source venv/bin/activate", "info")
        return True
    else:
        print_status("\nEnvironment setup completed with errors.", "header")
        return False

if __name__ == "__main__":
    main()