#!/usr/bin/env python
"""
Friday AI Trading System - Dependency Installation Script

This script installs all required dependencies for the Friday AI Trading System.
It checks if the required Python version is installed, creates a virtual environment if needed,
and installs all required packages from requirements.txt.
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path


def check_python_version():
    """Check if the Python version is 3.10 or higher."""
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 10):
        print("Error: Python 3.10 or higher is required.")
        print(f"Current Python version: {sys.version}")
        print("Please install Python 3.10 or higher and try again.")
        return False
    else:
        print(f"Python version check passed: {sys.version}")
        return True


def create_virtual_environment(venv_path):
    """Create a virtual environment if it doesn't exist."""
    venv_dir = Path(venv_path)
    
    if venv_dir.exists():
        print(f"Virtual environment already exists at {venv_dir}")
        return True
    
    print(f"Creating virtual environment at {venv_dir}...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        print("Virtual environment created successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return False


def get_pip_path(venv_path):
    """Get the path to pip in the virtual environment."""
    system = platform.system()
    if system == "Windows":
        pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        pip_path = os.path.join(venv_path, "bin", "pip")
    
    return pip_path


def get_python_path(venv_path):
    """Get the path to python in the virtual environment."""
    system = platform.system()
    if system == "Windows":
        python_path = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        python_path = os.path.join(venv_path, "bin", "python")
    
    return python_path


def install_dependencies(venv_path, requirements_file, upgrade=False):
    """Install dependencies from requirements.txt."""
    pip_path = get_pip_path(venv_path)
    
    if not os.path.exists(pip_path):
        print(f"Error: pip not found at {pip_path}")
        return False
    
    print("Upgrading pip...")
    try:
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error upgrading pip: {e}")
        return False
    
    print(f"Installing dependencies from {requirements_file}...")
    try:
        cmd = [pip_path, "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.extend(["-r", requirements_file])
        
        subprocess.run(cmd, check=True)
        print("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False


def verify_installation(venv_path, packages):
    """Verify that the required packages are installed."""
    python_path = get_python_path(venv_path)
    
    if not os.path.exists(python_path):
        print(f"Error: Python not found at {python_path}")
        return False
    
    print("Verifying installation...")
    all_installed = True
    
    for package in packages:
        try:
            result = subprocess.run(
                [python_path, "-c", f"import {package}; print({package}.__version__)"],
                capture_output=True,
                text=True,
                check=True
            )
            version = result.stdout.strip()
            print(f"✓ {package} {version}")
        except subprocess.CalledProcessError:
            print(f"✗ {package} not installed or failed to import")
            all_installed = False
    
    return all_installed


def main():
    """Main function to install dependencies."""
    parser = argparse.ArgumentParser(description="Install dependencies for Friday AI Trading System")
    parser.add_argument("--venv", default="venv", help="Path to virtual environment")
    parser.add_argument("--requirements", default="requirements.txt", help="Path to requirements.txt")
    parser.add_argument("--upgrade", action="store_true", help="Upgrade existing packages")
    parser.add_argument("--no-venv", action="store_true", help="Skip virtual environment creation")
    
    args = parser.parse_args()
    
    print("====================================================")
    print("Friday AI Trading System - Dependency Installation")
    print("====================================================")
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment if needed
    if not args.no_venv:
        if not create_virtual_environment(args.venv):
            sys.exit(1)
    else:
        print("Skipping virtual environment creation.")
    
    # Install dependencies
    if not install_dependencies(args.venv, args.requirements, args.upgrade):
        sys.exit(1)
    
    # Verify installation
    core_packages = ["pymongo", "redis", "fastapi", "uvicorn", "numpy", "pandas", "sklearn"]
    if not verify_installation(args.venv, core_packages):
        print("\nWarning: Some packages may not be installed correctly.")
        print("You can try running the script again with the --upgrade flag.")
    else:
        print("\nAll dependencies installed successfully.")
        print("You can now run the Friday AI Trading System.")
        print("Run 'python verify_databases.py' to check if MongoDB and Redis are running.")
    
    print("\nTo activate the virtual environment:")
    if platform.system() == "Windows":
        print(f"  {args.venv}\Scripts\activate.bat")
    else:
        print(f"  source {args.venv}/bin/activate")


if __name__ == "__main__":
    main()