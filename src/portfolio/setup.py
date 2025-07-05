#!/usr/bin/env python
"""Setup script for the Portfolio Management System."""

from setuptools import setup, find_packages
import os

# Read the contents of README.md
with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read the requirements from requirements.txt
with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="friday-portfolio",
    version="1.0.0",
    description="Portfolio Management System for the Friday AI Trading System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Friday AI Team",
    author_email="info@fridayai.com",
    url="https://github.com/fridayai/friday",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "friday-portfolio=portfolio.cli:main",
        ],
    },
)
