#!/usr/bin/env python

"""
Test script to verify TaxVisualizer functionality.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Try to import TaxVisualizer
try:
    from src.analytics.visualization import TaxVisualizer
    print("Successfully imported TaxVisualizer")
    
    # Create an instance of TaxVisualizer
    visualizer = TaxVisualizer()
    print("Successfully created TaxVisualizer instance")
    
    # Check if the required methods exist
    methods = [
        'plot_realized_gains',
        'plot_tax_impact',
        'plot_tax_optimization_metrics',
        'plot_tax_efficiency_metrics'
    ]
    
    for method in methods:
        if hasattr(visualizer, method):
            print(f"Method '{method}' exists")
        else:
            print(f"Method '{method}' does not exist")
    
    print("\nTaxVisualizer test completed successfully")
    
except Exception as e:
    print(f"Error: {e}")