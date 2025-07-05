#!/usr/bin/env python3
"""Test CLI functionality."""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def run_with_timeout(cmd, timeout=30):
    """Run a command with timeout."""
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(subprocess.run, cmd, shell=True, capture_output=True, text=True)
            result = future.result(timeout=timeout)
            return result.returncode, result.stdout, result.stderr
    except TimeoutError:
        return -1, "", "Command timed out"

def test_cli():
    """Test the CLI functionality."""
    print("Testing Market Data Pipeline CLI...")
    
    # Test 1: Help option
    print("\n1. Testing --help option...")
    cmd = "python -m pipelines.market_data_pipeline --help"
    returncode, stdout, stderr = run_with_timeout(cmd, 10)
    
    if returncode == 0:
        print("✅ Help option works")
    else:
        print(f"❌ Help option failed: {stderr}")
    
    # Test 2: Sample data with no storage
    print("\n2. Testing --sample --no-storage...")
    cmd = "python -m pipelines.market_data_pipeline --sample --outdir test_cli_output --warn-only --no-storage"
    returncode, stdout, stderr = run_with_timeout(cmd, 30)
    
    print(f"Return code: {returncode}")
    if stdout:
        print("STDOUT:")
        print(stdout[-500:])  # Last 500 chars
    if stderr:
        print("STDERR:")
        print(stderr[-500:])  # Last 500 chars
    
    if returncode == 0:
        print("✅ Sample data processing works")
    else:
        print(f"❌ Sample data processing failed")
    
    # Test 3: CSV file processing
    print("\n3. Testing CSV file processing...")
    # First create a simple CSV file
    try:
        import pandas as pd
        import numpy as np
        
        # Create simple test data
        dates = pd.date_range('2023-01-01', periods=50, freq='1min')
        data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'TEST',
            'open': 100 + np.random.randn(50) * 0.1,
            'high': 100.5 + np.random.randn(50) * 0.1,
            'low': 99.5 + np.random.randn(50) * 0.1,
            'close': 100 + np.random.randn(50) * 0.1,
            'volume': np.random.randint(1000, 10000, 50)
        })
        data.to_csv('test_cli_data.csv', index=False)
        print("Created test CSV file")
        
        cmd = "python -m pipelines.market_data_pipeline --input test_cli_data.csv --outdir test_cli_output --warn-only --no-storage"
        returncode, stdout, stderr = run_with_timeout(cmd, 30)
        
        print(f"Return code: {returncode}")
        if stdout:
            print("STDOUT:")
            print(stdout[-500:])  # Last 500 chars
        if stderr:
            print("STDERR:")
            print(stderr[-500:])  # Last 500 chars
        
        if returncode == 0:
            print("✅ CSV file processing works")
        else:
            print(f"❌ CSV file processing failed")
            
    except Exception as e:
        print(f"❌ CSV test setup failed: {e}")

if __name__ == "__main__":
    test_cli()
