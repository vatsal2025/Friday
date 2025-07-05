import os
import shutil
import sys

# Path to the original file
original_file = 'src/orchestration/knowledge_engine/strategy_generator.py'
fixed_file = 'strategy_generator_fixed.py'

# Check if the files exist
if not os.path.exists(original_file):
    print(f"Error: Original file {original_file} not found")
    sys.exit(1)

if not os.path.exists(fixed_file):
    print(f"Error: Fixed file {fixed_file} not found")
    sys.exit(1)

# Create a backup of the original file if not already done
backup_file = f"{original_file}.bak_final"
if not os.path.exists(backup_file):
    shutil.copy2(original_file, backup_file)
    print(f"Created backup at {backup_file}")

# Copy the fixed file to the original location
shutil.copy2(fixed_file, original_file)
print(f"Replaced {original_file} with fixed version")

# Run a simple test to verify the fix
print("\nRunning a test to verify the fix...")
try:
    # Try to import the module
    from src.orchestration.knowledge_engine.strategy_generator import StrategyGenerator
    print("Successfully imported StrategyGenerator class")
    
    # Create an instance
    sg = StrategyGenerator()
    print("Successfully created StrategyGenerator instance")
    
    # Test the _generate_python_code method with a simple strategy
    test_strategy = {
        "name": "Test Strategy",
        "description": "A test trading strategy",
        "type": "trend_following",
        "timeframes": ["daily", "weekly"],
        "indicators": [
            {"name": "moving_average", "description": "Simple Moving Average"}
        ],
        "components": {
            "entry_rules": {"rules": [{"content": "Buy when price crosses above MA"}]},
            "exit_rules": {"rules": [{"content": "Sell when price crosses below MA"}]},
            "risk_management": {"rules": [{"content": "Use 2% risk per trade"}]}
        }
    }
    
    # Generate a strategy
    strategy = sg.generate_strategy(test_strategy)
    print("Successfully generated a strategy")
    
    print("\nFix verification: SUCCESS - The fix has been applied successfully")
except SyntaxError as e:
    print(f"\nFix verification: FAILED - Syntax error: {e}")
except Exception as e:
    print(f"\nFix verification: FAILED - Other error: {e}")