import sys
import os

# Print Python version and path for debugging
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

# Add the current directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Backup the original strategy_generator.py module
original_module_path = os.path.join(os.path.dirname(__file__), 'src', 'services', 'strategy_generator.py')
backup_module_path = os.path.join(os.path.dirname(__file__), 'src', 'services', 'strategy_generator.py.bak')

fixed_module_path = os.path.join(os.path.dirname(__file__), 'fixed_strategy_generator.py')

try:
    # Check if we need to create a backup
    if os.path.exists(original_module_path) and not os.path.exists(backup_module_path):
        print(f"Creating backup of original strategy_generator.py")
        with open(original_module_path, 'r') as f:
            original_content = f.read()
        with open(backup_module_path, 'w') as f:
            f.write(original_content)
    
    # Replace the problematic file with our fixed version
    if os.path.exists(fixed_module_path):
        print(f"Replacing strategy_generator.py with fixed version")
        with open(fixed_module_path, 'r') as f:
            fixed_content = f.read()
        with open(original_module_path, 'w') as f:
            f.write(fixed_content)
    
    # Now try to import the problematic modules
    print("Attempting to import KnowledgeBaseBuilder...")
    from src.services.knowledge_base_builder import KnowledgeBaseBuilder
    print("Successfully imported KnowledgeBaseBuilder")
    
    print("Attempting to import StrategyGenerator...")
    from src.services.strategy_generator import StrategyGenerator
    print("Successfully imported StrategyGenerator")
    
    # Test if we can create instances
    strategy_generator = StrategyGenerator()
    print("Successfully created StrategyGenerator instance")
    
    # Test if we can call methods
    print("Testing StrategyGenerator methods:")
    print(f"generate_strategy: {strategy_generator.generate_strategy.__name__}")
    print(f"save_strategy: {strategy_generator.save_strategy.__name__}")
    print(f"load_strategy: {strategy_generator.load_strategy.__name__}")
    print(f"generate_code: {strategy_generator.generate_code.__name__}")
    print(f"trade: {strategy_generator.trade.__name__}")
    
    print("All imports and tests successful!")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Restore the original file if we created a backup
    if os.path.exists(backup_module_path):
        print(f"Restoring original strategy_generator.py from backup")
        with open(backup_module_path, 'r') as f:
            original_content = f.read()
        with open(original_module_path, 'w') as f:
            f.write(original_content)
        # os.remove(backup_module_path)  # Uncomment to remove the backup file