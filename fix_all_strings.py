import os
import re
import sys

# Path to the original file
original_file = 'src/orchestration/knowledge_engine/strategy_generator.py'

# Check if the file exists
if not os.path.exists(original_file):
    print(f"Error: Original file {original_file} not found")
    sys.exit(1)

# Create a backup of the original file
backup_file = f"{original_file}.bak2"
if not os.path.exists(backup_file):
    import shutil
    shutil.copy2(original_file, backup_file)
    print(f"Created backup at {backup_file}")

# Read the original file
with open(original_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix all potential string issues

# 1. Fix triple-quoted docstrings
content = re.sub(r'"""([^"]*?)"""', r'"""\1"""', content)

# 2. Fix f-strings with triple quotes
content = re.sub(r'f"""([^"]*?)"""', r'f"""\1"""', content)

# 3. Fix string literals with unescaped newlines
content = re.sub(r'"([^"\\]*?)\n([^"]*?)"', r'"\1\\n\2"', content)

# 4. Fix string literals with unescaped quotes
content = re.sub(r'"([^"\\]*?)"([^"]*?)"', r'"\1\\"\2"', content)

# 5. Fix specific issue with code_lines.append
content = re.sub(r"code_lines\.append\(f\"\s*\"\"\"([^\"]*?)\"\"\"\"\)", r'code_lines.append(f"    \"\"\"\1\"\"\"")', content)

# 6. Fix specific issue with self.name = "
# Find all instances of self.name = "strategy_name" and fix them
content = re.sub(r'self\.name = "([^"]*?)"', r'self.name = "\1"', content)

# 7. Fix specific issue with self.type = "
content = re.sub(r'self\.type = "([^"]*?)"', r'self.type = "\1"', content)

# 8. Fix the join issue
content = content.replace("code = '\n'.join(code_lines)", "code = '\\n'.join(code_lines)")

# Write the fixed content back to the file
with open(original_file, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Successfully updated {original_file} with fixed string literals")

# Run a simple test to verify the fix
print("\nRunning a test to verify the fix...")
try:
    # Try to import the module
    from src.orchestration.knowledge_engine.strategy_generator import StrategyGenerator
    print("Successfully imported StrategyGenerator class")
    
    # Create an instance
    sg = StrategyGenerator()
    print("Successfully created StrategyGenerator instance")
    
    print("\nFix verification: SUCCESS - The fix has been applied successfully")
except SyntaxError as e:
    print(f"\nFix verification: FAILED - Syntax error: {e}")
except Exception as e:
    print(f"\nFix verification: FAILED - Other error: {e}")