import os
import shutil
import sys

# Path to the original file
original_file = 'src/orchestration/knowledge_engine/strategy_generator.py'

# Check if the file exists
if not os.path.exists(original_file):
    print(f"Error: Original file {original_file} not found")
    sys.exit(1)

# Create a backup of the original file
backup_file = f"{original_file}.bak"
shutil.copy2(original_file, backup_file)
print(f"Created backup at {backup_file}")

# Read the original file
with open(original_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the _generate_python_code method
start_marker = 'def _generate_python_code(self, strategy: Dict[str, Any]) -> str:'
end_marker = '        return code'

# Extract the method content
start_index = content.find(start_marker)
if start_index == -1:
    print("Error: Could not find _generate_python_code method")
    sys.exit(1)

end_index = content.find(end_marker, start_index) + len(end_marker)
if end_index == -1:
    print("Error: Could not find end of _generate_python_code method")
    sys.exit(1)

original_method = content[start_index:end_index]

# Define the fixed method
fixed_method = '''
def _generate_python_code(self, strategy: Dict[str, Any]) -> str:
    """
    Generate Python code for a strategy.
    
    Args:
        strategy: Strategy to generate code for
        
    Returns:
        String with generated Python code
    """
    # In a production system, this would be more sophisticated
    # For this implementation, we'll generate a basic template
    
    strategy_name = strategy.get('name', 'UnnamedStrategy')
    class_name = ''.join(word.title() for word in strategy_name.split())
    
    # Start building the code as regular strings, not f-strings
    code_lines = []
    code_lines.append(f"# Generated Trading Strategy: {strategy_name}")
    code_lines.append(f"# Generated at: {datetime.now().isoformat()}")
    code_lines.append(f"# Strategy Type: {strategy.get('type', 'Unknown')}")
    code_lines.append("")
    code_lines.append("import pandas as pd")
    code_lines.append("import numpy as np")
    code_lines.append("from typing import Dict, List, Any, Optional")
    code_lines.append("")
    code_lines.append(f"class {class_name}:")
    code_lines.append(f"    \"\"\"{strategy.get('description', 'A trading strategy')}\"\"\"")
    code_lines.append("    ")
    code_lines.append("    def __init__(self, parameters: Optional[Dict[str, Any]] = None):")
    code_lines.append("        \"\"\"")
    code_lines.append("        Initialize the strategy with parameters.")
    code_lines.append("        ")
    code_lines.append("        Args:")
    code_lines.append("            parameters: Optional parameters to customize the strategy")
    code_lines.append("        \"\"\"")
    code_lines.append("        self.parameters = parameters if parameters else {}")
    code_lines.append(f"        self.name = \"{strategy_name}\"")
    code_lines.append(f"        self.type = \"{strategy.get('type', 'Unknown')}\"")
    code_lines.append(f"        self.timeframes = {strategy.get('timeframes', ['daily'])}")
    code_lines.append("        ")
    code_lines.append("        # Initialize indicators")
    code_lines.append("        self._setup_indicators()")
    
    # Add _setup_indicators method
    code_lines.append("    ")
    code_lines.append("    def _setup_indicators(self):")
    code_lines.append("        \"\"\"")
    code_lines.append("        Set up the indicators used by this strategy.")
    code_lines.append("        \"\"\"")
    code_lines.append("        self.indicators = {}")
    code_lines.append("        ")
    code_lines.append("        # Set up each indicator")
    
    # Add indicator setup code
    if 'indicators' in strategy and strategy['indicators']:
        for indicator in strategy['indicators']:
            indicator_name = indicator.get('name', 'unknown_indicator')
            code_lines.append(f"        # {indicator.get('description', 'An indicator')}")
            code_lines.append(f"        self.indicators['{indicator_name}'] = {{}}")
    
    # Add calculate_indicators method
    code_lines.append("    ")
    code_lines.append("    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:")
    code_lines.append("        \"\"\"")
    code_lines.append("        Calculate indicators for the strategy.")
    code_lines.append("        ")
    code_lines.append("        Args:")
    code_lines.append("            data: DataFrame with market data")
    code_lines.append("            ")
    code_lines.append("        Returns:")
    code_lines.append("            DataFrame with indicators added")
    code_lines.append("        \"\"\"")
    code_lines.append("        # Make a copy to avoid modifying the original data")
    code_lines.append("        df = data.copy()")
    code_lines.append("        ")
    code_lines.append("        # Calculate indicators")
    
    # Add indicator calculation code
    if 'indicators' in strategy and strategy['indicators']:
        for indicator in strategy['indicators']:
            indicator_name = indicator.get('name', 'unknown_indicator')
            code_lines.append(f"        # Calculate {indicator_name}")
            code_lines.append(f"        # TODO: Implement {indicator_name} calculation")
    
    code_lines.append("        return df")
    
    # Add generate_signals method
    code_lines.append("    ")
    code_lines.append("    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:")
    code_lines.append("        \"\"\"")
    code_lines.append("        Generate trading signals based on the strategy rules.")
    code_lines.append("        ")
    code_lines.append("        Args:")
    code_lines.append("            data: DataFrame with market data and indicators")
    code_lines.append("            ")
    code_lines.append("        Returns:")
    code_lines.append("            DataFrame with signals added")
    code_lines.append("        \"\"\"")
    code_lines.append("        # Make a copy to avoid modifying the original data")
    code_lines.append("        df = data.copy()")
    code_lines.append("        ")
    code_lines.append("        # Initialize signal column")
    code_lines.append("        df['signal'] = 0")
    
    # Add entry rules code
    if 'components' in strategy and 'entry_rules' in strategy['components']:
        entry_rules = strategy['components']['entry_rules'].get('rules', [])
        for i, rule in enumerate(entry_rules):
            code_lines.append(f"        # Entry Rule {i+1}: {rule.get('content', 'No description')}")
            code_lines.append(f"        # TODO: Implement entry rule {i+1}")
    
    code_lines.append("        # Apply exit rules")
    
    # Add exit rules code
    if 'components' in strategy and 'exit_rules' in strategy['components']:
        exit_rules = strategy['components']['exit_rules'].get('rules', [])
        for i, rule in enumerate(exit_rules):
            code_lines.append(f"        # Exit Rule {i+1}: {rule.get('content', 'No description')}")
            code_lines.append(f"        # TODO: Implement exit rule {i+1}")
    
    code_lines.append("        # Apply risk management rules")
    
    # Add risk management rules code
    if 'components' in strategy and 'risk_management' in strategy['components']:
        risk_rules = strategy['components']['risk_management'].get('rules', [])
        for i, rule in enumerate(risk_rules):
            code_lines.append(f"        # Risk Management Rule {i+1}: {rule.get('content', 'No description')}")
            code_lines.append(f"        # TODO: Implement risk management rule {i+1}")
    
    code_lines.append("        return df")
    
    # Add backtest method
    code_lines.append("    ")
    code_lines.append("    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:")
    code_lines.append("        \"\"\"")
    code_lines.append("        Backtest the strategy on historical data.")
    code_lines.append("        ")
    code_lines.append("        Args:")
    code_lines.append("            data: DataFrame with market data")
    code_lines.append("            initial_capital: Initial capital for the backtest")
    code_lines.append("            ")
    code_lines.append("        Returns:")
    code_lines.append("            Dictionary with backtest results")
    code_lines.append("        \"\"\"")
    code_lines.append("        # Calculate indicators")
    code_lines.append("        df = self.calculate_indicators(data)")
    code_lines.append("        ")
    code_lines.append("        # Generate signals")
    code_lines.append("        df = self.generate_signals(df)")
    code_lines.append("        ")
    code_lines.append("        # Initialize backtest results")
    code_lines.append("        results = {")
    code_lines.append("            'strategy_name': self.name,")
    code_lines.append("            'initial_capital': initial_capital,")
    code_lines.append("            'final_capital': initial_capital,")
    code_lines.append("            'returns': 0.0,")
    code_lines.append("            'trades': 0,")
    code_lines.append("            'win_rate': 0.0,")
    code_lines.append("            'max_drawdown': 0.0")
    code_lines.append("        }")
    code_lines.append("        ")
    code_lines.append("        # TODO: Implement backtest logic")
    code_lines.append("        ")
    code_lines.append("        return results")
    
    # Add trade method
    code_lines.append("    ")
    code_lines.append("    def trade(self, current_data: pd.DataFrame, position: Dict[str, Any]) -> Dict[str, Any]:")
    code_lines.append("        \"\"\"")
    code_lines.append("        Generate trading decision based on current market data and position.")
    code_lines.append("        ")
    code_lines.append("        Args:")
    code_lines.append("            current_data: DataFrame with current market data")
    code_lines.append("            position: Dictionary with current position information")
    code_lines.append("            ")
    code_lines.append("        Returns:")
    code_lines.append("            Dictionary with trading decision")
    code_lines.append("        \"\"\"")
    code_lines.append("        # Calculate indicators")
    code_lines.append("        df = self.calculate_indicators(current_data)")
    code_lines.append("        ")
    code_lines.append("        # Generate signals")
    code_lines.append("        df = self.generate_signals(df)")
    code_lines.append("        ")
    code_lines.append("        # Get the latest signal")
    code_lines.append("        latest_signal = df['signal'].iloc[-1]")
    code_lines.append("        ")
    code_lines.append("        # Generate trading decision")
    code_lines.append("        decision = {")
    code_lines.append("            'action': 'hold',  # buy, sell, or hold")
    code_lines.append("            'reason': 'No signal',")
    code_lines.append("            'timestamp': pd.Timestamp.now()")
    code_lines.append("        }")
    code_lines.append("        ")
    code_lines.append("        # TODO: Implement trading decision logic based on signals and position")
    code_lines.append("        ")
    code_lines.append("        return decision")
    
    # Join all lines with newlines
    code = '\n'.join(code_lines)
    
    return code'''

# Replace the original method with the fixed method
fixed_content = content.replace(original_method, fixed_method)

# Write the fixed content back to the file
with open(original_file, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print(f"Successfully updated {original_file} with the fixed _generate_python_code method")
print("The original file has been backed up in case you need to restore it")

# Run a simple test to verify the fix
print("\nRunning a test to verify the fix...")
try:
    # Try to import the module
    from src.orchestration.knowledge_engine.strategy_generator import StrategyGenerator
    print("Successfully imported StrategyGenerator class")
    
    # Create an instance
    sg = StrategyGenerator()
    print("Successfully created StrategyGenerator instance")
    
    # Test the _generate_python_code method
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
    
    code = sg._generate_python_code(test_strategy)
    print("Successfully generated code using the fixed method")
    
    # Test if the generated code is valid Python
    exec(code)
    print("Successfully executed the generated code")
    
    print("\nFix verification: SUCCESS - The fix has been applied successfully")
except SyntaxError as e:
    print(f"\nFix verification: FAILED - Syntax error: {e}")
except Exception as e:
    print(f"\nFix verification: FAILED - Other error: {e}")