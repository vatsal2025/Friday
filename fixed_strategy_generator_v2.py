import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# This is a fixed version of the strategy_generator.py file
# The main issue is in the _generate_python_code method where f-strings with triple quotes are used

def generate_python_code_fixed(strategy: Dict[str, Any]) -> str:
    """
    Generate Python code for a strategy with proper escaping of f-strings and triple quotes.
    
    Args:
        strategy: Strategy to generate code for
        
    Returns:
        String with generated Python code
    """
    strategy_name = strategy.get('name', 'UnnamedStrategy')
    class_name = ''.join(word.title() for word in strategy_name.split())
    
    # Start building the code as regular strings, not f-strings
    code = []
    code.append(f"# Generated Trading Strategy: {strategy_name}")
    code.append(f"# Generated at: {datetime.now().isoformat()}")
    code.append(f"# Strategy Type: {strategy.get('type', 'Unknown')}")
    code.append("")
    code.append("import pandas as pd")
    code.append("import numpy as np")
    code.append("from typing import Dict, List, Any, Optional")
    code.append("")
    code.append(f"class {class_name}:")
    code.append(f"    \"\"\"{strategy.get('description', 'A trading strategy')}\"\"\"")
    code.append("    ")
    code.append("    def __init__(self, parameters: Optional[Dict[str, Any]] = None):")
    code.append("        \"\"\"")
    code.append("        Initialize the strategy with parameters.")
    code.append("        ")
    code.append("        Args:")
    code.append("            parameters: Optional parameters to customize the strategy")
    code.append("        \"\"\"")
    code.append("        self.parameters = parameters if parameters else {}")
    code.append(f"        self.name = \"{strategy_name}\"")
    code.append(f"        self.type = \"{strategy.get('type', 'Unknown')}\"")
    code.append(f"        self.timeframes = {strategy.get('timeframes', ['daily'])}")
    code.append("        ")
    code.append("        # Initialize indicators")
    code.append("        self._setup_indicators()")
    
    # Add _setup_indicators method
    code.append("    ")
    code.append("    def _setup_indicators(self):")
    code.append("        \"\"\"")
    code.append("        Set up the indicators used by this strategy.")
    code.append("        \"\"\"")
    code.append("        self.indicators = {}")
    code.append("        ")
    code.append("        # Set up each indicator")
    
    # Add indicator setup code
    if 'indicators' in strategy and strategy['indicators']:
        for indicator in strategy['indicators']:
            indicator_name = indicator.get('name', 'unknown_indicator')
            code.append(f"        # {indicator.get('description', 'An indicator')}")
            code.append(f"        self.indicators['{indicator_name}'] = {{}}")
    
    # Add calculate_indicators method
    code.append("    ")
    code.append("    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:")
    code.append("        \"\"\"")
    code.append("        Calculate indicators for the strategy.")
    code.append("        ")
    code.append("        Args:")
    code.append("            data: DataFrame with market data")
    code.append("            ")
    code.append("        Returns:")
    code.append("            DataFrame with indicators added")
    code.append("        \"\"\"")
    code.append("        # Make a copy to avoid modifying the original data")
    code.append("        df = data.copy()")
    code.append("        ")
    code.append("        # Calculate indicators")
    
    # Add indicator calculation code
    if 'indicators' in strategy and strategy['indicators']:
        for indicator in strategy['indicators']:
            indicator_name = indicator.get('name', 'unknown_indicator')
            code.append(f"        # Calculate {indicator_name}")
            code.append(f"        # TODO: Implement {indicator_name} calculation")
    
    code.append("        return df")
    
    # Add generate_signals method
    code.append("    ")
    code.append("    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:")
    code.append("        \"\"\"")
    code.append("        Generate trading signals based on the strategy rules.")
    code.append("        ")
    code.append("        Args:")
    code.append("            data: DataFrame with market data and indicators")
    code.append("            ")
    code.append("        Returns:")
    code.append("            DataFrame with signals added")
    code.append("        \"\"\"")
    code.append("        # Make a copy to avoid modifying the original data")
    code.append("        df = data.copy()")
    code.append("        ")
    code.append("        # Initialize signal column")
    code.append("        df['signal'] = 0")
    
    # Add entry rules code
    if 'components' in strategy and 'entry_rules' in strategy['components']:
        entry_rules = strategy['components']['entry_rules'].get('rules', [])
        for i, rule in enumerate(entry_rules):
            code.append(f"        # Entry Rule {i+1}: {rule.get('content', 'No description')}")
            code.append(f"        # TODO: Implement entry rule {i+1}")
    
    code.append("        # Apply exit rules")
    
    # Add exit rules code
    if 'components' in strategy and 'exit_rules' in strategy['components']:
        exit_rules = strategy['components']['exit_rules'].get('rules', [])
        for i, rule in enumerate(exit_rules):
            code.append(f"        # Exit Rule {i+1}: {rule.get('content', 'No description')}")
            code.append(f"        # TODO: Implement exit rule {i+1}")
    
    code.append("        # Apply risk management rules")
    
    # Add risk management rules code
    if 'components' in strategy and 'risk_management' in strategy['components']:
        risk_rules = strategy['components']['risk_management'].get('rules', [])
        for i, rule in enumerate(risk_rules):
            code.append(f"        # Risk Management Rule {i+1}: {rule.get('content', 'No description')}")
            code.append(f"        # TODO: Implement risk management rule {i+1}")
    
    code.append("        return df")
    
    # Add backtest method
    code.append("    ")
    code.append("    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:")
    code.append("        \"\"\"")
    code.append("        Backtest the strategy on historical data.")
    code.append("        ")
    code.append("        Args:")
    code.append("            data: DataFrame with market data")
    code.append("            initial_capital: Initial capital for the backtest")
    code.append("            ")
    code.append("        Returns:")
    code.append("            Dictionary with backtest results")
    code.append("        \"\"\"")
    code.append("        # Calculate indicators")
    code.append("        df = self.calculate_indicators(data)")
    code.append("        ")
    code.append("        # Generate signals")
    code.append("        df = self.generate_signals(df)")
    code.append("        ")
    code.append("        # Initialize backtest results")
    code.append("        results = {")
    code.append("            'strategy_name': self.name,")
    code.append("            'initial_capital': initial_capital,")
    code.append("            'final_capital': initial_capital,")
    code.append("            'returns': 0.0,")
    code.append("            'trades': 0,")
    code.append("            'win_rate': 0.0,")
    code.append("            'max_drawdown': 0.0")
    code.append("        }")
    code.append("        ")
    code.append("        # TODO: Implement backtest logic")
    code.append("        ")
    code.append("        return results")
    
    # Add trade method
    code.append("    ")
    code.append("    def trade(self, current_data: pd.DataFrame, position: Dict[str, Any]) -> Dict[str, Any]:")
    code.append("        \"\"\"")
    code.append("        Generate trading decision based on current market data and position.")
    code.append("        ")
    code.append("        Args:")
    code.append("            current_data: DataFrame with current market data")
    code.append("            position: Dictionary with current position information")
    code.append("            ")
    code.append("        Returns:")
    code.append("            Dictionary with trading decision")
    code.append("        \"\"\"")
    code.append("        # Calculate indicators")
    code.append("        df = self.calculate_indicators(current_data)")
    code.append("        ")
    code.append("        # Generate signals")
    code.append("        df = self.generate_signals(df)")
    code.append("        ")
    code.append("        # Get the latest signal")
    code.append("        latest_signal = df['signal'].iloc[-1]")
    code.append("        ")
    code.append("        # Generate trading decision")
    code.append("        decision = {")
    code.append("            'action': 'hold',  # buy, sell, or hold")
    code.append("            'reason': 'No signal',")
    code.append("            'timestamp': pd.Timestamp.now()")
    code.append("        }")
    code.append("        ")
    code.append("        # TODO: Implement trading decision logic based on signals and position")
    code.append("        ")
    code.append("        return decision")
    
    # Join all lines with newlines
    return '\n'.join(code)

# Test the fixed function
if __name__ == "__main__":
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
    
    # Generate code using the fixed function
    generated_code = generate_python_code_fixed(test_strategy)
    
    # Print the generated code
    print("Generated code:")
    print(generated_code)
    
    # Test if the generated code is valid Python
    try:
        exec(generated_code)
        print("\nCode validation: SUCCESS - The generated code is valid Python")
    except SyntaxError as e:
        print(f"\nCode validation: FAILED - Syntax error: {e}")
    except Exception as e:
        print(f"\nCode validation: FAILED - Other error: {e}")