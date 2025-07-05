import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    print("Attempting to import StrategyGenerator...")
    from src.orchestration.knowledge_engine.strategy_generator import StrategyGenerator
    print("Successfully imported StrategyGenerator")
    print(f"StrategyGenerator methods: {dir(StrategyGenerator)}")
except Exception as e:
    print(f"Error importing StrategyGenerator: {e}")
    import traceback
    traceback.print_exc()