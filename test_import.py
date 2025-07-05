import importlib.util
import sys

# Define the path to the test file
test_file_path = 'e:/Friday/tests/orchestration/knowledge_engine/test_knowledge_base_builder.py'

# Load the module directly from the file path
spec = importlib.util.spec_from_file_location('test_knowledge_base_builder', test_file_path)
test_module = importlib.util.module_from_spec(spec)
sys.modules['test_knowledge_base_builder'] = test_module

try:
    spec.loader.exec_module(test_module)
    print(f"Successfully imported {test_file_path}")
    # Access the TestBookKnowledgeExtractor class
    test_class = test_module.TestKnowledgeBaseBuilder
    print(f"Successfully accessed {test_class.__name__}")
except Exception as e:
    print(f"Error importing {test_file_path}: {e}")
    import traceback
    traceback.print_exc()