# Test file to verify mongodb import and functionality
from src.infrastructure.database import mongodb

# Test getting a collection
try:
    collection = mongodb.get_collection('test_collection')
    print(f"Successfully got collection: {collection}")
except Exception as e:
    print(f"Error getting collection: {e}")
    import traceback
    traceback.print_exc()

# Test other mongodb functions
try:
    # List available functions in mongodb module
    print(f"Available functions in mongodb module: {dir(mongodb)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()