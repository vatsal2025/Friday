import sys

print(f'Python version: {sys.version}')
print('Testing f-string escaping in triple quotes:')

# Test 1: Basic f-string with triple quotes
try:
    test1 = f"""This is a test docstring with {{'variable'}} inside"""
    print('Test 1 passed:', test1)
except Exception as e:
    print('Test 1 failed:', str(e))

# Test 2: Nested quotes in f-string
try:
    test2 = f"""This is a test with "quotes" and {{'variable'}} inside"""
    print('Test 2 passed:', test2)
except Exception as e:
    print('Test 2 failed:', str(e))

# Test 3: Similar to what's in strategy_generator.py
try:
    variable = "test value"
    test3 = f"""A docstring with {{variable}} and "quotes""""
    print('Test 3 passed:', test3)
except Exception as e:
    print('Test 3 failed:', str(e))