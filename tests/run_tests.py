"""
Test runner for the recursive text editor.

This script runs all the tests for the recursive text editor.
"""

import unittest
import sys
import os

# Add the parent directory to the path so that we can import the rledit package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_tests():
    """Run all the tests."""
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(__file__), pattern="test_*.py")
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return the number of failures and errors
    return len(result.failures) + len(result.errors)


if __name__ == "__main__":
    # Run the tests
    exit_code = run_tests()
    
    # Exit with the number of failures and errors
    sys.exit(exit_code)
