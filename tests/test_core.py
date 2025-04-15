"""
Tests for the core module.
"""

import unittest
from horguesc.core import hello

class TestCore(unittest.TestCase):
    def test_hello(self):
        """Test the hello function."""
        self.assertEqual(hello(), "Hello from horguesc!")

if __name__ == "__main__":
    unittest.main()