"""
United Nations Financial Intelligence Dashboard - Source Package

This package contains the core modules for the UN Financial Intelligence Dashboard.
"""

__version__ = "1.0.0"
__author__ = "UN Financial Intelligence Team"

# Ensure proper module resolution for deployment environments
import sys
import os

# Add src directory to Python path if not already present
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
