"""
Pytest configuration for the Fama-French five-factor replication test suite.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress verbose logging during tests
logging.getLogger("src").setLevel(logging.WARNING)
