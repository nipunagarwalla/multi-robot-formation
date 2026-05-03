import os
import sys

# code/ is a flat dir, not a package — mirror how the existing scripts import.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
