import os
import sys
import tempfile
from pathlib import Path

# Never let a test write into the real data/ directory: point MIRA_DATA_DIR at
# a throwaway dir before any mira module (and mira.paths) is imported.
os.environ["MIRA_DATA_DIR"] = tempfile.mkdtemp(prefix="mira-test-data-")
sys.path.insert(0, str(Path(__file__).resolve().parent))
