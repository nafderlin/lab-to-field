import sys
from pathlib import Path

# Import lab_to_field from src/ without installing the full (PyTorch) project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
