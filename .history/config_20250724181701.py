# config.py
from pathlib import Path

# Automatically resolve the root directory (where this config file is)
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Adjust if needed

# Define commonly used paths
RAW_DATA_PATH = PROJECT_ROOT / "Data" / "Raw" / "Sample - Superstore.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "Data" / "Processed" / "cleaned_superstore.csv"
FEATURE_ENGINEERED_PATH = PROJECT_ROOT / "Data" / "Processed" / "feature_engineered_superstore.csv"