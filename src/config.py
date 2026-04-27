from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

class Config:
    RAW_DATA_DIR      = BASE_DIR / "data/raw"
    PROCESSED_DIR     = BASE_DIR / "data/processed"
    FEATURES_DIR      = BASE_DIR / "data/features"
    MODELS_DIR        = BASE_DIR / "models"
    REPORTS_DIR       = BASE_DIR / "reports"

    LOG_FILES         = ["logon", "device", "email", "file", "http"]

    AFTER_HOURS_START = 18
    AFTER_HOURS_END   = 8
    BASELINE_WINDOW   = 30

    CONTAMINATION     = 0.05
    ANOMALY_THRESHOLD = 0.7
    ZSCORE_ALERT      = 3.0

config = Config()
