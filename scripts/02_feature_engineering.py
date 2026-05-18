import sys
import os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import logging
from src.config import config
from src.features.session_builder import build_sessions
from src.features.feature_extractor import extract_features
from src.features.baseline import compute_rolling_baseline, flag_zscore_anomalies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading processed log files...")
    logs = {}
    for name in config.LOG_FILES:
        path = config.PROCESSED_DIR / f"{name}_processed.parquet"
        if not path.exists():
            logger.warning(f"Missing: {path} - skipping")
            continue
        logs[name] = pd.read_parquet(path)
        logger.info(f"  {name}: {len(logs[name]):,} rows")

    sessions = build_sessions(logs)
    logger.info(f"Session matrix: {sessions.shape}")

    features = extract_features(sessions)
    logger.info(f"Feature matrix: {features.shape}")

    features_with_baseline = compute_rolling_baseline(features)
    features_final = flag_zscore_anomalies(features_with_baseline)

    print("\n" + "="*60)
    print("FEATURE MATRIX SUMMARY")
    print("="*60)
    print(f"Total sessions    : {len(features_final):,}")
    print(f"Unique users      : {features_final['user'].nunique()}")
    print(f"Date range        : {features_final['date_only'].min()} to {features_final['date_only'].max()}")
    print(f"Total features    : {features_final.shape[1]}")
    print(f"Z-score anomalies : {features_final['zscore_anomaly'].sum():,}")

    print("\nTop 10 highest risk sessions:")
    cols = ["user", "date_only", "max_zscore", "composite_risk_score",
            "device_count", "email_to_external", "sensitive_file_count",
            "http_suspicious", "zscore_anomaly"]
    available = [c for c in cols if c in features_final.columns]
    print(features_final.nlargest(10, "max_zscore")[available].to_string(index=False))

    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FEATURES_DIR / "feature_matrix.parquet"
    features_final.to_parquet(out_path, index=False)
    logger.info(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
