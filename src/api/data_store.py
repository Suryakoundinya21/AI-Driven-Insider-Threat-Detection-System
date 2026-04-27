import pandas as pd
import json
import logging
from pathlib import Path
from src.config import config
from src.explainability.rule_explainer import generate_reasons
from src.explainability.alert_formatter import format_alert, get_risk_level

logger = logging.getLogger(__name__)

# Global state — loaded once at startup
_df = None
_alert_df = None
_shap_imp = None


def load_data():
    global _df, _alert_df, _shap_imp

    logger.info("Loading ensemble dataset...")

    # ✅ Build dataset path
    data_path = config.FEATURES_DIR / "feature_matrix_ensemble_scored.parquet"

    # ✅ Debug logs (VERY IMPORTANT for Render)
    logger.info(f"Looking for dataset at: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"❌ Missing dataset at {data_path}")

    # ✅ Load dataset
    _df = pd.read_parquet(data_path)

    # ✅ Basic transformations
    _df["date_only"] = pd.to_datetime(_df["date_only"])
    _df["reasons"] = _df.apply(generate_reasons, axis=1)

    logger.info(f"✅ Dataset loaded: {_df.shape}")

    # ==========================================================
    # ALERT TABLE
    # ==========================================================
    logger.info("Loading alert table...")

    alert_path = config.REPORTS_DIR / "alerts" / "alert_table.csv"
    logger.info(f"Looking for alerts at: {alert_path}")

    if alert_path.exists():
        _alert_df = pd.read_csv(alert_path)
        logger.info(f"✅ Alert table loaded: {len(_alert_df)} rows")
    else:
        logger.warning("⚠️ Alert table not found — using empty DataFrame")
        _alert_df = pd.DataFrame()

    # ==========================================================
    # SHAP IMPORTANCE
    # ==========================================================
    shap_path = config.MODELS_DIR / "isolation_forest" / "shap_importance.json"
    logger.info(f"Looking for SHAP at: {shap_path}")

    if shap_path.exists():
        with open(shap_path) as f:
            _shap_imp = json.load(f)
        logger.info("✅ SHAP importance loaded")
    else:
        logger.warning("⚠️ SHAP importance not found — using empty dict")
        _shap_imp = {}


# ==============================================================
# ACCESS FUNCTIONS
# ==============================================================

def get_df() -> pd.DataFrame:
    return _df


def get_alert_df() -> pd.DataFrame:
    return _alert_df


def get_shap_importance() -> dict:
    return _shap_imp or {}