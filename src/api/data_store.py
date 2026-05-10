import pandas as pd
import json
import logging
from pathlib import Path
from src.config import config
from src.explainability.rule_explainer import generate_reasons

logger = logging.getLogger(__name__)

_df       = None
_alert_df = None
_shap_imp = None


def load_data():
    global _df, _alert_df, _shap_imp

    # ── Dataset ────────────────────────────────────────────────
    data_path = config.FEATURES_DIR / "feature_matrix_ensemble_scored.parquet"
    logger.info(f"Loading dataset from: {data_path}")

    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        return

    _df = pd.read_parquet(data_path)
    _df["date_only"] = pd.to_datetime(_df["date_only"])
    _df["reasons"]   = _df.apply(generate_reasons, axis=1)
    logger.info(f"Dataset loaded: {_df.shape}")

    # ── Alert table ────────────────────────────────────────────
    alert_path = config.REPORTS_DIR / "alerts" / "alert_table.csv"
    logger.info(f"Loading alerts from: {alert_path}")

    if alert_path.exists():
        _alert_df = pd.read_csv(alert_path)
        logger.info(f"Alert table loaded: {len(_alert_df)} rows")
    else:
        logger.warning(f"Alert table not found at {alert_path}")
        _alert_df = pd.DataFrame()

    # ── SHAP importance ────────────────────────────────────────
    shap_path = config.MODELS_DIR / "isolation_forest" / "shap_importance.json"
    logger.info(f"Loading SHAP from: {shap_path}")

    if shap_path.exists():
        with open(shap_path) as f:
            _shap_imp = json.load(f)
        logger.info("SHAP importance loaded")
    else:
        logger.warning(f"SHAP not found at {shap_path}")
        _shap_imp = {}


def get_df():              return _df
def get_alert_df():        return _alert_df
def get_shap_importance(): return _shap_imp or {}
