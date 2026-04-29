import pandas as pd
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# All data served from deployment_data/ — tiny files only
DEPLOY_DIR = Path("deployment_data")

_alert_df    = None
_user_summary= None
_timeline    = None
_flagged     = None
_shap_imp    = None
_overview    = None
_counts      = None
_model_comp  = None
_top50       = None


def load_data():
    global _alert_df, _user_summary, _timeline, _flagged
    global _shap_imp, _overview, _counts, _model_comp, _top50

    logger.info(f"Loading from {DEPLOY_DIR.resolve()}")

    # Alert table
    p = DEPLOY_DIR / "alert_table.csv"
    if p.exists():
        _alert_df = pd.read_csv(p)
        logger.info(f"Alerts: {len(_alert_df)}")
    else:
        logger.warning("alert_table.csv not found")
        _alert_df = pd.DataFrame()

    # User summary
    p = DEPLOY_DIR / "user_summary.parquet"
    if p.exists():
        _user_summary = pd.read_parquet(p)
        logger.info(f"Users: {len(_user_summary)}")

    # Timeline (all sessions, minimal cols)
    p = DEPLOY_DIR / "timeline.parquet"
    if p.exists():
        _timeline = pd.read_parquet(p)
        logger.info(f"Timeline: {len(_timeline)}")

    # Flagged sessions only
    p = DEPLOY_DIR / "flagged_sessions.parquet"
    if p.exists():
        _flagged = pd.read_parquet(p)
        logger.info(f"Flagged: {len(_flagged)}")

    # Static JSONs
    for attr, fname in [
        ("_shap_imp",   "shap_importance.json"),
        ("_overview",   "overview_stats.json"),
        ("_counts",     "alert_counts.json"),
    ]:
        p = DEPLOY_DIR / fname
        if p.exists():
            with open(p) as f:
                globals()[attr] = json.load(f)

    # Model comparison
    p = DEPLOY_DIR / "model_comparison.csv"
    if p.exists():
        _model_comp = pd.read_csv(p)

    # Top 50 alerts JSON
    p = DEPLOY_DIR / "top50_alerts.json"
    if p.exists():
        with open(p) as f:
            _top50 = json.load(f)

    logger.info("All deployment data loaded successfully")


def get_alert_df():    return _alert_df
def get_user_summary():return _user_summary
def get_timeline():    return _timeline
def get_flagged():     return _flagged
def get_shap_imp():    return _shap_imp or {}
def get_overview():    return _overview or {}
def get_counts():      return _counts or {}
def get_model_comp():  return _model_comp
def get_top50():       return _top50 or []
