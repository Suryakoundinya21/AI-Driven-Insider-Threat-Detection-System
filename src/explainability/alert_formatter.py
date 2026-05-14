import pandas as pd
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

RISK_LEVELS = {
    (0.8, 1.01) : "CRITICAL",
    (0.6, 0.80) : "HIGH",
    (0.4, 0.60) : "MEDIUM",
    (0.0, 0.40) : "LOW",
}


def get_risk_level(score: float) -> str:
    for (lo, hi), label in RISK_LEVELS.items():
        if lo <= score < hi:
            return label
    return "LOW"


def format_alert(row: pd.Series) -> dict:
    score    = float(row.get("ensemble_score", 0))
    risk     = get_risk_level(score)
    reasons  = row.get("reasons", [])
    date_str = str(row.get("date_only", ""))[:10]

    # Ensure all reason fields are JSON-safe strings
    clean_reasons = []
    for r in reasons:
        clean_reasons.append({
            "severity" : str(r.get("severity", "LOW")),
            "feature"  : str(r.get("feature", "")),
            "reason"   : str(r.get("reason", "")),
            "value"    : str(r.get("value", "0")),
            "baseline" : str(r.get("baseline", "0")),
            "zscore"   : str(r.get("zscore", "0")),
        })

    return {
        "alert_id"       : f"ALERT-{str(row.get('user','')).upper()}-{date_str.replace('-','')}",
        "user"           : str(row.get("user", "")),
        "date"           : date_str,
        "risk_level"     : risk,
        "ensemble_score" : round(score, 4),
        "ae_score"       : round(float(row.get("ae_anomaly_score", 0) or 0), 4),
        "if_score"       : round(float(row.get("if_anomaly_score", 0) or 0), 4),
        "ae_flagged"     : bool(row.get("ae_anomaly_flag", 0)),
        "if_flagged"     : bool(row.get("if_anomaly_flag", 0)),
        "both_flagged"   : bool(
            row.get("ae_anomaly_flag", 0) and row.get("if_anomaly_flag", 0)
        ),
        "reasons"        : clean_reasons,
        "reason_summary" : "; ".join([r["reason"] for r in clean_reasons[:2]]),
        "stats"          : {
            "device_count"      : int(row.get("device_count", 0) or 0),
            "email_to_external" : int(row.get("email_to_external", 0) or 0),
            "http_suspicious"   : int(row.get("http_suspicious", 0) or 0),
            "sensitive_files"   : int(row.get("sensitive_file_count", 0) or 0),
            "after_hours_ratio" : round(float(row.get("after_hours_ratio", 0) or 0), 3),
            "first_logon_hour"  : int(row.get("first_logon_hour", 0) or 0),
            "total_events"      : int(row.get("total_events", 0) or 0),
        },
        "generated_at" : datetime.now().isoformat(),
    }
