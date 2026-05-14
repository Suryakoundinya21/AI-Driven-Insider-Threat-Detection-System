from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import pandas as pd
from src.api.data_store      import get_df, get_alert_df
from src.explainability.alert_formatter import format_alert, get_risk_level
from src.explainability.rule_explainer  import generate_reasons

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("/")
def get_alerts(
    risk_level : Optional[str] = None,
    user       : Optional[str] = None,
    min_score  : float         = 0.0,
    limit      : int           = Query(default=50, le=500),
    offset     : int           = 0,
):
    df = get_alert_df()
    if df is None or df.empty:
        return []

    filtered = df[df["ensemble_score"] >= min_score].copy()
    if risk_level:
        filtered = filtered[
            filtered["risk_level"].str.upper() == risk_level.upper()
        ]
    if user:
        filtered = filtered[
            filtered["user"].str.lower() == user.lower()
        ]

    filtered = filtered.sort_values("ensemble_score", ascending=False)
    page     = filtered.iloc[offset: offset + limit]
    return page.fillna("").to_dict(orient="records")


@router.get("/count")
def get_alert_count():
    df = get_alert_df()
    if df is None or df.empty:
        return {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
    return {
        "total"    : len(df),
        "critical" : int((df["ensemble_score"] >= 0.8).sum()),
        "high"     : int(((df["ensemble_score"] >= 0.6) &
                          (df["ensemble_score"] < 0.8)).sum()),
        "medium"   : int(((df["ensemble_score"] >= 0.4) &
                          (df["ensemble_score"] < 0.6)).sum()),
        "low"      : int((df["ensemble_score"] < 0.4).sum()),
    }


@router.get("/{alert_id}")
def get_alert_detail(alert_id: str):
    full_df = get_df()
    if full_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    parts = alert_id.split("-")
    if len(parts) < 3:
        raise HTTPException(status_code=400, detail="Invalid alert_id format")

    user      = parts[1].lower()
    date_part = parts[2]
    date_str  = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"

    matched = full_df[
        (full_df["user"] == user) &
        (full_df["date_only"].astype(str).str[:10] == date_str)
    ]

    if matched.empty:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

    row     = matched.iloc[0]
    reasons = generate_reasons(row)

    # Ensure all reason fields are strings for JSON safety
    clean_reasons = []
    for r in reasons:
        clean_reasons.append({
            "severity" : str(r.get("severity", "LOW")),
            "feature"  : str(r.get("feature", "")),
            "reason"   : str(r.get("reason", "")),
            "value"    : str(round(float(r.get("value", 0) or 0), 3)),
            "baseline" : str(round(float(r.get("baseline", 0) or 0), 3)),
            "zscore"   : str(round(float(r.get("zscore", 0) or 0), 2)),
        })

    return {
        "alert_id"      : alert_id,
        "user"          : user,
        "date"          : date_str,
        "risk_level"    : get_risk_level(float(row.get("ensemble_score", 0))),
        "ensemble_score": round(float(row.get("ensemble_score", 0)), 4),
        "ae_score"      : round(float(row.get("ae_anomaly_score", 0) or 0), 4),
        "if_score"      : round(float(row.get("if_anomaly_score", 0) or 0), 4),
        "both_flagged"  : bool(
            row.get("ae_anomaly_flag", 0) and row.get("if_anomaly_flag", 0)
        ),
        "reasons"       : clean_reasons,
        "stats"         : {
            "device_count"      : int(row.get("device_count", 0) or 0),
            "email_to_external" : int(row.get("email_to_external", 0) or 0),
            "http_suspicious"   : int(row.get("http_suspicious", 0) or 0),
            "sensitive_files"   : int(row.get("sensitive_file_count", 0) or 0),
            "after_hours_ratio" : round(float(row.get("after_hours_ratio", 0) or 0), 3),
            "first_logon_hour"  : int(row.get("first_logon_hour", 0) or 0),
            "total_events"      : int(row.get("total_events", 0) or 0),
        },
    }
