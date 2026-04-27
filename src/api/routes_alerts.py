from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
import pandas as pd
from src.api.data_store      import get_df, get_alert_df
from src.api.schemas         import AlertSummary, AlertDetail
from src.explainability.alert_formatter import format_alert, get_risk_level

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("/", response_model=List[AlertSummary])
def get_alerts(
    risk_level : Optional[str]  = None,
    user       : Optional[str]  = None,
    min_score  : float          = 0.0,
    limit      : int            = Query(default=50, le=500),
    offset     : int            = 0,
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
    page     = filtered.iloc[offset : offset + limit]

    results = []
    for _, row in page.iterrows():
        results.append(AlertSummary(
            alert_id        = str(row.get("alert_id", "")),
            user            = str(row.get("user", "")),
            date            = str(row.get("date", "")),
            risk_level      = str(row.get("risk_level", "LOW")),
            ensemble_score  = float(row.get("ensemble_score", 0)),
            ae_score        = float(row.get("ae_score", 0)),
            if_score        = float(row.get("if_score", 0)),
            both_flagged    = bool(row.get("both_flagged", False)),
            reason_count    = int(row.get("reason_count", 0)),
            reason_summary  = str(row.get("reason_summary", "")),
            high_reasons    = int(row.get("high_reasons", 0)),
            device_count    = int(row.get("device_count", 0)),
            email_external  = int(row.get("email_external", 0)),
            http_suspicious = int(row.get("http_suspicious", 0)),
            sensitive_files = int(row.get("sensitive_files", 0)),
        ))
    return results


@router.get("/count")
def get_alert_counts():
    df = get_alert_df()
    if df is None or df.empty:
        return {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
    return {
        "total"    : len(df),
        "critical" : int((df["ensemble_score"] >= 0.8).sum()),
        "high"     : int(((df["ensemble_score"] >= 0.6) & (df["ensemble_score"] < 0.8)).sum()),
        "medium"   : int(((df["ensemble_score"] >= 0.4) & (df["ensemble_score"] < 0.6)).sum()),
        "low"      : int((df["ensemble_score"] < 0.4).sum()),
    }


@router.get("/{alert_id}", response_model=AlertDetail)
def get_alert_detail(alert_id: str):
    df = get_df()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    user = alert_id.split("-")[1].lower() if "-" in alert_id else ""
    date_part = alert_id.split("-")[2] if len(alert_id.split("-")) > 2 else ""

    matched = df[df["user"] == user]
    if date_part:
        date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
        matched  = matched[
            matched["date_only"].dt.strftime("%Y-%m-%d") == date_str
        ]

    if matched.empty:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

    row   = matched.iloc[0]
    alert = format_alert(row)

    return AlertDetail(
        alert_id       = alert["alert_id"],
        user           = alert["user"],
        date           = alert["date"],
        risk_level     = alert["risk_level"],
        ensemble_score = alert["ensemble_score"],
        ae_score       = alert["ae_score"],
        if_score       = alert["if_score"],
        both_flagged   = alert["both_flagged"],
        reasons        = alert["reasons"],
        stats          = alert["stats"],
    )
