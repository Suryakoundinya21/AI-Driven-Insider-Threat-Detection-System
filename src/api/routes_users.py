from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import pandas as pd
from src.api.data_store  import get_df
from src.api.schemas     import UserTimeline
from src.explainability.alert_formatter import get_risk_level

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/top-risk")
def get_top_risk_users(limit: int = Query(default=20, le=100)):
    df = get_df()
    if df is None:
        return []

    top = (
        df.groupby("user")
        .agg(
            max_score       = ("ensemble_score", "max"),
            avg_score       = ("ensemble_score", "mean"),
            total_sessions  = ("ensemble_score", "count"),
            flagged_sessions= ("ensemble_flag_intersect", "sum"),
            total_device    = ("device_count", "sum"),
            total_external  = ("email_to_external", "sum"),
            total_http_susp = ("http_suspicious", "sum"),
        )
        .reset_index()
        .sort_values("max_score", ascending=False)
        .head(limit)
    )

    results = []
    for _, row in top.iterrows():
        results.append({
            "user"            : row["user"],
            "max_score"       : round(float(row["max_score"]), 4),
            "avg_score"       : round(float(row["avg_score"]), 4),
            "risk_level"      : get_risk_level(float(row["max_score"])),
            "total_sessions"  : int(row["total_sessions"]),
            "flagged_sessions": int(row["flagged_sessions"]),
            "total_usb"       : int(row["total_device"]),
            "total_ext_email" : int(row["total_external"]),
            "total_susp_http" : int(row["total_http_susp"]),
        })
    return results


@router.get("/{user_id}/timeline")
def get_user_timeline(
    user_id   : str,
    days      : int = Query(default=90, le=500),
):
    df = get_df()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    user_df = df[df["user"] == user_id.lower()].copy()
    if user_df.empty:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    user_df = user_df.sort_values("date_only").tail(days)

    timeline = []
    for _, row in user_df.iterrows():
        timeline.append({
            "date"            : str(row["date_only"])[:10],
            "ensemble_score"  : round(float(row["ensemble_score"]), 4),
            "ae_score"        : round(float(row.get("ae_anomaly_score", 0)), 4),
            "if_score"        : round(float(row.get("if_anomaly_score", 0)), 4),
            "flagged"         : bool(row.get("ensemble_flag_intersect", 0)),
            "device_count"    : int(row.get("device_count", 0)),
            "email_external"  : int(row.get("email_to_external", 0)),
            "http_suspicious" : int(row.get("http_suspicious", 0)),
            "after_hours_ratio": round(float(row.get("after_hours_ratio", 0)), 3),
            "total_events"    : int(row.get("total_events", 0)),
            "risk_level"      : get_risk_level(float(row["ensemble_score"])),
            "reason_count"    : len(row.get("reasons", [])),
        })

    max_score = float(user_df["ensemble_score"].max())
    return {
        "user"            : user_id,
        "total_sessions"  : len(user_df),
        "flagged_sessions": int(user_df["ensemble_flag_intersect"].sum()),
        "max_score"       : round(max_score, 4),
        "risk_level"      : get_risk_level(max_score),
        "timeline"        : timeline,
    }


@router.get("/{user_id}/summary")
def get_user_summary(user_id: str):
    df = get_df()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    user_df = df[df["user"] == user_id.lower()]
    if user_df.empty:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    flagged = user_df[user_df["ensemble_flag_intersect"] == 1]
    max_score = float(user_df["ensemble_score"].max())

    return {
        "user"              : user_id,
        "risk_level"        : get_risk_level(max_score),
        "max_ensemble_score": round(max_score, 4),
        "total_sessions"    : len(user_df),
        "flagged_sessions"  : len(flagged),
        "flag_rate"         : round(len(flagged) / len(user_df), 4),
        "total_usb_events"  : int(user_df["device_count"].sum()),
        "total_ext_emails"  : int(user_df["email_to_external"].sum()),
        "total_susp_http"   : int(user_df["http_suspicious"].sum()),
        "avg_after_hours"   : round(float(user_df["after_hours_ratio"].mean()), 3),
        "date_range"        : {
            "from": str(user_df["date_only"].min())[:10],
            "to"  : str(user_df["date_only"].max())[:10],
        },
    }
