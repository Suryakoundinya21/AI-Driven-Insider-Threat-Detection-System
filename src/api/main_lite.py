from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import logging

from src.api.data_store_lite import (
    load_data, get_alert_df, get_user_summary,
    get_timeline, get_shap_imp, get_overview,
    get_counts, get_model_comp, get_top50,
)
from src.api.feedback_store import (
    add_false_positive, add_confirmed,
    get_feedback_stats, get_user_threshold_boost,
    load_feedback,
)
from src.explainability.alert_formatter import get_risk_level

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    load_data()
    logger.info("Ready.")
    yield


app = FastAPI(
    title    = "Insider Threat Detection API",
    version  = "2.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────
@app.get("/")
def root():
    alert_df = get_alert_df()
    return {
        "status"       : "running",
        "version"      : "2.0.0",
        "total_alerts" : len(alert_df) if alert_df is not None else 0,
    }

@app.get("/health")
def health():
    df = get_alert_df()
    return {"status": "healthy" if df is not None else "loading",
            "data_loaded": df is not None and len(df) > 0}


# ── Alerts ─────────────────────────────────────────────────────
@app.get("/alerts/")
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
        filtered = filtered[filtered["user"].str.lower() == user.lower()]

    boost = get_user_threshold_boost(user) if user else 0.0
    if boost > 0:
        filtered = filtered[filtered["ensemble_score"] >= (min_score + boost)]

    filtered = filtered.sort_values("ensemble_score", ascending=False)
    return filtered.iloc[offset: offset + limit].fillna("").to_dict(orient="records")


@app.get("/alerts/count")
def get_alert_count():
    return get_counts()


@app.get("/alerts/{alert_id}")
def get_alert_detail(alert_id: str):
    df    = get_alert_df()
    parts = alert_id.split("-")
    if len(parts) < 3:
        raise HTTPException(400, "Invalid alert_id")
    user     = parts[1].lower()
    dp       = parts[2]
    date_str = f"{dp[:4]}-{dp[4:6]}-{dp[6:8]}"
    matched  = df[
        (df["user"] == user) &
        (df["date"].astype(str).str[:10] == date_str)
    ] if df is not None else pd.DataFrame()
    if matched.empty:
        raise HTTPException(404, f"Alert {alert_id} not found")
    row     = matched.iloc[0].fillna("").to_dict()
    reasons = []
    top50   = get_top50()
    if top50:
        for a in top50:
            if a.get("user") == user and a.get("date","")[:10] == date_str:
                reasons = a.get("reasons", [])
                break
    return {
        "alert_id"      : alert_id,
        "user"          : user,
        "date"          : date_str,
        "risk_level"    : str(row.get("risk_level","LOW")),
        "ensemble_score": float(row.get("ensemble_score",0)),
        "ae_score"      : float(row.get("ae_score",0)),
        "if_score"      : float(row.get("if_score",0)),
        "both_flagged"  : bool(row.get("both_flagged",False)),
        "reasons"       : reasons,
        "stats"         : {
            "device_count"      : int(row.get("device_count",0) or 0),
            "email_to_external" : int(row.get("email_external",0) or 0),
            "http_suspicious"   : int(row.get("http_suspicious",0) or 0),
            "sensitive_files"   : int(row.get("sensitive_files",0) or 0),
            "after_hours_ratio" : float(row.get("after_hours_ratio",0) or 0),
            "first_logon_hour"  : int(row.get("first_logon_hour",0) or 0),
            "total_events"      : int(row.get("total_events",0) or 0),
        },
    }


# ── Users ──────────────────────────────────────────────────────
@app.get("/users/top-risk")
def top_risk(limit: int = Query(default=20, le=100)):
    df = get_user_summary()
    if df is None or df.empty:
        return []
    return [
        {
            "user"            : r["user"],
            "max_score"       : round(float(r["max_score"]),4),
            "avg_score"       : round(float(r["avg_score"]),4),
            "risk_level"      : get_risk_level(float(r["max_score"])),
            "total_sessions"  : int(r["total_sessions"]),
            "flagged_sessions": int(r["flagged_sessions"]),
            "total_usb"       : int(r["total_usb"]),
            "total_ext_email" : int(r["total_ext_email"]),
            "total_susp_http" : int(r["total_susp_http"]),
        }
        for _, r in df.head(limit).fillna(0).iterrows()
    ]


@app.get("/users/{user_id}/timeline")
def user_timeline(user_id: str, days: int = Query(default=90, le=500)):
    df = get_timeline()
    if df is None:
        raise HTTPException(503, "Data not loaded")
    udf = df[df["user"] == user_id.lower()].sort_values("date_only").tail(days)
    if udf.empty:
        raise HTTPException(404, f"User {user_id} not found")
    max_s = float(udf["ensemble_score"].max())
    return {
        "user"            : user_id,
        "total_sessions"  : len(udf),
        "flagged_sessions": int(udf["ensemble_flag_intersect"].sum()),
        "max_score"       : round(max_s,4),
        "risk_level"      : get_risk_level(max_s),
        "timeline"        : [
            {
                "date"             : str(r["date_only"])[:10],
                "ensemble_score"   : round(float(r["ensemble_score"]),4),
                "ae_score"         : round(float(r.get("ae_anomaly_score",0) or 0),4),
                "if_score"         : round(float(r.get("if_anomaly_score",0) or 0),4),
                "flagged"          : bool(r.get("ensemble_flag_intersect",0)),
                "device_count"     : int(r.get("device_count",0) or 0),
                "email_external"   : int(r.get("email_to_external",0) or 0),
                "http_suspicious"  : int(r.get("http_suspicious",0) or 0),
                "after_hours_ratio": round(float(r.get("after_hours_ratio",0) or 0),3),
                "total_events"     : int(r.get("total_events",0) or 0),
                "risk_level"       : get_risk_level(float(r["ensemble_score"])),
            }
            for _, r in udf.iterrows()
        ],
    }


@app.get("/users/{user_id}/summary")
def user_summary(user_id: str):
    df = get_user_summary()
    if df is None:
        raise HTTPException(503, "Data not loaded")
    row = df[df["user"] == user_id.lower()]
    if row.empty:
        raise HTTPException(404, f"User {user_id} not found")
    r     = row.iloc[0]
    ms    = float(r["max_score"])
    ns    = int(r["total_sessions"])
    nf    = int(r["flagged_sessions"])
    tl    = get_timeline()
    utl   = tl[tl["user"]==user_id.lower()] if tl is not None else pd.DataFrame()
    return {
        "user"               : user_id,
        "risk_level"         : get_risk_level(ms),
        "max_ensemble_score" : round(ms,4),
        "total_sessions"     : ns,
        "flagged_sessions"   : nf,
        "flag_rate"          : round(nf/ns,4) if ns>0 else 0,
        "total_usb_events"   : int(r.get("total_usb",0) or 0),
        "total_ext_emails"   : int(r.get("total_ext_email",0) or 0),
        "total_susp_http"    : int(r.get("total_susp_http",0) or 0),
        "threshold_boost"    : get_user_threshold_boost(user_id),
        "date_range"         : {
            "from": str(utl["date_only"].min())[:10] if not utl.empty else "N/A",
            "to"  : str(utl["date_only"].max())[:10] if not utl.empty else "N/A",
        },
    }


# ── Stats ──────────────────────────────────────────────────────
@app.get("/stats/overview")
def stats_overview():
    ov  = get_overview()
    adf = get_alert_df()
    if not ov:
        return {}
    ov["total_alerts"]    = len(adf) if adf is not None else 0
    ov["critical_alerts"] = int((adf["ensemble_score"]>=0.8).sum()) if adf is not None and len(adf)>0 else 0
    return ov

@app.get("/stats/shap-importance")
def stats_shap():
    imp    = get_shap_imp()
    ranked = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    return {"features": [
        {"rank":i+1,"feature":f,"importance":round(v,5)}
        for i,(f,v) in enumerate(ranked)
    ]}

@app.get("/stats/model-comparison")
def stats_model():
    df = get_model_comp()
    if df is None:
        return {"error": "not found"}
    return {"models": df.fillna(0).to_dict(orient="records")}


# ── Feedback ───────────────────────────────────────────────────
class FPRequest(BaseModel):
    alert_id : str
    user     : str
    analyst  : str = "analyst"
    reason   : str = "Not anomalous"

class CFRequest(BaseModel):
    alert_id : str
    user     : str
    analyst  : str = "analyst"
    severity : str = "HIGH"

@app.post("/feedback/false-positive")
def mark_fp(req: FPRequest):
    add_false_positive(req.alert_id, req.user, req.analyst, req.reason)
    boost = get_user_threshold_boost(req.user)
    return {
        "status"          : "recorded",
        "user"            : req.user,
        "threshold_boost" : boost,
        "effect"          : f"Threshold raised by {boost:.0%} for {req.user}",
    }

@app.post("/feedback/confirm")
def confirm(req: CFRequest):
    add_confirmed(req.alert_id, req.user, req.analyst, req.severity)
    return {"status": "recorded", "user": req.user, "severity": req.severity}

@app.get("/feedback/stats")
def fb_stats():
    return get_feedback_stats()

@app.get("/feedback/user/{user_id}")
def fb_user(user_id: str):
    data  = load_feedback()
    boost = get_user_threshold_boost(user_id)
    fps   = [r for r in data["false_positives"] if r.get("user")==user_id]
    cfs   = [r for r in data["confirmed"]       if r.get("user")==user_id]
    return {
        "user"            : user_id,
        "false_positives" : len(fps),
        "confirmed"       : len(cfs),
        "threshold_boost" : boost,
        "adjusted"        : boost != 0.0,
    }
