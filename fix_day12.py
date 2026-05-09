import os

files = {}

# ─────────────────────────────────────────────
# FEEDBACK STORE
# ─────────────────────────────────────────────
files["src/api/feedback_store.py"] = """\
import json
import logging
from pathlib import Path
from datetime import datetime

logger    = logging.getLogger(__name__)
FEED_PATH = Path("reports/feedback.json")


def load_feedback() -> dict:
    if FEED_PATH.exists():
        with open(FEED_PATH) as f:
            return json.load(f)
    return {"false_positives": [], "confirmed": [], "user_adjustments": {}}


def save_feedback(data: dict):
    FEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEED_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def add_false_positive(alert_id: str, user: str, analyst: str, reason: str):
    data   = load_feedback()
    record = {
        "alert_id"   : alert_id,
        "user"       : user,
        "analyst"    : analyst,
        "reason"     : reason,
        "timestamp"  : datetime.now().isoformat(),
        "type"       : "false_positive",
    }
    data["false_positives"].append(record)

    # Increase threshold for this user
    adj = data["user_adjustments"].get(user, {"fp_count": 0, "threshold_boost": 0.0})
    adj["fp_count"]       += 1
    adj["threshold_boost"] = min(0.3, adj["fp_count"] * 0.05)
    adj["last_updated"]    = datetime.now().isoformat()
    data["user_adjustments"][user] = adj

    save_feedback(data)
    logger.info(f"False positive recorded: {alert_id} | user={user} | "
                f"threshold_boost={adj['threshold_boost']:.2f}")
    return record


def add_confirmed(alert_id: str, user: str, analyst: str, severity: str):
    data   = load_feedback()
    record = {
        "alert_id"  : alert_id,
        "user"      : user,
        "analyst"   : analyst,
        "severity"  : severity,
        "timestamp" : datetime.now().isoformat(),
        "type"      : "confirmed",
    }
    data["confirmed"].append(record)

    # Lower threshold for this user (more sensitive)
    adj = data["user_adjustments"].get(user, {"fp_count": 0, "threshold_boost": 0.0})
    adj["threshold_boost"] = max(-0.1, adj.get("threshold_boost", 0.0) - 0.02)
    adj["last_updated"]    = datetime.now().isoformat()
    data["user_adjustments"][user] = adj

    save_feedback(data)
    logger.info(f"Confirmed threat recorded: {alert_id} | user={user} | severity={severity}")
    return record


def get_user_threshold_boost(user: str) -> float:
    data = load_feedback()
    return data["user_adjustments"].get(user, {}).get("threshold_boost", 0.0)


def get_feedback_stats() -> dict:
    data = load_feedback()
    return {
        "total_false_positives" : len(data["false_positives"]),
        "total_confirmed"       : len(data["confirmed"]),
        "users_adjusted"        : len(data["user_adjustments"]),
        "user_adjustments"      : data["user_adjustments"],
    }
"""

# ─────────────────────────────────────────────
# FEEDBACK API ROUTES
# ─────────────────────────────────────────────
files["src/api/routes_feedback.py"] = """\
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.api.feedback_store import (
    add_false_positive, add_confirmed,
    get_feedback_stats, get_user_threshold_boost,
    load_feedback,
)

router = APIRouter(prefix="/feedback", tags=["Feedback"])


class FalsePositiveRequest(BaseModel):
    alert_id : str
    user     : str
    analyst  : str = "analyst"
    reason   : str = "Not anomalous"


class ConfirmRequest(BaseModel):
    alert_id : str
    user     : str
    analyst  : str = "analyst"
    severity : str = "HIGH"


@router.post("/false-positive")
def mark_false_positive(req: FalsePositiveRequest):
    record = add_false_positive(
        req.alert_id, req.user, req.analyst, req.reason
    )
    boost = get_user_threshold_boost(req.user)
    return {
        "status"          : "recorded",
        "message"         : f"Alert {req.alert_id} marked as false positive",
        "user"            : req.user,
        "threshold_boost" : boost,
        "effect"          : f"Threshold for {req.user} raised by {boost:.0%} to reduce future alerts",
    }


@router.post("/confirm")
def confirm_threat(req: ConfirmRequest):
    record = add_confirmed(
        req.alert_id, req.user, req.analyst, req.severity
    )
    return {
        "status"  : "recorded",
        "message" : f"Alert {req.alert_id} confirmed as {req.severity} threat",
        "user"    : req.user,
    }


@router.get("/stats")
def feedback_stats():
    return get_feedback_stats()


@router.get("/history")
def feedback_history():
    data = load_feedback()
    return {
        "false_positives" : data["false_positives"][-20:],
        "confirmed"       : data["confirmed"][-20:],
    }


@router.get("/user/{user_id}")
def user_feedback(user_id: str):
    data  = load_feedback()
    boost = get_user_threshold_boost(user_id)
    user_fp = [
        r for r in data["false_positives"]
        if r.get("user") == user_id
    ]
    user_cf = [
        r for r in data["confirmed"]
        if r.get("user") == user_id
    ]
    return {
        "user"             : user_id,
        "false_positives"  : len(user_fp),
        "confirmed"        : len(user_cf),
        "threshold_boost"  : boost,
        "adjusted"         : boost != 0.0,
        "history"          : user_fp[-5:] + user_cf[-5:],
    }


@router.delete("/reset/{user_id}")
def reset_user(user_id: str):
    data = load_feedback()
    if user_id in data["user_adjustments"]:
        del data["user_adjustments"][user_id]
        from src.api.feedback_store import save_feedback
        save_feedback(data)
        return {"status": "reset", "user": user_id}
    return {"status": "not_found", "user": user_id}
"""

# ─────────────────────────────────────────────
# UPDATE MAIN API TO INCLUDE FEEDBACK ROUTER
# ─────────────────────────────────────────────
files["src/api/main.py"] = """\
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.api.data_store     import load_data, get_df, get_alert_df
from src.api.routes_alerts  import router as alerts_router
from src.api.routes_users   import router as users_router
from src.api.routes_stats   import router as stats_router
from src.api.routes_feedback import router as feedback_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up - loading data...")
    load_data()
    logger.info("Data loaded. API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title       = "Insider Threat Detection API",
    description = "AI-powered insider threat detection with explainable alerts",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(alerts_router)
app.include_router(users_router)
app.include_router(stats_router)
app.include_router(feedback_router)


@app.get("/", tags=["Health"])
def root():
    df       = get_df()
    alert_df = get_alert_df()
    return {
        "status"         : "running",
        "service"        : "Insider Threat Detection API",
        "version"        : "2.0.0",
        "total_sessions" : len(df) if df is not None else 0,
        "total_alerts"   : len(alert_df) if alert_df is not None else 0,
        "endpoints"      : {
            "alerts"   : "/alerts/",
            "users"    : "/users/top-risk",
            "stats"    : "/stats/overview",
            "feedback" : "/feedback/stats",
            "docs"     : "/docs",
        }
    }


@app.get("/health", tags=["Health"])
def health():
    df = get_df()
    return {
        "status"      : "healthy" if df is not None else "loading",
        "data_loaded" : df is not None,
    }
"""

# ─────────────────────────────────────────────
# UPDATE MAIN_LITE FOR RENDER (also add feedback)
# ─────────────────────────────────────────────
files["src/api/main_lite.py"] = """\
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
"""

# ─────────────────────────────────────────────
# FEEDBACK PAGE IN DASHBOARD
# ─────────────────────────────────────────────
files["dashboard/views/feedback.py"] = """\
import sys, os
sys.path.insert(0, os.path.abspath("."))
import streamlit as st
import pandas as pd
import requests

API     = "http://127.0.0.1:8000"
TIMEOUT = 15

RC = {"CRITICAL":"#D85A30","HIGH":"#E8953A","MEDIUM":"#F5C842","LOW":"#1D9E75"}

def show():
    st.title("Analyst Feedback")
    st.markdown("Mark alerts as false positives or confirm threats to adapt detection thresholds.")

    tab1, tab2, tab3 = st.tabs([
        "Submit Feedback", "Feedback Stats", "User Adjustments"
    ])

    with tab1:
        st.subheader("Submit Alert Feedback")

        try:
            r      = requests.get(f"{API}/alerts/", params={"limit":100}, timeout=TIMEOUT)
            alerts = r.json()
        except:
            alerts = []

        if not alerts:
            st.warning("No alerts loaded. Make sure API is running.")
            return

        opts = {
            f"{a['user']} | {a['date']} | {a['ensemble_score']:.3f} | {a['risk_level']}":
            a for a in alerts
        }
        selected_label = st.selectbox("Select Alert", list(opts.keys()))
        selected_alert = opts[selected_label]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Ensemble Score", f"{selected_alert.get('ensemble_score',0):.4f}")
            st.metric("Risk Level",     selected_alert.get("risk_level",""))
        with col2:
            st.metric("AE Score", f"{selected_alert.get('ae_score',0):.4f}")
            st.metric("IF Score", f"{selected_alert.get('if_score',0):.4f}")

        st.markdown("---")
        analyst_name = st.text_input("Analyst Name", value="analyst_1")

        col_fp, col_cf = st.columns(2)

        with col_fp:
            st.markdown("### Mark as False Positive")
            fp_reason = st.selectbox("Reason", [
                "Normal behavior for this user",
                "Scheduled maintenance activity",
                "Authorized data transfer",
                "Test/training environment",
                "Other",
            ])
            if fp_reason == "Other":
                fp_reason = st.text_input("Specify reason")

            if st.button("Submit False Positive", type="secondary"):
                try:
                    resp = requests.post(f"{API}/feedback/false-positive", json={
                        "alert_id" : selected_alert["alert_id"],
                        "user"     : selected_alert["user"],
                        "analyst"  : analyst_name,
                        "reason"   : fp_reason,
                    }, timeout=TIMEOUT)
                    result = resp.json()
                    st.success(f"Recorded. {result.get('effect','')}")
                    boost = result.get("threshold_boost", 0)
                    st.info(f"User threshold raised by {boost:.0%}. Future alerts for this user require higher confidence.")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col_cf:
            st.markdown("### Confirm as Real Threat")
            severity = st.selectbox("Severity", ["CRITICAL","HIGH","MEDIUM","LOW"])

            if st.button("Confirm Threat", type="primary"):
                try:
                    resp = requests.post(f"{API}/feedback/confirm", json={
                        "alert_id" : selected_alert["alert_id"],
                        "user"     : selected_alert["user"],
                        "analyst"  : analyst_name,
                        "severity" : severity,
                    }, timeout=TIMEOUT)
                    result = resp.json()
                    st.success(f"Confirmed {severity} threat for user {selected_alert['user']}")
                    st.warning("Detection sensitivity increased for this user.")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        st.subheader("Feedback Statistics")
        try:
            stats = requests.get(f"{API}/feedback/stats", timeout=TIMEOUT).json()
            c1,c2,c3 = st.columns(3)
            c1.metric("Total False Positives", stats.get("total_false_positives",0))
            c2.metric("Total Confirmed",        stats.get("total_confirmed",0))
            c3.metric("Users Adjusted",         stats.get("users_adjusted",0))

            adj = stats.get("user_adjustments",{})
            if adj:
                st.subheader("User Threshold Adjustments")
                rows = []
                for user, data in adj.items():
                    rows.append({
                        "User"            : user,
                        "FP Count"        : data.get("fp_count",0),
                        "Threshold Boost" : f"+{data.get('threshold_boost',0):.0%}",
                        "Last Updated"    : str(data.get("last_updated",""))[:19],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No user adjustments yet.")
        except Exception as e:
            st.error(f"Cannot load stats: {e}")

    with tab3:
        st.subheader("User Adjustment Details")
        user_id = st.text_input("Enter User ID to check", "gko0078")
        if st.button("Check User"):
            try:
                r    = requests.get(f"{API}/feedback/user/{user_id}", timeout=TIMEOUT)
                data = r.json()
                if "error" in data:
                    st.warning(f"User {user_id} not found")
                else:
                    st.markdown(f"**User:** `{data['user']}`")
                    c1,c2,c3 = st.columns(3)
                    c1.metric("False Positives", data.get("false_positives",0))
                    c2.metric("Confirmed",        data.get("confirmed",0))
                    c3.metric("Threshold Boost",
                              f"+{data.get('threshold_boost',0):.0%}")
                    if data.get("adjusted"):
                        st.warning(f"This user's alert threshold has been raised. "
                                   f"Only alerts with score > {0.5 + data.get('threshold_boost',0):.2f} will appear.")
                    else:
                        st.success("No adjustments. Using default threshold.")
            except Exception as e:
                st.error(f"Error: {e}")
"""

# ─────────────────────────────────────────────
# UPDATE DASHBOARD APP TO INCLUDE FEEDBACK PAGE
# ─────────────────────────────────────────────
files["dashboard/app.py"] = """\
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import streamlit as st

st.set_page_config(
    page_title = "Insider Threat Detection System",
    page_icon  = "S",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

try:
    API_BASE = st.secrets.get("API_BASE", "http://127.0.0.1:8000")
except Exception:
    API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")

os.environ["ITDS_API_BASE"] = API_BASE

import importlib.util

def load_page(path):
    spec   = importlib.util.spec_from_file_location("page", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

BASE = os.path.dirname(os.path.abspath(__file__))

PAGES = {
    "Overview"           : os.path.join(BASE, "views", "overview.py"),
    "Alert Center"       : os.path.join(BASE, "views", "alerts.py"),
    "User Investigation" : os.path.join(BASE, "views", "users.py"),
    "Explainability"     : os.path.join(BASE, "views", "explainability.py"),
    "Model Report"       : os.path.join(BASE, "views", "model_report.py"),
    "Analyst Feedback"   : os.path.join(BASE, "views", "feedback.py"),
}

with st.sidebar:
    st.markdown("## Insider Threat Detection")
    st.markdown("---")
    selection = st.radio(
        "Navigate to",
        list(PAGES.keys()),
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**API Status**")
    try:
        import requests
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            st.success("API Online")
        else:
            st.error("API Error")
    except Exception:
        st.error("API Offline")
        st.caption(f"Target: {API_BASE}")
    st.markdown("---")
    st.caption("Insider Threat Detection v2.0")

page_path = PAGES[selection]
if os.path.exists(page_path):
    mod = load_page(page_path)
    mod.show()
else:
    st.error(f"Page not found: {page_path}")
"""

for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\nAll Day 12 files created.")
print("Run in two terminals:")
print("  Terminal 1: uvicorn src.api.main:app --reload --port 8000")
print("  Terminal 2: streamlit run dashboard/app.py")