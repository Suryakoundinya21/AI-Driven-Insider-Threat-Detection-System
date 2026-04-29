import os

files = {}

files["src/api/data_store_lite.py"] = """\
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
"""

files["src/api/main_lite.py"] = """\
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional, List
import pandas as pd
import logging

from src.api.data_store_lite import (
    load_data, get_alert_df, get_user_summary,
    get_timeline, get_shap_imp, get_overview,
    get_counts, get_model_comp, get_top50,
)
from src.explainability.alert_formatter import get_risk_level

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    load_data()
    logger.info("Ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title       = "Insider Threat Detection API",
    description = "Lightweight deployment API for CERT insider threat detection",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────
@app.get("/")
def root():
    alert_df = get_alert_df()
    return {
        "status"        : "running",
        "service"       : "Insider Threat Detection API",
        "version"       : "1.0.0",
        "total_alerts"  : len(alert_df) if alert_df is not None else 0,
        "endpoints"     : {
            "alerts"  : "/alerts/",
            "users"   : "/users/top-risk",
            "stats"   : "/stats/overview",
            "docs"    : "/docs",
        }
    }

@app.get("/health")
def health():
    df = get_alert_df()
    return {
        "status"     : "healthy" if df is not None else "loading",
        "data_loaded": df is not None and len(df) > 0,
    }


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
        filtered = filtered[filtered["risk_level"].str.upper() == risk_level.upper()]
    if user:
        filtered = filtered[filtered["user"].str.lower() == user.lower()]

    filtered = filtered.sort_values("ensemble_score", ascending=False)
    page     = filtered.iloc[offset: offset + limit]
    return page.fillna("").to_dict(orient="records")


@app.get("/alerts/count")
def get_alert_count():
    return get_counts()


@app.get("/alerts/{alert_id}")
def get_alert_detail(alert_id: str):
    df = get_alert_df()
    if df is None or df.empty:
        raise HTTPException(status_code=503, detail="Data not loaded")

    parts = alert_id.split("-")
    if len(parts) < 3:
        raise HTTPException(status_code=400, detail="Invalid alert_id format")

    user      = parts[1].lower()
    date_part = parts[2]
    date_str  = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"

    matched = df[
        (df["user"] == user) &
        (df["date"].astype(str).str[:10] == date_str)
    ]
    if matched.empty:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

    row = matched.iloc[0].fillna("").to_dict()

    # Build reasons from top50 if available
    reasons = []
    top50   = get_top50()
    if top50:
        for a in top50:
            if a.get("user") == user and a.get("date", "")[:10] == date_str:
                reasons = a.get("reasons", [])
                break

    return {
        "alert_id"      : alert_id,
        "user"          : user,
        "date"          : date_str,
        "risk_level"    : str(row.get("risk_level", "LOW")),
        "ensemble_score": float(row.get("ensemble_score", 0)),
        "ae_score"      : float(row.get("ae_score", 0)),
        "if_score"      : float(row.get("if_score", 0)),
        "both_flagged"  : bool(row.get("both_flagged", False)),
        "reasons"       : reasons,
        "stats"         : {
            "device_count"      : int(row.get("device_count", 0)),
            "email_to_external" : int(row.get("email_external", 0)),
            "http_suspicious"   : int(row.get("http_suspicious", 0)),
            "sensitive_files"   : int(row.get("sensitive_files", 0)),
            "after_hours_ratio" : float(row.get("after_hours_ratio", 0) or 0),
            "first_logon_hour"  : int(row.get("first_logon_hour", 0) or 0),
            "total_events"      : int(row.get("total_events", 0) or 0),
        },
    }


# ── Users ──────────────────────────────────────────────────────
@app.get("/users/top-risk")
def get_top_risk(limit: int = Query(default=20, le=100)):
    df = get_user_summary()
    if df is None or df.empty:
        return []
    top = df.head(limit).fillna(0)
    return [
        {
            "user"            : row["user"],
            "max_score"       : round(float(row["max_score"]), 4),
            "avg_score"       : round(float(row["avg_score"]), 4),
            "risk_level"      : get_risk_level(float(row["max_score"])),
            "total_sessions"  : int(row["total_sessions"]),
            "flagged_sessions": int(row["flagged_sessions"]),
            "total_usb"       : int(row["total_usb"]),
            "total_ext_email" : int(row["total_ext_email"]),
            "total_susp_http" : int(row["total_susp_http"]),
        }
        for _, row in top.iterrows()
    ]


@app.get("/users/{user_id}/timeline")
def get_user_timeline(user_id: str, days: int = Query(default=90, le=500)):
    df = get_timeline()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    user_df = df[df["user"] == user_id.lower()].copy()
    if user_df.empty:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    user_df = user_df.sort_values("date_only").tail(days)
    max_score = float(user_df["ensemble_score"].max())

    timeline = []
    for _, row in user_df.iterrows():
        timeline.append({
            "date"             : str(row["date_only"])[:10],
            "ensemble_score"   : round(float(row["ensemble_score"]), 4),
            "ae_score"         : round(float(row.get("ae_anomaly_score", 0) or 0), 4),
            "if_score"         : round(float(row.get("if_anomaly_score", 0) or 0), 4),
            "flagged"          : bool(row.get("ensemble_flag_intersect", 0)),
            "device_count"     : int(row.get("device_count", 0) or 0),
            "email_external"   : int(row.get("email_to_external", 0) or 0),
            "http_suspicious"  : int(row.get("http_suspicious", 0) or 0),
            "after_hours_ratio": round(float(row.get("after_hours_ratio", 0) or 0), 3),
            "total_events"     : int(row.get("total_events", 0) or 0),
            "risk_level"       : get_risk_level(float(row["ensemble_score"])),
        })

    return {
        "user"            : user_id,
        "total_sessions"  : len(user_df),
        "flagged_sessions": int(user_df["ensemble_flag_intersect"].sum()),
        "max_score"       : round(max_score, 4),
        "risk_level"      : get_risk_level(max_score),
        "timeline"        : timeline,
    }


@app.get("/users/{user_id}/summary")
def get_user_summary_endpoint(user_id: str):
    df = get_user_summary()
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    row = df[df["user"] == user_id.lower()]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    r         = row.iloc[0]
    max_score = float(r["max_score"])
    n_sess    = int(r["total_sessions"])
    n_flag    = int(r["flagged_sessions"])

    tl_df  = get_timeline()
    u_tl   = tl_df[tl_df["user"] == user_id.lower()] if tl_df is not None else pd.DataFrame()
    dr_from= str(u_tl["date_only"].min())[:10] if not u_tl.empty else "N/A"
    dr_to  = str(u_tl["date_only"].max())[:10] if not u_tl.empty else "N/A"

    return {
        "user"               : user_id,
        "risk_level"         : get_risk_level(max_score),
        "max_ensemble_score" : round(max_score, 4),
        "total_sessions"     : n_sess,
        "flagged_sessions"   : n_flag,
        "flag_rate"          : round(n_flag / n_sess, 4) if n_sess > 0 else 0,
        "total_usb_events"   : int(r.get("total_usb", 0) or 0),
        "total_ext_emails"   : int(r.get("total_ext_email", 0) or 0),
        "total_susp_http"    : int(r.get("total_susp_http", 0) or 0),
        "date_range"         : {"from": dr_from, "to": dr_to},
    }


# ── Stats ──────────────────────────────────────────────────────
@app.get("/stats/overview")
def stats_overview():
    ov       = get_overview()
    alert_df = get_alert_df()
    if not ov:
        return {}
    ov["total_alerts"]    = len(alert_df) if alert_df is not None else 0
    ov["critical_alerts"] = int((alert_df["ensemble_score"] >= 0.8).sum()) if alert_df is not None and len(alert_df) > 0 else 0
    return ov


@app.get("/stats/shap-importance")
def stats_shap():
    imp    = get_shap_imp()
    ranked = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    return {
        "features": [
            {"rank": i+1, "feature": feat, "importance": round(val, 5)}
            for i, (feat, val) in enumerate(ranked)
        ]
    }


@app.get("/stats/model-comparison")
def stats_model():
    df = get_model_comp()
    if df is None:
        return {"error": "model_comparison.csv not found"}
    return {"models": df.fillna(0).to_dict(orient="records")}
"""

# Update render start command
files["render.yaml"] = """\
services:
  - type: web
    name: insider-threat-api
    env: python
    buildCommand: pip install -r requirements_api.txt
    startCommand: uvicorn src.api.main_lite:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
"""

files["requirements_api.txt"] = """\
fastapi==0.110.0
uvicorn==0.29.0
pandas==2.2.0
numpy==1.26.4
pyarrow==15.0.0
pydantic==2.6.0
python-multipart==0.0.9
requests==2.31.0
"""

for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\nDone. Next steps:")
print("  1. python deployment_data.py")
print("  2. git add deployment_data/ src/api/main_lite.py src/api/data_store_lite.py render.yaml requirements_api.txt")
print("  3. git commit -m 'Add lightweight deployment API'")
print("  4. git push")