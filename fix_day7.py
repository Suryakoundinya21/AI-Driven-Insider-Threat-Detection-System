import os

files = {}

# ─────────────────────────────────────────────
# API MODELS (Pydantic schemas)
# ─────────────────────────────────────────────
files["src/api/__init__.py"] = ""

files["src/api/schemas.py"] = """\
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import date


class AlertSummary(BaseModel):
    alert_id        : str
    user            : str
    date            : str
    risk_level      : str
    ensemble_score  : float
    ae_score        : float
    if_score        : float
    both_flagged    : bool
    reason_count    : int
    reason_summary  : str
    high_reasons    : int
    device_count    : int
    email_external  : int
    http_suspicious : int
    sensitive_files : int


class AlertDetail(BaseModel):
    alert_id        : str
    user            : str
    date            : str
    risk_level      : str
    ensemble_score  : float
    ae_score        : float
    if_score        : float
    both_flagged    : bool
    reasons         : List[Dict[str, str]]
    stats           : Dict[str, Any]
    shap_features   : Optional[List[Dict[str, Any]]] = None


class UserTimeline(BaseModel):
    user            : str
    total_sessions  : int
    flagged_sessions: int
    max_score       : float
    risk_level      : str
    timeline        : List[Dict[str, Any]]


class ModelStats(BaseModel):
    total_sessions      : int
    ae_anomalies        : int
    if_anomalies        : int
    ensemble_union      : int
    ensemble_intersect  : int
    ae_anomaly_pct      : float
    if_anomaly_pct      : float
    top_risk_users      : List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status          : str
    total_sessions  : int
    total_alerts    : int
    model_loaded    : bool
"""

# ─────────────────────────────────────────────
# DATA STORE (loads data once at startup)
# ─────────────────────────────────────────────
files["src/api/data_store.py"] = """\
import pandas as pd
import json
import logging
from pathlib import Path
from src.config import config
from src.explainability.rule_explainer  import generate_reasons
from src.explainability.alert_formatter import format_alert, get_risk_level

logger = logging.getLogger(__name__)

# Global state — loaded once at startup
_df         = None
_alert_df   = None
_shap_imp   = None


def load_data():
    global _df, _alert_df, _shap_imp

    logger.info("Loading ensemble dataset...")
    _df = pd.read_parquet(
        config.FEATURES_DIR / "feature_matrix_ensemble_scored.parquet"
    )
    _df["date_only"] = pd.to_datetime(_df["date_only"])
    _df["reasons"]   = _df.apply(generate_reasons, axis=1)
    logger.info(f"Dataset loaded: {_df.shape}")

    logger.info("Loading alert table...")
    alert_path = Path("reports/alerts/alert_table.csv")
    if alert_path.exists():
        _alert_df = pd.read_csv(alert_path)
        logger.info(f"Alert table loaded: {len(_alert_df)} rows")
    else:
        logger.warning("Alert table not found — generating from dataset")
        _alert_df = pd.DataFrame()

    shap_path = config.MODELS_DIR / "isolation_forest" / "shap_importance.json"
    if shap_path.exists():
        with open(shap_path) as f:
            _shap_imp = json.load(f)
        logger.info("SHAP importance loaded")
    else:
        _shap_imp = {}


def get_df() -> pd.DataFrame:
    return _df


def get_alert_df() -> pd.DataFrame:
    return _alert_df


def get_shap_importance() -> dict:
    return _shap_imp or {}
"""

# ─────────────────────────────────────────────
# API ROUTERS
# ─────────────────────────────────────────────
files["src/api/routes_alerts.py"] = """\
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
"""

files["src/api/routes_users.py"] = """\
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
"""

files["src/api/routes_stats.py"] = """\
from fastapi import APIRouter
from src.api.data_store  import get_df, get_alert_df, get_shap_importance
from src.explainability.alert_formatter import get_risk_level

router = APIRouter(prefix="/stats", tags=["Statistics"])


@router.get("/overview")
def get_overview():
    df       = get_df()
    alert_df = get_alert_df()

    if df is None:
        return {}

    top_users = (
        df.groupby("user")["ensemble_score"]
        .max()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
        .rename(columns={"ensemble_score": "max_score"})
    )

    return {
        "total_sessions"     : len(df),
        "total_users"        : df["user"].nunique(),
        "ae_anomalies"       : int(df["ae_anomaly_flag"].sum()),
        "if_anomalies"       : int(df["if_anomaly_flag"].sum()),
        "ensemble_union"     : int(df["ensemble_flag_union"].sum()),
        "ensemble_intersect" : int(df["ensemble_flag_intersect"].sum()),
        "ae_anomaly_pct"     : round(df["ae_anomaly_flag"].mean() * 100, 2),
        "if_anomaly_pct"     : round(df["if_anomaly_flag"].mean() * 100, 2),
        "total_alerts"       : len(alert_df) if alert_df is not None else 0,
        "critical_alerts"    : int((alert_df["ensemble_score"] >= 0.8).sum())
                               if alert_df is not None and len(alert_df) > 0 else 0,
        "date_range"         : {
            "from" : str(df["date_only"].min())[:10],
            "to"   : str(df["date_only"].max())[:10],
        },
        "top_risk_users"     : [
            {
                "user"      : row["user"],
                "max_score" : round(float(row["max_score"]), 4),
                "risk_level": get_risk_level(float(row["max_score"])),
            }
            for _, row in top_users.iterrows()
        ],
    }


@router.get("/shap-importance")
def get_shap_importance_endpoint():
    imp = get_shap_importance()
    ranked = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    return {
        "features": [
            {"rank": i+1, "feature": feat, "importance": round(val, 5)}
            for i, (feat, val) in enumerate(ranked)
        ]
    }


@router.get("/model-comparison")
def get_model_comparison():
    import os
    import pandas as pd
    path = "reports/model_comparison.csv"
    if not os.path.exists(path):
        return {"error": "model_comparison.csv not found"}
    df = pd.read_csv(path)
    return {"models": df.to_dict(orient="records")}
"""

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
files["src/api/main.py"] = """\
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.api.data_store    import load_data, get_df, get_alert_df
from src.api.routes_alerts import router as alerts_router
from src.api.routes_users  import router as users_router
from src.api.routes_stats  import router as stats_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading data...")
    load_data()
    logger.info("Data loaded. API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title       = "Insider Threat Detection API",
    description = "AI-powered insider threat detection with explainable alerts",
    version     = "1.0.0",
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


@app.get("/", tags=["Health"])
def root():
    df       = get_df()
    alert_df = get_alert_df()
    return {
        "status"         : "running",
        "service"        : "Insider Threat Detection API",
        "version"        : "1.0.0",
        "total_sessions" : len(df) if df is not None else 0,
        "total_alerts"   : len(alert_df) if alert_df is not None else 0,
        "model_loaded"   : df is not None,
        "endpoints"      : {
            "alerts"     : "/alerts",
            "users"      : "/users/top-risk",
            "stats"      : "/stats/overview",
            "shap"       : "/stats/shap-importance",
            "docs"       : "/docs",
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
# STARTUP + TEST SCRIPTS
# ─────────────────────────────────────────────
files["scripts/run_day7.py"] = """\
import sys, os
sys.path.insert(0, os.path.abspath("."))

import logging
from src.api.data_store import load_data, get_df, get_alert_df, get_shap_importance

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    print("="*60)
    print("DAY 7 — API DATA VERIFICATION")
    print("="*60)

    load_data()

    df       = get_df()
    alert_df = get_alert_df()
    shap_imp = get_shap_importance()

    print(f"\\nDataset loaded        : {df.shape}")
    print(f"Alert table loaded    : {len(alert_df)} rows")
    print(f"SHAP features loaded  : {len(shap_imp)} features")
    print(f"Date range            : {df['date_only'].min()} to {df['date_only'].max()}")
    print(f"Unique users          : {df['user'].nunique()}")
    print(f"Flagged (intersect)   : {df['ensemble_flag_intersect'].sum():,}")

    print("\\nTop 5 risk users:")
    top5 = (
        df.groupby("user")["ensemble_score"]
        .max()
        .sort_values(ascending=False)
        .head(5)
    )
    for user, score in top5.items():
        print(f"  {user:12s}  score={score:.4f}")

    print("\\nAlert risk breakdown:")
    crit = (alert_df["ensemble_score"] >= 0.8).sum()
    high = ((alert_df["ensemble_score"] >= 0.6) &
            (alert_df["ensemble_score"] < 0.8)).sum()
    med  = ((alert_df["ensemble_score"] >= 0.4) &
            (alert_df["ensemble_score"] < 0.6)).sum()
    print(f"  CRITICAL : {crit}")
    print(f"  HIGH     : {high}")
    print(f"  MEDIUM   : {med}")

    print("\\nTop 5 SHAP features:")
    for feat, val in sorted(shap_imp.items(),
                             key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feat:35s} {val:.5f}")

    print("\\n" + "="*60)
    print("All data verified. Start the API with:")
    print("  uvicorn src.api.main:app --reload --port 8000")
    print("Then open: http://127.0.0.1:8000/docs")
    print("="*60)

if __name__ == "__main__":
    main()
"""

files["scripts/test_api.py"] = """\
import requests
import json

BASE = "http://127.0.0.1:8000"

def test(label, url, params=None):
    try:
        r = requests.get(url, params=params, timeout=10)
        print(f"\\n{'='*55}")
        print(f"TEST : {label}")
        print(f"URL  : {url}")
        print(f"STATUS: {r.status_code}")
        data = r.json()
        if isinstance(data, list):
            print(f"RESULT: [{len(data)} items]")
            if data:
                print(json.dumps(data[0], indent=2, default=str)[:600])
        else:
            print(json.dumps(data, indent=2, default=str)[:600])
    except Exception as e:
        print(f"ERROR: {e}")

print("\\ninsider Threat Detection API — Endpoint Tests")
print("Make sure server is running: uvicorn src.api.main:app --reload --port 8000\\n")

test("Health check",          f"{BASE}/")
test("System overview",       f"{BASE}/stats/overview")
test("Alert counts",          f"{BASE}/alerts/count")
test("All alerts (top 5)",    f"{BASE}/alerts", {"limit": 5})
test("CRITICAL alerts only",  f"{BASE}/alerts", {"risk_level": "CRITICAL", "limit": 5})
test("Top risk users",        f"{BASE}/users/top-risk", {"limit": 5})
test("User timeline",         f"{BASE}/users/dlm0051/timeline", {"days": 30})
test("User summary",          f"{BASE}/users/gko0078/summary")
test("SHAP importance",       f"{BASE}/stats/shap-importance")
test("Model comparison",      f"{BASE}/stats/model-comparison")
test("Alert detail",          f"{BASE}/alerts/ALERT-GKO0078-20110322")

print("\\nAll tests complete.")
"""

for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\nAll Day 7 files created.")
print("Run: python scripts/run_day7.py")
print("Then: uvicorn src.api.main:app --reload --port 8000")