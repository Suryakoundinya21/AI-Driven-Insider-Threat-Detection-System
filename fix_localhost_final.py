import os

files = {}

# ── 1. CONFIG FIX ─────────────────────────────────────────────
files["src/config.py"] = """\
from pathlib import Path
import os

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))

class Config:
    # Data paths — inside data/
    RAW_DATA_DIR  = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    FEATURES_DIR  = DATA_DIR / "features"

    # These are at project root level — NOT inside data/
    MODELS_DIR    = BASE_DIR / "models"
    REPORTS_DIR   = BASE_DIR / "reports"

    LOG_FILES         = ["logon", "device", "email", "file", "http"]
    AFTER_HOURS_START = 18
    AFTER_HOURS_END   = 8
    BASELINE_WINDOW   = 30
    CONTAMINATION     = 0.05
    ANOMALY_THRESHOLD = 0.7
    ZSCORE_ALERT      = 3.0

config = Config()
"""

# ── 2. DATA STORE FIX ─────────────────────────────────────────
files["src/api/data_store.py"] = """\
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
"""

# ── 3. ROUTES STATS FIX ───────────────────────────────────────
files["src/api/routes_stats.py"] = """\
from fastapi import APIRouter
import pandas as pd
import os
from src.api.data_store import get_df, get_alert_df, get_shap_importance
from src.explainability.alert_formatter import get_risk_level
from src.config import config

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
            "from": str(df["date_only"].min())[:10],
            "to"  : str(df["date_only"].max())[:10],
        },
        "top_risk_users": [
            {
                "user"      : row["user"],
                "max_score" : round(float(row["ensemble_score"]), 4),
                "risk_level": get_risk_level(float(row["ensemble_score"])),
            }
            for _, row in top_users.iterrows()
        ],
    }


@router.get("/shap-importance")
def get_shap_importance_endpoint():
    imp    = get_shap_importance()
    if not imp:
        return {"features": [], "error": "SHAP data not loaded"}
    ranked = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    return {
        "features": [
            {"rank": i+1, "feature": feat, "importance": round(val, 5)}
            for i, (feat, val) in enumerate(ranked)
        ]
    }


@router.get("/model-comparison")
def get_model_comparison():
    # Try v2 first (includes LSTM), fallback to v1
    for fname in ["model_comparison_v2.csv", "model_comparison.csv"]:
        path = config.REPORTS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            return {"models": df.fillna(0).to_dict(orient="records")}
    return {"error": "model_comparison.csv not found", "models": []}
"""

# ── 4. DASHBOARD VIEWS FIX — use env var for API ──────────────
API_LINE_OLD = 'API = "http://127.0.0.1:8000"'
API_LINE_NEW = 'API = os.environ.get("ITDS_API_BASE", "http://127.0.0.1:8000")'

view_files = [
    "dashboard/views/overview.py",
    "dashboard/views/alerts.py",
    "dashboard/views/users.py",
    "dashboard/views/explainability.py",
    "dashboard/views/model_report.py",
    "dashboard/views/feedback.py",
]

for vf in view_files:
    if os.path.exists(vf):
        content = open(vf, encoding="utf-8").read()
        # Fix API variable
        for old in [
            'API = "http://127.0.0.1:8000"',
            "API     = \"http://127.0.0.1:8000\"",
            'API   = "http://127.0.0.1:8000"',
        ]:
            content = content.replace(old, API_LINE_NEW)
        # Make sure os is imported
        if "import os" not in content:
            content = "import os\n" + content
        open(vf, "w", encoding="utf-8").write(content)
        print(f"Updated API base: {vf}")

# ── 5. EXPLAINABILITY VIEW — full fix ─────────────────────────
files["dashboard/views/explainability.py"] = """\
import sys, os
sys.path.insert(0, os.path.abspath("."))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

API = os.environ.get("ITDS_API_BASE", "http://127.0.0.1:8000")
T   = 15
RC  = {"CRITICAL":"#D85A30","HIGH":"#E8953A","MEDIUM":"#F5C842","LOW":"#1D9E75"}


def show():
    st.title("Explainability Center")
    tab1, tab2 = st.tabs(["SHAP Feature Importance", "Alert Reasoning"])

    with tab1:
        st.subheader("Global SHAP Feature Importance")
        try:
            r    = requests.get(f"{API}/stats/shap-importance", timeout=T)
            data = r.json()
            feats = data.get("features", [])
            if not feats and "error" in data:
                st.warning(f"SHAP: {data['error']}")
                feats = []
        except Exception as e:
            st.error(f"Cannot reach API: {e}")
            feats = []

        if feats:
            df_s = pd.DataFrame(feats).head(15)
            fig  = go.Figure(go.Bar(
                x=df_s["importance"],
                y=df_s["feature"],
                orientation="h",
                marker_color="#378ADD",
                text=[f"{v:.4f}" for v in df_s["importance"]],
                textposition="outside",
            ))
            fig.update_layout(
                height=480,
                xaxis_title="Mean |SHAP value|",
                yaxis=dict(autorange="reversed"),
                margin=dict(t=20, b=20, l=220),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("Key Finding: Temporal features dominate — insiders deviate in WHEN they work.")
        else:
            st.warning("SHAP data not available. Check that shap_importance.json exists in models/isolation_forest/")
            # Show file path for debugging
            st.caption(f"Expected: models/isolation_forest/shap_importance.json")

    with tab2:
        st.subheader("Alert-Level Explanation")
        try:
            r      = requests.get(f"{API}/alerts/", params={"limit": 50}, timeout=T)
            alerts = r.json()
        except Exception as e:
            st.error(f"Cannot reach API: {e}")
            alerts = []

        if not alerts:
            st.warning("No alerts loaded. Verify API is running and alert_table.csv exists.")
            st.caption(f"API: {API}/alerts/")
            return

        opts = {
            f"{a['user']} | {a['date']} | {a['ensemble_score']:.3f} | {a['risk_level']}":
            a["alert_id"] for a in alerts
        }
        label = st.selectbox("Select Alert", list(opts.keys()))
        aid   = opts[label]

        try:
            r   = requests.get(f"{API}/alerts/{aid}", timeout=T)
            det = r.json()
        except Exception as e:
            st.error(f"Cannot load detail: {e}")
            det = {}

        if not det or "alert_id" not in det:
            st.warning("Could not load alert detail.")
            return

        risk  = det.get("risk_level", "LOW")
        color = RC.get(risk, "#888")
        st.markdown(
            f"<h3 style='color:{color}'>{risk} ALERT - "
            f"{det.get('user','').upper()}</h3>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ensemble", f"{det.get('ensemble_score', 0):.4f}")
        c2.metric("AE Score", f"{det.get('ae_score', 0):.4f}")
        c3.metric("IF Score", f"{det.get('if_score', 0):.4f}")
        c4.metric("Both Flagged", "YES" if det.get("both_flagged") else "NO")

        st.markdown("**Detected Anomaly Reasons:**")
        reasons = det.get("reasons", [])
        tags    = {"HIGH": "[HIGH]", "MEDIUM": "[MED]", "LOW": "[LOW]"}
        if reasons:
            for r_item in reasons:
                sev = r_item.get("severity", "LOW")
                st.markdown(f"**{tags.get(sev, sev)}** {r_item.get('reason', '')}")
        else:
            st.info("No rule triggers for this session.")

        st.markdown("**Session Statistics:**")
        stats = det.get("stats", {})
        if stats:
            rows = [[k.replace("_", " ").title(), v] for k, v in stats.items()]
            st.dataframe(
                pd.DataFrame(rows, columns=["Metric", "Value"]),
                use_container_width=True,
                height=260,
            )
"""

# ── 6. ALERTS VIEW FIX ────────────────────────────────────────
files["dashboard/views/alerts.py"] = """\
import sys, os
sys.path.insert(0, os.path.abspath("."))
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

API = os.environ.get("ITDS_API_BASE", "http://127.0.0.1:8000")
T   = 15
RC  = {"CRITICAL":"#D85A30","HIGH":"#E8953A","MEDIUM":"#F5C842","LOW":"#1D9E75"}
RB  = {"CRITICAL":"#FAECE7","HIGH":"#FEF3E2","MEDIUM":"#FEFBE6","LOW":"#E8F8F2"}


def show():
    st.title("Alert Center")

    # Test API connection first
    try:
        r      = requests.get(f"{API}/health", timeout=5)
        health = r.json()
        if not health.get("data_loaded", False):
            st.error("API is running but data is not loaded yet. Wait a moment and refresh.")
            return
    except Exception as e:
        st.error(f"Cannot connect to API at {API}. Error: {e}")
        st.info("Make sure: uvicorn src.api.main:app --reload --port 8000")
        return

    try:
        counts = requests.get(f"{API}/alerts/count", timeout=T).json()
    except:
        counts = {}

    col1, col2, col3 = st.columns(3)
    with col1:
        risk = st.selectbox("Risk Level", ["All","CRITICAL","HIGH","MEDIUM","LOW"])
    with col2:
        min_score = st.slider("Min Ensemble Score", 0.0, 1.0, 0.0, 0.05)
    with col3:
        user_input = st.text_input("Filter by User ID", "")

    params = {"limit": 300, "min_score": min_score}
    if risk != "All":
        params["risk_level"] = risk
    if user_input.strip():
        params["user"] = user_input.strip().lower()

    try:
        r      = requests.get(f"{API}/alerts/", params=params, timeout=T)
        alerts = r.json()
    except Exception as e:
        st.error(f"Failed to load alerts: {e}")
        return

    if not alerts:
        st.warning("No alerts match filters.")
        st.caption(f"API returned 0 results for: {params}")
        # Show debug info
        try:
            cnt = requests.get(f"{API}/alerts/count", timeout=T).json()
            st.info(f"Total alerts in system: {cnt.get('total', 0)}")
        except:
            pass
        return

    df = pd.DataFrame(alerts)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Showing",     len(df))
    c2.metric("CRITICAL",    counts.get("critical", 0))
    c3.metric("HIGH",        counts.get("high", 0))
    c4.metric("Both Flagged",
              int(df["both_flagged"].sum()) if "both_flagged" in df.columns else 0)

    st.markdown("---")
    st.subheader(f"Alerts ({len(df)} results)")

    display_cols = ["user","date","risk_level","ensemble_score","ae_score",
                    "if_score","both_flagged","device_count",
                    "email_external","http_suspicious","reason_summary"]
    avail = [c for c in display_cols if c in df.columns]
    dfd   = df[avail].copy()

    for col in ["ensemble_score","ae_score","if_score"]:
        if col in dfd.columns:
            dfd[col] = dfd[col].round(4)

    def cr(val):
        return (f"background-color:{RB.get(str(val),'#fff')};"
                f"color:{RC.get(str(val),'#888')};font-weight:bold")

    st.dataframe(
        dfd.style.map(cr, subset=["risk_level"]),
        use_container_width=True,
        height=420,
    )

    if "ensemble_score" in df.columns and "risk_level" in df.columns:
        st.subheader("Score Distribution")
        fig = px.histogram(
            df, x="ensemble_score", color="risk_level",
            nbins=40, color_discrete_map=RC,
        )
        fig.update_layout(height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
"""

# ── 7. FEEDBACK VIEW FIX ──────────────────────────────────────
files["dashboard/views/feedback.py"] = """\
import sys, os
sys.path.insert(0, os.path.abspath("."))
import streamlit as st
import pandas as pd
import requests

API = os.environ.get("ITDS_API_BASE", "http://127.0.0.1:8000")
T   = 15
RC  = {"CRITICAL":"#D85A30","HIGH":"#E8953A","MEDIUM":"#F5C842","LOW":"#1D9E75"}


def show():
    st.title("Analyst Feedback")
    st.markdown("Mark alerts as false positives or confirm threats to adapt detection thresholds.")

    tab1, tab2, tab3 = st.tabs(["Submit Feedback","Feedback Stats","User Adjustments"])

    with tab1:
        st.subheader("Submit Alert Feedback")
        try:
            r      = requests.get(f"{API}/alerts/", params={"limit": 100}, timeout=T)
            alerts = r.json()
        except Exception as e:
            st.error(f"Cannot reach API: {e}")
            st.info(f"API target: {API}")
            return

        if not alerts:
            st.warning("No alerts loaded. Make sure API is running and data is loaded.")
            st.caption(f"API: {API}/alerts/")
            return

        opts = {
            f"{a['user']} | {a['date']} | {a['ensemble_score']:.3f} | {a['risk_level']}":
            a for a in alerts
        }
        selected_label = st.selectbox("Select Alert", list(opts.keys()))
        sel            = opts[selected_label]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Ensemble Score", f"{sel.get('ensemble_score',0):.4f}")
            st.metric("Risk Level",     sel.get("risk_level",""))
        with col2:
            st.metric("AE Score", f"{sel.get('ae_score',0):.4f}")
            st.metric("IF Score", f"{sel.get('if_score',0):.4f}")

        st.markdown("---")
        analyst = st.text_input("Analyst Name", value="analyst_1")

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
                        "alert_id": sel["alert_id"],
                        "user"    : sel["user"],
                        "analyst" : analyst,
                        "reason"  : fp_reason,
                    }, timeout=T)
                    result = resp.json()
                    st.success(f"Recorded. {result.get('effect','')}")
                    boost = result.get("threshold_boost", 0)
                    st.info(f"Threshold for {sel['user']} raised by {boost:.0%}")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col_cf:
            st.markdown("### Confirm as Real Threat")
            severity = st.selectbox("Severity", ["CRITICAL","HIGH","MEDIUM","LOW"])

            if st.button("Confirm Threat", type="primary"):
                try:
                    resp = requests.post(f"{API}/feedback/confirm", json={
                        "alert_id": sel["alert_id"],
                        "user"    : sel["user"],
                        "analyst" : analyst,
                        "severity": severity,
                    }, timeout=T)
                    result = resp.json()
                    st.success(f"Confirmed {severity} threat for {sel['user']}")
                    st.warning("Detection sensitivity increased for this user.")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        st.subheader("Feedback Statistics")
        try:
            stats = requests.get(f"{API}/feedback/stats", timeout=T).json()
            c1, c2, c3 = st.columns(3)
            c1.metric("False Positives", stats.get("total_false_positives", 0))
            c2.metric("Confirmed",       stats.get("total_confirmed", 0))
            c3.metric("Users Adjusted",  stats.get("users_adjusted", 0))

            adj = stats.get("user_adjustments", {})
            if adj:
                st.subheader("User Threshold Adjustments")
                rows = [{
                    "User"            : u,
                    "FP Count"        : d.get("fp_count", 0),
                    "Threshold Boost" : f"+{d.get('threshold_boost',0):.0%}",
                    "Last Updated"    : str(d.get("last_updated",""))[:19],
                } for u, d in adj.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No adjustments yet. Submit feedback above to see changes here.")
        except Exception as e:
            st.error(f"Cannot load stats: {e}")

    with tab3:
        st.subheader("User Adjustment Details")
        user_id = st.text_input("Enter User ID", "gko0078")
        if st.button("Check User"):
            try:
                r    = requests.get(f"{API}/feedback/user/{user_id}", timeout=T)
                data = r.json()
                st.markdown(f"**User:** `{data['user']}`")
                c1, c2, c3 = st.columns(3)
                c1.metric("False Positives", data.get("false_positives", 0))
                c2.metric("Confirmed",       data.get("confirmed", 0))
                c3.metric("Threshold Boost", f"+{data.get('threshold_boost',0):.0%}")
                if data.get("adjusted"):
                    st.warning(f"Threshold raised for this user.")
                else:
                    st.success("No adjustments. Using default threshold.")
            except Exception as e:
                st.error(f"Error: {e}")
"""

# Write all files
for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\nAll fixes applied.")
print("\nNow run:")
print("  Terminal 1: uvicorn src.api.main:app --reload --port 8000")
print("  Terminal 2: streamlit run dashboard/app.py")