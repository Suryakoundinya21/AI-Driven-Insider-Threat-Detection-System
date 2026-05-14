import os

files = {}

# ─────────────────────────────────────────────────────────────
# FIX 1: ALERT FORMATTER — zscore must be string for Pydantic
# ─────────────────────────────────────────────────────────────
files["src/explainability/alert_formatter.py"] = """\
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
"""

# ─────────────────────────────────────────────────────────────
# FIX 2: ROUTES ALERTS — fix alert detail endpoint
# ─────────────────────────────────────────────────────────────
files["src/api/routes_alerts.py"] = """\
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
"""

# ─────────────────────────────────────────────────────────────
# FIX 3: EXPLAINABILITY PAGE — fix zscore display + all errors
# ─────────────────────────────────────────────────────────────
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

    # ── Tab 1: SHAP ──────────────────────────────────────────
    with tab1:
        st.subheader("Global SHAP Feature Importance")
        st.markdown("Features ranked by average contribution to anomaly detection decisions.")

        try:
            r     = requests.get(f"{API}/stats/shap-importance", timeout=T)
            data  = r.json()
            feats = data.get("features", [])
        except Exception as e:
            st.error(f"Cannot reach API: {e}")
            feats = []

        if feats:
            df_s = pd.DataFrame(feats).head(15)
            colors = []
            for i in range(len(df_s)):
                if i < 3:
                    colors.append("#D85A30")
                elif i < 6:
                    colors.append("#E8953A")
                else:
                    colors.append("#378ADD")

            fig = go.Figure(go.Bar(
                x=df_s["importance"],
                y=df_s["feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.4f}" for v in df_s["importance"]],
                textposition="outside",
            ))
            fig.update_layout(
                height=500,
                xaxis_title="Mean |SHAP value|",
                yaxis=dict(autorange="reversed"),
                margin=dict(t=20, b=20, l=220),
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top 3 Anomaly Drivers:**")
                for i, row in df_s.head(3).iterrows():
                    st.markdown(f"**{i+1}. {row['feature']}** — `{row['importance']:.4f}`")
            with col2:
                st.info(
                    "**Key Finding:** Temporal features account for 45%+ "
                    "of total SHAP weight. Insiders deviate in WHEN they "
                    "work, not just WHAT they do."
                )
        else:
            st.warning("SHAP data not available.")

    # ── Tab 2: Alert Reasoning ───────────────────────────────
    with tab2:
        st.subheader("Alert-Level Explanation")
        st.markdown("Select any alert to see exactly why it was flagged — "
                    "with baseline comparisons for each anomalous feature.")

        try:
            r      = requests.get(f"{API}/alerts/", params={"limit": 100}, timeout=T)
            alerts = r.json()
            if not isinstance(alerts, list):
                alerts = []
        except Exception as e:
            st.error(f"Cannot reach API: {e}")
            return

        if not alerts:
            st.warning("No alerts loaded.")
            return

        # Risk level filter using radio (not selectbox — avoids horizontal error)
        risk_filter = st.radio(
            "Filter by Risk Level",
            ["All", "CRITICAL", "HIGH", "MEDIUM"],
            horizontal=True,
        )
        if risk_filter != "All":
            alerts = [a for a in alerts if a.get("risk_level","") == risk_filter]

        if not alerts:
            st.warning(f"No {risk_filter} alerts found.")
            return

        opts = {
            f"{a['user']} | {a['date']} | Score: {a['ensemble_score']:.3f} | {a['risk_level']}":
            a["alert_id"] for a in alerts
        }
        label = st.selectbox("Select Alert to Investigate", list(opts.keys()))
        aid   = opts[label]

        # Load detail
        try:
            r   = requests.get(f"{API}/alerts/{aid}", timeout=T)
            if r.status_code != 200:
                st.error(f"API returned status {r.status_code}: {r.text[:200]}")
                return
            det = r.json()
        except Exception as e:
            st.error(f"Cannot load detail: {e}")
            return

        if not det or "alert_id" not in det:
            st.warning("Alert detail not available.")
            return

        # ── Header ───────────────────────────────────────────
        risk  = det.get("risk_level", "LOW")
        color = RC.get(risk, "#888")
        st.markdown(
            f"<div style='background:{color}22;border-left:5px solid {color};"
            f"padding:14px;border-radius:6px;margin:10px 0'>"
            f"<h3 style='color:{color};margin:0'>{risk} ALERT — "
            f"{det.get('user','').upper()}</h3>"
            f"<p style='margin:4px 0;color:#aaa'>"
            f"Date: {det.get('date','')} | ID: {det.get('alert_id','')}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Scores ────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ensemble Score",   f"{det.get('ensemble_score',0):.4f}")
        c2.metric("Autoencoder",      f"{det.get('ae_score',0):.4f}")
        c3.metric("Isolation Forest", f"{det.get('if_score',0):.4f}")
        c4.metric("Both Models",      "YES ⚠️" if det.get("both_flagged") else "NO")

        # ── Reasons ───────────────────────────────────────────
        st.markdown("---")
        reasons = det.get("reasons", [])

        if reasons:
            high_r   = [r for r in reasons if r.get("severity") == "HIGH"]
            medium_r = [r for r in reasons if r.get("severity") == "MEDIUM"]
            low_r    = [r for r in reasons if r.get("severity") == "LOW"]

            st.markdown(f"### Why This Session Was Flagged ({len(reasons)} triggers)")

            if high_r:
                st.markdown("#### 🔴 High Severity")
                for r in high_r:
                    st.error(f"**[HIGH]** {r.get('reason','')}")
                    zs = r.get('zscore','0')
                    st.caption(
                        f"`{r.get('feature','')}` | "
                        f"Observed: **{r.get('value','')}** | "
                        f"Baseline: {r.get('baseline','')} | "
                        f"Z-Score: {zs}σ"
                    )

            if medium_r:
                st.markdown("#### 🟠 Medium Severity")
                for r in medium_r:
                    st.warning(f"**[MEDIUM]** {r.get('reason','')}")
                    zs = r.get('zscore','0')
                    st.caption(
                        f"`{r.get('feature','')}` | "
                        f"Observed: **{r.get('value','')}** | "
                        f"Baseline: {r.get('baseline','')} | "
                        f"Z-Score: {zs}σ"
                    )

            if low_r:
                st.markdown("#### 🟡 Low Severity")
                for r in low_r:
                    st.info(f"**[LOW]** {r.get('reason','')}")
        else:
            st.info("No specific rule triggers. Flagged by overall model reconstruction error.")

        # ── Baseline vs Observed ──────────────────────────────
        st.markdown("---")
        st.markdown("### Baseline vs Observed — Session Statistics")
        stats = det.get("stats", {})
        if stats:
            stat_labels = {
                "device_count"      : "USB/Device Events",
                "email_to_external" : "External Emails Sent",
                "http_suspicious"   : "Suspicious URL Visits",
                "sensitive_files"   : "Sensitive File Accesses",
                "after_hours_ratio" : "After-Hours Activity Ratio",
                "first_logon_hour"  : "First Login Hour",
                "total_events"      : "Total Session Events",
            }
            rows = []
            for key, label in stat_labels.items():
                val      = stats.get(key, 0)
                base_val = "—"
                for r in reasons:
                    if r.get("feature") == key:
                        base_val = r.get("baseline", "—")
                        break
                rows.append({
                    "Metric"   : label,
                    "Observed" : val,
                    "Baseline" : base_val,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=270)

        # ── Analyst Actions ───────────────────────────────────
        st.markdown("---")
        st.markdown("### Analyst Decision")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if st.button("Confirm as Real Threat", type="primary"):
                try:
                    requests.post(f"{API}/feedback/confirm", json={
                        "alert_id": aid,
                        "user"    : det.get("user",""),
                        "analyst" : "analyst_1",
                        "severity": risk,
                    }, timeout=T)
                    st.success("Confirmed and escalated.")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col_b:
            if st.button("Mark as False Positive", type="secondary"):
                try:
                    requests.post(f"{API}/feedback/false-positive", json={
                        "alert_id": aid,
                        "user"    : det.get("user",""),
                        "analyst" : "analyst_1",
                        "reason"  : "Reviewed and dismissed",
                    }, timeout=T)
                    st.success("Marked as false positive. Threshold adjusted.")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col_c:
            st.button("Flag for Investigation", type="secondary")
"""

# ─────────────────────────────────────────────────────────────
# FIX 4: FEEDBACK PAGE — user dropdown instead of text input
# ─────────────────────────────────────────────────────────────
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
            if not isinstance(alerts, list):
                alerts = []
        except Exception as e:
            st.error(f"Cannot reach API: {e}")
            return

        if not alerts:
            st.warning("No alerts loaded.")
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
                    resp   = requests.post(f"{API}/feedback/false-positive", json={
                        "alert_id": sel["alert_id"],
                        "user"    : sel["user"],
                        "analyst" : analyst,
                        "reason"  : fp_reason,
                    }, timeout=T)
                    result = resp.json()
                    boost  = result.get("threshold_boost", 0)
                    st.success(f"Recorded. Threshold raised by {boost:.0%} for {sel['user']}")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col_cf:
            st.markdown("### Confirm as Real Threat")
            severity = st.selectbox("Severity", ["CRITICAL","HIGH","MEDIUM","LOW"])

            if st.button("Confirm Threat", type="primary"):
                try:
                    requests.post(f"{API}/feedback/confirm", json={
                        "alert_id": sel["alert_id"],
                        "user"    : sel["user"],
                        "analyst" : analyst,
                        "severity": severity,
                    }, timeout=T)
                    st.success(f"Confirmed {severity} threat for {sel['user']}")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        st.subheader("Feedback Statistics")
        try:
            stats = requests.get(f"{API}/feedback/stats", timeout=T).json()
            c1,c2,c3 = st.columns(3)
            c1.metric("False Positives", stats.get("total_false_positives",0))
            c2.metric("Confirmed",       stats.get("total_confirmed",0))
            c3.metric("Users Adjusted",  stats.get("users_adjusted",0))

            adj = stats.get("user_adjustments",{})
            if adj:
                st.subheader("User Threshold Adjustments")
                rows = [{
                    "User"            : u,
                    "FP Count"        : d.get("fp_count",0),
                    "Threshold Boost" : f"+{d.get('threshold_boost',0):.0%}",
                    "Last Updated"    : str(d.get("last_updated",""))[:19],
                } for u, d in adj.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No adjustments yet.")
        except Exception as e:
            st.error(f"Cannot load stats: {e}")

    with tab3:
        st.subheader("User Adjustment Details")

        # Load top users for dropdown
        try:
            r         = requests.get(f"{API}/users/top-risk",
                                     params={"limit":50}, timeout=T)
            top_users = r.json()
            user_list = [u["user"] for u in top_users] if top_users else []
        except:
            user_list = []

        # Also load adjusted users from feedback
        try:
            stats    = requests.get(f"{API}/feedback/stats", timeout=T).json()
            adj_list = list(stats.get("user_adjustments", {}).keys())
        except:
            adj_list = []

        # Merge both lists, deduplicate
        all_users = list(dict.fromkeys(adj_list + user_list))

        if all_users:
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_user = st.selectbox(
                    "Select User to Check",
                    all_users,
                    help="Users with existing adjustments shown first"
                )
            with col2:
                manual_user = st.text_input(
                    "Or type User ID manually",
                    placeholder="e.g. dlm0051"
                )

            user_id = manual_user.strip().lower() if manual_user.strip() else selected_user
        else:
            user_id = st.text_input("Enter User ID", "gko0078")

        if st.button("Check User Adjustment"):
            try:
                r    = requests.get(f"{API}/feedback/user/{user_id}", timeout=T)
                data = r.json()

                st.markdown(f"### User: `{data['user']}`")
                c1, c2, c3 = st.columns(3)
                c1.metric("False Positives", data.get("false_positives",0))
                c2.metric("Confirmed",       data.get("confirmed",0))
                c3.metric("Threshold Boost", f"+{data.get('threshold_boost',0):.0%}")

                if data.get("adjusted"):
                    boost = data.get("threshold_boost", 0)
                    st.warning(
                        f"**Threshold adjusted for {user_id}.** "
                        f"Alerts only shown if score > "
                        f"{0.5 + boost:.2f} (raised by {boost:.0%})"
                    )
                else:
                    st.success("No adjustments. Using default threshold (0.50).")

                # Show history
                history = data.get("history", [])
                if history:
                    st.markdown("**Recent Feedback History:**")
                    hist_rows = [{
                        "Type"      : h.get("type","").replace("_"," ").title(),
                        "Alert"     : h.get("alert_id",""),
                        "Analyst"   : h.get("analyst",""),
                        "Timestamp" : str(h.get("timestamp",""))[:19],
                    } for h in history[-10:]]
                    st.dataframe(pd.DataFrame(hist_rows),
                                 use_container_width=True)

            except Exception as e:
                st.error(f"Error loading user {user_id}: {e}")
"""

for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\nAll fixes applied. Restart both services:")
print("  Terminal 1: uvicorn src.api.main:app --reload --port 8000")
print("  Terminal 2: streamlit run dashboard/app.py")