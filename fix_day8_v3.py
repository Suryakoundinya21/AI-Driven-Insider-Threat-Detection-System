import os

files = {}

files["dashboard/__init__.py"] = ""
files["dashboard/pages/__init__.py"] = ""

files["dashboard/config.py"] = """\
API_BASE = "http://127.0.0.1:8000"
APP_TITLE = "Insider Threat Detection System"

RISK_COLORS = {
    "CRITICAL": "#D85A30",
    "HIGH": "#E8953A",
    "MEDIUM": "#F5C842",
    "LOW": "#1D9E75",
}

RISK_BG = {
    "CRITICAL": "#FAECE7",
    "HIGH": "#FEF3E2",
    "MEDIUM": "#FEFBE6",
    "LOW": "#E8F8F2",
}
"""

files["dashboard/api_client.py"] = """\
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
TIMEOUT = 15

@st.cache_data(ttl=60)
def get_overview():
    try:
        r = requests.get(f"{API_BASE}/stats/overview", timeout=TIMEOUT)
        return r.json()
    except Exception as e:
        return {}

@st.cache_data(ttl=60)
def get_alert_counts():
    try:
        r = requests.get(f"{API_BASE}/alerts/count", timeout=TIMEOUT)
        return r.json()
    except:
        return {}

@st.cache_data(ttl=60)
def get_alerts(risk_level=None, user=None, min_score=0.0, limit=200):
    params = {"limit": limit, "min_score": min_score}
    if risk_level:
        params["risk_level"] = risk_level
    if user:
        params["user"] = user
    try:
        r = requests.get(f"{API_BASE}/alerts/", params=params, timeout=TIMEOUT)
        return r.json()
    except:
        return []

@st.cache_data(ttl=60)
def get_top_users(limit=20):
    try:
        r = requests.get(f"{API_BASE}/users/top-risk", params={"limit": limit}, timeout=TIMEOUT)
        return r.json()
    except:
        return []

@st.cache_data(ttl=60)
def get_user_timeline(user_id, days=365):
    try:
        r = requests.get(f"{API_BASE}/users/{user_id}/timeline", params={"days": days}, timeout=TIMEOUT)
        return r.json()
    except:
        return {}

@st.cache_data(ttl=60)
def get_user_summary(user_id):
    try:
        r = requests.get(f"{API_BASE}/users/{user_id}/summary", timeout=TIMEOUT)
        return r.json()
    except:
        return {}

@st.cache_data(ttl=300)
def get_shap_importance():
    try:
        r = requests.get(f"{API_BASE}/stats/shap-importance", timeout=TIMEOUT)
        return r.json()
    except:
        return {}

@st.cache_data(ttl=300)
def get_model_comparison():
    try:
        r = requests.get(f"{API_BASE}/stats/model-comparison", timeout=TIMEOUT)
        return r.json()
    except:
        return {}

def get_alert_detail(alert_id):
    try:
        r = requests.get(f"{API_BASE}/alerts/{alert_id}", timeout=TIMEOUT)
        return r.json()
    except:
        return {}
"""

files["dashboard/pages/overview.py"] = """\
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

RISK_COLORS = {
    "CRITICAL": "#D85A30",
    "HIGH": "#E8953A",
    "MEDIUM": "#F5C842",
    "LOW": "#1D9E75",
}

def get_overview():
    import requests
    try:
        r = requests.get("http://127.0.0.1:8000/stats/overview", timeout=15)
        return r.json()
    except:
        return {}

def get_alert_counts():
    import requests
    try:
        r = requests.get("http://127.0.0.1:8000/alerts/count", timeout=15)
        return r.json()
    except:
        return {}

def get_top_users(limit=10):
    import requests
    try:
        r = requests.get("http://127.0.0.1:8000/users/top-risk", params={"limit": limit}, timeout=15)
        return r.json()
    except:
        return []

def show():
    st.title("System Overview")
    st.markdown("Real-time insider threat detection dashboard")

    overview = get_overview()
    counts = get_alert_counts()

    if not overview:
        st.error("Cannot connect to API. Run: uvicorn src.api.main:app --reload --port 8000")
        st.code("uvicorn src.api.main:app --reload --port 8000")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sessions", f"{overview.get('total_sessions', 0):,}")
    c2.metric("Total Users", f"{overview.get('total_users', 0):,}")
    c3.metric("AE Anomalies", f"{overview.get('ae_anomalies', 0):,}")
    c4.metric("IF Anomalies", f"{overview.get('if_anomalies', 0):,}")
    c5.metric("High-Conf Alerts", f"{overview.get('ensemble_intersect', 0):,}")

    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Alert Risk Breakdown")
        labels = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        values = [
            counts.get("critical", 0),
            counts.get("high", 0),
            counts.get("medium", 0),
            counts.get("low", 0),
        ]
        colors = [RISK_COLORS[l] for l in labels]
        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.5,
            marker_colors=colors, textinfo="label+percent",
        ))
        fig.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Highest-Risk Users")
        top_users = get_top_users(10)
        if top_users:
            df_top = pd.DataFrame(top_users)
            colors = [RISK_COLORS.get(r, "#888") for r in df_top["risk_level"]]
            fig = go.Figure(go.Bar(
                x=df_top["user"],
                y=df_top["max_score"],
                marker_color=colors,
                text=[f"{s:.3f}" for s in df_top["max_score"]],
                textposition="outside",
            ))
            fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                          annotation_text="High-risk threshold")
            fig.update_layout(height=320, yaxis_range=[0, 1.1], margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    dr = overview.get("date_range", {})
    st.info(f"Dataset: {dr.get('from', 'N/A')} to {dr.get('to', 'N/A')}")
"""

files["dashboard/pages/alerts.py"] = """\
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import streamlit as st
import pandas as pd
import plotly.express as px
import requests

API_BASE = "http://127.0.0.1:8000"
TIMEOUT = 15

RISK_COLORS = {
    "CRITICAL": "#D85A30",
    "HIGH": "#E8953A",
    "MEDIUM": "#F5C842",
    "LOW": "#1D9E75",
}
RISK_BG = {
    "CRITICAL": "#FAECE7",
    "HIGH": "#FEF3E2",
    "MEDIUM": "#FEFBE6",
    "LOW": "#E8F8F2",
}

def fetch_alerts(risk_level=None, user=None, min_score=0.0, limit=200):
    params = {"limit": limit, "min_score": min_score}
    if risk_level:
        params["risk_level"] = risk_level
    if user:
        params["user"] = user
    try:
        r = requests.get(f"{API_BASE}/alerts/", params=params, timeout=TIMEOUT)
        return r.json()
    except:
        return []

def fetch_counts():
    try:
        r = requests.get(f"{API_BASE}/alerts/count", timeout=TIMEOUT)
        return r.json()
    except:
        return {}

def show():
    st.title("Alert Center")
    counts = fetch_counts()

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.selectbox("Risk Level", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with col2:
        min_score = st.slider("Min Ensemble Score", 0.0, 1.0, 0.0, 0.05)
    with col3:
        user_filter = st.text_input("Filter by User ID", "")

    risk_param = None if risk_filter == "All" else risk_filter
    user_param = user_filter.strip().lower() if user_filter.strip() else None

    alerts = fetch_alerts(risk_level=risk_param, user=user_param, min_score=min_score, limit=300)
    if not alerts:
        st.warning("No alerts match the current filters.")
        return

    df = pd.DataFrame(alerts)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Showing", len(df))
    c2.metric("CRITICAL", counts.get("critical", 0))
    c3.metric("HIGH", counts.get("high", 0))
    c4.metric("Both Flagged", int(df["both_flagged"].sum()) if "both_flagged" in df.columns else 0)

    st.markdown("---")
    st.subheader(f"Alerts ({len(df)} results)")

    display_cols = [
        "user", "date", "risk_level", "ensemble_score",
        "ae_score", "if_score", "both_flagged",
        "device_count", "email_external", "http_suspicious", "reason_summary",
    ]
    avail = [c for c in display_cols if c in df.columns]
    df_display = df[avail].copy()

    for col in ["ensemble_score", "ae_score", "if_score"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].round(4)

    def color_risk(val):
        color = RISK_COLORS.get(str(val), "#888")
        bg = RISK_BG.get(str(val), "#fff")
        return f"background-color: {bg}; color: {color}; font-weight: bold"

    styled = df_display.style.map(color_risk, subset=["risk_level"])
    st.dataframe(styled, use_container_width=True, height=420)

    if "ensemble_score" in df.columns and "risk_level" in df.columns:
        st.subheader("Score Distribution")
        fig = px.histogram(
            df, x="ensemble_score", color="risk_level",
            nbins=40, color_discrete_map=RISK_COLORS,
        )
        fig.update_layout(height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
"""

files["dashboard/pages/users.py"] = """\
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

API_BASE = "http://127.0.0.1:8000"
TIMEOUT = 15

RISK_COLORS = {
    "CRITICAL": "#D85A30",
    "HIGH": "#E8953A",
    "MEDIUM": "#F5C842",
    "LOW": "#1D9E75",
}

def show():
    st.title("User Investigation")

    try:
        r = requests.get(f"{API_BASE}/users/top-risk", params={"limit": 20}, timeout=TIMEOUT)
        top_users = r.json()
    except:
        st.error("Cannot reach API.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Top Risk Users")
        df_users = pd.DataFrame(top_users)
        show_cols = [c for c in ["user","risk_level","max_score","flagged_sessions","total_usb"] if c in df_users.columns]
        st.dataframe(df_users[show_cols], use_container_width=True, height=400)
        selected_user = st.selectbox("Investigate User", [u["user"] for u in top_users])

    with col2:
        if selected_user:
            try:
                r = requests.get(f"{API_BASE}/users/{selected_user}/summary", timeout=TIMEOUT)
                summary = r.json()
            except:
                summary = {}

            if summary and "error" not in summary:
                st.subheader(f"User: {selected_user}")
                risk = summary.get("risk_level", "LOW")
                color = RISK_COLORS.get(risk, "#888")
                st.markdown(f"<h3 style='color:{color}'>Risk Level: {risk}</h3>", unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Max Score", f"{summary.get('max_ensemble_score', 0):.4f}")
                m2.metric("Sessions", summary.get("total_sessions", 0))
                m3.metric("Flagged", summary.get("flagged_sessions", 0))
                m4.metric("Flag Rate", f"{summary.get('flag_rate', 0)*100:.1f}%")

                m5, m6, m7 = st.columns(3)
                m5.metric("USB Events", summary.get("total_usb_events", 0))
                m6.metric("Ext Emails", summary.get("total_ext_emails", 0))
                m7.metric("Susp URLs", summary.get("total_susp_http", 0))

    if selected_user:
        st.markdown("---")
        st.subheader(f"Activity Timeline - {selected_user}")

        try:
            r = requests.get(f"{API_BASE}/users/{selected_user}/timeline",
                             params={"days": 500}, timeout=TIMEOUT)
            timeline_data = r.json()
        except:
            st.error("Could not load timeline.")
            return

        tl = timeline_data.get("timeline", [])
        if not tl:
            st.info("No timeline data.")
            return

        df_tl = pd.DataFrame(tl)
        df_tl["date"] = pd.to_datetime(df_tl["date"])

        flagged_col = df_tl["flagged"].tolist() if "flagged" in df_tl.columns else [False]*len(df_tl)
        marker_sizes = [8 if f else 4 for f in flagged_col]
        marker_colors = ["#D85A30" if f else "#378ADD" for f in flagged_col]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_tl["date"],
            y=df_tl["ensemble_score"],
            mode="lines+markers",
            name="Ensemble Score",
            line=dict(color="#378ADD", width=1.5),
            marker=dict(size=marker_sizes, color=marker_colors),
        ))
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="CRITICAL")
        fig.add_hline(y=0.6, line_dash="dot", line_color="orange", annotation_text="HIGH")
        fig.update_layout(height=300, yaxis_range=[0, 1.05], margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Flagged Sessions")
        act_cols = ["date","device_count","email_external","http_suspicious","after_hours_ratio","ensemble_score"]
        avail = [c for c in act_cols if c in df_tl.columns]
        if "flagged" in df_tl.columns:
            flagged_df = df_tl[df_tl["flagged"] == True][avail]
        else:
            flagged_df = df_tl[avail]
        if not flagged_df.empty:
            st.dataframe(flagged_df.sort_values("ensemble_score", ascending=False),
                         use_container_width=True, height=250)
"""

files["dashboard/pages/explainability.py"] = """\
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

API_BASE = "http://127.0.0.1:8000"
TIMEOUT = 15

RISK_COLORS = {
    "CRITICAL": "#D85A30",
    "HIGH": "#E8953A",
    "MEDIUM": "#F5C842",
    "LOW": "#1D9E75",
}

def show():
    st.title("Explainability Center")
    tab1, tab2 = st.tabs(["SHAP Feature Importance", "Alert Reasoning"])

    with tab1:
        st.subheader("Global SHAP Feature Importance")
        try:
            r = requests.get(f"{API_BASE}/stats/shap-importance", timeout=TIMEOUT)
            shap_data = r.json()
            features = shap_data.get("features", [])
        except:
            features = []

        if not features:
            st.warning("SHAP data not available.")
        else:
            df_shap = pd.DataFrame(features).head(15)
            fig = go.Figure(go.Bar(
                x=df_shap["importance"],
                y=df_shap["feature"],
                orientation="h",
                marker_color="#378ADD",
                text=[f"{v:.4f}" for v in df_shap["importance"]],
                textposition="outside",
            ))
            fig.update_layout(
                height=480,
                xaxis_title="Mean |SHAP value|",
                yaxis=dict(autorange="reversed"),
                margin=dict(t=20, b=20, l=220),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("Key Insight: Temporal features dominate. Insiders deviate in WHEN they work, not just WHAT they do.")

    with tab2:
        st.subheader("Alert-Level Explanation")
        try:
            r = requests.get(f"{API_BASE}/alerts/", params={"limit": 50}, timeout=TIMEOUT)
            alerts = r.json()
        except:
            alerts = []

        if not alerts:
            st.warning("No alerts loaded.")
            return

        alert_options = {}
        for a in alerts:
            label = f"{a['user']} | {a['date']} | {a['ensemble_score']:.3f} | {a['risk_level']}"
            alert_options[label] = a["alert_id"]

        selected_label = st.selectbox("Select Alert", list(alert_options.keys()))
        alert_id = alert_options[selected_label]

        try:
            r = requests.get(f"{API_BASE}/alerts/{alert_id}", timeout=TIMEOUT)
            detail = r.json()
        except:
            detail = {}

        if not detail or "alert_id" not in detail:
            st.warning("Could not load alert detail.")
            return

        risk = detail.get("risk_level", "LOW")
        color = RISK_COLORS.get(risk, "#888")
        st.markdown(
            f"<h3 style='color:{color}'>{risk} ALERT - {detail.get('user','').upper()}</h3>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ensemble Score", f"{detail.get('ensemble_score', 0):.4f}")
        c2.metric("AE Score", f"{detail.get('ae_score', 0):.4f}")
        c3.metric("IF Score", f"{detail.get('if_score', 0):.4f}")
        c4.metric("Both Flagged", "YES" if detail.get("both_flagged") else "NO")

        st.markdown("**Detected Anomaly Reasons:**")
        reasons = detail.get("reasons", [])
        if reasons:
            for r_item in reasons:
                sev = r_item.get("severity", "LOW")
                icons = {"HIGH": "[HIGH]", "MEDIUM": "[MED]", "LOW": "[LOW]"}
                tag = icons.get(sev, sev)
                st.markdown(f"**{tag}** {r_item.get('reason', '')}")
        else:
            st.info("No specific rule triggers for this session.")

        st.markdown("**Session Statistics:**")
        stats = detail.get("stats", {})
        if stats:
            rows = [[k, v] for k, v in stats.items()]
            df_stats = pd.DataFrame(rows, columns=["Metric", "Value"])
            st.dataframe(df_stats, use_container_width=True, height=260)
"""

files["dashboard/pages/model_report.py"] = """\
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

API_BASE = "http://127.0.0.1:8000"
TIMEOUT = 15

def show():
    st.title("Model Performance Report")

    try:
        r = requests.get(f"{API_BASE}/stats/overview", timeout=TIMEOUT)
        overview = r.json()
    except:
        overview = {}

    try:
        r = requests.get(f"{API_BASE}/stats/model-comparison", timeout=TIMEOUT)
        mc_data = r.json()
        models = mc_data.get("models", [])
    except:
        models = []

    st.subheader("Detection Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sessions", f"{overview.get('total_sessions', 0):,}")
    c2.metric("AE Anomalies", f"{overview.get('ae_anomalies', 0):,}")
    c3.metric("IF Anomalies", f"{overview.get('if_anomalies', 0):,}")
    c4.metric("Ensemble (Both)", f"{overview.get('ensemble_intersect', 0):,}")

    if not models:
        st.warning("Model comparison data not available.")
        return

    df_models = pd.DataFrame(models)

    st.subheader("Model Comparison Table")
    display_cols = ["model","precision","recall","f1","roc_auc","avg_precision","fpr","tp","fp","fn"]
    avail = [c for c in display_cols if c in df_models.columns]
    num_cols = [c for c in ["precision","recall","f1","roc_auc","avg_precision","fpr"] if c in avail]
    fmt = {c: "{:.4f}" for c in num_cols}

    highlight_max_cols = [c for c in ["precision","recall","f1","roc_auc"] if c in avail]
    highlight_min_cols = [c for c in ["fpr"] if c in avail]

    styled = df_models[avail].style.format(fmt)
    if highlight_max_cols:
        styled = styled.highlight_max(subset=highlight_max_cols, color="#D4EFDF")
    if highlight_min_cols:
        styled = styled.highlight_min(subset=highlight_min_cols, color="#D4EFDF")

    st.dataframe(styled, use_container_width=True)

    if "roc_auc" in df_models.columns and "model" in df_models.columns:
        st.subheader("ROC-AUC Comparison")
        max_auc = df_models["roc_auc"].max()
        bar_colors = ["#D85A30" if v == max_auc else "#378ADD" for v in df_models["roc_auc"]]
        fig = go.Figure(go.Bar(
            x=df_models["model"],
            y=df_models["roc_auc"],
            marker_color=bar_colors,
            text=[f"{v:.4f}" for v in df_models["roc_auc"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=320,
            yaxis_range=[0, 1.0],
            yaxis_title="ROC-AUC Score",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Research Findings")
    findings = [
        "Autoencoder achieves highest ROC-AUC (0.8311) — best overall discriminative ability",
        "Ensemble Intersection achieves lowest FPR (0.97%) — highest precision alerts",
        "Temporal features dominate SHAP importance — insiders deviate in WHEN they work",
        "3,263 high-confidence sessions flagged independently by both models",
        "Dataset: 330,452 sessions | 1,000 users | 17 months | CERT Insider Threat v6.2",
    ]
    for finding in findings:
        st.markdown(f"- {finding}")
"""

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

from dashboard.pages import overview, alerts, users, explainability, model_report

PAGES = {
    "Overview"            : overview,
    "Alert Center"        : alerts,
    "User Investigation"  : users,
    "Explainability"      : explainability,
    "Model Report"        : model_report,
}

with st.sidebar:
    st.markdown("## Insider Threat Detection")
    st.markdown("---")
    selection = st.radio(
        "Navigation",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**API Status**")
    try:
        import requests
        r = requests.get("http://127.0.0.1:8000/health", timeout=3)
        if r.status_code == 200:
            st.success("API Online")
        else:
            st.error("API Error")
    except:
        st.error("API Offline")
        st.caption("Start: uvicorn src.api.main:app --reload --port 8000")
    st.markdown("---")
    st.caption("Insider Threat Detection System v1.0")

PAGES[selection].show()
"""

for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\nAll files created. Now run:")
print("  Terminal 1: uvicorn src.api.main:app --reload --port 8000")
print("  Terminal 2: streamlit run dashboard/app.py")