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
