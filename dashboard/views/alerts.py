import sys, os
sys.path.insert(0, os.path.abspath("."))

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from config import API_BASE

T = 15

RC = {"CRITICAL":"#D85A30","HIGH":"#E8953A","MEDIUM":"#F5C842","LOW":"#1D9E75"}
RB = {"CRITICAL":"#FAECE7","HIGH":"#FEF3E2","MEDIUM":"#FEFBE6","LOW":"#E8F8F2"}



def fetch_alerts(params=None):
    try:
        return requests.get(f"{API_BASE}/alerts/", params=params, timeout=5).json()
    except:
        return []

def fetch_counts():
    try:
        return requests.get(f"{API_BASE}/alerts/count", timeout=5).json()
    except:
        return {}


def show():
    st.title("Alert Center")

    counts = fetch_counts()

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


    alerts = fetch_alerts(params)

    if not alerts:
        st.warning("No alerts match filters.")
        return

    df = pd.DataFrame(alerts)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Showing", len(df))
    c2.metric("CRITICAL", counts.get("critical",0))
    c3.metric("HIGH", counts.get("high",0))
    c4.metric("Both Flagged",
              int(df["both_flagged"].sum()) if "both_flagged" in df.columns else 0)

    st.markdown("---")
    st.subheader(f"Alerts ({len(df)} results)")

    cols = ["user","date","risk_level","ensemble_score","ae_score","if_score",
            "both_flagged","device_count","email_external","http_suspicious","reason_summary"]

    avail = [c for c in cols if c in df.columns]
    dfd   = df[avail].copy()

    for c in ["ensemble_score","ae_score","if_score"]:
        if c in dfd.columns:
            dfd[c] = dfd[c].round(4)

    def cr(val):
        return (f"background-color:{RB.get(str(val),'#fff')};"
                f"color:{RC.get(str(val),'#888')};font-weight:bold")

    st.dataframe(dfd.style.map(cr, subset=["risk_level"]),
                 use_container_width=True, height=420)

    if "ensemble_score" in df.columns:
        st.subheader("Score Distribution")

        fig = px.histogram(
            df,
            x="ensemble_score",
            color="risk_level",
            nbins=40,
            color_discrete_map=RC
        )

        fig.update_layout(height=280, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)