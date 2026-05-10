import os
import sys, os
sys.path.insert(0, os.path.abspath("."))
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import requests

from config import API_BASE


T     = 15
RC    = {"CRITICAL":"#D85A30","HIGH":"#E8953A","MEDIUM":"#F5C842","LOW":"#1D9E75"}

@st.cache_data(ttl=60)
def _overview():
    try: return requests.get(f"{API_BASE}/stats/overview", timeout=T).json()
    except: return {}

@st.cache_data(ttl=60)
def _counts():
    try: return requests.get(f"{API_BASE}/alerts/count", timeout=T).json()
    except: return {}

@st.cache_data(ttl=60)
def _top_users(n=10):
    try: return requests.get(f"{API_BASE}/users/top-risk", params={"limit":n}, timeout=T).json()
    except: return []

def show():
    st.title("System Overview")
    st.markdown("Real-time insider threat detection dashboard")

    ov = _overview()
    ct = _counts()

    if not ov:
        st.error("Cannot connect to API.")
        st.code("uvicorn src.api.main:app --reload --port 8000")
        return

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Sessions",   f"{ov.get('total_sessions',0):,}")
    c2.metric("Total Users",      f"{ov.get('total_users',0):,}")
    c3.metric("AE Anomalies",     f"{ov.get('ae_anomalies',0):,}")
    c4.metric("IF Anomalies",     f"{ov.get('if_anomalies',0):,}")
    c5.metric("High-Conf Alerts", f"{ov.get('ensemble_intersect',0):,}")
    st.markdown("---")

    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Alert Risk Breakdown")
        labels = ["CRITICAL","HIGH","MEDIUM","LOW"]
        values = [ct.get("critical",0), ct.get("high",0),
                  ct.get("medium",0),   ct.get("low",0)]
        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.5,
            marker_colors=[RC[l] for l in labels],
            textinfo="label+percent",
        ))
        fig.update_layout(height=320, margin=dict(t=20,b=20,l=20,r=20),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Highest-Risk Users")
        tu = _top_users(10)
        if tu:
            df = pd.DataFrame(tu)
            fig = go.Figure(go.Bar(
                x=df["user"], y=df["max_score"],
                marker_color=[RC.get(r,"#888") for r in df["risk_level"]],
                text=[f"{s:.3f}" for s in df["max_score"]],
                textposition="outside",
            ))
            fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                          annotation_text="High-risk threshold")
            fig.update_layout(height=320, yaxis_range=[0,1.1],
                              margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)

    dr = ov.get("date_range",{})
    st.info(f"Dataset: {dr.get('from','N/A')} to {dr.get('to','N/A')}")
