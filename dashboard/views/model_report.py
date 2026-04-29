import sys, os
sys.path.insert(0, os.path.abspath("."))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from config import API_BASE

T   = 15

def show():
    st.title("Model Performance Report")

    try: ov = requests.get(f"{API_BASE}/stats/overview", timeout=T).json()
    except: ov = {}

    try:
        mc     = requests.get(f"{API_BASE}/stats/model-comparison", timeout=T).json()
        models = mc.get("models",[])
    except: models = []

    st.subheader("Detection Summary")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Sessions",  f"{ov.get('total_sessions',0):,}")
    c2.metric("AE Anomalies",    f"{ov.get('ae_anomalies',0):,}")
    c3.metric("IF Anomalies",    f"{ov.get('if_anomalies',0):,}")
    c4.metric("Ensemble (Both)", f"{ov.get('ensemble_intersect',0):,}")

    if not models:
        st.warning("Model comparison data not available.")
        return

    df_m = pd.DataFrame(models)
    st.subheader("Model Comparison Table")
    dc   = ["model","precision","recall","f1","roc_auc","avg_precision","fpr","tp","fp","fn"]
    avl  = [c for c in dc if c in df_m.columns]
    nc   = [c for c in ["precision","recall","f1","roc_auc","avg_precision","fpr"] if c in avl]
    stl  = df_m[avl].style.format({c:"{:.4f}" for c in nc})
    hmax = [c for c in ["precision","recall","f1","roc_auc"] if c in avl]
    hmin = [c for c in ["fpr"] if c in avl]
    if hmax: stl = stl.highlight_max(subset=hmax, color="#D4EFDF")
    if hmin: stl = stl.highlight_min(subset=hmin, color="#D4EFDF")
    st.dataframe(stl, use_container_width=True)

    if "roc_auc" in df_m.columns:
        st.subheader("ROC-AUC Comparison")
        mx  = df_m["roc_auc"].max()
        clr = ["#D85A30" if v==mx else "#378ADD" for v in df_m["roc_auc"]]
        fig = go.Figure(go.Bar(
            x=df_m["model"], y=df_m["roc_auc"],
            marker_color=clr,
            text=[f"{v:.4f}" for v in df_m["roc_auc"]],
            textposition="outside",
        ))
        fig.update_layout(height=320, yaxis_range=[0,1.0],
                          yaxis_title="ROC-AUC Score", margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Research Findings")
    for f in [
        "Autoencoder achieves highest ROC-AUC (0.8311) — best overall discriminative ability",
        "Ensemble Intersection achieves lowest FPR (0.97%) — highest precision alerts",
        "Temporal features dominate SHAP importance — insiders deviate in WHEN they work",
        "3,263 high-confidence sessions flagged independently by both models",
        "Dataset: 330,452 sessions | 1,000 users | 17 months | CERT Insider Threat v4.2",
    ]:
        st.markdown(f"- {f}")
