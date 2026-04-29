import sys, os
sys.path.insert(0, os.path.abspath("."))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from config import API_BASE

T   = 15
RC  = {"CRITICAL":"#D85A30","HIGH":"#E8953A","MEDIUM":"#F5C842","LOW":"#1D9E75"}

def show():
    st.title("Explainability Center")
    tab1, tab2 = st.tabs(["SHAP Feature Importance","Alert Reasoning"])

    with tab1:
        st.subheader("Global SHAP Feature Importance")
        try:
            data = requests.get(f"{API_BASE}/stats/shap-importance", timeout=T).json()
            feats = data.get("features",[])
        except: feats = []

        if not feats:
            st.warning("SHAP data not available.")
        else:
            df_s = pd.DataFrame(feats).head(15)
            fig  = go.Figure(go.Bar(
                x=df_s["importance"], y=df_s["feature"],
                orientation="h", marker_color="#378ADD",
                text=[f"{v:.4f}" for v in df_s["importance"]],
                textposition="outside",
            ))
            fig.update_layout(height=480, xaxis_title="Mean |SHAP value|",
                              yaxis=dict(autorange="reversed"),
                              margin=dict(t=20,b=20,l=220))
            st.plotly_chart(fig, use_container_width=True)
            st.info("Temporal features dominate: insiders deviate in WHEN they work.")

    with tab2:
        st.subheader("Alert-Level Explanation")
        try: alerts = requests.get(f"{API_BASE}/alerts/", params={"limit":50}, timeout=T).json()
        except: alerts = []

        if not alerts:
            st.warning("No alerts loaded.")
            return

        opts = {f"{a['user']} | {a['date']} | {a['ensemble_score']:.3f} | {a['risk_level']}":
                a["alert_id"] for a in alerts}
        label = st.selectbox("Select Alert", list(opts.keys()))
        aid   = opts[label]

        try: det = requests.get(f"{API_BASE}/alerts/{aid}", timeout=T).json()
        except: det = {}

        if not det or "alert_id" not in det:
            st.warning("Could not load alert detail.")
            return

        risk  = det.get("risk_level","LOW")
        color = RC.get(risk,"#888")
        st.markdown(f"<h3 style='color:{color}'>{risk} ALERT - "
                    f"{det.get('user','').upper()}</h3>", unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Ensemble", f"{det.get('ensemble_score',0):.4f}")
        c2.metric("AE Score", f"{det.get('ae_score',0):.4f}")
        c3.metric("IF Score", f"{det.get('if_score',0):.4f}")
        c4.metric("Both Flagged","YES" if det.get("both_flagged") else "NO")

        st.markdown("**Detected Anomaly Reasons:**")
        reasons = det.get("reasons",[])
        tags    = {"HIGH":"[HIGH]","MEDIUM":"[MED]","LOW":"[LOW]"}
        if reasons:
            for r in reasons:
                st.markdown(f"**{tags.get(r.get('severity','LOW'),'')}** {r.get('reason','')}")
        else:
            st.info("No rule triggers for this session.")

        st.markdown("**Session Statistics:**")
        stats = det.get("stats",{})
        if stats:
            st.dataframe(pd.DataFrame(list(stats.items()), columns=["Metric","Value"]),
                         use_container_width=True, height=260)
