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
    st.title("User Investigation")

    try: top = requests.get(f"{API_BASE}/users/top-risk", params={"limit":20}, timeout=T).json()
    except:
        st.error("Cannot reach API.")
        return

    col1, col2 = st.columns([1,2])

    with col1:
        st.subheader("Top Risk Users")
        df_u = pd.DataFrame(top)
        sc   = [c for c in ["user","risk_level","max_score","flagged_sessions","total_usb"]
                if c in df_u.columns]
        st.dataframe(df_u[sc], use_container_width=True, height=420)
        sel = st.selectbox("Investigate User", [u["user"] for u in top])

    with col2:
        if sel:
            try: s = requests.get(f"{API_BASE}/users/{sel}/summary", timeout=T).json()
            except: s = {}
            if s and "error" not in s:
                risk  = s.get("risk_level","LOW")
                color = RC.get(risk,"#888")
                st.subheader(f"User: {sel}")
                st.markdown(f"<h3 style='color:{color}'>Risk Level: {risk}</h3>",
                            unsafe_allow_html=True)
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Max Score",  f"{s.get('max_ensemble_score',0):.4f}")
                m2.metric("Sessions",   s.get("total_sessions",0))
                m3.metric("Flagged",    s.get("flagged_sessions",0))
                m4.metric("Flag Rate",  f"{s.get('flag_rate',0)*100:.1f}%")
                m5,m6,m7 = st.columns(3)
                m5.metric("USB Events", s.get("total_usb_events",0))
                m6.metric("Ext Emails", s.get("total_ext_emails",0))
                m7.metric("Susp URLs",  s.get("total_susp_http",0))

    if sel:
        st.markdown("---")
        st.subheader(f"Activity Timeline - {sel}")
        try:
            td = requests.get(f"{API_BASE}/users/{sel}/timeline",
                              params={"days":500}, timeout=T).json()
        except:
            st.error("Could not load timeline.")
            return

        tl = td.get("timeline",[])
        if not tl:
            st.info("No timeline data.")
            return

        df_t = pd.DataFrame(tl)
        df_t["date"] = pd.to_datetime(df_t["date"])
        fl   = df_t["flagged"].tolist() if "flagged" in df_t.columns else [False]*len(df_t)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_t["date"], y=df_t["ensemble_score"],
            mode="lines+markers", name="Ensemble Score",
            line=dict(color="#378ADD", width=1.5),
            marker=dict(size=[8 if f else 4 for f in fl],
                        color=["#D85A30" if f else "#378ADD" for f in fl]),
        ))
        fig.add_hline(y=0.8, line_dash="dash", line_color="red",   annotation_text="CRITICAL")
        fig.add_hline(y=0.6, line_dash="dot",  line_color="orange", annotation_text="HIGH")
        fig.update_layout(height=300, yaxis_range=[0,1.05], margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Flagged Sessions Detail")
        ac   = ["date","device_count","email_external","http_suspicious",
                "after_hours_ratio","ensemble_score"]
        avl  = [c for c in ac if c in df_t.columns]
        fdf  = df_t[df_t["flagged"]==True][avl] if "flagged" in df_t.columns else df_t[avl]
        if not fdf.empty:
            st.dataframe(fdf.sort_values("ensemble_score", ascending=False),
                         use_container_width=True, height=250)
