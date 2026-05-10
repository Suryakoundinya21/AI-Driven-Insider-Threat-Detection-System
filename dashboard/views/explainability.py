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
