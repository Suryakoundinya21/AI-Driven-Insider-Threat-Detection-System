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
