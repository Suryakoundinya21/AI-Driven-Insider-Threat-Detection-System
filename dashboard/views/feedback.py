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
        except Exception as e:
            st.error(f"Cannot reach API: {e}")
            st.info(f"API target: {API}")
            return

        if not alerts:
            st.warning("No alerts loaded. Make sure API is running and data is loaded.")
            st.caption(f"API: {API}/alerts/")
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
                    resp = requests.post(f"{API}/feedback/false-positive", json={
                        "alert_id": sel["alert_id"],
                        "user"    : sel["user"],
                        "analyst" : analyst,
                        "reason"  : fp_reason,
                    }, timeout=T)
                    result = resp.json()
                    st.success(f"Recorded. {result.get('effect','')}")
                    boost = result.get("threshold_boost", 0)
                    st.info(f"Threshold for {sel['user']} raised by {boost:.0%}")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col_cf:
            st.markdown("### Confirm as Real Threat")
            severity = st.selectbox("Severity", ["CRITICAL","HIGH","MEDIUM","LOW"])

            if st.button("Confirm Threat", type="primary"):
                try:
                    resp = requests.post(f"{API}/feedback/confirm", json={
                        "alert_id": sel["alert_id"],
                        "user"    : sel["user"],
                        "analyst" : analyst,
                        "severity": severity,
                    }, timeout=T)
                    result = resp.json()
                    st.success(f"Confirmed {severity} threat for {sel['user']}")
                    st.warning("Detection sensitivity increased for this user.")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        st.subheader("Feedback Statistics")
        try:
            stats = requests.get(f"{API}/feedback/stats", timeout=T).json()
            c1, c2, c3 = st.columns(3)
            c1.metric("False Positives", stats.get("total_false_positives", 0))
            c2.metric("Confirmed",       stats.get("total_confirmed", 0))
            c3.metric("Users Adjusted",  stats.get("users_adjusted", 0))

            adj = stats.get("user_adjustments", {})
            if adj:
                st.subheader("User Threshold Adjustments")
                rows = [{
                    "User"            : u,
                    "FP Count"        : d.get("fp_count", 0),
                    "Threshold Boost" : f"+{d.get('threshold_boost',0):.0%}",
                    "Last Updated"    : str(d.get("last_updated",""))[:19],
                } for u, d in adj.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No adjustments yet. Submit feedback above to see changes here.")
        except Exception as e:
            st.error(f"Cannot load stats: {e}")

    with tab3:
        st.subheader("User Adjustment Details")
        user_id = st.text_input("Enter User ID", "gko0078")
        if st.button("Check User"):
            try:
                r    = requests.get(f"{API}/feedback/user/{user_id}", timeout=T)
                data = r.json()
                st.markdown(f"**User:** `{data['user']}`")
                c1, c2, c3 = st.columns(3)
                c1.metric("False Positives", data.get("false_positives", 0))
                c2.metric("Confirmed",       data.get("confirmed", 0))
                c3.metric("Threshold Boost", f"+{data.get('threshold_boost',0):.0%}")
                if data.get("adjusted"):
                    st.warning(f"Threshold raised for this user.")
                else:
                    st.success("No adjustments. Using default threshold.")
            except Exception as e:
                st.error(f"Error: {e}")
