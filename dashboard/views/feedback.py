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
            if not isinstance(alerts, list):
                alerts = []
        except Exception as e:
            st.error(f"Cannot reach API: {e}")
            return

        if not alerts:
            st.warning("No alerts loaded.")
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
                    resp   = requests.post(f"{API}/feedback/false-positive", json={
                        "alert_id": sel["alert_id"],
                        "user"    : sel["user"],
                        "analyst" : analyst,
                        "reason"  : fp_reason,
                    }, timeout=T)
                    result = resp.json()
                    boost  = result.get("threshold_boost", 0)
                    st.success(f"Recorded. Threshold raised by {boost:.0%} for {sel['user']}")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col_cf:
            st.markdown("### Confirm as Real Threat")
            severity = st.selectbox("Severity", ["CRITICAL","HIGH","MEDIUM","LOW"])

            if st.button("Confirm Threat", type="primary"):
                try:
                    requests.post(f"{API}/feedback/confirm", json={
                        "alert_id": sel["alert_id"],
                        "user"    : sel["user"],
                        "analyst" : analyst,
                        "severity": severity,
                    }, timeout=T)
                    st.success(f"Confirmed {severity} threat for {sel['user']}")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        st.subheader("Feedback Statistics")
        try:
            stats = requests.get(f"{API}/feedback/stats", timeout=T).json()
            c1,c2,c3 = st.columns(3)
            c1.metric("False Positives", stats.get("total_false_positives",0))
            c2.metric("Confirmed",       stats.get("total_confirmed",0))
            c3.metric("Users Adjusted",  stats.get("users_adjusted",0))

            adj = stats.get("user_adjustments",{})
            if adj:
                st.subheader("User Threshold Adjustments")
                rows = [{
                    "User"            : u,
                    "FP Count"        : d.get("fp_count",0),
                    "Threshold Boost" : f"+{d.get('threshold_boost',0):.0%}",
                    "Last Updated"    : str(d.get("last_updated",""))[:19],
                } for u, d in adj.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No adjustments yet.")
        except Exception as e:
            st.error(f"Cannot load stats: {e}")

    with tab3:
        st.subheader("User Adjustment Details")

        # Load top users for dropdown
        try:
            r         = requests.get(f"{API}/users/top-risk",
                                     params={"limit":50}, timeout=T)
            top_users = r.json()
            user_list = [u["user"] for u in top_users] if top_users else []
        except:
            user_list = []

        # Also load adjusted users from feedback
        try:
            stats    = requests.get(f"{API}/feedback/stats", timeout=T).json()
            adj_list = list(stats.get("user_adjustments", {}).keys())
        except:
            adj_list = []

        # Merge both lists, deduplicate
        all_users = list(dict.fromkeys(adj_list + user_list))

        if all_users:
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_user = st.selectbox(
                    "Select User to Check",
                    all_users,
                    help="Users with existing adjustments shown first"
                )
            with col2:
                manual_user = st.text_input(
                    "Or type User ID manually",
                    placeholder="e.g. dlm0051"
                )

            user_id = manual_user.strip().lower() if manual_user.strip() else selected_user
        else:
            user_id = st.text_input("Enter User ID", "gko0078")

        if st.button("Check User Adjustment"):
            try:
                r    = requests.get(f"{API}/feedback/user/{user_id}", timeout=T)
                data = r.json()

                st.markdown(f"### User: `{data['user']}`")
                c1, c2, c3 = st.columns(3)
                c1.metric("False Positives", data.get("false_positives",0))
                c2.metric("Confirmed",       data.get("confirmed",0))
                c3.metric("Threshold Boost", f"+{data.get('threshold_boost',0):.0%}")

                if data.get("adjusted"):
                    boost = data.get("threshold_boost", 0)
                    st.warning(
                        f"**Threshold adjusted for {user_id}.** "
                        f"Alerts only shown if score > "
                        f"{0.5 + boost:.2f} (raised by {boost:.0%})"
                    )
                else:
                    st.success("No adjustments. Using default threshold (0.50).")

                # Show history
                history = data.get("history", [])
                if history:
                    st.markdown("**Recent Feedback History:**")
                    hist_rows = [{
                        "Type"      : h.get("type","").replace("_"," ").title(),
                        "Alert"     : h.get("alert_id",""),
                        "Analyst"   : h.get("analyst",""),
                        "Timestamp" : str(h.get("timestamp",""))[:19],
                    } for h in history[-10:]]
                    st.dataframe(pd.DataFrame(hist_rows),
                                 use_container_width=True)

            except Exception as e:
                st.error(f"Error loading user {user_id}: {e}")
