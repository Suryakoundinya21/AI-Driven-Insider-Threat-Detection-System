import sys
import os
sys.path.insert(0, os.path.abspath("."))
import streamlit as st

st.set_page_config(
    page_title = "Insider Threat Detection System",
    page_icon  = "S",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

try:
    API_BASE = st.secrets.get("API_BASE", "http://127.0.0.1:8000")
except Exception:
    API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")

os.environ["ITDS_API_BASE"] = API_BASE

import importlib.util

def load_page(path):
    spec   = importlib.util.spec_from_file_location("page", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

BASE = os.path.dirname(os.path.abspath(__file__))

PAGES = {
    "Overview"           : os.path.join(BASE, "views", "overview.py"),
    "Alert Center"       : os.path.join(BASE, "views", "alerts.py"),
    "User Investigation" : os.path.join(BASE, "views", "users.py"),
    "Explainability"     : os.path.join(BASE, "views", "explainability.py"),
    "Model Report"       : os.path.join(BASE, "views", "model_report.py"),
    "Analyst Feedback"   : os.path.join(BASE, "views", "feedback.py"),
}

with st.sidebar:
    st.markdown("## Insider Threat Detection")
    st.markdown("---")
    selection = st.radio(
        "Navigate to",
        list(PAGES.keys()),
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**API Status**")
    try:
        import requests
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            st.success("API Online")
        else:
            st.error("API Error")
    except Exception:
        st.error("API Offline")
        st.caption(f"Target: {API_BASE}")
    st.markdown("---")
    st.caption("Insider Threat Detection v2.0")

page_path = PAGES[selection]
if os.path.exists(page_path):
    mod = load_page(page_path)
    mod.show()
else:
    st.error(f"Page not found: {page_path}")
