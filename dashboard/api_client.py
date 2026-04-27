import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
TIMEOUT = 15

@st.cache_data(ttl=60)
def get_overview():
    try:
        r = requests.get(f"{API_BASE}/stats/overview", timeout=TIMEOUT)
        return r.json()
    except Exception as e:
        return {}

@st.cache_data(ttl=60)
def get_alert_counts():
    try:
        r = requests.get(f"{API_BASE}/alerts/count", timeout=TIMEOUT)
        return r.json()
    except:
        return {}

@st.cache_data(ttl=60)
def get_alerts(risk_level=None, user=None, min_score=0.0, limit=200):
    params = {"limit": limit, "min_score": min_score}
    if risk_level:
        params["risk_level"] = risk_level
    if user:
        params["user"] = user
    try:
        r = requests.get(f"{API_BASE}/alerts/", params=params, timeout=TIMEOUT)
        return r.json()
    except:
        return []

@st.cache_data(ttl=60)
def get_top_users(limit=20):
    try:
        r = requests.get(f"{API_BASE}/users/top-risk", params={"limit": limit}, timeout=TIMEOUT)
        return r.json()
    except:
        return []

@st.cache_data(ttl=60)
def get_user_timeline(user_id, days=365):
    try:
        r = requests.get(f"{API_BASE}/users/{user_id}/timeline", params={"days": days}, timeout=TIMEOUT)
        return r.json()
    except:
        return {}

@st.cache_data(ttl=60)
def get_user_summary(user_id):
    try:
        r = requests.get(f"{API_BASE}/users/{user_id}/summary", timeout=TIMEOUT)
        return r.json()
    except:
        return {}

@st.cache_data(ttl=300)
def get_shap_importance():
    try:
        r = requests.get(f"{API_BASE}/stats/shap-importance", timeout=TIMEOUT)
        return r.json()
    except:
        return {}

@st.cache_data(ttl=300)
def get_model_comparison():
    try:
        r = requests.get(f"{API_BASE}/stats/model-comparison", timeout=TIMEOUT)
        return r.json()
    except:
        return {}

def get_alert_detail(alert_id):
    try:
        r = requests.get(f"{API_BASE}/alerts/{alert_id}", timeout=TIMEOUT)
        return r.json()
    except:
        return {}
