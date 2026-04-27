import os
import shutil

print("Step 1: Removing old pages folder that causes Streamlit auto-navigation...")
pages_dir = "dashboard/pages"
if os.path.exists(pages_dir):
    shutil.rmtree(pages_dir)
    print(f"  Deleted: {pages_dir}")
else:
    print(f"  Already removed: {pages_dir}")

print("\nStep 2: Creating clean dashboard structure...")

files = {}

files["dashboard/__init__.py"] = ""
files["dashboard/views/__init__.py"] = ""

files["dashboard/app.py"] = """\
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import streamlit as st

st.set_page_config(
    page_title="Insider Threat Detection System",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dynamically import page modules from views/
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
        r = requests.get("http://127.0.0.1:8000/health", timeout=3)
        st.success("API Online") if r.status_code == 200 else st.error("API Error")
    except:
        st.error("API Offline")
        st.caption("Run: uvicorn src.api.main:app --reload --port 8000")
    st.markdown("---")
    st.caption("Insider Threat Detection v1.0")

page_path = PAGES[selection]
if os.path.exists(page_path):
    mod = load_page(page_path)
    mod.show()
else:
    st.error(f"Page file not found: {page_path}")
"""

files["dashboard/views/overview.py"] = """\
import sys, os
sys.path.insert(0, os.path.abspath("."))
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import requests

API   = "http://127.0.0.1:8000"
T     = 15
RC    = {"CRITICAL":"#D85A30","HIGH":"#E8953A","MEDIUM":"#F5C842","LOW":"#1D9E75"}

@st.cache_data(ttl=60)
def _overview():
    try: return requests.get(f"{API}/stats/overview", timeout=T).json()
    except: return {}

@st.cache_data(ttl=60)
def _counts():
    try: return requests.get(f"{API}/alerts/count", timeout=T).json()
    except: return {}

@st.cache_data(ttl=60)
def _top_users(n=10):
    try: return requests.get(f"{API}/users/top-risk", params={"limit":n}, timeout=T).json()
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
"""

files["dashboard/views/alerts.py"] = """\
import sys, os
sys.path.insert(0, os.path.abspath("."))
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

API = "http://127.0.0.1:8000"
T   = 15
RC  = {"CRITICAL":"#D85A30","HIGH":"#E8953A","MEDIUM":"#F5C842","LOW":"#1D9E75"}
RB  = {"CRITICAL":"#FAECE7","HIGH":"#FEF3E2","MEDIUM":"#FEFBE6","LOW":"#E8F8F2"}

def show():
    st.title("Alert Center")

    try: counts = requests.get(f"{API}/alerts/count", timeout=T).json()
    except: counts = {}

    col1, col2, col3 = st.columns(3)
    with col1:
        risk = st.selectbox("Risk Level", ["All","CRITICAL","HIGH","MEDIUM","LOW"])
    with col2:
        min_score = st.slider("Min Ensemble Score", 0.0, 1.0, 0.0, 0.05)
    with col3:
        user_input = st.text_input("Filter by User ID", "")

    params = {"limit": 300, "min_score": min_score}
    if risk != "All":
        params["risk_level"] = risk
    if user_input.strip():
        params["user"] = user_input.strip().lower()

    try: alerts = requests.get(f"{API}/alerts/", params=params, timeout=T).json()
    except: alerts = []

    if not alerts:
        st.warning("No alerts match filters.")
        return

    df = pd.DataFrame(alerts)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Showing",     len(df))
    c2.metric("CRITICAL",    counts.get("critical",0))
    c3.metric("HIGH",        counts.get("high",0))
    c4.metric("Both Flagged",
              int(df["both_flagged"].sum()) if "both_flagged" in df.columns else 0)

    st.markdown("---")
    st.subheader(f"Alerts ({len(df)} results)")

    cols = ["user","date","risk_level","ensemble_score","ae_score","if_score",
            "both_flagged","device_count","email_external","http_suspicious","reason_summary"]
    avail = [c for c in cols if c in df.columns]
    dfd   = df[avail].copy()
    for c in ["ensemble_score","ae_score","if_score"]:
        if c in dfd.columns: dfd[c] = dfd[c].round(4)

    def cr(val):
        return (f"background-color:{RB.get(str(val),'#fff')};"
                f"color:{RC.get(str(val),'#888')};font-weight:bold")

    st.dataframe(dfd.style.map(cr, subset=["risk_level"]),
                 use_container_width=True, height=420)

    if "ensemble_score" in df.columns:
        st.subheader("Score Distribution")
        fig = px.histogram(df, x="ensemble_score", color="risk_level",
                           nbins=40, color_discrete_map=RC)
        fig.update_layout(height=280, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
"""

files["dashboard/views/users.py"] = """\
import sys, os
sys.path.insert(0, os.path.abspath("."))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

API = "http://127.0.0.1:8000"
T   = 15
RC  = {"CRITICAL":"#D85A30","HIGH":"#E8953A","MEDIUM":"#F5C842","LOW":"#1D9E75"}

def show():
    st.title("User Investigation")

    try: top = requests.get(f"{API}/users/top-risk", params={"limit":20}, timeout=T).json()
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
            try: s = requests.get(f"{API}/users/{sel}/summary", timeout=T).json()
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
            td = requests.get(f"{API}/users/{sel}/timeline",
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
"""

files["dashboard/views/explainability.py"] = """\
import sys, os
sys.path.insert(0, os.path.abspath("."))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

API = "http://127.0.0.1:8000"
T   = 15
RC  = {"CRITICAL":"#D85A30","HIGH":"#E8953A","MEDIUM":"#F5C842","LOW":"#1D9E75"}

def show():
    st.title("Explainability Center")
    tab1, tab2 = st.tabs(["SHAP Feature Importance","Alert Reasoning"])

    with tab1:
        st.subheader("Global SHAP Feature Importance")
        try:
            data = requests.get(f"{API}/stats/shap-importance", timeout=T).json()
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
        try: alerts = requests.get(f"{API}/alerts/", params={"limit":50}, timeout=T).json()
        except: alerts = []

        if not alerts:
            st.warning("No alerts loaded.")
            return

        opts = {f"{a['user']} | {a['date']} | {a['ensemble_score']:.3f} | {a['risk_level']}":
                a["alert_id"] for a in alerts}
        label = st.selectbox("Select Alert", list(opts.keys()))
        aid   = opts[label]

        try: det = requests.get(f"{API}/alerts/{aid}", timeout=T).json()
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
"""

files["dashboard/views/model_report.py"] = """\
import sys, os
sys.path.insert(0, os.path.abspath("."))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

API = "http://127.0.0.1:8000"
T   = 15

def show():
    st.title("Model Performance Report")

    try: ov = requests.get(f"{API}/stats/overview", timeout=T).json()
    except: ov = {}

    try:
        mc     = requests.get(f"{API}/stats/model-comparison", timeout=T).json()
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
"""

for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\nDone. Now run:")
print("  streamlit run dashboard/app.py")