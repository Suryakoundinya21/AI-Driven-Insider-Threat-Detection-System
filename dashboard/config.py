import os
API_BASE = os.getenv(
    "API_BASE",
    "http://127.0.0.1:8000"   
)
APP_TITLE = "Insider Threat Detection System"

RISK_COLORS = {
    "CRITICAL": "#D85A30",
    "HIGH": "#E8953A",
    "MEDIUM": "#F5C842",
    "LOW": "#1D9E75",
}

RISK_BG = {
    "CRITICAL": "#FAECE7",
    "HIGH": "#FEF3E2",
    "MEDIUM": "#FEFBE6",
    "LOW": "#E8F8F2",
}
