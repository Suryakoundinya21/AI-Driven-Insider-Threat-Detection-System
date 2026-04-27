from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import date


class AlertSummary(BaseModel):
    alert_id        : str
    user            : str
    date            : str
    risk_level      : str
    ensemble_score  : float
    ae_score        : float
    if_score        : float
    both_flagged    : bool
    reason_count    : int
    reason_summary  : str
    high_reasons    : int
    device_count    : int
    email_external  : int
    http_suspicious : int
    sensitive_files : int


class AlertDetail(BaseModel):
    alert_id        : str
    user            : str
    date            : str
    risk_level      : str
    ensemble_score  : float
    ae_score        : float
    if_score        : float
    both_flagged    : bool
    reasons         : List[Dict[str, str]]
    stats           : Dict[str, Any]
    shap_features   : Optional[List[Dict[str, Any]]] = None


class UserTimeline(BaseModel):
    user            : str
    total_sessions  : int
    flagged_sessions: int
    max_score       : float
    risk_level      : str
    timeline        : List[Dict[str, Any]]


class ModelStats(BaseModel):
    total_sessions      : int
    ae_anomalies        : int
    if_anomalies        : int
    ensemble_union      : int
    ensemble_intersect  : int
    ae_anomaly_pct      : float
    if_anomaly_pct      : float
    top_risk_users      : List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status          : str
    total_sessions  : int
    total_alerts    : int
    model_loaded    : bool
