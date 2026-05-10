from fastapi import APIRouter
import pandas as pd
import os
from src.api.data_store import get_df, get_alert_df, get_shap_importance
from src.explainability.alert_formatter import get_risk_level
from src.config import config

router = APIRouter(prefix="/stats", tags=["Statistics"])


@router.get("/overview")
def get_overview():
    df       = get_df()
    alert_df = get_alert_df()

    if df is None:
        return {}

    top_users = (
        df.groupby("user")["ensemble_score"]
        .max()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )

    return {
        "total_sessions"     : len(df),
        "total_users"        : df["user"].nunique(),
        "ae_anomalies"       : int(df["ae_anomaly_flag"].sum()),
        "if_anomalies"       : int(df["if_anomaly_flag"].sum()),
        "ensemble_union"     : int(df["ensemble_flag_union"].sum()),
        "ensemble_intersect" : int(df["ensemble_flag_intersect"].sum()),
        "ae_anomaly_pct"     : round(df["ae_anomaly_flag"].mean() * 100, 2),
        "if_anomaly_pct"     : round(df["if_anomaly_flag"].mean() * 100, 2),
        "total_alerts"       : len(alert_df) if alert_df is not None else 0,
        "critical_alerts"    : int((alert_df["ensemble_score"] >= 0.8).sum())
                               if alert_df is not None and len(alert_df) > 0 else 0,
        "date_range"         : {
            "from": str(df["date_only"].min())[:10],
            "to"  : str(df["date_only"].max())[:10],
        },
        "top_risk_users": [
            {
                "user"      : row["user"],
                "max_score" : round(float(row["ensemble_score"]), 4),
                "risk_level": get_risk_level(float(row["ensemble_score"])),
            }
            for _, row in top_users.iterrows()
        ],
    }


@router.get("/shap-importance")
def get_shap_importance_endpoint():
    imp    = get_shap_importance()
    if not imp:
        return {"features": [], "error": "SHAP data not loaded"}
    ranked = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    return {
        "features": [
            {"rank": i+1, "feature": feat, "importance": round(val, 5)}
            for i, (feat, val) in enumerate(ranked)
        ]
    }


@router.get("/model-comparison")
def get_model_comparison():
    # Try v2 first (includes LSTM), fallback to v1
    for fname in ["model_comparison_v2.csv", "model_comparison.csv"]:
        path = config.REPORTS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            return {"models": df.fillna(0).to_dict(orient="records")}
    return {"error": "model_comparison.csv not found", "models": []}
