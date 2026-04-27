from fastapi import APIRouter
from src.api.data_store  import get_df, get_alert_df, get_shap_importance
from src.explainability.alert_formatter import get_risk_level

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
        .rename(columns={"ensemble_score": "max_score"})
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
            "from" : str(df["date_only"].min())[:10],
            "to"   : str(df["date_only"].max())[:10],
        },
        "top_risk_users"     : [
            {
                "user"      : row["user"],
                "max_score" : round(float(row["max_score"]), 4),
                "risk_level": get_risk_level(float(row["max_score"])),
            }
            for _, row in top_users.iterrows()
        ],
    }


@router.get("/shap-importance")
def get_shap_importance_endpoint():
    imp = get_shap_importance()
    ranked = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    return {
        "features": [
            {"rank": i+1, "feature": feat, "importance": round(val, 5)}
            for i, (feat, val) in enumerate(ranked)
        ]
    }


@router.get("/model-comparison")
def get_model_comparison():
    import os
    import pandas as pd
    path = "reports/model_comparison.csv"
    if not os.path.exists(path):
        return {"error": "model_comparison.csv not found"}
    df = pd.read_csv(path)
    return {"models": df.to_dict(orient="records")}
