import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import json
import os

os.makedirs("deployment_data", exist_ok=True)

print("Loading full dataset...")
df = pd.read_parquet("data/features/feature_matrix_ensemble_scored.parquet")

# ── 1. Alert table (already small) ────────────────────────────
import shutil
shutil.copy("reports/alerts/alert_table.csv",
            "deployment_data/alert_table.csv")
print(f"Alert table: {len(pd.read_csv('deployment_data/alert_table.csv'))} rows")

# ── 2. User risk summary (aggregated — tiny) ──────────────────
user_summary = (
    df.groupby("user")
    .agg(
        max_score        = ("ensemble_score", "max"),
        avg_score        = ("ensemble_score", "mean"),
        total_sessions   = ("ensemble_score", "count"),
        flagged_sessions = ("ensemble_flag_intersect", "sum"),
        total_usb        = ("device_count", "sum"),
        total_ext_email  = ("email_to_external", "sum"),
        total_susp_http  = ("http_suspicious", "sum"),
        avg_after_hours  = ("after_hours_ratio", "mean"),
    )
    .reset_index()
    .sort_values("max_score", ascending=False)
)
user_summary.to_parquet("deployment_data/user_summary.parquet", index=False)
print(f"User summary: {len(user_summary)} users, "
      f"{user_summary.memory_usage(deep=True).sum()/1024:.1f} KB")

# ── 3. Timeline data — only flagged sessions ──────────────────
flagged = df[df["ensemble_flag_intersect"] == 1][[
    "user", "date_only", "ensemble_score", "ae_anomaly_score",
    "if_anomaly_score", "ae_anomaly_flag", "if_anomaly_flag",
    "ensemble_flag_intersect", "device_count", "email_to_external",
    "http_suspicious", "sensitive_file_count", "after_hours_ratio",
    "first_logon_hour", "total_events", "composite_risk_score",
]].copy()
flagged["date_only"] = flagged["date_only"].astype(str)
flagged.to_parquet("deployment_data/flagged_sessions.parquet", index=False)
print(f"Flagged sessions: {len(flagged)} rows, "
      f"{flagged.memory_usage(deep=True).sum()/1024/1024:.1f} MB")

# ── 4. User timeline — all sessions but minimal columns ───────
timeline = df[[
    "user", "date_only", "ensemble_score", "ae_anomaly_score",
    "if_anomaly_score", "ensemble_flag_intersect",
    "device_count", "email_to_external", "http_suspicious",
    "after_hours_ratio", "total_events",
]].copy()
timeline["date_only"] = timeline["date_only"].astype(str)
timeline.to_parquet("deployment_data/timeline.parquet", index=False)
mb = timeline.memory_usage(deep=True).sum()/1024/1024
print(f"Timeline: {len(timeline)} rows, {mb:.1f} MB")

# ── 5. SHAP importance ────────────────────────────────────────
shutil.copy("models/isolation_forest/shap_importance.json",
            "deployment_data/shap_importance.json")

# ── 6. Model comparison ───────────────────────────────────────
shutil.copy("reports/model_comparison.csv",
            "deployment_data/model_comparison.csv")

# ── 7. Top 50 alerts JSON ─────────────────────────────────────
shutil.copy("reports/alerts/top50_alerts.json",
            "deployment_data/top50_alerts.json")

# ── 8. Overview stats (static JSON) ──────────────────────────
from src.explainability.alert_formatter import get_risk_level

top5 = user_summary.head(5)[["user","max_score"]].copy()
overview = {
    "total_sessions"    : int(len(df)),
    "total_users"       : int(df["user"].nunique()),
    "ae_anomalies"      : int(df["ae_anomaly_flag"].sum()),
    "if_anomalies"      : int(df["if_anomaly_flag"].sum()),
    "ensemble_union"    : int(df["ensemble_flag_union"].sum()),
    "ensemble_intersect": int(df["ensemble_flag_intersect"].sum()),
    "ae_anomaly_pct"    : round(df["ae_anomaly_flag"].mean()*100, 2),
    "if_anomaly_pct"    : round(df["if_anomaly_flag"].mean()*100, 2),
    "date_range"        : {
        "from": str(df["date_only"].min())[:10],
        "to"  : str(df["date_only"].max())[:10],
    },
    "top_risk_users": [
        {
            "user"      : row["user"],
            "max_score" : round(float(row["max_score"]), 4),
            "risk_level": get_risk_level(float(row["max_score"])),
        }
        for _, row in top5.iterrows()
    ],
}
with open("deployment_data/overview_stats.json", "w") as f:
    json.dump(overview, f, indent=2)

# ── 9. Alert counts (static JSON) ────────────────────────────
alert_df = pd.read_csv("deployment_data/alert_table.csv")
counts = {
    "total"   : len(alert_df),
    "critical": int((alert_df["ensemble_score"] >= 0.8).sum()),
    "high"    : int(((alert_df["ensemble_score"] >= 0.6) &
                     (alert_df["ensemble_score"] < 0.8)).sum()),
    "medium"  : int(((alert_df["ensemble_score"] >= 0.4) &
                     (alert_df["ensemble_score"] < 0.6)).sum()),
    "low"     : int((alert_df["ensemble_score"] < 0.4).sum()),
}
with open("deployment_data/alert_counts.json", "w") as f:
    json.dump(counts, f, indent=2)

print("\nDeployment data sizes:")
total = 0
for f in os.listdir("deployment_data"):
    size = os.path.getsize(f"deployment_data/{f}")
    total += size
    print(f"  {f:45s} {size/1024/1024:.2f} MB")

print(f"\nTotal deployment_data size: {total/1024/1024:.1f} MB")
print("Done. Now commit deployment_data/ and push to GitHub.")