import os

files = {}

files["src/explainability/__init__.py"] = ""

files["src/explainability/shap_explainer.py"] = """\
import numpy as np
import pandas as pd
import shap
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def build_shap_explainer(clf, X_background: np.ndarray):
    logger.info("Building SHAP TreeExplainer for Isolation Forest...")
    explainer = shap.TreeExplainer(clf)
    logger.info("SHAP explainer ready.")
    return explainer


def compute_shap_values(
    explainer,
    X: np.ndarray,
    feature_names: list,
    max_samples: int = 3000,
) -> pd.DataFrame:
    n = min(len(X), max_samples)
    logger.info(f"Computing SHAP values for {n:,} sessions...")
    X_sample  = X[:n]
    shap_vals = explainer.shap_values(X_sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    logger.info(f"SHAP values computed. Shape: {shap_df.shape}")
    return shap_df


def plot_shap_summary(
    shap_df: pd.DataFrame,
    X: np.ndarray,
    feature_names: list,
    save_path: str,
    max_samples: int = 3000,
):
    n = min(len(X), max_samples)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_df.values,
        X[:n],
        feature_names=feature_names,
        show=False,
        plot_type="bar",
    )
    plt.title("SHAP Feature Importance — Isolation Forest")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved: {save_path}")


def get_top_shap_features(
    shap_df: pd.DataFrame,
    idx: int,
    top_n: int = 5,
) -> list:
    if idx >= len(shap_df):
        return []
    row = shap_df.iloc[idx]
    top = row.abs().nlargest(top_n)
    return [(feat, float(row[feat])) for feat in top.index]
"""

files["src/explainability/rule_explainer.py"] = """\
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

RULES = [
    {
        "feature"  : "device_count",
        "baseline" : "device_count_baseline_mean",
        "threshold": 3.0,
        "direction": "ratio",
        "template" : "USB/device activity ({val:.0f} events) is {ratio:.1f}x above this user's normal ({base:.1f} avg)",
        "severity" : "HIGH",
    },
    {
        "feature"  : "email_to_external",
        "baseline" : "email_to_external_baseline_mean",
        "threshold": 3.0,
        "direction": "ratio",
        "template" : "External emails ({val:.0f}) is {ratio:.1f}x above user baseline ({base:.1f} avg)",
        "severity" : "HIGH",
    },
    {
        "feature"  : "http_suspicious",
        "baseline" : None,
        "threshold": 3,
        "direction": "absolute",
        "template" : "Visited {val:.0f} suspicious URLs (job boards, cloud storage, competitor sites)",
        "severity" : "MEDIUM",
    },
    {
        "feature"  : "sensitive_file_count",
        "baseline" : None,
        "threshold": 1,
        "direction": "absolute",
        "template" : "Accessed {val:.0f} sensitive file(s) in HR, finance, or executive directories",
        "severity" : "HIGH",
    },
    {
        "feature"  : "after_hours_ratio",
        "baseline" : "after_hours_ratio_baseline_mean",
        "threshold": 0.4,
        "direction": "absolute",
        "template" : "{pct:.0f}% of session activity occurred outside business hours (baseline: {base_pct:.0f}%)",
        "severity" : "MEDIUM",
    },
    {
        "feature"  : "first_logon_hour",
        "baseline" : "first_logon_hour_baseline_mean",
        "threshold": 3.0,
        "direction": "deviation",
        "template" : "Login at hour {val:.0f}:00 deviates {dev:.1f} hours from personal baseline ({base:.1f}h avg)",
        "severity" : "MEDIUM",
    },
    {
        "feature"  : "email_size_total",
        "baseline" : "email_size_total_baseline_mean",
        "threshold": 3.0,
        "direction": "ratio",
        "template" : "Email data volume ({val:.0f} bytes) is {ratio:.1f}x above user baseline",
        "severity" : "MEDIUM",
    },
    {
        "feature"  : "unique_pcs",
        "baseline" : None,
        "threshold": 3,
        "direction": "absolute",
        "template" : "Logged into {val:.0f} different machines in one day — possible lateral movement",
        "severity" : "HIGH",
    },
    {
        "feature"  : "logon_count",
        "baseline" : "logon_count_baseline_mean",
        "threshold": 3.0,
        "direction": "ratio",
        "template" : "Logon count ({val:.0f}) is {ratio:.1f}x above personal normal ({base:.1f} avg)",
        "severity" : "LOW",
    },
    {
        "feature"  : "activity_entropy",
        "baseline" : "activity_entropy_baseline_mean",
        "threshold": 0.0,
        "direction": "low",
        "template" : "Activity entropy ({val:.2f}) lower than baseline ({base:.2f}) — unusually concentrated behavior",
        "severity" : "LOW",
    },
]

SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def generate_reasons(row: pd.Series) -> list:
    reasons = []
    for rule in RULES:
        feat      = rule["feature"]
        baseline  = rule["baseline"]
        threshold = rule["threshold"]
        direction = rule["direction"]
        template  = rule["template"]
        severity  = rule["severity"]

        if feat not in row.index:
            continue

        val  = float(row[feat]) if pd.notna(row.get(feat)) else 0.0
        base = float(row[baseline]) if (
            baseline and baseline in row.index and pd.notna(row.get(baseline))
        ) else 0.0

        triggered   = False
        reason_text = ""

        if direction == "ratio":
            ratio = val / base if base > 0.001 else 0.0
            if ratio >= threshold and val > 0:
                triggered   = True
                reason_text = template.format(
                    val=val, base=base, ratio=ratio,
                    pct=val*100, base_pct=base*100,
                )

        elif direction == "absolute":
            if val >= threshold:
                triggered   = True
                reason_text = template.format(
                    val=val, base=base, ratio=1,
                    pct=val*100, base_pct=base*100,
                )

        elif direction == "deviation":
            dev = abs(val - base)
            if dev >= threshold:
                triggered   = True
                reason_text = template.format(val=val, base=base, dev=dev)

        elif direction == "low":
            if base > 0 and val < base * 0.5:
                triggered   = True
                reason_text = template.format(val=val, base=base)

        if triggered:
            reasons.append({
                "severity": severity,
                "feature" : feat,
                "reason"  : reason_text,
            })

    reasons.sort(key=lambda x: SEVERITY_ORDER.get(x["severity"], 3))
    return reasons


def generate_all_reasons(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating human-readable reasons for all sessions...")
    df = df.copy()
    df["reasons"]      = df.apply(generate_reasons, axis=1)
    df["reason_count"] = df["reasons"].apply(len)
    logger.info(f"Done. Avg triggers per session: {df['reason_count'].mean():.2f}")
    return df
"""

files["src/explainability/alert_formatter.py"] = """\
import pandas as pd
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

SEVERITY_EMOJI = {"HIGH": "[HIGH]", "MEDIUM": "[MED]", "LOW": "[LOW]"}

RISK_LEVELS = {
    (0.8, 1.01) : "CRITICAL",
    (0.6, 0.80) : "HIGH",
    (0.4, 0.60) : "MEDIUM",
    (0.0, 0.40) : "LOW",
}


def get_risk_level(score: float) -> str:
    for (lo, hi), label in RISK_LEVELS.items():
        if lo <= score < hi:
            return label
    return "LOW"


def format_alert(row: pd.Series) -> dict:
    score    = float(row.get("ensemble_score", 0))
    risk     = get_risk_level(score)
    reasons  = row.get("reasons", [])
    date_str = str(row.get("date_only", ""))[:10]

    return {
        "alert_id"       : f"ALERT-{str(row.get('user','')).upper()}-{date_str.replace('-','')}",
        "user"           : str(row.get("user", "")),
        "date"           : date_str,
        "risk_level"     : risk,
        "ensemble_score" : round(score, 4),
        "ae_score"       : round(float(row.get("ae_anomaly_score", 0)), 4),
        "if_score"       : round(float(row.get("if_anomaly_score", 0)), 4),
        "ae_flagged"     : bool(row.get("ae_anomaly_flag", 0)),
        "if_flagged"     : bool(row.get("if_anomaly_flag", 0)),
        "both_flagged"   : bool(
            row.get("ae_anomaly_flag", 0) and row.get("if_anomaly_flag", 0)
        ),
        "reasons"        : reasons,
        "reason_summary" : "; ".join([r["reason"] for r in reasons[:3]]),
        "high_reasons"   : [r for r in reasons if r["severity"] == "HIGH"],
        "stats"          : {
            "device_count"      : int(row.get("device_count", 0)),
            "email_to_external" : int(row.get("email_to_external", 0)),
            "http_suspicious"   : int(row.get("http_suspicious", 0)),
            "sensitive_files"   : int(row.get("sensitive_file_count", 0)),
            "after_hours_ratio" : round(float(row.get("after_hours_ratio", 0)), 3),
            "first_logon_hour"  : int(row.get("first_logon_hour", 0)),
            "total_events"      : int(row.get("total_events", 0)),
        },
        "generated_at"   : datetime.now().isoformat(),
    }


def format_alert_text(alert: dict) -> str:
    risk_icons = {"CRITICAL": "*** CRITICAL ***", "HIGH": "** HIGH **",
                  "MEDIUM": "* MEDIUM *", "LOW": "LOW"}
    icon = risk_icons.get(alert["risk_level"], alert["risk_level"])
    lines = [
        "=" * 62,
        f"ALERT ID    : {alert['alert_id']}",
        f"USER        : {alert['user']}",
        f"DATE        : {alert['date']}",
        f"RISK LEVEL  : {icon}",
        f"SCORES      : Ensemble={alert['ensemble_score']:.3f} | "
        f"AE={alert['ae_score']:.3f} | IF={alert['if_score']:.3f}",
        f"BOTH MODELS : {'YES - HIGH CONFIDENCE' if alert['both_flagged'] else 'NO'}",
        "-" * 62,
        f"REASONS ({len(alert['reasons'])} triggers):",
    ]
    for r in alert["reasons"]:
        tag = SEVERITY_EMOJI.get(r["severity"], "•")
        lines.append(f"  {tag} {r['reason']}")

    lines += [
        "-" * 62,
        "SESSION STATS:",
        f"  USB events        : {alert['stats']['device_count']}",
        f"  External emails   : {alert['stats']['email_to_external']}",
        f"  Suspicious URLs   : {alert['stats']['http_suspicious']}",
        f"  Sensitive files   : {alert['stats']['sensitive_files']}",
        f"  After-hours ratio : {alert['stats']['after_hours_ratio']*100:.1f}%",
        f"  First logon hour  : {alert['stats']['first_logon_hour']}:00",
        f"  Total events      : {alert['stats']['total_events']}",
        "=" * 62,
    ]
    return "\\n".join(lines)


def build_alert_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building alert dataframe from intersection-flagged sessions...")
    flagged = df[df["ensemble_flag_intersect"] == 1].copy()
    logger.info(f"Processing {len(flagged):,} high-confidence alerts...")

    rows = []
    for _, row in flagged.iterrows():
        alert = format_alert(row)
        rows.append({
            "alert_id"        : alert["alert_id"],
            "user"            : alert["user"],
            "date"            : alert["date"],
            "risk_level"      : alert["risk_level"],
            "ensemble_score"  : alert["ensemble_score"],
            "ae_score"        : alert["ae_score"],
            "if_score"        : alert["if_score"],
            "both_flagged"    : alert["both_flagged"],
            "reason_count"    : len(alert["reasons"]),
            "reason_summary"  : alert["reason_summary"],
            "high_reasons"    : len(alert["high_reasons"]),
            "device_count"    : alert["stats"]["device_count"],
            "email_external"  : alert["stats"]["email_to_external"],
            "http_suspicious" : alert["stats"]["http_suspicious"],
            "sensitive_files" : alert["stats"]["sensitive_files"],
        })

    alert_df = pd.DataFrame(rows).sort_values(
        "ensemble_score", ascending=False
    ).reset_index(drop=True)
    logger.info(f"Alert dataframe: {len(alert_df)} rows")
    return alert_df
"""

files["scripts/run_day6.py"] = """\
import sys, os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging, json

from src.config import config
from src.modeling.preprocessor         import prepare_data, split_normal_sessions
from src.modeling.isolation_forest     import load_if_model
from src.explainability.shap_explainer import (
    build_shap_explainer, compute_shap_values,
    plot_shap_summary, get_top_shap_features,
)
from src.explainability.rule_explainer  import generate_all_reasons
from src.explainability.alert_formatter import (
    format_alert, format_alert_text, build_alert_dataframe,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    os.makedirs("reports/plots",  exist_ok=True)
    os.makedirs("reports/alerts", exist_ok=True)

    # 1. Load ensemble dataset
    logger.info("Loading ensemble-scored dataset...")
    df = pd.read_parquet(
        config.FEATURES_DIR / "feature_matrix_ensemble_scored.parquet"
    )
    logger.info(f"Loaded: {df.shape}")

    # 2. Prepare features
    _, all_df          = split_normal_sessions(df)
    X_all, features, _ = prepare_data(all_df, fit=False)

    # 3. Load IF model + build SHAP explainer
    clf      = load_if_model()
    explainer = build_shap_explainer(clf, X_all[:500])

    # 4. Compute SHAP values
    shap_df = compute_shap_values(
        explainer, X_all, features, max_samples=3000
    )

    # 5. SHAP summary plot
    plot_shap_summary(
        shap_df, X_all, features,
        save_path="reports/plots/shap_summary.png",
        max_samples=3000,
    )

    # 6. Global feature importance
    global_importance = shap_df.abs().mean().sort_values(ascending=False)
    print("\\n" + "="*62)
    print("SHAP GLOBAL FEATURE IMPORTANCE (Top 15)")
    print("="*62)
    for feat, val in global_importance.head(15).items():
        bar = "█" * max(1, int(val * 300))
        print(f"  {feat:35s} {val:.5f}  {bar}")

    imp_path = config.MODELS_DIR / "isolation_forest" / "shap_importance.json"
    imp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(imp_path, "w") as f:
        json.dump(global_importance.to_dict(), f, indent=2)
    logger.info(f"Importance saved: {imp_path}")

    # 7. Generate rule-based reasons for ALL sessions
    logger.info("Generating human-readable reasons for all sessions...")
    df_with_reasons = generate_all_reasons(df)

    # 8. Build alert table
    alert_df   = build_alert_dataframe(df_with_reasons)
    alert_path = "reports/alerts/alert_table.csv"
    alert_df.to_csv(alert_path, index=False)
    logger.info(f"Alert table saved: {alert_path}")

    # 9. Print top 10 formatted alerts
    print("\\n" + "="*62)
    print("TOP 10 FORMATTED ANALYST ALERTS")
    print("="*62)
    top10 = df_with_reasons.nlargest(10, "ensemble_score")
    for _, row in top10.iterrows():
        alert = format_alert(row)
        print(format_alert_text(alert))

    # 10. Save top 50 as JSON
    top50     = df_with_reasons.nlargest(50, "ensemble_score")
    all_alerts = []
    for _, row in top50.iterrows():
        a = format_alert(row)
        all_alerts.append(a)

    json_path = "reports/alerts/top50_alerts.json"
    with open(json_path, "w") as f:
        json.dump(all_alerts, f, indent=2, default=str)
    logger.info(f"Top 50 alerts JSON saved: {json_path}")

    # 11. SHAP per-session for top 5 flagged
    print("\\n" + "="*62)
    print("SHAP EXPLANATION — TOP 5 FLAGGED SESSIONS")
    print("="*62)
    top_flagged = df_with_reasons[
        df_with_reasons["ensemble_flag_intersect"] == 1
    ].nlargest(5, "ensemble_score")

    all_idx = list(range(len(df_with_reasons)))
    for i, (_, row) in enumerate(top_flagged.iterrows()):
        print(f"\\nUser: {row['user']} | "
              f"Date: {str(row['date_only'])[:10]} | "
              f"Score: {row['ensemble_score']:.3f}")
        if i < len(shap_df):
            top_feats = get_top_shap_features(shap_df, i, top_n=5)
            for feat, shap_val in top_feats:
                direction = "anomaly contribution" if shap_val > 0 else "normal contribution"
                print(f"  {feat:35s} SHAP={shap_val:+.4f}  ({direction})")
        reasons = row.get("reasons", [])
        if reasons:
            print(f"  Rule triggers ({len(reasons)}):")
            for r in reasons[:4]:
                print(f"    -> [{r['severity']}] {r['reason']}")

    # 12. Summary
    print("\\n" + "="*62)
    print("DAY 6 SUMMARY")
    print("="*62)
    print(f"SHAP values computed    : {len(shap_df):,} sessions")
    print(f"Total rule triggers     : {df_with_reasons['reason_count'].sum():,}")
    print(f"Avg triggers/session    : {df_with_reasons['reason_count'].mean():.2f}")
    print(f"High-confidence alerts  : {len(alert_df):,}")
    crit = (alert_df['ensemble_score'] >= 0.8).sum()
    high = ((alert_df['ensemble_score'] >= 0.6) &
            (alert_df['ensemble_score'] < 0.8)).sum()
    med  = ((alert_df['ensemble_score'] >= 0.4) &
            (alert_df['ensemble_score'] < 0.6)).sum()
    print(f"  CRITICAL (score>=0.8) : {crit}")
    print(f"  HIGH     (0.6-0.8)    : {high}")
    print(f"  MEDIUM   (0.4-0.6)    : {med}")
    print(f"\\nFiles saved:")
    print(f"  reports/alerts/alert_table.csv")
    print(f"  reports/alerts/top50_alerts.json")
    print(f"  reports/plots/shap_summary.png")
    print("\\nDay 6 Complete.")


if __name__ == "__main__":
    main()
"""

for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\nAll Day 6 files created. Run: python scripts/run_day6.py")