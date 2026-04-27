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
    return "\n".join(lines)


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
