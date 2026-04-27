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
