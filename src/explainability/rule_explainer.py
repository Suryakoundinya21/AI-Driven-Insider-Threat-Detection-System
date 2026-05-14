import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

RULES = [
    {
        "feature"   : "device_count",
        "baseline"  : "device_count_baseline_mean",
        "std"       : "device_count_baseline_std",
        "threshold" : 2.0,
        "direction" : "ratio",
        "template"  : "USB/device activity ({val:.0f} events) is {ratio:.1f}x above this user's 30-day baseline ({base:.1f} avg). Z-score: {zscore:.1f} standard deviations from normal.",
        "severity"  : "HIGH",
    },
    {
        "feature"   : "email_to_external",
        "baseline"  : "email_to_external_baseline_mean",
        "std"       : "email_to_external_baseline_std",
        "threshold" : 2.0,
        "direction" : "ratio",
        "template"  : "External email volume ({val:.0f} emails sent outside organisation) is {ratio:.1f}x above personal baseline ({base:.1f} avg). Potential data exfiltration via email.",
        "severity"  : "HIGH",
    },
    {
        "feature"   : "http_suspicious",
        "baseline"  : None,
        "std"       : None,
        "threshold" : 3,
        "direction" : "absolute",
        "template"  : "Visited {val:.0f} high-risk URLs (job boards, cloud storage, competitor sites) in a single session. Normal baseline: 0-2 suspicious visits.",
        "severity"  : "MEDIUM",
    },
    {
        "feature"   : "sensitive_file_count",
        "baseline"  : None,
        "std"       : None,
        "threshold" : 1,
        "direction" : "absolute",
        "template"  : "Accessed {val:.0f} file(s) in sensitive directories (HR, Finance, Executive, Payroll). This user does not normally access these directories.",
        "severity"  : "HIGH",
    },
    {
        "feature"   : "after_hours_ratio",
        "baseline"  : "after_hours_ratio_baseline_mean",
        "std"       : "after_hours_ratio_baseline_std",
        "threshold" : 0.35,
        "direction" : "absolute",
        "template"  : "{pct:.0f}% of session activity occurred outside business hours (before 8am or after 6pm). This user's personal baseline is {base_pct:.0f}% after-hours activity.",
        "severity"  : "MEDIUM",
    },
    {
        "feature"   : "first_logon_hour",
        "baseline"  : "first_logon_hour_baseline_mean",
        "std"       : "first_logon_hour_baseline_std",
        "threshold" : 3.0,
        "direction" : "deviation",
        "template"  : "Login occurred at {val:.0f}:00 — {dev:.1f} hours outside this user's normal login window (typically {base:.1f}:00). Unusual access time detected.",
        "severity"  : "MEDIUM",
    },
    {
        "feature"   : "email_size_total",
        "baseline"  : "email_size_total_baseline_mean",
        "std"       : "email_size_total_baseline_std",
        "threshold" : 3.0,
        "direction" : "ratio",
        "template"  : "Total email data volume ({val:.0f} bytes) is {ratio:.1f}x above personal 30-day average ({base:.0f} bytes). Large volume email transfer detected.",
        "severity"  : "MEDIUM",
    },
    {
        "feature"   : "unique_pcs",
        "baseline"  : None,
        "std"       : None,
        "threshold" : 3,
        "direction" : "absolute",
        "template"  : "Logged into {val:.0f} different machines in a single day. Normal behaviour is 1-2 machines. Possible lateral movement or credential sharing.",
        "severity"  : "HIGH",
    },
    {
        "feature"   : "logon_count",
        "baseline"  : "logon_count_baseline_mean",
        "std"       : "logon_count_baseline_std",
        "threshold" : 3.0,
        "direction" : "ratio",
        "template"  : "Login frequency ({val:.0f} logons) is {ratio:.1f}x above personal baseline ({base:.1f} avg). Unusually high system access frequency.",
        "severity"  : "LOW",
    },
    {
        "feature"   : "activity_entropy",
        "baseline"  : "activity_entropy_baseline_mean",
        "std"       : "activity_entropy_baseline_std",
        "threshold" : 0.0,
        "direction" : "low",
        "template"  : "Activity entropy ({val:.2f}) is lower than baseline ({base:.2f}). Behaviour is unusually concentrated in one activity type — possible focused data collection.",
        "severity"  : "LOW",
    },
    {
        "feature"   : "device_after_hours",
        "baseline"  : None,
        "std"       : None,
        "threshold" : 1,
        "direction" : "absolute",
        "template"  : "USB/removable device connected {val:.0f} time(s) outside business hours. After-hours device activity is a known data exfiltration indicator.",
        "severity"  : "HIGH",
    },
    {
        "feature"   : "email_attachments",
        "baseline"  : "email_attachments_baseline_mean",
        "std"       : "email_attachments_baseline_std",
        "threshold" : 3.0,
        "direction" : "ratio",
        "template"  : "Email attachments sent ({val:.0f}) is {ratio:.1f}x above personal baseline ({base:.1f} avg). High attachment volume may indicate data exfiltration.",
        "severity"  : "MEDIUM",
    },
]

SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def generate_reasons(row: pd.Series) -> list:
    reasons = []

    for rule in RULES:
        feat      = rule["feature"]
        baseline  = rule["baseline"]
        std_col   = rule.get("std")
        threshold = rule["threshold"]
        direction = rule["direction"]
        template  = rule["template"]
        severity  = rule["severity"]

        if feat not in row.index:
            continue

        val  = float(row.get(feat, 0) or 0)
        base = float(row.get(baseline, 0) or 0) if baseline and baseline in row.index else 0.0
        std  = float(row.get(std_col, 1) or 1)  if std_col and std_col in row.index else 1.0
        if std == 0:
            std = 1.0

        zscore    = (val - base) / std if std > 0 else 0.0
        triggered = False
        reason_text = ""

        if direction == "ratio":
            ratio = val / base if base > 0.001 else 0.0
            if ratio >= threshold and val > 0:
                triggered   = True
                reason_text = template.format(
                    val=val, base=base, ratio=ratio,
                    pct=val*100, base_pct=base*100,
                    zscore=zscore, dev=abs(val-base),
                )

        elif direction == "absolute":
            if val >= threshold:
                triggered   = True
                reason_text = template.format(
                    val=val, base=base, ratio=1,
                    pct=val*100, base_pct=base*100,
                    zscore=zscore, dev=abs(val-base),
                )

        elif direction == "deviation":
            dev = abs(val - base)
            if dev >= threshold:
                triggered   = True
                reason_text = template.format(
                    val=val, base=base, dev=dev,
                    ratio=1, pct=val*100, base_pct=base*100,
                    zscore=zscore,
                )

        elif direction == "low":
            if base > 0 and val < base * 0.5:
                triggered   = True
                reason_text = template.format(
                    val=val, base=base,
                    ratio=1, pct=val*100, base_pct=base*100,
                    zscore=zscore, dev=abs(val-base),
                )

        if triggered:
            reasons.append({
                "severity"   : severity,
                "feature"    : feat,
                "reason"     : reason_text,
                "value"      : round(val, 3),
                "baseline"   : round(base, 3),
                "zscore"     : round(zscore, 2),
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
