import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def build_sessions(logs: dict) -> pd.DataFrame:
    logger.info("Building user-day sessions...")
    frames = []

    if "logon" in logs:
        logon = logs["logon"].copy()
        s = logon.groupby(["user", "date_only"]).agg(
            logon_count       = ("id", "count"),
            logon_after_hours = ("is_after_hours", "sum"),
            logon_weekend     = ("is_weekend", "sum"),
            first_logon_hour  = ("hour", "min"),
            last_logon_hour   = ("hour", "max"),
            unique_pcs        = ("pc", "nunique"),
        ).reset_index()
        s["session_span_hours"] = (s["last_logon_hour"] - s["first_logon_hour"]).clip(lower=0)
        frames.append(("logon", s))

    if "device" in logs:
        device = logs["device"].copy()
        s = device.groupby(["user", "date_only"]).agg(
            device_count       = ("id", "count"),
            device_after_hours = ("is_after_hours", "sum"),
            device_weekend     = ("is_weekend", "sum"),
        ).reset_index()
        frames.append(("device", s))

    if "email" in logs:
        email = logs["email"].copy()
        email["cc"]  = email["cc"].fillna("") if "cc" in email.columns else ""
        email["bcc"] = email["bcc"].fillna("") if "bcc" in email.columns else ""
        if "to" in email.columns:
            email["to_external"] = email["to"].str.contains(
                r"@(?!dtaa\.com)[a-zA-Z]", regex=True, na=False
            ).astype(int)
        else:
            email["to_external"] = 0
        agg_dict = {"id": "count", "is_after_hours": "sum", "to_external": "sum"}
        if "size" in email.columns:
            agg_dict["size"] = "sum"
        if "attachments" in email.columns:
            agg_dict["attachments"] = "sum"
        s = email.groupby(["user", "date_only"]).agg(
            email_count       = ("id", "count"),
            email_after_hours = ("is_after_hours", "sum"),
            email_to_external = ("to_external", "sum"),
        ).reset_index()
        if "size" in email.columns:
            s2 = email.groupby(["user", "date_only"])["size"].sum().reset_index()
            s2.columns = ["user", "date_only", "email_size_total"]
            s = s.merge(s2, on=["user", "date_only"], how="left")
        if "attachments" in email.columns:
            s3 = email.groupby(["user", "date_only"])["attachments"].sum().reset_index()
            s3.columns = ["user", "date_only", "email_attachments"]
            s = s.merge(s3, on=["user", "date_only"], how="left")
        frames.append(("email", s))

    if "file" in logs:
        file = logs["file"].copy()
        sensitive_patterns = r"(hr|finance|executive|payroll|confidential|secret)"
        if "filename" in file.columns:
            file["sensitive_file"] = file["filename"].str.lower().str.contains(
                sensitive_patterns, regex=True, na=False
            ).astype(int)
        else:
            file["sensitive_file"] = 0
        s = file.groupby(["user", "date_only"]).agg(
            file_count           = ("id", "count"),
            file_after_hours     = ("is_after_hours", "sum"),
            sensitive_file_count = ("sensitive_file", "sum"),
        ).reset_index()
        frames.append(("file", s))

    if "http" in logs:
        http = logs["http"].copy()
        suspicious_patterns = r"(linkedin|monster|indeed|careerbuilder|dropbox|wikileaks)"
        if "url" in http.columns:
            http["suspicious_url"] = http["url"].str.lower().str.contains(
                suspicious_patterns, regex=True, na=False
            ).astype(int)
        else:
            http["suspicious_url"] = 0
        s = http.groupby(["user", "date_only"]).agg(
            http_count       = ("id", "count"),
            http_after_hours = ("is_after_hours", "sum"),
            http_suspicious  = ("suspicious_url", "sum"),
        ).reset_index()
        frames.append(("http", s))

    if not frames:
        raise ValueError("No log data available to build sessions.")

    session = frames[0][1].copy()
    for name, df in frames[1:]:
        session = session.merge(df, on=["user", "date_only"], how="outer")

    fill_cols = [c for c in session.columns if c not in ["user", "date_only"]]
    session[fill_cols] = session[fill_cols].fillna(0)

    session["date_only"]   = pd.to_datetime(session["date_only"])
    session["day_of_week"] = session["date_only"].dt.dayofweek
    session["is_weekend"]  = session["day_of_week"].isin([5, 6]).astype(int)
    session["month"]       = session["date_only"].dt.month
    session = session.sort_values(["user", "date_only"]).reset_index(drop=True)

    logger.info(f"Sessions built: {len(session):,} rows | {session['user'].nunique()} users")
    return session
