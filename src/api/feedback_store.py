import json
import logging
from pathlib import Path
from datetime import datetime

logger    = logging.getLogger(__name__)
FEED_PATH = Path("reports/feedback.json")


def load_feedback() -> dict:
    if FEED_PATH.exists():
        with open(FEED_PATH) as f:
            return json.load(f)
    return {"false_positives": [], "confirmed": [], "user_adjustments": {}}


def save_feedback(data: dict):
    FEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEED_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def add_false_positive(alert_id: str, user: str, analyst: str, reason: str):
    data   = load_feedback()
    record = {
        "alert_id"   : alert_id,
        "user"       : user,
        "analyst"    : analyst,
        "reason"     : reason,
        "timestamp"  : datetime.now().isoformat(),
        "type"       : "false_positive",
    }
    data["false_positives"].append(record)

    # Increase threshold for this user
    adj = data["user_adjustments"].get(user, {"fp_count": 0, "threshold_boost": 0.0})
    adj["fp_count"]       += 1
    adj["threshold_boost"] = min(0.3, adj["fp_count"] * 0.05)
    adj["last_updated"]    = datetime.now().isoformat()
    data["user_adjustments"][user] = adj

    save_feedback(data)
    logger.info(f"False positive recorded: {alert_id} | user={user} | "
                f"threshold_boost={adj['threshold_boost']:.2f}")
    return record


def add_confirmed(alert_id: str, user: str, analyst: str, severity: str):
    data   = load_feedback()
    record = {
        "alert_id"  : alert_id,
        "user"      : user,
        "analyst"   : analyst,
        "severity"  : severity,
        "timestamp" : datetime.now().isoformat(),
        "type"      : "confirmed",
    }
    data["confirmed"].append(record)

    # Lower threshold for this user (more sensitive)
    adj = data["user_adjustments"].get(user, {"fp_count": 0, "threshold_boost": 0.0})
    adj["threshold_boost"] = max(-0.1, adj.get("threshold_boost", 0.0) - 0.02)
    adj["last_updated"]    = datetime.now().isoformat()
    data["user_adjustments"][user] = adj

    save_feedback(data)
    logger.info(f"Confirmed threat recorded: {alert_id} | user={user} | severity={severity}")
    return record


def get_user_threshold_boost(user: str) -> float:
    data = load_feedback()
    return data["user_adjustments"].get(user, {}).get("threshold_boost", 0.0)


def get_feedback_stats() -> dict:
    data = load_feedback()
    return {
        "total_false_positives" : len(data["false_positives"]),
        "total_confirmed"       : len(data["confirmed"]),
        "users_adjusted"        : len(data["user_adjustments"]),
        "user_adjustments"      : data["user_adjustments"],
    }
