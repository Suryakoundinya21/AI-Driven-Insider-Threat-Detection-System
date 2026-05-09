from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.api.feedback_store import (
    add_false_positive, add_confirmed,
    get_feedback_stats, get_user_threshold_boost,
    load_feedback,
)

router = APIRouter(prefix="/feedback", tags=["Feedback"])


class FalsePositiveRequest(BaseModel):
    alert_id : str
    user     : str
    analyst  : str = "analyst"
    reason   : str = "Not anomalous"


class ConfirmRequest(BaseModel):
    alert_id : str
    user     : str
    analyst  : str = "analyst"
    severity : str = "HIGH"


@router.post("/false-positive")
def mark_false_positive(req: FalsePositiveRequest):
    record = add_false_positive(
        req.alert_id, req.user, req.analyst, req.reason
    )
    boost = get_user_threshold_boost(req.user)
    return {
        "status"          : "recorded",
        "message"         : f"Alert {req.alert_id} marked as false positive",
        "user"            : req.user,
        "threshold_boost" : boost,
        "effect"          : f"Threshold for {req.user} raised by {boost:.0%} to reduce future alerts",
    }


@router.post("/confirm")
def confirm_threat(req: ConfirmRequest):
    record = add_confirmed(
        req.alert_id, req.user, req.analyst, req.severity
    )
    return {
        "status"  : "recorded",
        "message" : f"Alert {req.alert_id} confirmed as {req.severity} threat",
        "user"    : req.user,
    }


@router.get("/stats")
def feedback_stats():
    return get_feedback_stats()


@router.get("/history")
def feedback_history():
    data = load_feedback()
    return {
        "false_positives" : data["false_positives"][-20:],
        "confirmed"       : data["confirmed"][-20:],
    }


@router.get("/user/{user_id}")
def user_feedback(user_id: str):
    data  = load_feedback()
    boost = get_user_threshold_boost(user_id)
    user_fp = [
        r for r in data["false_positives"]
        if r.get("user") == user_id
    ]
    user_cf = [
        r for r in data["confirmed"]
        if r.get("user") == user_id
    ]
    return {
        "user"             : user_id,
        "false_positives"  : len(user_fp),
        "confirmed"        : len(user_cf),
        "threshold_boost"  : boost,
        "adjusted"         : boost != 0.0,
        "history"          : user_fp[-5:] + user_cf[-5:],
    }


@router.delete("/reset/{user_id}")
def reset_user(user_id: str):
    data = load_feedback()
    if user_id in data["user_adjustments"]:
        del data["user_adjustments"][user_id]
        from src.api.feedback_store import save_feedback
        save_feedback(data)
        return {"status": "reset", "user": user_id}
    return {"status": "not_found", "user": user_id}
