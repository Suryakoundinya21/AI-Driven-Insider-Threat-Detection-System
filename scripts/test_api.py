import requests
import json

BASE = "http://127.0.0.1:8000"

def test(label, url, params=None):
    try:
        r = requests.get(url, params=params, timeout=10)
        print(f"\n{'='*55}")
        print(f"TEST : {label}")
        print(f"URL  : {url}")
        print(f"STATUS: {r.status_code}")
        data = r.json()
        if isinstance(data, list):
            print(f"RESULT: [{len(data)} items]")
            if data:
                print(json.dumps(data[0], indent=2, default=str)[:600])
        else:
            print(json.dumps(data, indent=2, default=str)[:600])
    except Exception as e:
        print(f"ERROR: {e}")

print("\ninsider Threat Detection API — Endpoint Tests")
print("Make sure server is running: uvicorn src.api.main:app --reload --port 8000\n")

test("Health check",          f"{BASE}/")
test("System overview",       f"{BASE}/stats/overview")
test("Alert counts",          f"{BASE}/alerts/count")
test("All alerts (top 5)",    f"{BASE}/alerts", {"limit": 5})
test("CRITICAL alerts only",  f"{BASE}/alerts", {"risk_level": "CRITICAL", "limit": 5})
test("Top risk users",        f"{BASE}/users/top-risk", {"limit": 5})
test("User timeline",         f"{BASE}/users/dlm0051/timeline", {"days": 30})
test("User summary",          f"{BASE}/users/gko0078/summary")
test("SHAP importance",       f"{BASE}/stats/shap-importance")
test("Model comparison",      f"{BASE}/stats/model-comparison")
test("Alert detail",          f"{BASE}/alerts/ALERT-GKO0078-20110322")

print("\nAll tests complete.")
