import sys, os
sys.path.insert(0, os.path.abspath("."))

import logging
from src.api.data_store import load_data, get_df, get_alert_df, get_shap_importance

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    print("="*60)
    print("DAY 7 — API DATA VERIFICATION")
    print("="*60)

    load_data()

    df       = get_df()
    alert_df = get_alert_df()
    shap_imp = get_shap_importance()

    print(f"\nDataset loaded        : {df.shape}")
    print(f"Alert table loaded    : {len(alert_df)} rows")
    print(f"SHAP features loaded  : {len(shap_imp)} features")
    print(f"Date range            : {df['date_only'].min()} to {df['date_only'].max()}")
    print(f"Unique users          : {df['user'].nunique()}")
    print(f"Flagged (intersect)   : {df['ensemble_flag_intersect'].sum():,}")

    print("\nTop 5 risk users:")
    top5 = (
        df.groupby("user")["ensemble_score"]
        .max()
        .sort_values(ascending=False)
        .head(5)
    )
    for user, score in top5.items():
        print(f"  {user:12s}  score={score:.4f}")

    print("\nAlert risk breakdown:")
    crit = (alert_df["ensemble_score"] >= 0.8).sum()
    high = ((alert_df["ensemble_score"] >= 0.6) &
            (alert_df["ensemble_score"] < 0.8)).sum()
    med  = ((alert_df["ensemble_score"] >= 0.4) &
            (alert_df["ensemble_score"] < 0.6)).sum()
    print(f"  CRITICAL : {crit}")
    print(f"  HIGH     : {high}")
    print(f"  MEDIUM   : {med}")

    print("\nTop 5 SHAP features:")
    for feat, val in sorted(shap_imp.items(),
                             key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feat:35s} {val:.5f}")

    print("\n" + "="*60)
    print("All data verified. Start the API with:")
    print("  uvicorn src.api.main:app --reload --port 8000")
    print("Then open: http://127.0.0.1:8000/docs")
    print("="*60)

if __name__ == "__main__":
    main()
