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
    print("\n" + "="*62)
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
    print("\n" + "="*62)
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
    print("\n" + "="*62)
    print("SHAP EXPLANATION — TOP 5 FLAGGED SESSIONS")
    print("="*62)
    top_flagged = df_with_reasons[
        df_with_reasons["ensemble_flag_intersect"] == 1
    ].nlargest(5, "ensemble_score")

    all_idx = list(range(len(df_with_reasons)))
    for i, (_, row) in enumerate(top_flagged.iterrows()):
        print(f"\nUser: {row['user']} | "
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
    print("\n" + "="*62)
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
    print(f"\nFiles saved:")
    print(f"  reports/alerts/alert_table.csv")
    print(f"  reports/alerts/top50_alerts.json")
    print(f"  reports/plots/shap_summary.png")
    print("\nDay 6 Complete.")


if __name__ == "__main__":
    main()
