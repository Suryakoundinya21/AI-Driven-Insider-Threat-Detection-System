# scripts/run_day2.py

from src.ingestion.loader import load_all_logs, save_processed, load_ground_truth
from src.ingestion.eda import (
    null_audit, user_activity_summary,
    plot_hourly_distribution, plot_user_volume_distribution,
    date_range_audit, after_hours_analysis
)
from src.config import config

if __name__ == "__main__":

    # ── Step 1: Load all logs ─────────────────────────────────────────────────
    print("Loading CERT dataset logs...")
    logs = load_all_logs()        # use nrows=50000 while testing

    # ── Step 2: EDA ───────────────────────────────────────────────────────────
    date_range_audit(logs)
    null_audit(logs)
    summary = user_activity_summary(logs)
    after_hours_analysis(logs)

    # ── Step 3: Plots ─────────────────────────────────────────────────────────
    plot_hourly_distribution(logs, save_path="reports/plots/hourly_dist.png")
    plot_user_volume_distribution(summary, save_path="reports/plots/user_volume.png")

    # ── Step 4: Save processed data ───────────────────────────────────────────
    save_processed(logs)

    # ── Step 5: Ground truth ─────────────────────────────────────────────────
    gt = load_ground_truth()
    if not gt.empty:
        print(f"\nInsider scenarios in dataset: {gt['scenario'].nunique()}")
        print(gt["scenario"].value_counts())

    print("\nDay 2 Complete. Processed data saved to data/processed/")