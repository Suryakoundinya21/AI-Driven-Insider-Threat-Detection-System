# src/ingestion/eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import config


def null_audit(logs: dict):
    """Print null counts for every column in every log."""
    print("\n" + "="*60)
    print("NULL AUDIT")
    print("="*60)
    for name, df in logs.items():
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        pct   = (nulls / len(df) * 100).round(2)
        print(f"\n── {name.upper()} ({len(df):,} rows) ──")
        if nulls.empty:
            print("  No nulls.")
        else:
            for col in nulls.index:
                print(f"  {col:30s} {nulls[col]:6,}  ({pct[col]}%)")


def user_activity_summary(logs: dict) -> pd.DataFrame:
    """Build a per-user event count matrix across all log types."""
    frames = []
    for name, df in logs.items():
        if "user" in df.columns:
            counts = df.groupby("user").size().rename(f"{name}_count")
            frames.append(counts)

    summary = pd.concat(frames, axis=1).fillna(0).astype(int)
    summary["total_events"] = summary.sum(axis=1)
    summary = summary.sort_values("total_events", ascending=False)

    print("\n" + "="*60)
    print("USER ACTIVITY SUMMARY (top 20)")
    print("="*60)
    print(summary.head(20).to_string())
    return summary


def plot_hourly_distribution(logs: dict, save_path=None):
    """Plot event frequency by hour of day for each log type."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, (name, df) in enumerate(logs.items()):
        ax = axes[i]
        df["hour"].value_counts().sort_index().plot(
            kind="bar", ax=ax, color="#378ADD", alpha=0.8
        )
        ax.set_title(f"{name} — events by hour", fontsize=11)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Event count")
        ax.set_xticks(range(0, 24, 2))

    # Hide unused subplot
    for j in range(len(logs), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("CERT Dataset — Hourly Activity Distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_user_volume_distribution(summary: pd.DataFrame, save_path=None):
    """Plot distribution of total events per user — detect power users."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(summary["total_events"], bins=50, ax=ax, color="#1D9E75")
    ax.set_title("Distribution of total events per user")
    ax.set_xlabel("Total events")
    ax.set_ylabel("Number of users")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def date_range_audit(logs: dict):
    """Print date range and record count per log type."""
    print("\n" + "="*60)
    print("DATE RANGE AUDIT")
    print("="*60)
    for name, df in logs.items():
        print(f"\n{name.upper()}")
        print(f"  Records  : {len(df):,}")
        if "timestamp" in df.columns:
            print(f"  From     : {df['timestamp'].min()}")
            print(f"  To       : {df['timestamp'].max()}")
            print(f"  Duration : {(df['timestamp'].max() - df['timestamp'].min()).days} days")

        if "user" in df.columns:
            print(f"  Users    : {df['user'].nunique()}")


def after_hours_analysis(logs: dict):
    """What % of each log type happens after hours?"""
    print("\n" + "="*60)
    print("AFTER-HOURS ACTIVITY")
    print("="*60)
    for name, df in logs.items():
        if "is_after_hours" in df.columns:
            pct = df["is_after_hours"].mean() * 100
            print(f"  {name:10s}: {pct:.1f}% of events after hours")
if __name__ == "__main__":

    print("Loading processed logs for EDA...\n")
    logs = {
        name: pd.read_parquet(config.PROCESSED_DIR / f"{name}_processed.parquet")
        for name in config.LOG_FILES
    }
    import os
    os.makedirs("reports/plots", exist_ok=True)

    # Run EDA functions
    null_audit(logs)
    summary = user_activity_summary(logs)
    date_range_audit(logs)
    after_hours_analysis(logs)

    # Create plots
    plot_hourly_distribution(logs, save_path="reports/plots/hourly.png")
    plot_user_volume_distribution(summary, save_path="reports/plots/user_volume.png")