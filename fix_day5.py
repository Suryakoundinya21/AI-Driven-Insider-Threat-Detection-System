import os

files = {}

# ─────────────────────────────────────────────
# ISOLATION FOREST MODEL
# ─────────────────────────────────────────────
files["src/modeling/isolation_forest.py"] = """\
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from src.config import config

logger = logging.getLogger(__name__)


def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
) -> IsolationForest:

    logger.info(f"Training Isolation Forest...")
    logger.info(f"  Samples      : {len(X_train):,}")
    logger.info(f"  Contamination: {contamination}")
    logger.info(f"  n_estimators : {n_estimators}")

    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train)

    save_path = config.MODELS_DIR / "isolation_forest" / "if_model.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, save_path)
    logger.info(f"IF model saved: {save_path}")
    return clf


def score_isolation_forest(
    clf: IsolationForest,
    X: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # Raw scores: more negative = more anomalous
    raw_scores = clf.score_samples(X)
    predictions = clf.predict(X)   # -1 = anomaly, 1 = normal

    # Normalize to 0-1 (flip so higher = more anomalous)
    lo, hi = raw_scores.min(), raw_scores.max()
    if hi - lo == 0:
        normalized = np.zeros_like(raw_scores)
    else:
        normalized = 1 - (raw_scores - lo) / (hi - lo)

    df["if_raw_score"]     = raw_scores
    df["if_anomaly_score"] = normalized
    df["if_anomaly_flag"]  = (predictions == -1).astype(int)

    n = df["if_anomaly_flag"].sum()
    logger.info(f"IF anomalies flagged: {n:,} ({n/len(df)*100:.2f}%)")
    return df


def load_if_model() -> IsolationForest:
    path = config.MODELS_DIR / "isolation_forest" / "if_model.pkl"
    clf  = joblib.load(path)
    logger.info(f"IF model loaded from {path}")
    return clf
"""

# ─────────────────────────────────────────────
# ENSEMBLE SCORER
# ─────────────────────────────────────────────
files["src/modeling/ensemble.py"] = """\
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def build_ensemble_score(
    df: pd.DataFrame,
    ae_weight: float = 0.5,
    if_weight: float = 0.5,
) -> pd.DataFrame:

    df = df.copy()

    if "ae_anomaly_score" not in df.columns:
        raise ValueError("ae_anomaly_score missing. Run AE scorer first.")
    if "if_anomaly_score" not in df.columns:
        raise ValueError("if_anomaly_score missing. Run IF scorer first.")

    # Weighted ensemble score
    df["ensemble_score"] = (
        ae_weight * df["ae_anomaly_score"] +
        if_weight * df["if_anomaly_score"]
    )

    # Flag if EITHER model flags it (union — high recall)
    df["ensemble_flag_union"] = (
        (df["ae_anomaly_flag"] == 1) | (df["if_anomaly_flag"] == 1)
    ).astype(int)

    # Flag if BOTH models flag it (intersection — high precision)
    df["ensemble_flag_intersect"] = (
        (df["ae_anomaly_flag"] == 1) & (df["if_anomaly_flag"] == 1)
    ).astype(int)

    # Top percentile ensemble flag (score > 95th percentile)
    threshold_95 = df["ensemble_score"].quantile(0.95)
    df["ensemble_flag_top5pct"] = (
        df["ensemble_score"] >= threshold_95
    ).astype(int)

    n_union     = df["ensemble_flag_union"].sum()
    n_intersect = df["ensemble_flag_intersect"].sum()
    n_top5      = df["ensemble_flag_top5pct"].sum()

    logger.info(f"Ensemble union flags        : {n_union:,} ({n_union/len(df)*100:.2f}%)")
    logger.info(f"Ensemble intersection flags : {n_intersect:,} ({n_intersect/len(df)*100:.2f}%)")
    logger.info(f"Ensemble top-5% flags       : {n_top5:,} ({n_top5/len(df)*100:.2f}%)")

    return df
"""

# ─────────────────────────────────────────────
# EVALUATOR (with ground truth)
# ─────────────────────────────────────────────
files["src/modeling/evaluator.py"] = """\
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)
from src.config import config

logger = logging.getLogger(__name__)


def load_ground_truth() -> pd.DataFrame:
    path = config.RAW_DATA_DIR / "answers" / "insiders.csv"
    if not path.exists():
        logger.warning(f"Ground truth not found at {path}")
        return pd.DataFrame()
    gt = pd.read_csv(path)
    gt.columns = [c.strip().lower() for c in gt.columns]
    logger.info(f"Ground truth loaded: {len(gt)} records")
    logger.info(f"Columns: {list(gt.columns)}")
    return gt


def attach_ground_truth(
    df: pd.DataFrame,
    gt: pd.DataFrame,
) -> pd.DataFrame:

    if gt.empty:
        logger.warning("No ground truth — adding dummy label column (all zeros)")
        df["is_insider"] = 0
        return df

    df = df.copy()
    df["date_only"] = pd.to_datetime(df["date_only"]).dt.date.astype(str)

    # Build insider label: user appears in ground truth
    insider_users = set(gt["user"].str.strip().str.lower().unique())
    df["is_insider"] = df["user"].isin(insider_users).astype(int)

    n_insider_sessions = df["is_insider"].sum()
    logger.info(f"Insider sessions labeled: {n_insider_sessions:,} "
                f"({n_insider_sessions/len(df)*100:.2f}%)")
    return df


def evaluate_model(
    df: pd.DataFrame,
    flag_col: str,
    score_col: str,
    model_name: str,
) -> dict:

    if "is_insider" not in df.columns or df["is_insider"].sum() == 0:
        logger.warning(f"No ground truth labels — skipping evaluation for {model_name}")
        return {}

    y_true  = df["is_insider"].values
    y_pred  = df[flag_col].values
    y_score = df[score_col].values

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "model"         : model_name,
        "precision"     : round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"        : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1"            : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc"       : round(roc_auc_score(y_true, y_score), 4),
        "avg_precision" : round(average_precision_score(y_true, y_score), 4),
        "tp"            : int(tp),
        "fp"            : int(fp),
        "tn"            : int(tn),
        "fn"            : int(fn),
        "fpr"           : round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),
    }

    print(f"\\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")
    print(f"  Precision     : {metrics['precision']}")
    print(f"  Recall        : {metrics['recall']}")
    print(f"  F1 Score      : {metrics['f1']}")
    print(f"  ROC-AUC       : {metrics['roc_auc']}")
    print(f"  Avg Precision : {metrics['avg_precision']}")
    print(f"  FPR           : {metrics['fpr']}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    return metrics


def compare_models(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results)
    print("\\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    print(df[["model","precision","recall","f1",
              "roc_auc","fpr","tp","fp","fn"]].to_string(index=False))
    return df
"""

# ─────────────────────────────────────────────
# RUN DAY 5 SCRIPT
# ─────────────────────────────────────────────
files["scripts/run_day5.py"] = """\
import sys, os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

from src.config import config
from src.modeling.preprocessor    import prepare_data, split_normal_sessions
from src.modeling.isolation_forest import train_isolation_forest, score_isolation_forest
from src.modeling.ensemble        import build_ensemble_score
from src.modeling.evaluator       import (
    load_ground_truth, attach_ground_truth,
    evaluate_model, compare_models
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_score_comparison(df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(df["ae_anomaly_score"], bins=60,
                 color="#378ADD", alpha=0.7, label="AE Score")
    axes[0].set_title("Autoencoder Anomaly Score")
    axes[0].set_xlabel("Score (0-1)")
    axes[0].set_ylabel("Count")

    axes[1].hist(df["if_anomaly_score"], bins=60,
                 color="#1D9E75", alpha=0.7, label="IF Score")
    axes[1].set_title("Isolation Forest Anomaly Score")
    axes[1].set_xlabel("Score (0-1)")

    axes[2].hist(df["ensemble_score"], bins=60,
                 color="#D85A30", alpha=0.7, label="Ensemble")
    axes[2].set_title("Ensemble Score (AE + IF)")
    axes[2].set_xlabel("Score (0-1)")

    plt.suptitle("Anomaly Score Distributions — All Models", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_venn_overlap(df, save_path):
    ae_flags = set(df[df["ae_anomaly_flag"] == 1].index)
    if_flags = set(df[df["if_anomaly_flag"] == 1].index)

    ae_only   = len(ae_flags - if_flags)
    if_only   = len(if_flags - ae_flags)
    both      = len(ae_flags & if_flags)

    fig, ax = plt.subplots(figsize=(7, 5))
    from matplotlib.patches import Circle
    c1 = Circle((0.38, 0.5), 0.28, color="#378ADD", alpha=0.5, label=f"AE only: {ae_only}")
    c2 = Circle((0.62, 0.5), 0.28, color="#1D9E75", alpha=0.5, label=f"IF only: {if_only}")
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.text(0.28, 0.5, str(ae_only),  ha="center", va="center",
            fontsize=14, fontweight="bold", color="white")
    ax.text(0.72, 0.5, str(if_only),  ha="center", va="center",
            fontsize=14, fontweight="bold", color="white")
    ax.text(0.50, 0.5, str(both),     ha="center", va="center",
            fontsize=14, fontweight="bold", color="white")
    ax.text(0.28, 0.82, "Autoencoder", ha="center", fontsize=11)
    ax.text(0.72, 0.82, "Isolation Forest", ha="center", fontsize=11)
    ax.text(0.50, 0.18, f"Both: {both}", ha="center", fontsize=10,
            color="#555")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Model Agreement — Flagged Sessions Overlap", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_top_users(df, save_path):
    top = (
        df.groupby("user")["ensemble_score"]
        .max()
        .sort_values(ascending=False)
        .head(20)
    )
    colors = ["#D85A30" if s >= 0.8 else "#378ADD" for s in top.values]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(top)), top.values, color=colors)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Max ensemble score")
    ax.set_title("Top 20 Users by Maximum Ensemble Anomaly Score")
    ax.axhline(y=0.8, color="red", linestyle="--", linewidth=1,
               label="High-risk threshold (0.8)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_roc_curves(df, results, save_path):
    if not results or "is_insider" not in df.columns:
        logger.warning("Skipping ROC — no ground truth")
        return
    if df["is_insider"].sum() == 0:
        logger.warning("Skipping ROC — no insider sessions labeled")
        return

    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(7, 6))

    configs = [
        ("ae_anomaly_score",  "#378ADD", "Autoencoder"),
        ("if_anomaly_score",  "#1D9E75", "Isolation Forest"),
        ("ensemble_score",    "#D85A30", "Ensemble"),
    ]
    for score_col, color, label in configs:
        if score_col not in df.columns:
            continue
        fpr, tpr, _ = roc_curve(df["is_insider"], df[score_col])
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)

    ax.plot([0,1],[0,1],"k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {save_path}")


def main():
    os.makedirs("reports/plots", exist_ok=True)

    # 1. Load AE-scored feature matrix
    logger.info("Loading AE-scored feature matrix...")
    df = pd.read_parquet(
        config.FEATURES_DIR / "feature_matrix_ae_scored.parquet"
    )
    logger.info(f"Loaded: {df.shape}")

    # 2. Prepare features for IF (reuse AE scaler)
    normal_df, all_df = split_normal_sessions(df)
    X_train, features, scaler = prepare_data(normal_df, fit=False)
    X_all,   _,        _      = prepare_data(all_df,   scaler=scaler, fit=False)

    # 3. Train Isolation Forest
    clf = train_isolation_forest(
        X_train,
        contamination=config.CONTAMINATION,
        n_estimators=200,
    )

    # 4. Score all sessions with IF
    df_if = score_isolation_forest(clf, X_all, all_df)

    # 5. Merge AE scores into IF-scored df
    df_if["ae_anomaly_score"] = df["ae_anomaly_score"].values
    df_if["ae_anomaly_flag"]  = df["ae_anomaly_flag"].values
    df_if["reconstruction_error"] = df["reconstruction_error"].values

    # 6. Build ensemble score
    df_ensemble = build_ensemble_score(df_if, ae_weight=0.5, if_weight=0.5)

    # 7. Load and attach ground truth
    gt = load_ground_truth()
    df_labeled = attach_ground_truth(df_ensemble, gt)

    # 8. Evaluate all models
    results = []
    for flag_col, score_col, name in [
        ("ae_anomaly_flag",        "ae_anomaly_score", "Autoencoder"),
        ("if_anomaly_flag",        "if_anomaly_score", "Isolation Forest"),
        ("ensemble_flag_union",    "ensemble_score",   "Ensemble (Union)"),
        ("ensemble_flag_intersect","ensemble_score",   "Ensemble (Intersect)"),
        ("ensemble_flag_top5pct",  "ensemble_score",   "Ensemble (Top 5%)"),
    ]:
        r = evaluate_model(df_labeled, flag_col, score_col, name)
        if r:
            results.append(r)

    if results:
        comparison_df = compare_models(results)
        comparison_df.to_csv("reports/model_comparison.csv", index=False)
        logger.info("Model comparison saved: reports/model_comparison.csv")

    # 9. Plots
    plot_score_comparison(df_ensemble, "reports/plots/score_comparison.png")
    plot_venn_overlap(df_ensemble,     "reports/plots/venn_overlap.png")
    plot_top_users(df_ensemble,        "reports/plots/top_users_risk.png")
    plot_roc_curves(df_labeled, results,"reports/plots/roc_curves.png")

    # 10. Summary
    print("\\n" + "="*60)
    print("DAY 5 SUMMARY")
    print("="*60)
    print(f"Total sessions       : {len(df_ensemble):,}")
    print(f"AE anomalies         : {df_ensemble['ae_anomaly_flag'].sum():,} "
          f"({df_ensemble['ae_anomaly_flag'].mean()*100:.2f}%)")
    print(f"IF anomalies         : {df_ensemble['if_anomaly_flag'].sum():,} "
          f"({df_ensemble['if_anomaly_flag'].mean()*100:.2f}%)")
    print(f"Ensemble union       : {df_ensemble['ensemble_flag_union'].sum():,} "
          f"({df_ensemble['ensemble_flag_union'].mean()*100:.2f}%)")
    print(f"Ensemble intersection: {df_ensemble['ensemble_flag_intersect'].sum():,} "
          f"({df_ensemble['ensemble_flag_intersect'].mean()*100:.2f}%)")

    print("\\nTop 15 highest-risk sessions (ensemble):")
    cols = ["user", "date_only", "ensemble_score",
            "ae_anomaly_score", "if_anomaly_score",
            "device_count", "email_to_external",
            "http_suspicious", "composite_risk_score"]
    avail = [c for c in cols if c in df_ensemble.columns]
    print(df_ensemble.nlargest(15, "ensemble_score")[avail].to_string(index=False))

    # 11. Save final scored dataset
    out = config.FEATURES_DIR / "feature_matrix_ensemble_scored.parquet"
    df_ensemble.to_parquet(out, index=False)
    logger.info(f"Final ensemble dataset saved: {out}")
    print("\\nDay 5 Complete.")


if __name__ == "__main__":
    main()
"""

for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\nAll Day 5 files created. Run: python scripts/run_day5.py")