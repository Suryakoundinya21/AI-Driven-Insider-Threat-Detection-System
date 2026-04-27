import numpy as np
import pandas as pd
import logging
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix,
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
    logger.info(f"Unique insider users: {gt['user'].nunique()}")
    logger.info(f"Scenarios: {sorted(gt['scenario'].unique())}")
    return gt


def attach_ground_truth(df: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    if gt.empty:
        logger.warning("No ground truth — adding dummy label column (all zeros)")
        df["is_insider"] = 0
        return df

    df = df.copy()
    df["date_only"] = pd.to_datetime(df["date_only"])

    insider_users = set(gt["user"].str.strip().str.lower().unique())
    logger.info(f"Total insider users in ground truth: {len(insider_users)}")

    gt_parsed = gt.copy()
    gt_parsed["user"] = gt_parsed["user"].str.strip().str.lower()

    has_window = "start" in gt_parsed.columns and "end" in gt_parsed.columns

    if has_window:
        gt_parsed["start_dt"] = pd.to_datetime(gt_parsed["start"], errors="coerce")
        gt_parsed["end_dt"]   = pd.to_datetime(gt_parsed["end"],   errors="coerce")
        logger.info("Using date-window matching for ground truth labels")

    df["is_insider"] = 0

    for _, gt_row in gt_parsed.iterrows():
        user = gt_row["user"]
        mask = df["user"] == user

        if has_window and pd.notna(gt_row.get("start_dt")):
            start     = gt_row["start_dt"].date()
            end       = gt_row["end_dt"].date()
            date_mask = (
                (df["date_only"].dt.date >= start) &
                (df["date_only"].dt.date <= end)
            )
            df.loc[mask & date_mask, "is_insider"] = 1
        else:
            df.loc[mask, "is_insider"] = 1

    n_insider = df["is_insider"].sum()
    n_users   = df[df["is_insider"] == 1]["user"].nunique()
    logger.info(f"Insider sessions labeled : {n_insider:,} ({n_insider/len(df)*100:.2f}%)")
    logger.info(f"Insider users matched    : {n_users}")

    if n_insider == 0:
        logger.warning("NO sessions matched — checking user ID format...")
        sample_gt   = list(insider_users)[:5]
        sample_df   = df["user"].unique()[:5].tolist()
        logger.warning(f"GT users sample  : {sample_gt}")
        logger.warning(f"DF users sample  : {sample_df}")

    return df


def evaluate_model(
    df: pd.DataFrame,
    flag_col: str,
    score_col: str,
    model_name: str,
) -> dict:

    if "is_insider" not in df.columns or df["is_insider"].sum() == 0:
        logger.warning(f"No ground truth labels — skipping {model_name}")
        return {}

    y_true  = df["is_insider"].values
    y_pred  = df[flag_col].values
    y_score = df[score_col].values

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, int(y_pred.sum())

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

    print(f"\n{'='*55}")
    print(f"MODEL : {model_name}")
    print(f"{'='*55}")
    print(f"  Precision     : {metrics['precision']}")
    print(f"  Recall        : {metrics['recall']}")
    print(f"  F1 Score      : {metrics['f1']}")
    print(f"  ROC-AUC       : {metrics['roc_auc']}")
    print(f"  Avg Precision : {metrics['avg_precision']}")
    print(f"  FPR           : {metrics['fpr']}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    return metrics


def compare_models(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results)
    print("\n" + "="*75)
    print("MODEL COMPARISON TABLE")
    print("="*75)
    cols  = ["model","precision","recall","f1","roc_auc","avg_precision","fpr","tp","fp","fn"]
    avail = [c for c in cols if c in df.columns]
    print(df[avail].to_string(index=False))
    return df
