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
