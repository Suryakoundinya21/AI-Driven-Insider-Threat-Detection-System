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
