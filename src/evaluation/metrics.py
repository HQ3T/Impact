"""
Evaluation Metrics Module for Environmental Score Modeling.
Computes comprehensive statistical performance indicators for:
- Continuous Environmental Score Regression (R2, MAE, RMSE, Pearson r, Spearman rank)
- Environmental Leadership Tier Classification (Accuracy, Macro F1, Per-Tier Metrics)
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes rigorous statistical regression metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Filter out NaNs if any
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]

    r2 = r2_score(y_t, y_p)
    mae = mean_absolute_error(y_t, y_p)
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    ev = explained_variance_score(y_t, y_p)
    
    pr_corr, _ = pearsonr(y_t, y_p)
    sp_corr, _ = spearmanr(y_t, y_p)

    # Within 5-point and 10-point accuracy tolerance
    within_5 = float(np.mean(np.abs(y_t - y_p) <= 5.0) * 100)
    within_10 = float(np.mean(np.abs(y_t - y_p) <= 10.0) * 100)

    return {
        "R2_Score": round(float(r2), 4),
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "Explained_Variance": round(float(ev), 4),
        "Pearson_Correlation": round(float(pr_corr), 4),
        "Spearman_Rank_Correlation": round(float(sp_corr), 4),
        "Within_5_Points_Pct": round(within_5, 2),
        "Within_10_Points_Pct": round(within_10, 2)
    }


def compute_tier_classification_metrics(y_true_code: np.ndarray, y_pred_code: np.ndarray) -> Dict[str, Any]:
    """
    Computes classification metrics for 3-Tier Environmental Leadership:
    0: Laggard, 1: Moderate, 2: Leader.
    """
    y_t = np.asarray(y_true_code, dtype=int)
    y_p = np.asarray(y_pred_code, dtype=int)

    acc = accuracy_score(y_t, y_p)
    f1_macro = f1_score(y_t, y_p, average="macro", zero_division=0)
    prec_macro = precision_score(y_t, y_p, average="macro", zero_division=0)
    rec_macro = recall_score(y_t, y_p, average="macro", zero_division=0)
    cm = confusion_matrix(y_t, y_p, labels=[0, 1, 2])

    return {
        "Accuracy": round(float(acc), 4),
        "Macro_F1": round(float(f1_macro), 4),
        "Macro_Precision": round(float(prec_macro), 4),
        "Macro_Recall": round(float(rec_macro), 4),
        "Confusion_Matrix": cm.tolist()
    }
