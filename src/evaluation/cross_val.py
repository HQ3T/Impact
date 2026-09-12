"""
Cross-Validation and Benchmarking Suite.
Runs rigorous multi-model validation across:
1. Ridge Linear Baseline
2. Random Forest Regressor
3. HistGradientBoosting
4. LightGBM Regressor
5. GPU XGBoost (NVIDIA CUDA)
6. GPU CatBoost (NVIDIA CUDA)
7. Tabular Neural Network (PyTorch)
8. Super GPU Ensemble Blend
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold

from src.models.baselines import build_ridge_pipeline
from src.models.tree_models import (
    build_hist_gradient_boosting_regressor, 
    build_random_forest_regressor,
    build_gpu_xgboost_regressor,
    build_gpu_catboost_regressor,
    build_lightgbm_regressor
)
from src.models.neural_net import TabularNeuralNetRegressor
from src.evaluation.metrics import compute_regression_metrics, compute_tier_classification_metrics


def run_multi_model_cv(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_splits: int = 5, 
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Runs 5-fold cross-validation on all core and GPU model architectures.
    """
    models = {
        "Ridge Linear Baseline": build_ridge_pipeline(),
        "Random Forest": build_random_forest_regressor(random_state=random_state),
        "Tabular Neural Net": TabularNeuralNetRegressor(epochs=25, random_state=random_state),
        "HistGradientBoosting": build_hist_gradient_boosting_regressor(random_state=random_state),
        "LightGBM": build_lightgbm_regressor(random_state=random_state),
        "GPU XGBoost (CUDA)": build_gpu_xgboost_regressor(random_state=random_state),
        "GPU CatBoost (CUDA)": build_gpu_catboost_regressor(random_state=random_state)
    }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}

    for name, model in models.items():
        print(f"[CV] Evaluating {name}...")
        fold_metrics = []
        oof_preds = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model.fit(X_tr, y_tr)
            preds = np.clip(model.predict(X_val), 20.0, 98.0)
            oof_preds[val_idx] = preds

            m = compute_regression_metrics(y_val.values, preds)
            fold_metrics.append(m)

        avg_metrics = {
            k: round(float(np.mean([fm[k] for fm in fold_metrics])), 4)
            for k in fold_metrics[0].keys()
        }
        results[name] = {
            "metrics": avg_metrics,
            "oof_predictions": oof_preds
        }
        print(f"  -> {name:<22} | R2: {avg_metrics['R2_Score']:.3f}, MAE: {avg_metrics['MAE']:.2f}, RMSE: {avg_metrics['RMSE']:.2f}")

    # Compute Super GPU Ensemble Blend from OOF predictions
    p_xgb = results["GPU XGBoost (CUDA)"]["oof_predictions"]
    p_cat = results["GPU CatBoost (CUDA)"]["oof_predictions"]
    p_lgb = results["LightGBM"]["oof_predictions"]
    p_hgb = results["HistGradientBoosting"]["oof_predictions"]
    p_nn = results["Tabular Neural Net"]["oof_predictions"]

    ensemble_oof = np.clip(
        0.40 * p_xgb + 0.20 * p_cat + 0.20 * p_lgb + 0.10 * p_hgb + 0.10 * p_nn,
        20.0, 98.0
    )

    ensemble_metrics = compute_regression_metrics(y.values, ensemble_oof)
    results["Super GPU Ensemble (XGB+Cat+LGB+NN)"] = {
        "metrics": ensemble_metrics,
        "oof_predictions": ensemble_oof
    }
    print(f"  -> {'Super GPU Ensemble':<22} | R2: {ensemble_metrics['R2_Score']:.3f}, MAE: {ensemble_metrics['MAE']:.2f}, RMSE: {ensemble_metrics['RMSE']:.2f}")

    return results


def run_group_cv(
    X: pd.DataFrame, 
    y: pd.Series, 
    groups: pd.Series, 
    n_splits: int = 5
) -> Dict[str, float]:
    """
    Evaluates GPU XGBoost generalization to completely unseen companies (GroupKFold by Ticker).
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = build_gpu_xgboost_regressor()
        model.fit(X_tr, y_tr)
        preds = np.clip(model.predict(X_val), 20.0, 98.0)

        m = compute_regression_metrics(y_val.values, preds)
        fold_metrics.append(m)

    avg_metrics = {
        k: round(float(np.mean([fm[k] for fm in fold_metrics])), 4)
        for k in fold_metrics[0].keys()
    }
    return avg_metrics
