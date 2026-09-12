"""
Explainability & Interpretability Module.
Computes permutation feature importances and isolates the core financial drivers
that govern corporate environmental ratings:
- Financial Slack & Cash Flow Margin
- Capital Intensity & Tangible Assets
- Solvency & Debt Burden
- Operating Spread & Efficiency
- Industry Sector Context
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def compute_model_feature_importances(
    model, 
    X: pd.DataFrame, 
    y: pd.Series, 
    n_repeats: int = 5, 
    random_state: int = 42
) -> pd.DataFrame:
    """
    Computes permutation feature importance to measure real out-of-sample impact of each feature.
    """
    result = permutation_importance(
        model, 
        X, 
        y, 
        n_repeats=n_repeats, 
        random_state=random_state, 
        n_jobs=-1
    )
    
    df_importance = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    # Add economic interpretation tags
    df_importance["category"] = df_importance["feature"].apply(categorize_feature)
    return df_importance


def categorize_feature(feature_name: str) -> str:
    """Categorizes features into economic financial domains."""
    name = feature_name.lower()
    if name.startswith("sector_"):
        return "Sector & Industry"
    elif any(k in name for k in ["cash", "slack", "ocf"]):
        return "Cash Flow & Slack"
    elif any(k in name for k in ["asset", "capital", "turnover", "inventory"]):
        return "Capital & Asset Intensity"
    elif any(k in name for k in ["debt", "liab", "equity", "leverage", "current"]):
        return "Solvency & Capital Structure"
    elif any(k in name for k in ["margin", "ebit", "ebitda", "spread", "profit", "overhead"]):
        return "Profitability & Cost Structure"
    elif any(k in name for k in ["log_", "volume", "close", "revenue"]):
        return "Firm Scale & Visibility"
    else:
        return "Other Financial Signals"
