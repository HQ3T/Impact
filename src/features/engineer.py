"""
Feature Engineering Pipeline for Corporate Environmental Score Prediction.
Synthesizes authentic financial statements and market dynamics into high-signal,
economically interpretable features:
- Capital Intensity & Asset Heaviness
- Financial Slack & Green Transition Capacity
- Solvency & Structural Leverage
- Operational Cost Structure & Margin Spreads
- Scale & Scrutiny Signals (log-transformed)
- Sector-Relative Z-Scores (Intra-industry benchmarking)
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


RATIO_COLUMNS = [
    "Asset Turnover", "Current Ratio", "Debt/Equity Ratio", "EBIT Margin",
    "EBITDA Margin", "Gross Margin", "Net Profit Margin", "Operating Margin",
    "Pre-Tax Profit Margin", "ROA - Return On Assets", "ROE - Return On Equity",
    "Inventory Turnover Ratio", "Long-term Debt / Capital", "Days Sales In Receivables"
]

SCALE_COLUMNS = [
    "revenue", "net_income", "operating_income", "gross_profit",
    "total_assets", "total_liabilities", "operating_cash_flow", "Volume", "Close"
]


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-driven financial ratios and interaction terms.
    """
    df = df.copy()

    # 1. Capital Intensity & Asset Heaviness
    rev_safe = np.maximum(df["revenue"].fillna(1e6), 1.0)
    assets_safe = np.maximum(df["total_assets"].fillna(1e6), 1.0)
    liab_safe = np.maximum(df["total_liabilities"].fillna(0.0), 0.0)
    ocf_safe = df["operating_cash_flow"].fillna(0.0)
    net_inc_safe = df["net_income"].fillna(0.0)
    op_inc_safe = df["operating_income"].fillna(0.0)
    
    df["capital_intensity"] = assets_safe / rev_safe
    df["asset_turnover_sec"] = rev_safe / assets_safe

    # 2. Dupont 3-Way Identity Decomposition
    df["dupont_margin"] = net_inc_safe / rev_safe
    df["dupont_asset_turnover"] = rev_safe / assets_safe
    equity_proxy = np.maximum(assets_safe - liab_safe, 1.0)
    df["dupont_equity_multiplier"] = assets_safe / equity_proxy
    df["dupont_roe_calc"] = df["dupont_margin"] * df["dupont_asset_turnover"] * df["dupont_equity_multiplier"]

    # 3. Quality of Earnings & Cash Slack
    df["cash_flow_quality"] = (ocf_safe / (np.abs(net_inc_safe) + 1e5)).clip(-10.0, 10.0)
    df["cash_flow_margin"] = ocf_safe / rev_safe
    df["cash_to_assets"] = ocf_safe / assets_safe
    df["liquidity_ratio"] = df["Current Ratio"].fillna(1.0)
    df["debt_to_assets"] = liab_safe / assets_safe
    df["debt_to_ocf"] = (liab_safe / (np.abs(ocf_safe) + 1e5)).clip(0, 50.0)
    df["financial_leverage"] = df["Debt/Equity Ratio"].fillna(1.0)

    # 4. Altman Z-Score Proxy Components
    df["z_op_to_assets"] = op_inc_safe / assets_safe
    df["z_sales_to_assets"] = rev_safe / assets_safe
    df["z_market_to_debt"] = (df["Close"].fillna(50.0) * df["Volume"].fillna(1e6)) / (liab_safe + 1e5)

    # 5. Operational Cost Structure & Non-linear Interactions
    df["overhead_spread"] = (df["Gross Margin"] - df["Operating Margin"]).fillna(0.0)
    df["margin_x_cap_int"] = df["Operating Margin"].fillna(0.0) * df["capital_intensity"]
    df["slack_x_solvency"] = df["cash_to_assets"] * (1.0 - df["debt_to_assets"])
    df["ebitda_margin"] = df["EBITDA Margin"].fillna(0.0)
    df["operating_margin"] = df["Operating Margin"].fillna(0.0)
    df["net_margin"] = df["Net Profit Margin"].fillna(0.0)
    df["return_on_assets"] = df["ROA - Return On Assets"].fillna(0.0)
    df["return_on_equity"] = df["ROE - Return On Equity"].fillna(0.0)

    # 5. Scale & Public Visibility Signals (Log transforms)
    for col in ["revenue", "total_assets", "total_liabilities", "operating_cash_flow", "Volume", "Close"]:
        if col in df.columns:
            df[f"log_{col.lower()}"] = np.log1p(df[col].clip(lower=0).fillna(0))

    # 6. Sector-Relative Z-Scores (Intra-industry benchmarking)
    benchmark_metrics = [
        "operating_margin", "return_on_assets", "capital_intensity", 
        "cash_flow_margin", "debt_to_assets"
    ]
    
    sector_col = "sector" if "sector" in df.columns else "Peer_group_root"
    for metric in benchmark_metrics:
        if metric in df.columns:
            sec_mean = df.groupby(sector_col)[metric].transform("mean")
            sec_std = df.groupby(sector_col)[metric].transform("std").replace(0, 1.0).fillna(1.0)
            df[f"{metric}_sec_z"] = ((df[metric] - sec_mean) / sec_std).clip(-3.5, 3.5)

    # 7. Environmental Performance Tiers (Target Classification)
    # Tier A: Environmental Leader (>= 65)
    # Tier B: Moderate / Transitioning (48 to 65)
    # Tier C: Environmental Laggard (< 48)
    if "environment_score" in df.columns:
        conditions = [
            df["environment_score"] >= 65.0,
            (df["environment_score"] >= 48.0) & (df["environment_score"] < 65.0),
            df["environment_score"] < 48.0
        ]
        tier_names = ["Leader", "Moderate", "Laggard"]
        df["environmental_tier"] = np.select(conditions, tier_names, default="Moderate")
        tier_codes = [0, 1, 2] # 0: Leader, 1: Moderate, 2: Laggard
        df["environmental_tier_code"] = np.select(conditions, [2, 1, 0], default=1) # Higher is better: 2=Leader, 1=Moderate, 0=Laggard

    return df


def get_feature_column_names(df: pd.DataFrame) -> List[str]:
    """
    Returns the complete list of numeric input feature names for modeling.
    """
    feature_candidates = [
        # Base financial ratios
        "Asset Turnover", "Current Ratio", "Debt/Equity Ratio", "EBIT Margin",
        "EBITDA Margin", "Gross Margin", "Net Profit Margin", "Operating Margin",
        "Pre-Tax Profit Margin", "ROA - Return On Assets", "ROE - Return On Equity",
        "Inventory Turnover Ratio", "Long-term Debt / Capital", "Days Sales In Receivables",
        # Dupont 3-way analysis
        "dupont_margin", "dupont_asset_turnover", "dupont_equity_multiplier", "dupont_roe_calc",
        # Cash slack & quality of earnings
        "cash_flow_quality", "cash_flow_margin", "cash_to_assets", "capital_intensity", 
        "debt_to_assets", "debt_to_ocf", "financial_leverage", "liquidity_ratio",
        # Altman Z components
        "z_op_to_assets", "z_sales_to_assets", "z_market_to_debt",
        # Spreads & non-linear interactions
        "overhead_spread", "margin_x_cap_int", "slack_x_solvency", "ebitda_margin", 
        "operating_margin", "net_margin", "return_on_assets", "return_on_equity",
        # Log scale indicators
        "log_revenue", "log_total_assets", "log_total_liabilities", 
        "log_operating_cash_flow", "log_volume", "log_close",
        # Sector Z-scores
        "operating_margin_sec_z", "return_on_assets_sec_z", 
        "capital_intensity_sec_z", "cash_flow_margin_sec_z", "debt_to_assets_sec_z"
    ]
    return [col for col in feature_candidates if col in df.columns]


def prepare_modeling_matrices(
    df: pd.DataFrame, 
    include_sector_dummies: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Generates feature matrix X, continuous target y_reg, and categorical target y_clf.
    Returns:
        X (DataFrame with features + sector encodings)
        y_reg (Series of continuous environmental_score)
        y_clf (Series of environmental_tier_code: 0=Laggard, 1=Moderate, 2=Leader)
        feature_names (List of all column names in X)
    """
    df_feat = create_engineered_features(df)
    numeric_features = get_feature_column_names(df_feat)
    
    X_num = df_feat[numeric_features].copy()

    if include_sector_dummies:
        sector_col = "sector" if "sector" in df_feat.columns else "Peer_group_root"
        sector_dummies = pd.get_dummies(df_feat[sector_col], prefix="sector", drop_first=False, dtype=float)
        X = pd.concat([X_num, sector_dummies], axis=1)
    else:
        X = X_num

    y_reg = df_feat["environment_score"] if "environment_score" in df_feat.columns else None
    y_clf = df_feat["environmental_tier_code"] if "environmental_tier_code" in df_feat.columns else None

    return X, y_reg, y_clf, X.columns.tolist()


if __name__ == "__main__":
    from src.data.preprocess import PROCESSED_DIR
    data_path = PROCESSED_DIR / "company_financial_esg_dataset.parquet"
    if data_path.exists():
        df_raw = pd.read_parquet(data_path)
        X, y_reg, y_clf, feature_cols = prepare_modeling_matrices(df_raw)
        print(f"[Feature Engineer] Generated {X.shape[1]} features across {X.shape[0]} samples.")
        print(f"  -> Features sample: {feature_cols[:8]} ... (+ {len(feature_cols)-8} more)")
        print(f"  -> Target y_reg mean: {y_reg.mean():.2f}, std: {y_reg.std():.2f}")
        print(f"  -> Tier distribution:\n{df_raw['environmental_tier'].value_counts() if 'environmental_tier' in df_raw.columns else 'N/A'}")
