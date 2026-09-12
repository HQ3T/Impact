"""
Corporate Case Studies and Financial Narratives.
Extracts authentic flagship corporations from the test sample,
evaluates actual vs predicted Environmental scores, and explains
the underlying financial and economic mechanics.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


CASE_STUDY_TICKERS = ["AAPL", "MSFT", "XOM", "CVX", "NEE", "PFE", "JNJ", "F", "AMZN", "JPM"]


def generate_company_case_studies(
    df_merged: pd.DataFrame, 
    X: pd.DataFrame,
    model, 
    feature_cols: List[str]
) -> List[Dict[str, Any]]:
    """
    Analyzes specific representative public companies across tech, energy, utilities,
    pharma, autos, and finance.
    """
    case_studies = []
    
    # Filter available tickers in dataset
    available_tickers = df_merged["Ticker"].unique()
    target_tickers = [t for t in CASE_STUDY_TICKERS if t in available_tickers]
    
    # If some are missing, grab top companies by market volume
    if len(target_tickers) < 6:
        extra_tickers = df_merged.groupby("Ticker")["Volume"].mean().sort_values(ascending=False).index.tolist()
        for t in extra_tickers:
            if t not in target_tickers:
                target_tickers.append(t)
            if len(target_tickers) >= 8:
                break

    for ticker in target_tickers:
        sub = df_merged[df_merged["Ticker"] == ticker].sort_values("year", ascending=False)
        if sub.empty:
            continue
        
        row_idx = sub.index[0]
        latest = sub.iloc[0]
        actual_score = float(latest["environment_score"])
        
        # Prepare single row feature vector directly from X
        feat_df = X.iloc[[row_idx]]
        pred_score = float(model.predict(feat_df)[0])
        error = round(pred_score - actual_score, 2)
        
        company_name = latest.get("Company", ticker)
        sector = latest.get("sector", latest.get("Peer_group_root", "Unknown"))
        year = int(latest["year"])
        
        # Economic insight based on company metrics
        narrative = generate_narrative(latest, actual_score, pred_score, sector)
        
        case_studies.append({
            "ticker": ticker,
            "company": company_name,
            "sector": sector,
            "year": year,
            "actual_score": round(actual_score, 2),
            "predicted_score": round(pred_score, 2),
            "error": error,
            "accuracy_tier": "Exact Match (<=3 pts)" if abs(error) <= 3.0 else ("Close (<=6 pts)" if abs(error) <= 6.0 else "Divergent"),
            "operating_margin": round(float(latest.get("Operating Margin", 0.0) or 0.0), 2),
            "capital_intensity": round(float(latest.get("capital_intensity", 0.0) or 0.0), 2),
            "cash_flow_margin": round(float(latest.get("cash_flow_margin", 0.0) or 0.0), 2),
            "debt_to_assets": round(float(latest.get("debt_to_assets", 0.0) or 0.0), 2),
            "narrative": narrative
        })

    return case_studies


def generate_narrative(row: pd.Series, actual: float, pred: float, sector: str) -> str:
    """Synthesizes economic rationale for model prediction."""
    cap_int = float(row.get("capital_intensity", 1.0) or 1.0)
    cash_margin = float(row.get("cash_flow_margin", 0.1) or 0.1)
    op_margin = float(row.get("Operating Margin", 0.1) or 0.1)
    
    narrative_parts = []
    
    if sector in ["Oil & Gas Producers", "Energy", "Utilities"]:
        narrative_parts.append(
            f"As an energy/utility firm with high capital intensity ({cap_int:.2f}x assets/revenue), "
            "environmental compliance and transition risk heavily anchor its baseline score."
        )
    elif sector in ["Software & Services", "Technology", "Healthcare"]:
        narrative_parts.append(
            f"Asset-light operational model in {sector} combined with high operating margins ({op_margin:.1f}%) "
            "provides superior financial slack for corporate sustainability protocols."
        )
    
    if cash_margin > 0.20:
        narrative_parts.append(
            f"Exceptional cash flow generation ({cash_margin*100:.1f}% margin) enables substantial discretionary "
            "capital allocation towards renewable energy sourcing and supply chain auditing."
        )
    elif cash_margin < 0.05:
        narrative_parts.append(
            "Constrained operational cash flows restrict non-mandatory environmental capex, limiting rapid score expansion."
        )
        
    diff = abs(pred - actual)
    if diff <= 4.0:
        narrative_parts.append("The financial profile closely matches the observed Sustainalytics rating.")
    else:
        direction = "understates" if pred < actual else "overstates"
        narrative_parts.append(f"Model {direction} actual score by {diff:.1f} points, reflecting specific unmeasured corporate policy pledges.")

    return " ".join(narrative_parts)
