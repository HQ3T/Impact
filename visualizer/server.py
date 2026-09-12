"""
FastAPI Backend Server for Corporate Environmental Score Predictor & Visualizer.
Serves:
- Real-time ML inference for custom corporate financial profiles
- Precomputed benchmarks, feature importances, and historical S&P 500 company records
- Interactive Green Transition "What-If" simulation
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(
    title="Corporate Environmental Score Predictor API",
    description="Predicts corporate ESG Environmental ratings from authentic financial fundamentals."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR.parent / "models" / "best_model.joblib"
SUMMARY_PATH = STATIC_DIR / "pipeline_summary.json"

# In-memory model cache
loaded_model = None
model_metadata = None


def get_model():
    global loaded_model, model_metadata
    if loaded_model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run pipeline first.")
        artifact = joblib.load(MODEL_PATH)
        loaded_model = artifact["model"]
        model_metadata = artifact["metadata"]
    return loaded_model, model_metadata


class PredictionRequest(BaseModel):
    sector: str = Field(default="Software & Services", description="Industry Sector")
    operating_margin: float = Field(default=0.25, description="Operating Margin (e.g. 0.25 = 25%)")
    capital_intensity: float = Field(default=1.2, description="Total Assets / Revenue")
    cash_flow_margin: float = Field(default=0.20, description="Operating Cash Flow / Revenue")
    debt_to_assets: float = Field(default=0.45, description="Total Liabilities / Total Assets")
    asset_turnover: float = Field(default=0.85, description="Revenue / Total Assets")
    current_ratio: float = Field(default=1.60, description="Current Assets / Current Liabilities")
    debt_equity_ratio: float = Field(default=0.80, description="Debt to Equity Ratio")
    ebitda_margin: float = Field(default=0.30, description="EBITDA Margin")
    gross_margin: float = Field(default=0.55, description="Gross Margin")
    return_on_assets: float = Field(default=0.10, description="ROA")
    revenue: float = Field(default=25_000_000_000, description="Annual Revenue in USD")
    total_assets: float = Field(default=30_000_000_000, description="Total Assets in USD")


@app.get("/api/summary")
def get_pipeline_summary():
    """Returns dataset summary, benchmarks, feature importances, and company data."""
    if not SUMMARY_PATH.exists():
        raise HTTPException(status_code=404, detail="Pipeline summary payload not found. Run pipeline first.")
    with open(SUMMARY_PATH, "r") as f:
        data = json.load(f)
    return data


@app.post("/api/predict")
def predict_score(req: PredictionRequest):
    """
    Runs real-time machine learning inference for user-provided financial profile.
    """
    model, metadata = get_model()
    feature_cols = metadata["feature_cols"]

    # Construct input vector matching trained features
    row_dict = {}
    
    # Base ratios
    row_dict["Asset Turnover"] = req.asset_turnover
    row_dict["Current Ratio"] = req.current_ratio
    row_dict["Debt/Equity Ratio"] = req.debt_equity_ratio
    row_dict["EBIT Margin"] = req.operating_margin
    row_dict["EBITDA Margin"] = req.ebitda_margin
    row_dict["Gross Margin"] = req.gross_margin
    row_dict["Net Profit Margin"] = req.operating_margin * 0.75
    row_dict["Operating Margin"] = req.operating_margin
    row_dict["Pre-Tax Profit Margin"] = req.operating_margin * 0.90
    row_dict["ROA - Return On Assets"] = req.return_on_assets
    row_dict["ROE - Return On Equity"] = req.return_on_assets * (1.0 + req.debt_equity_ratio)
    row_dict["Inventory Turnover Ratio"] = 5.0
    row_dict["Long-term Debt / Capital"] = req.debt_to_assets * 0.6
    row_dict["Days Sales In Receivables"] = 55.0

    # Engineered interaction metrics
    row_dict["capital_intensity"] = req.capital_intensity
    row_dict["asset_turnover_sec"] = req.asset_turnover
    row_dict["cash_flow_margin"] = req.cash_flow_margin
    row_dict["cash_to_assets"] = req.cash_flow_margin * req.asset_turnover
    row_dict["liquidity_ratio"] = req.current_ratio
    row_dict["debt_to_assets"] = req.debt_to_assets
    row_dict["financial_leverage"] = req.debt_equity_ratio
    row_dict["overhead_spread"] = req.gross_margin - req.operating_margin
    row_dict["ebitda_margin"] = req.ebitda_margin
    row_dict["operating_margin"] = req.operating_margin
    row_dict["net_margin"] = req.operating_margin * 0.75
    row_dict["return_on_assets"] = req.return_on_assets
    row_dict["return_on_equity"] = req.return_on_assets * (1.0 + req.debt_equity_ratio)

    # Dupont 3-Way Identity
    net_margin_est = req.operating_margin * 0.75
    row_dict["dupont_margin"] = net_margin_est
    row_dict["dupont_asset_turnover"] = req.asset_turnover
    equity_proxy = max(req.total_assets * (1.0 - req.debt_to_assets), 1.0)
    row_dict["dupont_equity_multiplier"] = req.total_assets / equity_proxy
    row_dict["dupont_roe_calc"] = net_margin_est * req.asset_turnover * row_dict["dupont_equity_multiplier"]

    # Cash slack, quality of earnings, & Altman Z proxies
    row_dict["cash_flow_quality"] = float(np.clip(req.cash_flow_margin / (abs(net_margin_est) + 0.01), -5.0, 5.0))
    liab_est = req.total_assets * req.debt_to_assets
    ocf_est = req.revenue * req.cash_flow_margin
    row_dict["debt_to_ocf"] = float(np.clip(liab_est / (abs(ocf_est) + 1e5), 0.0, 40.0))
    row_dict["z_op_to_assets"] = req.return_on_assets
    row_dict["z_sales_to_assets"] = req.asset_turnover
    row_dict["z_market_to_debt"] = float(np.clip(req.revenue / (liab_est + 1e5) * 2.5, 0.0, 50.0))
    row_dict["margin_x_cap_int"] = req.operating_margin * req.capital_intensity
    row_dict["slack_x_solvency"] = (req.cash_flow_margin * req.asset_turnover) * (1.0 - req.debt_to_assets)

    # Log metrics
    row_dict["log_revenue"] = np.log1p(max(0, req.revenue))
    row_dict["log_total_assets"] = np.log1p(max(0, req.total_assets))
    row_dict["log_total_liabilities"] = np.log1p(max(0, req.total_assets * req.debt_to_assets))
    row_dict["log_operating_cash_flow"] = np.log1p(max(0, req.revenue * req.cash_flow_margin))
    row_dict["log_volume"] = 15.0
    row_dict["log_close"] = 4.5

    # Sector Z-scores proxies (relative to sector baseline)
    row_dict["operating_margin_sec_z"] = (req.operating_margin - 0.15) / 0.10
    row_dict["return_on_assets_sec_z"] = (req.return_on_assets - 0.08) / 0.06
    row_dict["capital_intensity_sec_z"] = (req.capital_intensity - 1.5) / 1.0
    row_dict["cash_flow_margin_sec_z"] = (req.cash_flow_margin - 0.15) / 0.08
    row_dict["debt_to_assets_sec_z"] = (req.debt_to_assets - 0.50) / 0.15

    # Sector one-hot encoding
    for col in feature_cols:
        if col.startswith("sector_"):
            sector_name = col[len("sector_"):]
            row_dict[col] = 1.0 if sector_name.lower() == req.sector.lower() else 0.0

    # Ensure all feature_cols exist in order
    X_input = pd.DataFrame([[row_dict.get(c, 0.0) for c in feature_cols]], columns=feature_cols)
    
    # Run prediction
    pred_score = float(np.clip(model.predict(X_input)[0], 0.0, 100.0))
    
    # Determine Tier
    if pred_score >= 65.0:
        tier = "Leader"
        tier_description = "High Environmental Performance (Top Tier) — Strong ESG posture and decarbonization readiness."
        tier_color = "#10b981"
    elif pred_score >= 48.0:
        tier = "Moderate"
        tier_description = "Moderate / Transitioning — Baseline environmental compliance with room for operational upgrades."
        tier_color = "#f59e0b"
    else:
        tier = "Laggard"
        tier_description = "Environmental Laggard — High exposure to regulatory sanctions, fossil dependency, or weak disclosure."
        tier_color = "#ef4444"

    # Evaluate primary drivers
    drivers = []
    if req.cash_flow_margin > 0.18:
        drivers.append({"factor": "Cash Flow Slack", "impact": "Positive (+)", "desc": "High cash flow margin supplies capital for green infrastructure."})
    elif req.cash_flow_margin < 0.08:
        drivers.append({"factor": "Cash Flow Constraints", "impact": "Negative (-)", "desc": "Constrained operating cash limits discretionary sustainability capex."})

    if req.capital_intensity > 2.5:
        drivers.append({"factor": "Heavy Physical Assets", "impact": "Anchoring", "desc": "High asset-to-revenue ratio reflects heavy physical plant footprint."})
    elif req.capital_intensity < 0.8:
        drivers.append({"factor": "Asset-Light Structure", "impact": "Positive (+)", "desc": "Lower physical direct emissions intensity."})

    if req.debt_to_assets > 0.65:
        drivers.append({"factor": "High Debt Burden", "impact": "Negative (-)", "desc": "Elevated debt service crowds out long-term green investments."})

    return {
        "predicted_score": round(pred_score, 2),
        "tier": tier,
        "tier_color": tier_color,
        "tier_description": tier_description,
        "sector": req.sector,
        "drivers": drivers
    }


# Mount visualizer static assets
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
