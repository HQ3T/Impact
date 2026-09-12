"""
Data Preprocessing and Harmonization Pipeline.
Merges authentic Sustainalytics corporate environmental ratings with:
- 20-year quarterly/daily financial ratios
- SEC EDGAR 10-K corporate fundamentals
Produces a unified, high-integrity master dataset for ML modeling.
"""

import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.download import RAW_DIR

PROCESSED_DIR = Path("data/processed")


def load_and_preprocess_data(
    raw_dir: Path = RAW_DIR, 
    output_dir: Path = PROCESSED_DIR
) -> pd.DataFrame:
    """
    Harmonizes and merges authentic datasets on (Ticker, Fiscal Year).
    Saves clean master parquet and CSV to output_dir.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Preprocess] 1. Loading Sustainalytics ESG ratings...")
    esg_file = raw_dir / "sustainalytics_spx_esg.parquet"
    df_esg = pd.read_parquet(esg_file)
    
    # Extract reporting year
    df_esg["year"] = df_esg["Date"].astype(str).str[:4].astype(int)
    df_esg["Ticker"] = df_esg["Ticker"].astype(str).str.strip().str.upper()

    # Aggregate to annual observations per company
    esg_agg = df_esg.groupby(["Ticker", "year"]).agg({
        "Company": "first",
        "Peer_group_root": "first",
        "Region": "first",
        "Country": "first",
        "environment_score": "mean",
        "social_score": "mean",
        "governance_score": "mean",
        "total_esg_score": "mean"
    }).dropna(subset=["environment_score"]).reset_index()

    print(f"  -> Extracted {len(esg_agg)} company-year ESG records across {esg_agg['Ticker'].nunique()} companies.")

    print("[Preprocess] 2. Loading and aggregating 20-Year Financial Ratios...")
    ratios_zip = raw_dir / "sp500_daily_ratios_20yrs.zip"
    with zipfile.ZipFile(ratios_zip, "r") as zf:
        with zf.open("sp500_daily_ratios_20yrs.csv") as f:
            ratio_cols = [
                "Ticker", "year", "Asset Turnover", "Current Ratio", 
                "Debt/Equity Ratio", "EBIT Margin", "EBITDA Margin", 
                "Gross Margin", "Net Profit Margin", "Operating Margin", 
                "Pre-Tax Profit Margin", "ROA - Return On Assets", 
                "ROE - Return On Equity", "Inventory Turnover Ratio",
                "Long-term Debt / Capital", "Days Sales In Receivables",
                "Close", "Volume"
            ]
            df_ratios = pd.read_csv(f, usecols=ratio_cols)

    df_ratios["Ticker"] = df_ratios["Ticker"].astype(str).str.strip().str.upper()
    df_ratios["year"] = pd.to_numeric(df_ratios["year"], errors="coerce")
    df_ratios = df_ratios.dropna(subset=["Ticker", "year"])
    df_ratios["year"] = df_ratios["year"].astype(int)

    # Median aggregation per ticker and year
    ratios_agg = df_ratios.groupby(["Ticker", "year"]).median().reset_index()
    print(f"  -> Processed {len(ratios_agg)} annual ratio profiles across {ratios_agg['Ticker'].nunique()} companies.")

    print("[Preprocess] 3. Loading SEC EDGAR 10-K Fundamentals...")
    sec_file = raw_dir / "sec_financials_annual.parquet"
    df_sec = pd.read_parquet(sec_file)
    df_sec["symbol"] = df_sec["symbol"].astype(str).str.strip().str.upper()
    df_sec["fiscal_year"] = pd.to_numeric(df_sec["fiscal_year"], errors="coerce")
    df_sec = df_sec[df_sec["symbol"] != ""].dropna(subset=["fiscal_year"])
    df_sec["fiscal_year"] = df_sec["fiscal_year"].astype(int)

    sec_agg = df_sec.groupby(["symbol", "fiscal_year"]).agg({
        "revenue": "max",
        "net_income": "max",
        "operating_income": "max",
        "gross_profit": "max",
        "total_assets": "max",
        "total_liabilities": "max",
        "operating_cash_flow": "max",
        "shares_outstanding": "max"
    }).reset_index()
    print(f"  -> Aggregated SEC fundamental filings across {sec_agg['symbol'].nunique()} corporate entities.")

    print("[Preprocess] 4. Merging datasets on (Ticker, Year)...")
    # Step 4a: Merge ESG with Financial Ratios
    merged = pd.merge(esg_agg, ratios_agg, on=["Ticker", "year"], how="inner")
    
    # Step 4b: Merge with SEC Fundamentals
    master = pd.merge(
        merged, 
        sec_agg, 
        left_on=["Ticker", "year"], 
        right_on=["symbol", "fiscal_year"], 
        how="left"
    )
    master.drop(columns=["symbol", "fiscal_year"], errors="ignore", inplace=True)

    # Normalize Sector column
    master["sector"] = master["Peer_group_root"].fillna("Diversified Industrials")

    # Clean extreme ratio outliers (1st to 99th percentile winsorization on financial ratios)
    financial_ratio_cols = [
        "Asset Turnover", "Current Ratio", "Debt/Equity Ratio", "EBIT Margin", 
        "EBITDA Margin", "Gross Margin", "Net Profit Margin", "Operating Margin", 
        "Pre-Tax Profit Margin", "ROA - Return On Assets", "ROE - Return On Equity",
        "Inventory Turnover Ratio", "Long-term Debt / Capital", "Days Sales In Receivables"
    ]
    for col in financial_ratio_cols:
        if col in master.columns:
            low_q = master[col].quantile(0.01)
            high_q = master[col].quantile(0.99)
            master[col] = master[col].clip(lower=low_q, upper=high_q)

    # Save to disk
    output_parquet = output_dir / "company_financial_esg_dataset.parquet"
    output_csv = output_dir / "company_financial_esg_dataset.csv"
    master.to_parquet(output_parquet, index=False)
    master.to_csv(output_csv, index=False)

    print(f"[Preprocess] Completed! Master dataset saved to {output_parquet}")
    print(f"  -> Total corporate observations: {len(master):,}")
    print(f"  -> Distinct public companies: {master['Ticker'].nunique()}")
    print(f"  -> Distinct industry sectors: {master['sector'].nunique()}")
    print(f"  -> Time coverage: {master['year'].min()} to {master['year'].max()}")
    print(f"  -> Target Environmental Score mean: {master['environment_score'].mean():.2f} (std: {master['environment_score'].std():.2f}, range: {master['environment_score'].min():.1f} - {master['environment_score'].max():.1f})")

    return master


if __name__ == "__main__":
    load_and_preprocess_data()
