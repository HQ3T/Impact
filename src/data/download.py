"""
Authentic Data Acquisition Module.
Downloads and locally caches authentic corporate ESG ratings and financial datasets:
1. Sustainalytics S&P 500 ESG Database (13,237 historical observations from Morningstar/Sustainalytics).
2. S&P 500 20-Year Financial Ratios Database (over 1M daily/quarterly financial ratio records).
3. SEC EDGAR 10-K Corporate Financial Fundamentals (verified US SEC annual filings).
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

RAW_DIR = Path("data/raw")

ESG_REPO = "nlp-esg-scoring/spx-sustainalytics-esg-scores"
ESG_FILENAME = "data/train-00000-of-00001-7f89b36539207c5e.parquet"

RATIOS_REPO = "pmoe7/SP_500_Stocks_Data-ratios_news_price_10_yrs"
RATIOS_FILENAME = "sp500_daily_ratios_20yrs.zip"

SEC_REPO = "ZanderL1337/financial-fundamentals"
SEC_FILENAME = "financials_annual_latest.parquet"


def download_all_datasets(raw_dir: Path = RAW_DIR) -> dict:
    """
    Downloads and caches all authentic raw datasets into raw_dir.
    Returns dictionary with local file paths.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # 1. Sustainalytics ESG Scores
    local_esg = raw_dir / "sustainalytics_spx_esg.parquet"
    if not local_esg.exists():
        print(f"[1/3] Downloading authentic Sustainalytics ESG ratings from {ESG_REPO}...")
        downloaded = hf_hub_download(repo_id=ESG_REPO, filename=ESG_FILENAME, repo_type="dataset")
        shutil.copy2(downloaded, local_esg)
        print(f"  [OK] Saved to {local_esg} ({local_esg.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"  [OK] Found cached ESG ratings at {local_esg}")
    paths["esg"] = local_esg

    # 2. 20-Year S&P 500 Financial Ratios
    local_ratios = raw_dir / "sp500_daily_ratios_20yrs.zip"
    if not local_ratios.exists():
        print(f"[2/3] Downloading authentic S&P 500 20-Year Financial Ratios from {RATIOS_REPO}...")
        downloaded = hf_hub_download(repo_id=RATIOS_REPO, filename=RATIOS_FILENAME, repo_type="dataset")
        shutil.copy2(downloaded, local_ratios)
        print(f"  [OK] Saved to {local_ratios} ({local_ratios.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"  [OK] Found cached financial ratios at {local_ratios}")
    paths["ratios"] = local_ratios

    # 3. SEC EDGAR 10-K Fundamentals
    local_sec = raw_dir / "sec_financials_annual.parquet"
    if not local_sec.exists():
        print(f"[3/3] Downloading authentic SEC EDGAR 10-K corporate fundamentals from {SEC_REPO}...")
        downloaded = hf_hub_download(repo_id=SEC_REPO, filename=SEC_FILENAME, repo_type="dataset")
        shutil.copy2(downloaded, local_sec)
        print(f"  [OK] Saved to {local_sec} ({local_sec.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"  [OK] Found cached SEC EDGAR fundamentals at {local_sec}")
    paths["sec"] = local_sec

    return paths


if __name__ == "__main__":
    download_all_datasets()
