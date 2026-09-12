"""
Master Pipeline Runner.
Orchestrates the entire end-to-end workflow:
1. Authentic Data Ingestion & Caching
2. Harmonization & Preprocessing (Sustainalytics + 20yr Ratios + SEC EDGAR)
3. Domain-driven Feature Engineering (Capital intensity, Slack, Sector Z-Scores)
4. Multi-Model Benchmark (Ridge, RF, HistGradientBoosting, Tabular Neural Net, Ensemble)
5. Out-of-sample GroupKFold (Generalization to unseen corporations)
6. Feature Importance & Explainability Breakdown
7. Corporate Case Studies (Flagship S&P 500 corporations)
8. Artifact Serialization for Production Inference & Interactive Visualizer
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.data.download import download_all_datasets
from src.data.preprocess import load_and_preprocess_data, PROCESSED_DIR
from src.features.engineer import (
    create_engineered_features, 
    prepare_modeling_matrices, 
    get_feature_column_names
)
from src.models.tree_models import build_hist_gradient_boosting_regressor
from src.models.ensemble import EnvironmentalScoreEnsemble
from src.evaluation.cross_val import run_multi_model_cv, run_group_cv
from src.explainability.feature_importance import compute_model_feature_importances
from src.explainability.case_studies import generate_company_case_studies


def run_full_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("CORPORATE ENVIRONMENTAL SCORE PREDICTION PIPELINE")
    print("Powered by Authentic Sustainalytics ESG Ratings & SEC 10-K Fundamentals")
    print("=" * 80)

    # Step 1: Ingest Data
    print("\n[Step 1/6] Ingesting Authentic Datasets...")
    download_all_datasets()

    # Step 2: Preprocess & Harmonize
    print("\n[Step 2/6] Harmonizing ESG Ratings with Corporate Fundamentals...")
    master_df = load_and_preprocess_data()

    # Step 3: Feature Engineering
    print("\n[Step 3/6] Engineering Domain Financial Features...")
    df_engineered = create_engineered_features(master_df)
    X, y_reg, y_clf, feature_cols = prepare_modeling_matrices(df_engineered)
    print(f"  -> Engineered Matrix: {X.shape[0]} rows x {X.shape[1]} features.")

    # Step 4: Multi-Model Cross Validation
    print("\n[Step 4/6] Running 5-Fold Multi-Model Benchmark...")
    benchmark_results = run_multi_model_cv(X, y_reg, n_splits=5, random_state=42)

    # Step 4b: Evaluate Generalization to Unseen Companies (GroupKFold)
    print("\n[Step 4b/6] Evaluating Zero-Shot Generalization to Unseen Companies (GroupKFold)...")
    unseen_metrics = run_group_cv(X, y_reg, groups=master_df["Ticker"], n_splits=5)
    print(f"  -> Unseen Company Transfer | MAE: {unseen_metrics['MAE']:.2f} pts, Within 10 pts: {unseen_metrics['Within_10_Points_Pct']:.1f}%")

    # Step 5: Fit Production Models on Full Dataset
    print("\n[Step 5/6] Training Production High-Performance GPU Models...")
    # Best GPU single model
    from src.models.tree_models import build_gpu_xgboost_regressor
    best_gpu_xgb = build_gpu_xgboost_regressor(random_state=42)
    best_gpu_xgb.fit(X, y_reg)

    # Production Super GPU Ensemble
    ensemble = EnvironmentalScoreEnsemble(random_state=42)
    ensemble.fit(X, y_reg)

    # Step 6: Explainability & Case Studies
    print("\n[Step 6/6] Computing Feature Importances & Corporate Case Studies...")
    importance_df = compute_model_feature_importances(best_gpu_xgb, X, y_reg, n_repeats=5)
    print("\nTop 10 Most Predictive Financial Features for Corporate Environmental Score:")
    print("-" * 75)
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {idx+1:2d}. {row['feature']:<30} | {row['category']:<25} | Imp: {row['importance_mean']:.4f}")
    print("-" * 75)

    case_studies = generate_company_case_studies(df_engineered, X, best_gpu_xgb, feature_cols)
    print("\nCorporate Case Studies (Actual vs Predicted Environmental Scores):")
    print("-" * 75)
    for cs in case_studies:
        print(f"  {cs['ticker']:<5} ({cs['company'][:22]:<22}) | Actual: {cs['actual_score']:5.1f} | Pred: {cs['predicted_score']:5.1f} | Err: {cs['error']:+5.1f} | {cs['accuracy_tier']}")
    print("-" * 75)

    # Export Trained Models and Visualizer Payload
    Path("models").mkdir(parents=True, exist_ok=True)
    model_path = Path("models/best_model.joblib")
    ensemble_path = Path("models/ensemble_model.joblib")
    
    metadata = {
        "feature_cols": feature_cols,
        "numeric_cols": get_feature_column_names(df_engineered),
        "sectors": sorted(master_df["sector"].unique().tolist()),
        "target_stats": {
            "mean": float(y_reg.mean()),
            "std": float(y_reg.std()),
            "min": float(y_reg.min()),
            "max": float(y_reg.max())
        }
    }
    joblib.dump({"model": best_gpu_xgb, "metadata": metadata}, model_path)
    joblib.dump({"model": ensemble, "metadata": metadata}, ensemble_path)
    print(f"\n[OK] Serialized production GPU model to {model_path}")

    # Prepare Payload for Interactive Visualizer
    summary_payload = {
        "dataset_stats": {
            "total_observations": int(len(master_df)),
            "distinct_companies": int(master_df["Ticker"].nunique()),
            "distinct_sectors": int(master_df["sector"].nunique()),
            "year_min": int(master_df["year"].min()),
            "year_max": int(master_df["year"].max()),
            "mean_score": round(float(y_reg.mean()), 2),
            "std_score": round(float(y_reg.std()), 2)
        },
        "model_benchmarks": {
            model_name: data["metrics"]
            for model_name, data in benchmark_results.items()
        },
        "unseen_company_metrics": unseen_metrics,
        "top_features": importance_df.head(15).to_dict(orient="records"),
        "case_studies": case_studies,
        "sectors": metadata["sectors"],
        "companies_list": (
            df_engineered.groupby("Ticker")
            .agg({
                "Company": "first",
                "sector": "first",
                "environment_score": "last",
                "year": "max",
                "Operating Margin": "last",
                "capital_intensity": "last",
                "cash_flow_margin": "last",
                "debt_to_assets": "last",
                "revenue": "last",
                "total_assets": "last"
            })
            .reset_index()
            .to_dict(orient="records")
        )
    }

    # Clean any NaNs in payload for valid JSON
    def clean_json(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        elif isinstance(obj, dict):
            return {k: clean_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_json(v) for v in obj]
        return obj

    cleaned_payload = clean_json(summary_payload)
    
    vis_dir = Path("visualizer/static")
    vis_dir.mkdir(parents=True, exist_ok=True)
    with open(vis_dir / "pipeline_summary.json", "w") as f:
        json.dump(cleaned_payload, f, indent=2)
    print(f"[OK] Exported interactive visualizer payload to {vis_dir / 'pipeline_summary.json'}")

    elapsed = time.time() - start_time
    print(f"\nPipeline successfully completed in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    run_full_pipeline()
