"""
Unit and Integration Tests for Environmental Score Prediction Pipeline.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.features.engineer import (
    create_engineered_features, 
    prepare_modeling_matrices, 
    get_feature_column_names
)
from src.models.tree_models import build_hist_gradient_boosting_regressor
from src.models.baselines import build_ridge_pipeline
from src.models.neural_net import TabularNeuralNetRegressor
from src.evaluation.metrics import compute_regression_metrics, compute_tier_classification_metrics


class TestPipeline(unittest.TestCase):

    def setUp(self):
        """Creates synthetic corporate financial and ESG dataset for unit tests."""
        np.random.seed(42)
        n = 50
        self.sample_df = pd.DataFrame({
            "Ticker": np.random.choice(["AAPL", "MSFT", "XOM", "NEE", "PFE"], n),
            "year": np.random.choice([2015, 2016, 2017, 2018], n),
            "Company": "Sample Corp",
            "sector": np.random.choice(["Technology", "Energy", "Utilities"], n),
            "environment_score": np.random.uniform(30.0, 90.0, n),
            "revenue": np.random.uniform(1e8, 1e11, n),
            "net_income": np.random.uniform(-1e7, 1e10, n),
            "operating_income": np.random.uniform(1e7, 2e10, n),
            "gross_profit": np.random.uniform(2e7, 4e10, n),
            "total_assets": np.random.uniform(1e8, 2e11, n),
            "total_liabilities": np.random.uniform(5e7, 1e11, n),
            "operating_cash_flow": np.random.uniform(1e7, 3e10, n),
            "Asset Turnover": np.random.uniform(0.3, 2.5, n),
            "Current Ratio": np.random.uniform(0.8, 3.5, n),
            "Debt/Equity Ratio": np.random.uniform(0.2, 3.0, n),
            "EBIT Margin": np.random.uniform(0.05, 0.40, n),
            "EBITDA Margin": np.random.uniform(0.10, 0.50, n),
            "Gross Margin": np.random.uniform(0.20, 0.70, n),
            "Net Profit Margin": np.random.uniform(0.02, 0.30, n),
            "Operating Margin": np.random.uniform(0.05, 0.35, n),
            "Pre-Tax Profit Margin": np.random.uniform(0.05, 0.35, n),
            "ROA - Return On Assets": np.random.uniform(0.01, 0.20, n),
            "ROE - Return On Equity": np.random.uniform(0.02, 0.40, n),
            "Volume": np.random.uniform(1e5, 1e7, n),
            "Close": np.random.uniform(20.0, 300.0, n)
        })

    def test_feature_engineering(self):
        """Verifies that engineered financial metrics and sector Z-scores are computed correctly."""
        df_eng = create_engineered_features(self.sample_df)
        
        self.assertIn("capital_intensity", df_eng.columns)
        self.assertIn("cash_flow_margin", df_eng.columns)
        self.assertIn("debt_to_assets", df_eng.columns)
        self.assertIn("overhead_spread", df_eng.columns)
        self.assertIn("environmental_tier", df_eng.columns)
        self.assertIn("environmental_tier_code", df_eng.columns)
        self.assertIn("operating_margin_sec_z", df_eng.columns)

        # Check values are finite and non-empty
        self.assertEqual(len(df_eng), len(self.sample_df))
        self.assertFalse(df_eng["capital_intensity"].isna().all())

    def test_modeling_matrices_generation(self):
        """Checks X matrix and target extraction."""
        X, y_reg, y_clf, feature_cols = prepare_modeling_matrices(self.sample_df)
        self.assertEqual(X.shape[0], len(self.sample_df))
        self.assertGreater(X.shape[1], 15)
        self.assertEqual(len(y_reg), len(self.sample_df))
        self.assertEqual(len(y_clf), len(self.sample_df))
        self.assertTrue(all(isinstance(c, str) for c in feature_cols))

    def test_tree_regressor_training_and_inference(self):
        """Tests HistGradientBoosting training and prediction."""
        X, y_reg, _, _ = prepare_modeling_matrices(self.sample_df)
        model = build_hist_gradient_boosting_regressor(max_iter=10)
        model.fit(X, y_reg)
        preds = model.predict(X)

        self.assertEqual(len(preds), len(y_reg))
        self.assertFalse(np.isnan(preds).any())
        self.assertTrue(all(0.0 <= p <= 100.0 for p in preds))

    def test_ridge_baseline_training(self):
        """Tests Ridge pipeline training."""
        X, y_reg, _, _ = prepare_modeling_matrices(self.sample_df)
        ridge = build_ridge_pipeline()
        ridge.fit(X, y_reg)
        preds = ridge.predict(X)
        self.assertEqual(len(preds), len(y_reg))

    def test_neural_net_regressor(self):
        """Tests TabularNeuralNetRegressor training for a minimal epoch count."""
        X, y_reg, _, _ = prepare_modeling_matrices(self.sample_df)
        nn_model = TabularNeuralNetRegressor(epochs=2, batch_size=16, hidden_dim=32, num_blocks=1)
        nn_model.fit(X, y_reg)
        preds = nn_model.predict(X)
        self.assertEqual(len(preds), len(y_reg))
        self.assertFalse(np.isnan(preds).any())

    def test_regression_metrics(self):
        """Validates statistical metrics computation."""
        y_true = np.array([50.0, 60.0, 70.0, 80.0])
        y_pred = np.array([52.0, 58.0, 69.0, 82.0])
        metrics = compute_regression_metrics(y_true, y_pred)

        self.assertIn("R2_Score", metrics)
        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("Pearson_Correlation", metrics)
        self.assertGreater(metrics["R2_Score"], 0.90)
        self.assertLess(metrics["MAE"], 3.0)


if __name__ == "__main__":
    unittest.main()
