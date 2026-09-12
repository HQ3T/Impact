import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def build_ridge_pipeline() -> Pipeline:
    """Returns a Ridge regression pipeline with median imputation, standardization, and logspace alphas."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", RidgeCV(alphas=np.logspace(2, 6, 25)))
    ])


def build_elastic_net_pipeline() -> Pipeline:
    """Returns an ElasticNet pipeline with median imputation and scaling."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=42, max_iter=2000))
    ])
