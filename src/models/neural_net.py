"""
Deep Tabular Neural Network for Corporate Environmental Score Prediction.
Implements a Residual Tabular Multi-Layer Perceptron (TabularResNet) in PyTorch
with LayerNorm, GELU activations, dropout regularization, and skip connections.
Provides a scikit-learn compatible estimator interface.
"""

from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class TabularResidualBlock(nn.Module):
    """Residual building block with LayerNorm, GELU, and Dropout."""
    def __init__(self, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.act2(out + residual)
        return out


class TabularResNet(nn.Module):
    """Full deep tabular architecture with input projection, residual blocks, and regression head."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_blocks: int = 3, dropout: float = 0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList([
            TabularResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_proj(x)
        for block in self.blocks:
            out = block(out)
        out = self.head(out)
        return out.squeeze(-1)


class TabularNeuralNetRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper around TabularResNet.
    Automatically handles missing values, feature standardization,
    and mini-batch PyTorch training.
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.15,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 40,
        random_state: int = 42
    ):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

        self.imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        self.model_ = None
        
        # Test if PyTorch has compatible CUDA kernels for this device architecture
        device = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                test_t = torch.randn(2, 2, device="cuda")
                _ = test_t @ test_t
                device = torch.device("cuda")
            except Exception:
                device = torch.device("cpu")
        self.device_ = device

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.array(X)

        if isinstance(y, pd.Series):
            y_arr = y.values.astype(np.float32)
        else:
            y_arr = np.array(y, dtype=np.float32)

        # Preprocessing: Impute + Scale
        X_imputed = self.imputer_.fit_transform(X_arr)
        X_scaled = self.scaler_.fit_transform(X_imputed).astype(np.float32)

        input_dim = X_scaled.shape[1]
        self.model_ = TabularResNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            dropout=self.dropout
        ).to(self.device_)

        dataset = TensorDataset(torch.tensor(X_scaled), torch.tensor(y_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.SmoothL1Loss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        self.model_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)

                optimizer.zero_grad()
                preds = self.model_(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

        return self

    def predict(self, X) -> np.ndarray:
        self.model_.eval()
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.array(X)

        X_imputed = self.imputer_.transform(X_arr)
        X_scaled = self.scaler_.transform(X_imputed).astype(np.float32)

        tensor_x = torch.tensor(X_scaled).to(self.device_)
        with torch.no_grad():
            preds = self.model_(tensor_x).cpu().numpy()
        return preds
