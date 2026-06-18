"""Mitigations avancées optionnelles: Fairlearn, seuils et adversarial PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class OptionalResult:
    available: bool
    method: str
    reason: str = ""
    payload: dict | None = None


def fairlearn_reductions_available() -> bool:
    try:
        import fairlearn  # noqa: F401
        return True
    except Exception:
        return False


def run_fairlearn_reduction(estimator, X_train, y_train, sensitive_train, X_test,
                            constraint_name: str = "equalized_odds") -> OptionalResult:
    """Entraîne ExponentiatedGradient si Fairlearn est disponible."""
    try:
        from fairlearn.reductions import (
            DemographicParity,
            EqualizedOdds,
            ExponentiatedGradient,
        )
    except Exception as exc:
        return OptionalResult(False, f"fairlearn_{constraint_name}", str(exc))

    constraints = {
        "demographic_parity": DemographicParity(),
        "equalized_odds": EqualizedOdds(),
    }
    if constraint_name not in constraints:
        return OptionalResult(False, f"fairlearn_{constraint_name}", "Contrainte inconnue")

    mitigator = ExponentiatedGradient(
        estimator=estimator,
        constraints=constraints[constraint_name],
        sample_weight_name="classifier__sample_weight",
    )
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
    y_pred = mitigator.predict(X_test)

    if hasattr(mitigator, "_pmf_predict"):
        proba = mitigator._pmf_predict(X_test)
        y_scores = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else y_pred
    else:
        y_scores = y_pred

    return OptionalResult(
        True,
        f"fairlearn_{constraint_name}",
        payload={"model": mitigator, "y_pred": y_pred, "y_scores": y_scores},
    )


def group_threshold_predictions(y_scores: np.ndarray, sensitive_features,
                                thresholds: Dict[str, float]) -> np.ndarray:
    y_scores = np.asarray(y_scores)
    sensitive_features = np.asarray(sensitive_features)
    y_pred = np.zeros(len(y_scores), dtype=int)
    for group, threshold in thresholds.items():
        mask = sensitive_features.astype(str) == str(group)
        y_pred[mask] = (y_scores[mask] >= threshold).astype(int)
    return y_pred


def tune_group_thresholds(y_val: np.ndarray, y_scores_val: np.ndarray,
                          sensitive_val, grid: Iterable[float] = None) -> Dict[str, float]:
    """
    Choisit des seuils par groupe pour rapprocher les taux de sélection du taux global.

    Cette méthode est volontairement simple et documentée comme post-processing sensible:
    elle sert à comparer le compromis technique, pas à recommander un usage juridique brut.
    """
    y_scores_val = np.asarray(y_scores_val)
    sensitive_val = pd.Series(sensitive_val).astype(str)
    grid = list(grid or np.linspace(0.05, 0.95, 19))
    target_selection_rate = float(np.mean(y_scores_val >= 0.5))
    thresholds = {}

    for group in sensitive_val.unique():
        mask = sensitive_val == group
        best_threshold = 0.5
        best_gap = float("inf")
        for threshold in grid:
            selection_rate = float(np.mean(y_scores_val[mask] >= threshold))
            gap = abs(selection_rate - target_selection_rate)
            if gap < best_gap:
                best_gap = gap
                best_threshold = float(threshold)
        thresholds[str(group)] = best_threshold

    return thresholds


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def run_adversarial_debiasing_torch(
    preprocessor,
    X_train,
    y_train,
    sensitive_train,
    X_test,
    lambdas: List[float] = None,
    epochs: int = 20,
    random_state: int = 42,
) -> OptionalResult:
    """Entraîne un prédicteur/adversaire PyTorch avec gradient reversal."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.preprocessing import LabelEncoder
    except Exception as exc:
        return OptionalResult(False, "adversarial_pytorch", str(exc))

    lambdas = lambdas or [0.0, 0.1, 0.5, 1.0]
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    X_train_np = preprocessor.fit_transform(X_train).astype("float32")
    X_test_np = preprocessor.transform(X_test).astype("float32")
    y_train_np = np.asarray(y_train).astype("float32").reshape(-1, 1)

    encoder = LabelEncoder()
    sensitive_np = encoder.fit_transform(pd.Series(sensitive_train).astype(str))
    n_sensitive = len(encoder.classes_)

    class GradientReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lambda_value):
            ctx.lambda_value = lambda_value
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.lambda_value * grad_output, None

    class AdvNet(nn.Module):
        def __init__(self, input_dim: int, n_groups: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 48),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(48, 24),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(24, 1)
            self.adversary = nn.Linear(24, n_groups)

        def forward(self, x, lambda_value: float):
            z = self.encoder(x)
            y_logit = self.classifier(z)
            z_rev = GradientReverse.apply(z, lambda_value)
            s_logit = self.adversary(z_rev)
            return y_logit, s_logit

    X_tensor = torch.tensor(X_train_np)
    y_tensor = torch.tensor(y_train_np)
    s_tensor = torch.tensor(sensitive_np, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_np)

    results = {}
    for lambda_value in lambdas:
        model = AdvNet(X_train_np.shape[1], n_sensitive)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss()
        ce = nn.CrossEntropyLoss()

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            y_logit, s_logit = model(X_tensor, float(lambda_value))
            loss = bce(y_logit, y_tensor) + ce(s_logit, s_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits, _ = model(X_test_tensor, float(lambda_value))
            scores = torch.sigmoid(test_logits).cpu().numpy().reshape(-1)
            preds = (scores >= 0.5).astype(int)

        results[str(lambda_value)] = {
            "y_pred": preds,
            "y_scores": scores,
            "lambda": float(lambda_value),
        }

    return OptionalResult(True, "adversarial_pytorch", payload=results)
