"""Tests des garde-fous pour les dépendances optionnelles."""

import numpy as np

from src.advanced_mitigation import (
    fairlearn_reductions_available,
    group_threshold_predictions,
    torch_available,
    tune_group_thresholds,
)


def test_optional_dependency_checks_return_bool():
    assert isinstance(torch_available(), bool)
    assert isinstance(fairlearn_reductions_available(), bool)


def test_group_threshold_predictions():
    scores = np.array([0.2, 0.8, 0.4, 0.7])
    sensitive = np.array(["A", "A", "B", "B"])
    thresholds = tune_group_thresholds(
        np.array([0, 1, 0, 1]),
        scores,
        sensitive,
        grid=[0.3, 0.5, 0.7],
    )
    preds = group_threshold_predictions(scores, sensitive, thresholds)

    assert set(thresholds) == {"A", "B"}
    assert preds.shape == scores.shape
    assert set(preds).issubset({0, 1})
