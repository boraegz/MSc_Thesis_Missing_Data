"""Common interface and scorecard learners for reject inference methods."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.data.simulation import TrainingView


def make_learner(kind: str = "logistic", random_state: int = 0) -> ClassifierMixin:
    """Scorecard learner shared by all methods, so only the label handling differs.

    Parameters
    ----------
    kind : {"logistic", "gbm"}
        ``logistic`` is the industry-standard scorecard model and the main
        learner of the thesis; ``gbm`` (histogram gradient boosting) is a
        robustness check with a flexible model.
    random_state : int
        Seed for learners with internal randomness.
    """
    if kind == "logistic":
        return LogisticRegression(C=1.0, max_iter=2000)
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=random_state)
    raise ValueError(f"Unknown learner kind: {kind!r}")


class RejectInferenceMethod(ABC):
    """Base class: turn a biased historical sample into a scorecard.

    Subclasses implement :meth:`fit` using only the :class:`TrainingView`
    (features, Z, acceptance flag and the labels of accepted applicants).

    Parameters
    ----------
    learner : sklearn classifier, optional
        Unfitted scorecard model; defaults to logistic regression.

    Attributes
    ----------
    model_ : fitted classifier
    diagnostics_ : dict
        Method-specific quantities worth reporting (e.g. effective sample size).
    """

    name: str = "base"

    def __init__(self, learner: Optional[ClassifierMixin] = None) -> None:
        self.learner = learner if learner is not None else make_learner()
        self.model_: Optional[ClassifierMixin] = None
        self.diagnostics_: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, data: TrainingView, rng: np.random.Generator) -> "RejectInferenceMethod":
        """Fit the scorecard from the historical sample."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicted probability of default for each row of X."""
        if self.model_ is None:
            raise RuntimeError(f"{self.name}: call fit() before predict_proba().")
        return self.model_.predict_proba(X)[:, 1]

    def _fit_learner(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> None:
        y = np.asarray(y)
        if np.isnan(y.astype(float)).any():
            raise ValueError(f"{self.name}: training labels contain NaN.")
        model = clone(self.learner)
        model.fit(X, y.astype(int), sample_weight=sample_weight)
        self.model_ = model
