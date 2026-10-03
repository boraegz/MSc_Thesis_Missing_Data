"""Inverse probability weighting (reweighting / "augmentation" in Banasik & Crook, 2007)."""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression

from src.data.simulation import TrainingView
from src.reject_inference.base import RejectInferenceMethod


class InverseProbabilityWeighting(RejectInferenceMethod):
    """Weight accepted applicants by the inverse of their acceptance probability.

    A propensity model estimates e(x, z) = P(S = 1 | X = x, Z = z) on the whole
    historical sample. The scorecard is then fit on accepts with weights
    1 / e, normalised to mean 1. Accepts who resemble rejects get more weight.

    Identification: valid if S is independent of Y given (X, Z) (MAR) and
    e(x, z) > 0 wherever applicants occur (positivity). Under MNAR (rho > 0)
    the weights cannot correct the bias, because the lender's private
    information is not in (X, Z).

    Parameters
    ----------
    learner : sklearn classifier, optional
        Scorecard model.
    propensity_model : sklearn classifier, optional
        Model for e(x, z); logistic regression by default.
    min_propensity : float
        Lower clip for e to avoid exploding weights.
    use_z : bool
        Include Z in the propensity model (it drives acceptance).
    """

    name = "ipw"

    def __init__(
        self,
        learner: Optional[ClassifierMixin] = None,
        propensity_model: Optional[ClassifierMixin] = None,
        min_propensity: float = 0.01,
        use_z: bool = True,
    ) -> None:
        super().__init__(learner)
        self.propensity_model = propensity_model if propensity_model is not None else LogisticRegression(max_iter=2000)
        self.min_propensity = min_propensity
        self.use_z = use_z

    def _selection_features(self, data: TrainingView) -> np.ndarray:
        return np.column_stack([data.X, data.z]) if self.use_z else data.X

    def fit(self, data: TrainingView, rng: np.random.Generator) -> "InverseProbabilityWeighting":
        W = self._selection_features(data)
        self.propensity_model.fit(W, data.s)
        e = self.propensity_model.predict_proba(W)[:, 1]
        acc = data.accepted
        e_acc = np.clip(e[acc], self.min_propensity, 1.0)
        w = 1.0 / e_acc
        w = w / w.mean()
        self._fit_learner(data.X[acc], data.y[acc], sample_weight=w)
        self.diagnostics_ = {
            "n_train": int(acc.sum()),
            "share_clipped": float(np.mean(e[acc] < self.min_propensity)),
            "max_weight": float(w.max()),
            # Kish effective sample size relative to the number of accepts
            "ess_ratio": float(w.sum() ** 2 / (w**2).sum() / acc.sum()),
        }
        return self
