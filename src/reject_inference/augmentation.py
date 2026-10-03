"""Label-assignment reject inference: fuzzy augmentation and parceling."""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import ClassifierMixin, clone

from src.data.simulation import TrainingView
from src.reject_inference.base import RejectInferenceMethod


def _accepts_model_pd(data: TrainingView, learner: ClassifierMixin) -> np.ndarray:
    """Fit on accepts, return predicted PD for every applicant."""
    acc = data.accepted
    model = clone(learner).fit(data.X[acc], data.y[acc].astype(int))
    return model.predict_proba(data.X)[:, 1]


class FuzzyAugmentation(RejectInferenceMethod):
    """Add each reject twice, as default and as non-default, weighted by its PD.

    The PD of a reject comes from a model fit on accepts. Each reject enters
    the final training set as (x, 1) with weight p and as (x, 0) with weight
    1 - p. Assumes the accepts model extrapolates correctly to rejects, i.e.
    MAR given X.

    Parameters
    ----------
    learner : sklearn classifier, optional
        Scorecard model (also used as the accepts model).
    """

    name = "fuzzy_augmentation"

    def fit(self, data: TrainingView, rng: np.random.Generator) -> "FuzzyAugmentation":
        p = _accepts_model_pd(data, self.learner)
        acc, rej = data.accepted, data.rejected
        X = np.vstack([data.X[acc], data.X[rej], data.X[rej]])
        y = np.r_[data.y[acc], np.ones(rej.sum()), np.zeros(rej.sum())]
        w = np.r_[np.ones(acc.sum()), p[rej], 1 - p[rej]]
        self._fit_learner(X, y, sample_weight=w)
        self.diagnostics_ = {"n_train": int(len(data.s)), "mean_pd_rejects": float(p[rej].mean())}
        return self


class Parceling(RejectInferenceMethod):
    """Assign rejects random labels using bad rates of accepts in the same score band.

    1. Fit a model on accepts and score all applicants.
    2. Cut the score into ``n_bands`` quantile bands (on all applicants).
    3. In each band, draw each reject's label as Bernoulli(min(1, k * b)),
       where b is the band's bad rate among accepts and k the inflation factor.
       Bands without accepts borrow the bad rate of the nearest band with accepts.

    ``inflation > 1`` encodes the practitioner's belief that rejects are riskier
    than similar accepts (a manual MNAR adjustment).

    Parameters
    ----------
    learner : sklearn classifier, optional
    n_bands : int
        Number of score bands.
    inflation : float
        Multiplier k on band bad rates for rejects.
    """

    name = "parceling"

    def __init__(self, learner: Optional[ClassifierMixin] = None, n_bands: int = 10, inflation: float = 1.0) -> None:
        super().__init__(learner)
        self.n_bands = n_bands
        self.inflation = inflation

    def fit(self, data: TrainingView, rng: np.random.Generator) -> "Parceling":
        p = _accepts_model_pd(data, self.learner)
        edges = np.quantile(p, np.linspace(0, 1, self.n_bands + 1)[1:-1])
        band = np.searchsorted(edges, p, side="right")
        acc, rej = data.accepted, data.rejected

        band_rate = np.full(self.n_bands, np.nan)
        for b in range(self.n_bands):
            in_b = acc & (band == b)
            if in_b.any():
                band_rate[b] = data.y[in_b].mean()
        known = np.flatnonzero(~np.isnan(band_rate))
        for b in np.flatnonzero(np.isnan(band_rate)):
            band_rate[b] = band_rate[known[np.argmin(np.abs(known - b))]]

        prob = np.minimum(1.0, self.inflation * band_rate[band[rej]])
        y = data.y.copy()
        y[rej] = (rng.random(rej.sum()) < prob).astype(float)
        self._fit_learner(data.X, y)
        self.diagnostics_ = {"n_train": int(len(y)), "assigned_bad_rate_rejects": float(y[rej].mean())}
        return self
