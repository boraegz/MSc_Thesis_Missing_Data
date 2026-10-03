"""Reference and naive baselines."""
from __future__ import annotations

import numpy as np

from src.data.simulation import TrainingView
from src.reject_inference.base import RejectInferenceMethod


class AcceptsOnly(RejectInferenceMethod):
    """Fit the scorecard on accepted applicants only (no reject inference).

    This is what a lender does by default and the lower reference point: it is
    unbiased only if selection is ignorable for the model.
    """

    name = "accepts_only"

    def fit(self, data: TrainingView, rng: np.random.Generator) -> "AcceptsOnly":
        acc = data.accepted
        self._fit_learner(data.X[acc], data.y[acc])
        self.diagnostics_ = {"n_train": int(acc.sum())}
        return self


class RejectsAsBad(RejectInferenceMethod):
    """Label every rejected applicant as a default.

    A naive reject inference rule (cf. Crook & Banasik, 2004). It is correct
    only if all rejects would have defaulted, which overstates their risk.
    """

    name = "rejects_as_bad"

    def fit(self, data: TrainingView, rng: np.random.Generator) -> "RejectsAsBad":
        y = np.where(data.accepted, data.y, 1.0)
        self._fit_learner(data.X, y)
        self.diagnostics_ = {"n_train": len(y)}
        return self
