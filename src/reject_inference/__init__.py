"""Reject inference methods. Each turns a biased historical sample into a scorecard."""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

from src.reject_inference.augmentation import FuzzyAugmentation, Parceling
from src.reject_inference.base import RejectInferenceMethod, make_learner
from src.reject_inference.baselines import AcceptsOnly, RejectsAsBad
from src.reject_inference.reweighting import InverseProbabilityWeighting

METHODS: Dict[str, Type[RejectInferenceMethod]] = {
    cls.name: cls
    for cls in (AcceptsOnly, RejectsAsBad, InverseProbabilityWeighting, FuzzyAugmentation, Parceling)
}


def build_method(name: str, learner_kind: str = "logistic", params: Optional[Dict[str, Any]] = None) -> RejectInferenceMethod:
    """Instantiate a registered method with a fresh scorecard learner.

    Parameters
    ----------
    name : str
        Key in :data:`METHODS`.
    learner_kind : str
        Passed to :func:`make_learner`.
    params : dict, optional
        Extra keyword arguments for the method's constructor.
    """
    if name not in METHODS:
        raise KeyError(f"Unknown method {name!r}. Available: {sorted(METHODS)}")
    return METHODS[name](learner=make_learner(learner_kind), **(params or {}))


__all__ = ["METHODS", "RejectInferenceMethod", "build_method", "make_learner"]
