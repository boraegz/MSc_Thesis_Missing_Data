"""Scorecard metrics evaluated on the through-the-door (Oracle) test set."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between the score distributions of bads and goods.

    Parameters
    ----------
    y_true : ndarray of {0, 1}
        True labels (1 = default).
    score : ndarray
        Predicted probability of default (higher = riskier).

    Returns
    -------
    float
        max_t |F_bad(t) - F_good(t)|, in [0, 1].
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    order = np.argsort(score, kind="mergesort")
    s_sorted, y_sorted = score[order], y_true[order]
    n_bad = y_sorted.sum()
    n_good = len(y_sorted) - n_bad
    if n_bad == 0 or n_good == 0:
        return float("nan")
    cdf_bad = np.cumsum(y_sorted) / n_bad
    cdf_good = np.cumsum(1 - y_sorted) / n_good
    # Evaluate only at the last index of each tied score value.
    last_of_tie = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    return float(np.max(np.abs(cdf_bad - cdf_good)[last_of_tie]))


def bad_rate_at_approval(y_true: np.ndarray, score: np.ndarray, approval_rate: float) -> float:
    """Default rate among applicants a scorecard would approve.

    The ``approval_rate`` share with the lowest predicted PD is approved; the
    business question is how many of those default. Lower is better.

    Parameters
    ----------
    y_true : ndarray of {0, 1}
    score : ndarray
        Predicted probability of default.
    approval_rate : float
        Share of applicants approved, in (0, 1].
    """
    if not 0 < approval_rate <= 1:
        raise ValueError("approval_rate must lie in (0, 1].")
    n_approve = int(round(approval_rate * len(score)))
    order = np.argsort(score, kind="mergesort")
    return float(np.asarray(y_true)[order[:n_approve]].mean())


def compute_metrics(y_true: np.ndarray, pd_hat: np.ndarray, approval_rate: float = 0.5) -> Dict[str, float]:
    """All scorecard metrics used in the thesis.

    Parameters
    ----------
    y_true : ndarray of {0, 1}
        True default labels of the test applicants.
    pd_hat : ndarray
        Predicted probabilities of default.
    approval_rate : float
        Approval rate for :func:`bad_rate_at_approval`.

    Returns
    -------
    dict
        ``auc``, ``gini`` (= 2 AUC - 1), ``brier``, ``ks``, ``bad_rate``
        and ``mean_pd`` (predicted default rate, to compare with the true one).
    """
    pd_hat = np.clip(np.asarray(pd_hat, dtype=float), 0.0, 1.0)
    auc = roc_auc_score(y_true, pd_hat)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "brier": float(brier_score_loss(y_true, pd_hat)),
        "ks": ks_statistic(y_true, pd_hat),
        "bad_rate": bad_rate_at_approval(y_true, pd_hat, approval_rate),
        "mean_pd": float(pd_hat.mean()),
    }
