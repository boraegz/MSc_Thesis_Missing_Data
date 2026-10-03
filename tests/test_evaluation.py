"""Tests for scorecard metrics."""
import numpy as np
from scipy.stats import ks_2samp

from src.evaluation import bad_rate_at_approval, compute_metrics, ks_statistic


def test_ks_matches_scipy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    score = rng.normal(size=2000) + 0.8 * y
    expected = ks_2samp(score[y == 1], score[y == 0]).statistic
    assert abs(ks_statistic(y, score) - expected) < 1e-12


def test_ks_extremes():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.0


def test_bad_rate_at_approval():
    y = np.array([0, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.3, 0.9])
    assert bad_rate_at_approval(y, score, 0.5) == 0.5   # approves rows 0, 1
    assert bad_rate_at_approval(y, score, 0.25) == 0.0  # approves row 0


def test_compute_metrics_consistency():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.2, 1000), 0, 1)
    m = compute_metrics(y, p)
    assert abs(m["gini"] - (2 * m["auc"] - 1)) < 1e-12
    assert 0 <= m["brier"] <= 1
    assert abs(m["mean_pd"] - p.mean()) < 1e-12
