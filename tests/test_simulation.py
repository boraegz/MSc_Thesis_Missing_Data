"""Tests for the label-selection data generator."""
import numpy as np
import pytest

from src.data.simulation import (
    PolicyConfig,
    PopulationConfig,
    TrainingView,
    feature_covariance,
    risk_index,
    simulate_experiment,
    true_pd,
)

POP = PopulationConfig()


def _sim(rho=0.0, alpha=0.5, er=True, n_train=20000, n_test=5000, rep=0, pop=POP):
    policy = PolicyConfig(rho=rho, acceptance_rate=alpha, exclusion_restriction=er)
    return simulate_experiment(pop, policy, n_train, n_test, base_seed=123, replication=rep)


def test_acceptance_and_default_rates_match_targets():
    for alpha in (0.3, 0.5, 0.7):
        d = _sim(alpha=alpha)
        assert abs(d.meta["acceptance_rate_realised"] - alpha) < 0.02
        # nonlinear index is not exactly normal: allow a small deviation
        assert abs(d.meta["default_rate_population"] - POP.default_rate) < 0.025


def test_labels_missing_exactly_for_rejects():
    d = _sim()
    tv = d.train
    assert np.isnan(tv.y[tv.rejected]).all()
    assert not np.isnan(tv.y[tv.accepted]).any()
    np.testing.assert_array_equal(tv.y[tv.accepted], d.y_train_oracle[tv.accepted])


def test_training_view_rejects_inconsistent_labels():
    with pytest.raises(ValueError):
        TrainingView(X=np.zeros((2, 1)), z=np.zeros(2), s=np.array([1, 0]), y=np.array([1.0, 0.0]))


def test_reproducible_and_paired_across_policies():
    a = _sim(rho=0.0, alpha=0.3)
    b = _sim(rho=0.6, alpha=0.7, er=False)
    c = _sim(rho=0.0, alpha=0.3)
    # same replication -> same applicants and outcomes, different selection
    np.testing.assert_array_equal(a.train.X, b.train.X)
    np.testing.assert_array_equal(a.y_train_oracle, b.y_train_oracle)
    np.testing.assert_array_equal(a.X_test, b.X_test)
    assert not np.array_equal(a.train.s, b.train.s)
    # same arguments -> identical result
    np.testing.assert_array_equal(a.train.s, c.train.s)
    # different replication -> different applicants
    assert not np.array_equal(a.train.X, _sim(rep=1).train.X)


def test_mar_when_rho_zero_and_mnar_when_rho_positive():
    """Accepted applicants default as predicted by X under MAR, less under MNAR."""
    for rho, expect_mnar in ((0.0, False), (0.6, True)):
        d = _sim(rho=rho, alpha=0.5, n_train=60000)
        acc = d.train.accepted
        observed = d.train.y[acc].mean()
        predicted = true_pd(d.train.X[acc], POP).mean()
        se = np.sqrt(predicted * (1 - predicted) / acc.sum())
        if expect_mnar:
            assert observed < predicted - 10 * se
        else:
            assert abs(observed - predicted) < 4 * se


def test_exclusion_restriction_switch():
    on = _sim(er=True)
    off = _sim(er=False)
    assert abs(np.corrcoef(on.train.z, on.train.s)[0, 1]) > 0.2
    assert abs(np.corrcoef(off.train.z, off.train.s)[0, 1]) < 0.03
    # same selection strength: realised acceptance and accepted default rate close
    assert abs(on.meta["default_rate_accepted"] - off.meta["default_rate_accepted"]) < 0.02


def test_risk_index_has_target_scale():
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(np.zeros(POP.n_features), feature_covariance(POP), size=200000)
    eta = risk_index(X, POP)
    assert abs(eta.mean()) < 0.01
    assert abs(eta.std() - POP.signal_sd) < 0.01


def test_invalid_configs_raise():
    with pytest.raises(ValueError):
        PopulationConfig(default_rate=1.2)
    with pytest.raises(ValueError):
        PopulationConfig(n_informative=9, n_features=10)
    with pytest.raises(ValueError):
        PolicyConfig(rho=1.0)
    with pytest.raises(ValueError):
        PolicyConfig(acceptance_rate=0.0)
