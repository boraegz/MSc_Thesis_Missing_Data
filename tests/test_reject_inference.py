"""Tests for reject inference methods, including leakage guards."""
import copy

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data.simulation import PolicyConfig, PopulationConfig, simulate_experiment
from src.experiments import evaluate_method
from src.reject_inference import METHODS, build_method
from src.utils import make_rng

POP = PopulationConfig()


def _data(rho=0.0, alpha=0.5, n_train=8000, seed=11):
    return simulate_experiment(POP, PolicyConfig(rho=rho, acceptance_rate=alpha), n_train, 4000, base_seed=seed, replication=0)


def test_every_method_fits_and_predicts_probabilities():
    d = _data()
    for name in METHODS:
        m = build_method(name).fit(d.train, make_rng(0))
        p = m.predict_proba(d.X_test)
        assert p.shape == (len(d.X_test),)
        assert np.all((p >= 0) & (p <= 1)), name


def test_no_leakage_of_reject_labels():
    """Scrambling the hidden labels of rejects must not change any method's output."""
    d = _data()
    scrambled = copy.deepcopy(d)
    rej = d.train.rejected
    scrambled.y_train_oracle[rej] = 1 - scrambled.y_train_oracle[rej]
    for name in METHODS:
        spec = {"name": name}
        a = evaluate_method(spec, d, "logistic", 0.5, make_rng(5))
        b = evaluate_method(spec, scrambled, "logistic", 0.5, make_rng(5))
        for key in ("auc", "brier", "ks", "bad_rate", "mean_pd"):
            assert a[key] == b[key], (name, key)


def test_ipw_recovers_population_fit_under_mar():
    """Under MAR, IPW is consistent for the full-sample logistic fit; accepts-only is not.

    With a misspecified (logistic) scorecard, accepts-only converges to a
    different model; IPW's distance to the Oracle fit shrinks with n.
    """
    d = _data(rho=0.0, alpha=0.3, n_train=150000)
    oracle = LogisticRegression(max_iter=3000).fit(d.train.X, d.y_train_oracle)
    acc = build_method("accepts_only").fit(d.train, make_rng(0))
    ipw = build_method("ipw").fit(d.train, make_rng(0))
    dist_acc = np.linalg.norm(acc.model_.coef_ - oracle.coef_)
    dist_ipw = np.linalg.norm(ipw.model_.coef_ - oracle.coef_)
    assert dist_ipw < 0.5 * dist_acc
    assert 0 < ipw.diagnostics_["ess_ratio"] <= 1


def test_rejects_as_bad_overstates_risk():
    d = _data()
    acc = build_method("accepts_only").fit(d.train, make_rng(0))
    rab = build_method("rejects_as_bad").fit(d.train, make_rng(0))
    assert rab.predict_proba(d.X_test).mean() > acc.predict_proba(d.X_test).mean() + 0.1


def test_fuzzy_augmentation_reproduces_accepts_only_for_logistic():
    """Known property: fuzzy augmentation with the same logistic model adds no information."""
    d = _data()
    acc = build_method("accepts_only").fit(d.train, make_rng(0)).predict_proba(d.X_test)
    fuz = build_method("fuzzy_augmentation").fit(d.train, make_rng(0)).predict_proba(d.X_test)
    assert np.corrcoef(acc, fuz)[0, 1] > 0.999


def test_parceling_inflation_raises_assigned_bad_rate():
    d = _data()
    base = build_method("parceling", params={"inflation": 1.0}).fit(d.train, make_rng(1))
    infl = build_method("parceling", params={"inflation": 2.0}).fit(d.train, make_rng(1))
    assert infl.diagnostics_["assigned_bad_rate_rejects"] > base.diagnostics_["assigned_bad_rate_rejects"]
    assert 0 <= base.diagnostics_["assigned_bad_rate_rejects"] <= 1
