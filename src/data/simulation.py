"""Synthetic credit population with a past acceptance policy (label selection).

Data-generating process
-----------------------
For applicant i with features X_i (multivariate normal, Toeplitz correlation):

    Y*_i = eta(X_i) + eps_Y,i                       Y_i = 1[Y*_i > t_Y]   (1 = default)
    R*_i = X_i gamma + delta * W_i + eps_S,i + tau * nu_i
                                                    S_i = 1[R*_i <= t_S]  (1 = accepted)
    (eps_Y, eps_S) ~ N(0, [[1, rho], [rho, 1]]),    nu_i ~ N(0, 1)

* ``eta(x) = x beta + g(x)`` is the true risk index. ``g`` adds a squared term
  and an interaction, so a linear (logistic) scorecard is misspecified, as
  real scorecards are. ``nonlinear_share`` sets g's share of the index variance.
* ``R*`` is the lender's perceived risk from its *old* scorecard ``x gamma``,
  which differs from the truth: it ignores g, misses the last two informative
  features and puts weight on two irrelevant ones. The lender accepts the
  share ``alpha`` with the lowest perceived risk.
* ``rho`` is the MNAR dial. With ``rho = 0`` selection depends on recorded
  data only (S independent of Y given X): labels are MAR. With ``rho > 0`` the
  lender sees part of ``eps_Y`` (e.g. a loan officer's judgement), so accepted
  applicants default less than their X predicts: labels are MNAR.
* ``W`` moves acceptance but not default (e.g. branch capacity). With an
  exclusion restriction, ``W = Z`` is recorded in the data; without one, ``W``
  is an unrecorded draw. Both versions select with exactly the same strength,
  so switching the restriction off changes only what the analyst can observe.
* ``tau * nu`` is pure policy noise (overrides, inconsistent decisions). It
  keeps acceptance probabilities away from 0 and 1.
* ``t_Y`` and ``t_S`` are set analytically so the population default rate is
  ``default_rate`` and the acceptance rate is ``alpha``.

Within one replication the applicants (X, Z, W, all errors) are identical
across policy settings; only the selection changes. Comparisons across cells
and methods are therefore paired (common random numbers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from scipy.stats import norm

from src.utils import make_rng

# Stream ids for make_rng, so each random step has its own independent stream.
_STREAM_TRAIN = 0
_STREAM_TEST = 1


@dataclass(frozen=True)
class PopulationConfig:
    """Parameters of the through-the-door applicant population.

    Attributes
    ----------
    n_features : int
        Number of applicant features p (at least n_informative + 2).
    n_informative : int
        Number of features with a linear effect on default (at least 3).
    feature_corr : float
        Toeplitz correlation, corr(X_j, X_k) = feature_corr ** |j - k|.
    default_rate : float
        Population default rate P(Y = 1).
    signal_sd : float
        Standard deviation of the risk index eta(X). Larger means an easier
        prediction problem (higher Oracle AUC).
    nonlinear_share : float
        Share of var(eta) due to the nonlinear part g, in [0, 1). 0 makes the
        logistic scorecard (nearly) correctly specified.
    """

    n_features: int = 10
    n_informative: int = 6
    feature_corr: float = 0.3
    default_rate: float = 0.2
    signal_sd: float = 1.0
    nonlinear_share: float = 0.3

    def __post_init__(self) -> None:
        if not 3 <= self.n_informative <= self.n_features - 2:
            raise ValueError("Need 3 <= n_informative <= n_features - 2.")
        if not 0 < self.default_rate < 1:
            raise ValueError("default_rate must lie in (0, 1).")
        if not -1 < self.feature_corr < 1:
            raise ValueError("feature_corr must lie in (-1, 1).")
        if self.signal_sd <= 0:
            raise ValueError("signal_sd must be positive.")
        if not 0 <= self.nonlinear_share < 1:
            raise ValueError("nonlinear_share must lie in [0, 1).")


@dataclass(frozen=True)
class PolicyConfig:
    """Parameters of the past acceptance policy that censors labels.

    Attributes
    ----------
    rho : float
        Correlation of the default and acceptance errors; 0 = MAR, > 0 = MNAR.
    acceptance_rate : float
        Share of historical applicants accepted (alpha).
    exclusion_restriction : bool
        Whether the acceptance shifter W is recorded as Z (True) or unobserved.
    z_strength : float
        Effect delta of W on perceived risk.
    policy_noise_sd : float
        Standard deviation tau of pure policy noise.
    """

    rho: float = 0.0
    acceptance_rate: float = 0.5
    exclusion_restriction: bool = True
    z_strength: float = 1.0
    policy_noise_sd: float = 0.5

    def __post_init__(self) -> None:
        if not 0 <= self.rho < 1:
            raise ValueError("rho must lie in [0, 1).")
        if not 0 < self.acceptance_rate < 1:
            raise ValueError("acceptance_rate must lie in (0, 1).")
        if self.z_strength < 0 or self.policy_noise_sd < 0:
            raise ValueError("z_strength and policy_noise_sd must be non-negative.")


@dataclass
class TrainingView:
    """Everything a reject inference method may see: the historical sample.

    Rejected applicants have ``y = NaN``. True labels of rejects are *not* part
    of this object, so no method can leak them by construction.

    Attributes
    ----------
    X : ndarray of shape (n, p)
        Applicant features.
    z : ndarray of shape (n,)
        Recorded acceptance shifter (exclusion restriction). When the
        restriction is switched off, z is recorded but irrelevant to acceptance.
    s : ndarray of shape (n,)
        1 if accepted, 0 if rejected.
    y : ndarray of shape (n,)
        Observed outcome: 0/1 for accepted, NaN for rejected.
    """

    X: np.ndarray
    z: np.ndarray
    s: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        if not np.array_equal(np.isnan(self.y), self.s == 0):
            raise ValueError("y must be NaN exactly for rejected applicants.")

    @property
    def accepted(self) -> np.ndarray:
        return self.s == 1

    @property
    def rejected(self) -> np.ndarray:
        return self.s == 0


@dataclass
class ExperimentData:
    """One simulated replication: biased training sample plus Oracle test set.

    Attributes
    ----------
    train : TrainingView
        Historical sample as the lender sees it.
    y_train_oracle : ndarray
        True labels of *all* historical applicants. Used only to fit the Oracle
        reference model and for diagnostics; never passed to a method.
    X_test, y_test : ndarray
        Fresh through-the-door applicants with all labels (evaluation only).
    meta : dict
        Ground-truth quantities (thresholds, realised rates).
    """

    train: TrainingView
    y_train_oracle: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Population structure (deterministic: it describes the world, not a sample)
# --------------------------------------------------------------------------

def feature_covariance(pop: PopulationConfig) -> np.ndarray:
    """Toeplitz covariance with unit variances."""
    idx = np.arange(pop.n_features)
    return pop.feature_corr ** np.abs(idx[:, None] - idx[None, :])


def _unit_sd(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return w / np.sqrt(w @ cov @ w)


def _raw_nonlinear(X: np.ndarray) -> np.ndarray:
    """Nonlinear part before scaling: (x0^2 - 1)/sqrt(2) + x1 * x2."""
    return (X[:, 0] ** 2 - 1) / np.sqrt(2) + X[:, 1] * X[:, 2]


def _nonlinear_moments(cov: np.ndarray) -> tuple[float, float]:
    """Exact mean and sd of :func:`_raw_nonlinear` under X ~ N(0, cov).

    Uses Isserlis' theorem: E[x1 x2] = c12, var(x1 x2) = 1 + c12^2,
    cov(x0^2, x1 x2) = 2 c01 c02. Both terms are uncorrelated with any linear
    combination of X (odd moments vanish), so var(eta) adds up exactly.
    """
    c01, c02, c12 = cov[0, 1], cov[0, 2], cov[1, 2]
    mean = c12
    var = 1.0 + (1.0 + c12**2) + 2 * (2 * c01 * c02) / np.sqrt(2)
    return float(mean), float(np.sqrt(var))


def true_coefficients(pop: PopulationConfig) -> np.ndarray:
    """Linear default coefficients beta, scaled to the linear share of signal_sd.

    Informative coefficients alternate in sign and decay from 1.0 to 0.3.
    """
    beta = np.zeros(pop.n_features)
    k = pop.n_informative
    beta[:k] = np.linspace(1.0, 0.3, k) * np.where(np.arange(k) % 2 == 0, 1.0, -1.0)
    return _unit_sd(beta, feature_covariance(pop)) * pop.signal_sd * np.sqrt(1 - pop.nonlinear_share)


def policy_coefficients(pop: PopulationConfig) -> np.ndarray:
    """The lender's old scorecard gamma (perceived-risk weights), sd(X gamma) = signal_sd.

    It keeps the linear weights of the first n_informative - 2 features, misses
    the last two informative features and puts weight on two irrelevant ones.
    """
    beta = true_coefficients(pop)
    k = pop.n_informative
    gamma = np.zeros(pop.n_features)
    gamma[: k - 2] = beta[: k - 2]
    gamma[k : k + 2] = 0.5 * np.abs(beta[: k - 2]).mean() * np.array([1.0, -1.0])
    return _unit_sd(gamma, feature_covariance(pop)) * pop.signal_sd


def risk_index(X: np.ndarray, pop: PopulationConfig) -> np.ndarray:
    """True risk index eta(X) with mean 0 and sd signal_sd."""
    lin = X @ true_coefficients(pop)
    if pop.nonlinear_share == 0:
        return lin
    m, sd = _nonlinear_moments(feature_covariance(pop))
    g = (_raw_nonlinear(X) - m) / sd * pop.signal_sd * np.sqrt(pop.nonlinear_share)
    return lin + g


def default_threshold(pop: PopulationConfig) -> float:
    """t_Y with P(eta + eps_Y > t_Y) = default_rate, using a normal approximation.

    With a nonlinear part, eta is not exactly normal, so the realised default
    rate deviates slightly from the target; it is reported in ``meta``.
    """
    return float(norm.ppf(1 - pop.default_rate) * np.sqrt(pop.signal_sd**2 + 1))


def acceptance_threshold(pop: PopulationConfig, policy: PolicyConfig) -> float:
    """t_S with P(R* <= t_S) = acceptance_rate (R* is exactly normal)."""
    sd = np.sqrt(pop.signal_sd**2 + policy.z_strength**2 + 1 + policy.policy_noise_sd**2)
    return float(norm.ppf(policy.acceptance_rate) * sd)


def true_pd(X: np.ndarray, pop: PopulationConfig) -> np.ndarray:
    """Population probability of default P(Y = 1 | X)."""
    return norm.sf(default_threshold(pop) - risk_index(X, pop))


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def _draw_applicants(pop: PopulationConfig, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Draw features, the acceptance shifters and the error components."""
    X = rng.multivariate_normal(np.zeros(pop.n_features), feature_covariance(pop), size=n)
    return {
        "X": X,
        "z": rng.standard_normal(n),   # recorded acceptance shifter
        "w": rng.standard_normal(n),   # unrecorded acceptance shifter
        "u1": rng.standard_normal(n),  # default error eps_Y
        "u2": rng.standard_normal(n),  # independent part of eps_S
        "nu": rng.standard_normal(n),  # policy noise
    }


def simulate_experiment(
    pop: PopulationConfig,
    policy: PolicyConfig,
    n_train: int,
    n_test: int,
    base_seed: int,
    replication: int,
) -> ExperimentData:
    """Simulate one replication of the label-selection experiment.

    Parameters
    ----------
    pop : PopulationConfig
        Applicant population.
    policy : PolicyConfig
        Past acceptance policy applied to the historical sample.
    n_train, n_test : int
        Sizes of the historical sample and of the Oracle test set.
    base_seed : int
        Global seed of the study.
    replication : int
        Replication index; together with ``base_seed`` it fixes the applicants.

    Returns
    -------
    ExperimentData
    """
    t_y = default_threshold(pop)
    t_s = acceptance_threshold(pop, policy)
    gamma = policy_coefficients(pop)

    tr = _draw_applicants(pop, n_train, make_rng(base_seed, replication, _STREAM_TRAIN))
    te = _draw_applicants(pop, n_test, make_rng(base_seed, replication, _STREAM_TEST))

    # Default outcome: identical for every policy within a replication.
    y_train = (risk_index(tr["X"], pop) + tr["u1"] > t_y).astype(int)
    y_test = (risk_index(te["X"], pop) + te["u1"] > t_y).astype(int)

    # Acceptance by the old scorecard, with eps_S = rho * eps_Y + sqrt(1 - rho^2) * u2.
    eps_s = policy.rho * tr["u1"] + np.sqrt(1 - policy.rho**2) * tr["u2"]
    shifter = tr["z"] if policy.exclusion_restriction else tr["w"]
    perceived_risk = tr["X"] @ gamma + policy.z_strength * shifter + eps_s + policy.policy_noise_sd * tr["nu"]
    s = (perceived_risk <= t_s).astype(int)

    y_obs = y_train.astype(float)
    y_obs[s == 0] = np.nan

    meta = {
        "t_y": t_y,
        "t_s": t_s,
        "rho": policy.rho,
        "acceptance_rate_target": policy.acceptance_rate,
        "acceptance_rate_realised": float(s.mean()),
        "exclusion_restriction": policy.exclusion_restriction,
        "default_rate_population": float(y_train.mean()),
        "default_rate_accepted": float(y_train[s == 1].mean()) if s.any() else float("nan"),
        "default_rate_rejected": float(y_train[s == 0].mean()) if (s == 0).any() else float("nan"),
        "default_rate_test": float(y_test.mean()),
    }
    return ExperimentData(
        train=TrainingView(X=tr["X"], z=tr["z"], s=s, y=y_obs),
        y_train_oracle=y_train,
        X_test=te["X"],
        y_test=y_test,
        meta=meta,
    )
