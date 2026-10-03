"""Run the simulation study: grid of policy settings x replications x methods."""
from __future__ import annotations

import itertools
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.base import clone

from src.data.simulation import ExperimentData, PolicyConfig, PopulationConfig, simulate_experiment
from src.evaluation import compute_metrics
from src.reject_inference import build_method, make_learner
from src.utils import PROJECT_ROOT, make_rng

ORACLE = "oracle"


def grid_cells(config: Dict[str, Any]) -> List[PolicyConfig]:
    """All policy settings (cells) defined by the config grid, in a fixed order."""
    g = config["grid"]
    fixed = config.get("policy", {})
    return [
        PolicyConfig(rho=rho, acceptance_rate=alpha, exclusion_restriction=bool(er), **fixed)
        for rho, alpha, er in itertools.product(g["rho"], g["acceptance_rate"], g["exclusion_restriction"])
    ]


def evaluate_method(
    spec: Dict[str, Any],
    data: ExperimentData,
    learner_kind: str,
    approval_rate: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Fit one reject inference method and score it on the Oracle test set.

    The method receives only ``data.train`` (the censored historical sample);
    ``data.y_train_oracle`` is never passed on, so the true labels of rejected
    applicants cannot leak into training.
    """
    method = build_method(spec["name"], learner_kind, spec.get("params"))
    t0 = time.perf_counter()
    method.fit(data.train, rng)
    fit_seconds = time.perf_counter() - t0
    metrics = compute_metrics(data.y_test, method.predict_proba(data.X_test), approval_rate)
    diag = {f"diag_{k}": v for k, v in method.diagnostics_.items()}
    return {"method": spec["name"], "fit_seconds": fit_seconds, **metrics, **diag}


def evaluate_oracle(data: ExperimentData, learner_kind: str, approval_rate: float) -> Dict[str, Any]:
    """Upper reference: the same learner fit on all true labels of the historical sample."""
    t0 = time.perf_counter()
    model = clone(make_learner(learner_kind)).fit(data.train.X, data.y_train_oracle)
    fit_seconds = time.perf_counter() - t0
    metrics = compute_metrics(data.y_test, model.predict_proba(data.X_test)[:, 1], approval_rate)
    return {"method": ORACLE, "fit_seconds": fit_seconds, **metrics}


def run_replication(config: Dict[str, Any], cell_id: int, policy: PolicyConfig, replication: int) -> List[Dict[str, Any]]:
    """Simulate one replication of one cell and evaluate the Oracle and every method.

    All methods see the same simulated data, so method comparisons are paired.

    Returns
    -------
    list of dict
        One row per method with cell settings, metrics and diagnostics.
    """
    study = config["study"]
    pop = PopulationConfig(**config["population"])
    data = simulate_experiment(
        pop,
        policy,
        n_train=config["samples"]["n_train"],
        n_test=config["samples"]["n_test"],
        base_seed=study["base_seed"],
        replication=replication,
    )
    approval_rate = config["evaluation"]["approval_rate"]
    learner_kind = config.get("learner", "logistic")
    base = {
        "cell_id": cell_id,
        "replication": replication,
        "rho": policy.rho,
        "acceptance_rate": policy.acceptance_rate,
        "exclusion_restriction": policy.exclusion_restriction,
        "acceptance_rate_realised": data.meta["acceptance_rate_realised"],
        "default_rate_accepted": data.meta["default_rate_accepted"],
        "default_rate_rejected": data.meta["default_rate_rejected"],
        "default_rate_test": data.meta["default_rate_test"],
        "learner": learner_kind,
    }
    rows = [{**base, **evaluate_oracle(data, learner_kind, approval_rate)}]
    for m_idx, spec in enumerate(config["methods"]):
        rng = make_rng(study["base_seed"], replication, 1000 + cell_id, m_idx)
        rows.append({**base, **evaluate_method(spec, data, learner_kind, approval_rate, rng)})
    return rows


def run_study(config: Dict[str, Any], replications: Iterable[int] | None = None, verbose: int = 0) -> pd.DataFrame:
    """Run every (cell, replication) pair, in parallel.

    Parameters
    ----------
    config : dict
        Parsed experiment configuration.
    replications : iterable of int, optional
        Replication indices; defaults to ``range(n_replications)``.
    verbose : int
        joblib verbosity.
    """
    cells = grid_cells(config)
    reps = list(replications) if replications is not None else list(range(config["study"]["n_replications"]))
    jobs = [(cid, cell, r) for cid, cell in enumerate(cells) for r in reps]
    results = Parallel(n_jobs=config["study"].get("n_jobs", 1), verbose=verbose)(
        delayed(run_replication)(config, cid, cell, r) for cid, cell, r in jobs
    )
    return pd.DataFrame([row for rows in results for row in rows])


def save_results(df: pd.DataFrame, config: Dict[str, Any], tag: str | None = None) -> Path:
    """Write results as CSV next to a copy of the config that produced them."""
    out_dir = PROJECT_ROOT / config.get("output_dir", "results/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{config['study']['name']}_{stamp}.csv"
    df.to_csv(path, index=False)
    with open(path.with_suffix(".yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


def summarise(df: pd.DataFrame, metrics: Iterable[str] = ("auc", "brier", "ks", "bad_rate", "mean_pd")) -> pd.DataFrame:
    """Mean and 95% confidence half-width per cell and method.

    Besides the raw metric, two *paired* differences are reported, computed
    within each replication before averaging (so shared noise cancels):

    * ``<metric>_vs_oracle``: method minus Oracle (the cost of missing labels);
    * ``<metric>_vs_accepts``: method minus accepts-only (the gain from reject
      inference; positive is better for AUC and KS, negative for Brier and
      bad rate).

    Half-widths use the normal approximation 1.96 * sd / sqrt(n_reps).
    """
    metrics = list(metrics)
    keys = ["rho", "acceptance_rate", "exclusion_restriction", "method"]
    cell_rep = ["rho", "acceptance_rate", "exclusion_restriction", "replication"]

    paired = df.copy()
    for ref_name, suffix in ((ORACLE, "vs_oracle"), ("accepts_only", "vs_accepts")):
        ref = df[df["method"] == ref_name][cell_rep + metrics]
        if ref.empty:
            continue
        ref = ref.rename(columns={m: f"{m}__ref" for m in metrics})
        paired = paired.merge(ref, on=cell_rep, how="left")
        for m in metrics:
            paired[f"{m}_{suffix}"] = paired[m] - paired[f"{m}__ref"]
        paired = paired.drop(columns=[f"{m}__ref" for m in metrics])

    value_cols = [c for c in paired.columns if any(c == m or c.startswith(f"{m}_vs_") for m in metrics)]
    g = paired.groupby(keys)
    n = g.size()
    out = g[value_cols].mean().add_suffix("_mean")
    half = (1.96 * g[value_cols].std(ddof=1)).div(np.sqrt(n), axis=0).add_suffix("_ci95")
    out = out.join(half)
    out["n_reps"] = n
    return out.reset_index()
