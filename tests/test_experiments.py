"""Tests for the study runner and the summary table."""
import copy

import numpy as np

from src.experiments import ORACLE, grid_cells, run_replication, run_study, summarise
from src.utils import load_config


def _small_config():
    cfg = copy.deepcopy(load_config("configs/experiment_config.yaml"))
    cfg["samples"] = {"n_train": 3000, "n_test": 2000}
    cfg["grid"] = {"rho": [0.0, 0.6], "acceptance_rate": [0.5], "exclusion_restriction": [True]}
    cfg["study"]["n_replications"] = 2
    cfg["study"]["n_jobs"] = 1
    return cfg


def test_full_config_has_24_cells():
    assert len(grid_cells(load_config("configs/experiment_config.yaml"))) == 24


def test_run_replication_rows():
    cfg = _small_config()
    rows = run_replication(cfg, 0, grid_cells(cfg)[0], replication=0)
    methods = [r["method"] for r in rows]
    assert methods[0] == ORACLE
    assert methods[1:] == [m["name"] for m in cfg["methods"]]
    assert all(np.isfinite(r["auc"]) for r in rows)


def test_study_is_reproducible_and_summary_is_paired():
    cfg = _small_config()
    a = run_study(cfg)
    b = run_study(cfg)
    np.testing.assert_allclose(a["auc"].to_numpy(), b["auc"].to_numpy())

    s = summarise(a)
    assert (s["n_reps"] == 2).all()
    oracle = s[s["method"] == ORACLE]
    np.testing.assert_allclose(oracle["auc_vs_oracle_mean"], 0.0)
    acc = s[s["method"] == "accepts_only"]
    np.testing.assert_allclose(acc["auc_vs_accepts_mean"], 0.0)
    assert {"auc_mean", "auc_ci95", "mean_pd_vs_oracle_mean"} <= set(s.columns)
