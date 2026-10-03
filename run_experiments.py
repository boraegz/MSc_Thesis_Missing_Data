"""Run the simulation study and save raw results plus a summary table.

Usage (from the project root)::

    python run_experiments.py                       # full grid from the config
    python run_experiments.py --quick               # 2 replications, smoke test
    python run_experiments.py --config configs/experiment_config.yaml --reps 5
"""
from __future__ import annotations

import argparse
import time

from src.experiments import run_study, save_results, summarise
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--reps", type=int, default=None, help="override number of replications")
    parser.add_argument("--quick", action="store_true", help="2 replications for a fast check")
    parser.add_argument("--tag", default=None, help="suffix for the output file name")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.quick:
        config["study"]["n_replications"] = 2
    elif args.reps is not None:
        config["study"]["n_replications"] = args.reps

    t0 = time.perf_counter()
    df = run_study(config, verbose=5)
    path = save_results(df, config, tag=args.tag)
    summary = summarise(df)
    summary.to_csv(path.with_name(path.stem + "_summary.csv"), index=False)
    print(f"\n{len(df)} rows in {time.perf_counter() - t0:.0f}s -> {path}")

    cols = ["rho", "acceptance_rate", "method", "auc_mean", "auc_vs_accepts_mean", "auc_vs_accepts_ci95", "mean_pd_mean"]
    view = summary[summary["exclusion_restriction"]][cols].round(4)
    print(view.to_string(index=False))


if __name__ == "__main__":
    main()
