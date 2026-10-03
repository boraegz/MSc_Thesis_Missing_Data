"""Draw the main thesis figures from a results file written by run_experiments.py.

Usage (from the project root)::

    python make_figures.py                                  # newest results file
    python make_figures.py results/raw/synthetic_main_X.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.experiments import ORACLE, summarise  # noqa: E402
from src.utils import PROJECT_ROOT  # noqa: E402

FIG_DIR = PROJECT_ROOT / "results" / "figures"

# Fixed method -> colour/marker assignment (identity never depends on order or filters).
STYLE = {
    "accepts_only": ("#2a78d6", "o", "Accepts only"),
    "rejects_as_bad": ("#eb6834", "s", "Rejects = bad"),
    "ipw": ("#1baf7a", "^", "IPW"),
    "fuzzy_augmentation": ("#eda100", "D", "Fuzzy augmentation"),
    "parceling": ("#e87ba4", "v", "Parceling"),
    "mice_label": ("#008300", "P", "MICE (labels)"),
    "self_learning": ("#4a3aa7", "X", "Self-learning"),
    "heckman": ("#e34948", "*", "Heckman"),
}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e0"


def _latest_results() -> Path:
    files = sorted((PROJECT_ROOT / "results" / "raw").glob("*.csv"), key=lambda p: p.stat().st_mtime)
    files = [f for f in files if not f.stem.endswith("_summary")]
    if not files:
        raise FileNotFoundError("No results in results/raw. Run run_experiments.py first.")
    return files[-1]


def _style_axes(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def metric_vs_rho(summary: pd.DataFrame, metric: str, ylabel: str, title: str, out: Path,
                  exclusion_restriction: bool = True, reference: str | None = ORACLE,
                  exclude: tuple[str, ...] = (), dodge: float = 0.012) -> None:
    """One panel per acceptance rate: metric (mean with 95% CI) against MNAR strength rho.

    Methods are shifted slightly along x (``dodge``) so overlapping lines, such
    as accepts-only and fuzzy augmentation, stay visible.
    """
    s = summary[summary["exclusion_restriction"] == exclusion_restriction]
    alphas = sorted(s["acceptance_rate"].unique())
    fig, axes = plt.subplots(1, len(alphas), figsize=(3.6 * len(alphas), 3.4), sharey=True)
    for ax, alpha in zip(axes, alphas):
        cell = s[s["acceptance_rate"] == alpha]
        if reference is not None:
            ref = cell[cell["method"] == reference].sort_values("rho")
            ax.plot(ref["rho"], ref[f"{metric}_mean"], color=TEXT, linestyle="--", linewidth=1.5, label="Oracle")
        present = [k for k in STYLE if k not in exclude and (cell["method"] == k).any()]
        for i, method in enumerate(present):
            color, marker, label = STYLE[method]
            m = cell[cell["method"] == method].sort_values("rho")
            shift = (i - (len(present) - 1) / 2) * dodge
            ax.errorbar(m["rho"] + shift, m[f"{metric}_mean"], yerr=m[f"{metric}_ci95"], color=color, marker=marker,
                        markersize=6, linewidth=2, capsize=3, label=label)
        ax.set_title(f"Acceptance rate {alpha:.0%}", fontsize=10, color=TEXT)
        ax.set_xlabel("MNAR strength ρ (0 = MAR)", fontsize=9, color=TEXT)
        ax.set_xticks(sorted(cell["rho"].unique()))
        _style_axes(ax)
    axes[0].set_ylabel(ylabel, fontsize=9, color=TEXT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_results()
    df = pd.read_csv(path)
    summary = summarise(df)
    true_rate = df["default_rate_test"].mean()
    metric_vs_rho(summary, "auc", "AUC on all applicants",
                  "Ranking quality falls with MNAR strength and lower acceptance", FIG_DIR / "auc_vs_rho.png")
    metric_vs_rho(summary, "mean_pd", f"Mean predicted PD (true ≈ {true_rate:.2f})",
                  "Under MNAR, models trained on accepts underestimate the default rate",
                  FIG_DIR / "calibration_vs_rho.png", exclude=("rejects_as_bad",))
    metric_vs_rho(summary, "bad_rate", "Default rate among approved (50% approval)",
                  "Business impact: defaults among approved applicants", FIG_DIR / "bad_rate_vs_rho.png")
    print(f"Figures from {path.name} written to {FIG_DIR}")


if __name__ == "__main__":
    main()
