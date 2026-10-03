"""Shared helpers: configuration loading, seeding and paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = "configs/experiment_config.yaml") -> Dict[str, Any]:
    """Load a YAML experiment configuration.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file, absolute or relative to the project root.

    Returns
    -------
    dict
        Parsed configuration.
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_rng(*keys: int | Sequence[int]) -> np.random.Generator:
    """Create an independent, reproducible random generator from integer keys.

    Every random step in the project draws from a generator created here, so a
    run is fully determined by its keys (e.g. ``base_seed``, replication index,
    stream id). No global seeding (``np.random.seed``) is used anywhere.

    Parameters
    ----------
    *keys : int or sequence of int
        Entropy keys, combined via :class:`numpy.random.SeedSequence`.

    Returns
    -------
    numpy.random.Generator
    """
    flat: list[int] = []
    for k in keys:
        if isinstance(k, (list, tuple, np.ndarray)):
            flat.extend(int(v) for v in k)
        else:
            flat.append(int(k))
    return np.random.default_rng(np.random.SeedSequence(flat))
