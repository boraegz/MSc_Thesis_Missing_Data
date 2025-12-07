import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across numpy, random, and torch.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

def load_config(config_path: str = "configs/experiment_config.yaml") -> dict:
    """
    Load YAML configuration file.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded from: {path}")
    return config

def get_project_root() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).parent.parent
