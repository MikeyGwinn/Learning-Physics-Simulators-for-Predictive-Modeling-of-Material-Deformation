"""Configuration and shared training helpers."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def get_config(path):
    with Path(path).open() as stream:
        cfg = yaml.safe_load(stream)
    for section, key in [
        (cfg["dataset"], "data_dir"),
        (cfg, "exp_dir"),
        (cfg, "output_dir"),
    ]:
        value = Path(section[key]).expanduser()
        section[key] = str(value if value.is_absolute() else ROOT / value)
    if cfg["training"]["rollout_num"] < 2:
        raise ValueError("training.rollout_num must be at least 2.")
    return cfg


def seed_everything(seed, cpu_threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
