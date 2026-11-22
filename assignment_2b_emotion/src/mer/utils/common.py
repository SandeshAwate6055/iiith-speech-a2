"""Shared utilities: seeding, config loading, split loading, emotion labels."""
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_split(protocol: str, splits_dir: str | Path = "splits") -> dict:
    p = Path(splits_dir) / f"protocol_{protocol.upper()}.json"
    return json.loads(p.read_text(encoding="utf-8"))
