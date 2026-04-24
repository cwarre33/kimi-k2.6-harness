"""SOTA score registry loader."""

from pathlib import Path
from typing import Any, Dict

import yaml

_SCORES_PATH = Path(__file__).parent / "sota_scores.yaml"


def load_sota_scores() -> Dict[str, Any]:
    """Load published SOTA scores from YAML registry."""
    with _SCORES_PATH.open("r") as f:
        return yaml.safe_load(f)
