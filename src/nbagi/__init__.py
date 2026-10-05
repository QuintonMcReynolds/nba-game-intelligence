"""NBA play-by-play intelligence: lake, win probability, clutch, fact-checked AI recaps."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LAKE = DATA / "lake"
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)
