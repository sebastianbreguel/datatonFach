"""Centralized paths and constants."""

from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT / "data"
MODELS_DIR: Path = ROOT / "models"
FEATURES_DIR: Path = DATA_DIR / "features"
PREDICTIONS_DIR: Path = DATA_DIR / "predictions"

LABEL_NAMES: list[str] = [
    "high",
    "low",
    "moderate",
    "non-burnable",
    "very_high",
    "very_low",
    "water",
]

DEFAULT_PATCH_SIZE: tuple[int, int] = (224, 224)
DEFAULT_STRIDE: int = 112


def ensure_dirs() -> None:
    """Create standard output directories if missing."""
    for d in (DATA_DIR, MODELS_DIR, FEATURES_DIR, PREDICTIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
