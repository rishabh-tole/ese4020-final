"""Project-wide constants. Hyperparameters live in configs/*.yaml."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SLOTCOURSE_DATA", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("SLOTCOURSE_RESULTS", ROOT / "results"))

NUM_FRAMES = 24  # MOVi-A clip length
FRAME_SIZE = 64  # frames downsampled from 128x128
PATCH = 2  # 2x2 pixel patches
GRID = FRAME_SIZE // PATCH  # 32x32 = 1024 patches per frame
NUM_PATCHES = GRID * GRID
PIXEL_DIM = PATCH * PATCH * 3  # 12 pixel values per patch
FEATURE_DIM = PIXEL_DIM + 2  # + row, column

VAL_HOLDOUT = 500  # training videos held out for model selection
SPLIT_SEED = 4020
