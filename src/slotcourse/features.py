"""Patch features shared by every method, plus PCA for the clustering baselines.

Each 64x64 frame is cut into 2x2 patches -> 32x32 = 1024 patches. A patch's feature is
its 12 pixel values (in [0, 1]) followed by its row and column (in [0, 1]).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from slotcourse.config import FEATURE_DIM, FRAME_SIZE, GRID, PATCH


def patchify(frames: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """[..., 64, 64, 3] -> [..., 1024, 12] with per-patch order (dy, dx, channel)."""
    lead = frames.shape[:-3]
    x = frames.reshape(*lead, GRID, PATCH, GRID, PATCH, 3)
    if isinstance(x, torch.Tensor):
        x = x.permute(*range(len(lead)), -5, -3, -4, -2, -1)
    else:
        x = np.moveaxis(x, -4, -3)  # (gy, py, gx, px, c) -> (gy, gx, py, px, c)
    return x.reshape(*lead, GRID * GRID, PATCH * PATCH * 3)


def patch_positions() -> np.ndarray:
    """[1024, 2] (row, col) of each patch, scaled to [0, 1]."""
    r, c = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    return np.stack([r.ravel(), c.ravel()], axis=-1).astype(np.float32) / (GRID - 1)


def patch_features(frames_uint8: np.ndarray) -> np.ndarray:
    """[..., 64, 64, 3] uint8 -> [..., 1024, 14] float32 (pixels then row, col)."""
    pix = patchify(frames_uint8).astype(np.float32) / 255.0
    pos = np.broadcast_to(patch_positions(), pix.shape[:-1] + (2,))
    return np.concatenate([pix, pos], axis=-1)


def labels_to_pixels(labels: np.ndarray | torch.Tensor) -> np.ndarray:
    """Patch labels [..., 1024] -> pixel labels [..., 64, 64] (each patch covers 2x2 px)."""
    lab = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    grid = lab.reshape(*lab.shape[:-1], GRID, GRID)
    return grid.repeat(PATCH, axis=-2).repeat(PATCH, axis=-1).reshape(*lab.shape[:-1], FRAME_SIZE, FRAME_SIZE)


@dataclass(frozen=True)
class PCAFeatures:
    """Standardize -> (optionally up-weight position) -> project onto top principal axes."""

    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    components: np.ndarray  # [n_components, 14]
    explained_variance_ratio: np.ndarray

    @classmethod
    def fit(cls, feats: np.ndarray, n_components: int, pos_weight: float = 1.0) -> PCAFeatures:
        x = feats.reshape(-1, feats.shape[-1]).astype(np.float64)
        if n_components < 1 or n_components > x.shape[1]:
            raise ValueError(f"n_components must be in [1, {x.shape[1]}], got {n_components}")
        mean, std = x.mean(0), x.std(0) + 1e-8
        weights = np.ones(x.shape[1])
        weights[-2:] = pos_weight
        z = (x - mean) / std * weights
        _, s, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
        var = s**2
        return cls(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            weights=weights.astype(np.float32),
            components=vt[:n_components].astype(np.float32),
            explained_variance_ratio=(var[:n_components] / var.sum()).astype(np.float32),
        )

    def transform(self, feats: np.ndarray) -> np.ndarray:
        z = (feats - self.mean) / self.std * self.weights
        return (z @ self.components.T).astype(np.float32)


def sample_features(videos: np.ndarray, n_frames: int, seed: int = 0) -> np.ndarray:
    """Random frames from [V, T, 64, 64, 3] -> stacked features [n_frames * 1024, 14] for PCA fitting."""
    rng = np.random.default_rng(seed)
    v = rng.integers(0, videos.shape[0], n_frames)
    t = rng.integers(0, videos.shape[1], n_frames)
    order = np.argsort(v)
    frames = np.stack([np.asarray(videos[v[i], t[i]]) for i in order])
    return patch_features(frames).reshape(-1, FEATURE_DIM)
