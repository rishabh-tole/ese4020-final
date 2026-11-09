"""Nonparametric bootstrap over test videos (Module 3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    mean: float
    lo: float
    hi: float

    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0

    def fmt(self, scale: float = 100.0, digits: int = 1) -> str:
        m, lo, hi = (round(v * scale, digits) for v in (self.mean, self.lo, self.hi))
        return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _resample_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ci(scores: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> Interval:
    """Percentile CI for the mean of per-video scores (NaN videos ignored)."""
    x = np.asarray(scores, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    means = x[_resample_indices(x.size, n_boot, seed)].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(float(x.mean()), float(lo), float(hi))


def paired_diff_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> Interval:
    """CI for mean(b - a) resampling the same videos for both methods."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs scores on the same videos")
    keep = np.isfinite(a) & np.isfinite(b)
    return bootstrap_ci(b[keep] - a[keep], n_boot=n_boot, alpha=alpha, seed=seed)
