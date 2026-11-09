"""Segmentation metrics. Ground-truth masks are used only here.

All metrics ignore the background (GT id 0) and return values in [0, 1]; the report
multiplies by 100. Per-video scores are what the bootstrap resamples.

    fg_ari     per-frame FG-ARI, averaged over frames with >=2 foreground pixels
    fg_ari_v   FG-ARI over all foreground pixels of the whole video (penalizes label switches)
    mbo_f      per-frame mean best-overlap IoU, averaged over frames with any object
    mbo_v      mean best-overlap IoU of spatiotemporal (whole-video) masks
"""

from __future__ import annotations

import numpy as np


def _contingency(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Dense contingency table [n_gt_ids, n_pred_ids] for flat non-negative int arrays."""
    g = gt.astype(np.int64).ravel()
    p = pred.astype(np.int64).ravel()
    n_g, n_p = int(g.max(initial=0)) + 1, int(p.max(initial=0)) + 1
    return np.bincount(g * n_p + p, minlength=n_g * n_p).reshape(n_g, n_p)


def _comb2(x: np.ndarray | float) -> np.ndarray | float:
    return x * (x - 1) / 2.0


def ari(gt: np.ndarray, pred: np.ndarray) -> float:
    """Adjusted Rand index (matches sklearn.metrics.adjusted_rand_score)."""
    n = gt.size
    if n < 2:
        return float("nan")
    c = _contingency(gt, pred)
    index = _comb2(c).sum()
    sum_a, sum_b = _comb2(c.sum(1)).sum(), _comb2(c.sum(0)).sum()
    expected = sum_a * sum_b / _comb2(float(n))
    maximum = (sum_a + sum_b) / 2.0
    if np.isclose(maximum, expected):
        return 1.0
    return float((index - expected) / (maximum - expected))


def fg_ari(gt: np.ndarray, pred: np.ndarray) -> float:
    """FG-ARI over the given pixels (any shape); NaN if <2 foreground pixels."""
    fg = gt > 0
    if fg.sum() < 2:
        return float("nan")
    return ari(gt[fg], pred[fg])


def _best_overlap(gt: np.ndarray, pred: np.ndarray) -> float:
    """Mean over GT objects (ids > 0 present) of the best IoU with any predicted segment."""
    c = _contingency(gt, pred)
    gt_area, pred_area = c.sum(1), c.sum(0)
    present = np.nonzero(gt_area[1:])[0] + 1
    if present.size == 0:
        return float("nan")
    inter = c[present]
    union = gt_area[present, None] + pred_area[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return float(iou.max(1).mean())


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def score_video(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Score one video. gt, pred: [T, H, W] integer masks with matching shapes."""
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt {gt.shape} vs pred {pred.shape}")
    return {
        "fg_ari": _nanmean([fg_ari(g, p) for g, p in zip(gt, pred)]),
        "fg_ari_v": fg_ari(gt, pred),
        "mbo_f": _nanmean([_best_overlap(g, p) for g, p in zip(gt, pred)]),
        "mbo_v": _best_overlap(gt, pred),
    }


def mbo_v(gt: np.ndarray, pred: np.ndarray) -> float:
    return _best_overlap(gt, pred)


METRICS = ("fg_ari", "fg_ari_v", "mbo_f", "mbo_v")
