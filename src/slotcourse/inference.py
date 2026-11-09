"""Turn any method into pixel-level segment ids for a batch of videos."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from slotcourse.baselines.hierarchical import hierarchical_frames
from slotcourse.baselines.kmeans import kmeans_frames, temporal_kmeans
from slotcourse.features import PCAFeatures, labels_to_pixels, patch_features, sample_features


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def segment_videos(model: torch.nn.Module, videos: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    """Learned model: uint8 [V, T, 64, 64, 3] -> pixel labels [V, T, 64, 64]."""
    was_training = model.training
    model.eval()
    out = []
    for s in range(0, len(videos), batch):
        x = torch.from_numpy(np.asarray(videos[s : s + batch])).to(device).float() / 255.0
        out.append(labels_to_pixels(model.segment(x)))
    model.train(was_training)
    return np.concatenate(out).astype(np.uint8)


def fit_pca(train_videos: np.ndarray, ccfg: dict[str, Any], seed: int = 0) -> PCAFeatures:
    feats = sample_features(train_videos, ccfg["pca_fit_frames"], seed=seed)
    return PCAFeatures.fit(feats, ccfg["pca_components"], ccfg["pos_weight"])


def cluster_videos(method: str, videos: np.ndarray, pca: PCAFeatures, ccfg: dict[str, Any], seed: int = 0) -> np.ndarray:
    """Clustering baseline C1/C2/C3: uint8 [V, T, 64, 64, 3] -> pixel labels [V, T, 64, 64]."""
    v, t = videos.shape[:2]
    feats = pca.transform(patch_features(np.asarray(videos)))  # [V, T, N, d]
    k, iters = ccfg["num_clusters"], ccfg["kmeans_iters"]
    if method == "C1":
        flat = torch.from_numpy(feats.reshape(v * t, *feats.shape[2:]))
        labels = kmeans_frames(flat, k, seed=seed, iters=iters).numpy().reshape(v, t, -1)
    elif method == "C2":
        labels = hierarchical_frames(feats.reshape(v * t, *feats.shape[2:]), k, ccfg["linkage"]).reshape(v, t, -1)
    elif method == "C3":
        labels = temporal_kmeans(torch.from_numpy(feats), k, seed=seed, iters=iters).numpy()
    else:
        raise ValueError(f"unknown clustering method {method!r}")
    return labels_to_pixels(labels).astype(np.uint8)
