"""C2: Ward agglomerative clustering per frame, tree cut into K clusters."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering


def _cluster_frame(x: np.ndarray, k: int, linkage: str) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(x).astype(np.int64)


def hierarchical_frames(x: np.ndarray, k: int, linkage: str = "ward", n_jobs: int = -1) -> np.ndarray:
    """x [B, N, D] -> labels [B, N]; frames run in parallel processes."""
    labels = Parallel(n_jobs=n_jobs, batch_size=16)(delayed(_cluster_frame)(f, k, linkage) for f in x)
    return np.stack(labels)
