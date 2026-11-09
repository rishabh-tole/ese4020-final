"""Batched K-means (C1) and temporal K-means (C3) on PCA patch features.

Many frames are clustered at once: X is [B, N, D] (B frames, N=1024 patches each).
"""

from __future__ import annotations

import torch


def _sq_dists(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """[B, N, D], [B, K, D] -> squared distances [B, N, K]."""
    return (x.pow(2).sum(-1, keepdim=True) - 2 * x @ c.transpose(1, 2) + c.pow(2).sum(-1).unsqueeze(1)).clamp_min(0)


def kmeans_pp_init(x: torch.Tensor, k: int, gen: torch.Generator | None = None) -> torch.Tensor:
    """k-means++ seeding for every frame in the batch -> [B, K, D].

    Seed choice is non-differentiable, but the returned centers are gathered from x, so
    gradients still flow to the chosen points (used inside the learned slot module).
    """
    b, n, _ = x.shape
    rows = torch.arange(b, device=x.device)
    xd = x.detach()
    first = torch.randint(0, n, (b,), generator=gen, device="cpu").to(x.device)
    idx_list = [first]
    d2 = _sq_dists(xd, xd[rows, first].unsqueeze(1)).squeeze(-1)
    for _ in range(1, k):
        probs = d2 / d2.sum(1, keepdim=True).clamp_min(1e-12)
        probs = torch.where(probs.sum(1, keepdim=True) > 0, probs, torch.full_like(probs, 1.0 / n))
        idx = torch.multinomial(probs.cpu(), 1, generator=gen).squeeze(1).to(x.device)
        idx_list.append(idx)
        d2 = torch.minimum(d2, _sq_dists(xd, xd[rows, idx].unsqueeze(1)).squeeze(-1))
    idx_all = torch.stack(idx_list, dim=1)  # [B, K]
    return torch.gather(x, 1, idx_all.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def lloyd(x: torch.Tensor, centers: torch.Tensor, iters: int = 50, tol: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd iterations from given centers. Empty clusters keep their previous center.

    Returns (labels [B, N], centers [B, K, D]).
    """
    k = centers.shape[1]
    c = centers
    for _ in range(iters):
        labels = _sq_dists(x, c).argmin(-1)
        onehot = torch.nn.functional.one_hot(labels, k).to(x.dtype)  # [B, N, K]
        counts = onehot.sum(1)  # [B, K]
        sums = onehot.transpose(1, 2) @ x  # [B, K, D]
        new_c = torch.where(counts.unsqueeze(-1) > 0, sums / counts.clamp_min(1).unsqueeze(-1), c)
        shift = (new_c - c).pow(2).sum(-1).max()
        c = new_c
        if shift < tol:
            break
    return _sq_dists(x, c).argmin(-1), c


def kmeans_frames(x: torch.Tensor, k: int, seed: int = 0, iters: int = 50, chunk: int = 2048) -> torch.Tensor:
    """C1: independent K-means per frame. x [B, N, D] -> labels [B, N]."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for s in range(0, x.shape[0], chunk):
        xb = x[s : s + chunk]
        labels, _ = lloyd(xb, kmeans_pp_init(xb, k, gen), iters)
        out.append(labels)
    return torch.cat(out)


def temporal_kmeans(x: torch.Tensor, k: int, seed: int = 0, iters: int = 50) -> torch.Tensor:
    """C3: x [V, T, N, D]. Frame 0 uses k-means++; frame t starts from frame t-1's centroids.

    Because centroid k at frame t descends from centroid k at frame t-1, cluster ids carry
    across frames and can track objects without any learning.
    """
    gen = torch.Generator().manual_seed(seed)
    labels_t, centers = lloyd(x[:, 0], kmeans_pp_init(x[:, 0], k, gen), iters)
    labels = [labels_t]
    for t in range(1, x.shape[1]):
        labels_t, centers = lloyd(x[:, t], centers, iters)
        labels.append(labels_t)
    return torch.stack(labels, dim=1)
