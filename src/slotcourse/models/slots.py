"""Building blocks: patch encoder, soft K-means slot module, spatial-broadcast MLP decoder.

Soft K-means is our course stand-in for slot attention: each iteration softly assigns every
patch to the K slots (softmax over slots of negative squared distance), then moves each slot
to the assignment-weighted mean of its patches. Slot attention (Slot-BERT eqs. 1-3) is the
same two steps with learned projections and a GRU update.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from slotcourse.baselines.kmeans import kmeans_pp_init


def mlp(sizes: list[int], final_act: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2 or final_act:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class PatchEncoder(nn.Module):
    """Per-patch MLP: 14 raw features (12 pixels + row, col) -> D-dim key."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = mlp([in_dim, hidden, hidden, hidden], final_act=True)
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)  # keeps key scale fixed so the temperature means something

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.proj(self.norm(self.net(x))))


class SoftKMeans(nn.Module):
    """Differentiable K-means over patch keys with a learned Gaussian slot initialisation."""

    def __init__(self, dim: int, num_slots: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.mu = nn.Parameter(torch.zeros(1, 1, dim))
        self.log_sigma = nn.Parameter(torch.zeros(1, 1, dim))  # unit spread, matching LayerNorm'd keys
        # learned inverse temperature, initialised so logits = -squared distance (sharp enough
        # that k-means++ seeds stay distinct; a soft start collapses every slot to the mean)
        self.log_beta = nn.Parameter(torch.tensor(0.5 * math.log(dim)))
        self.scale = 1.0 / math.sqrt(dim)
        self.eps = eps

    def init_slots(self, keys: torch.Tensor) -> torch.Tensor:
        """Fresh slots for keys [B, N, D]: k-means++ seeds on the keys (the same seeding C1
        uses) plus a small learned offset. Learned-Gaussian init alone collapses: all slots
        drift to the global mean under soft assignments and never separate again.
        """
        seeds = kmeans_pp_init(keys, self.num_slots)
        noise = torch.randn_like(seeds)
        return seeds + self.mu + 0.1 * self.log_sigma.exp() * noise

    def assign(self, keys: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """Soft assignment [B, N, K]: softmax over slots of -beta * ||key - slot||^2 / sqrt(D).

        A near-uniform assignment would pull every slot to the global mean, where they stay
        identical forever; the 1/sqrt(D) scale (as in slot attention) keeps it sharp enough
        that randomly initialised slots break symmetry.
        """
        d2 = (keys.pow(2).sum(-1, keepdim=True) - 2 * keys @ slots.transpose(1, 2)
              + slots.pow(2).sum(-1).unsqueeze(1)).clamp_min(0)
        return torch.softmax(-self.log_beta.exp() * self.scale * d2, dim=-1)

    def forward(self, keys: torch.Tensor, slots: torch.Tensor, iters: int) -> tuple[torch.Tensor, torch.Tensor]:
        """keys [B, N, D], slots [B, K, D] -> (slots [B, K, D], assignment [B, N, K])."""
        for _ in range(iters):
            attn = self.assign(keys, slots)
            weights = attn / (attn.sum(dim=1, keepdim=True) + self.eps)
            slots = weights.transpose(1, 2) @ keys
        return slots, self.assign(keys, slots)


class BroadcastDecoder(nn.Module):
    """Decode each slot at every patch position -> 12 pixel values per (slot, patch).

    The model reconstructs patch n as sum_k assign[n, k] * decode(slot_k, position_n), so the
    soft K-means assignment is the only route from pixels to reconstruction: the clustering is
    a true bottleneck and its argmax is the segmentation. (A decoder that predicts its own
    alpha masks learned to paint by position alone and ignore the slots.)
    """

    def __init__(self, slot_dim: int, hidden: int, num_patches: int, out_dim: int) -> None:
        super().__init__()
        self.slot_proj = nn.Linear(slot_dim, hidden)
        self.pos = nn.Parameter(torch.randn(num_patches, hidden) * 0.02)
        self.net = mlp([hidden, hidden, hidden, out_dim])

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """slots [B, K, D] -> per-slot reconstructions [B, K, N, out_dim]."""
        return self.net(torch.relu(self.slot_proj(slots).unsqueeze(2) + self.pos))
