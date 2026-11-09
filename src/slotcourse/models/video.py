"""The learned rungs L1-L4 as one configurable video model.

    L1  encoder -> soft K-means slots -> decoder, each frame alone (fresh k-means++ init per frame)
    L2  + recurrence: frame t's slots start from frame t-1's final slots
    L3  + bidirectional GRU over each slot's track, trained with random frames masked
    L4  + penalty on cosine similarity between slots (ridge-style term added to the loss)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import NamedTuple

import torch
from torch import nn

from slotcourse.config import FEATURE_DIM, NUM_PATCHES, PIXEL_DIM
from slotcourse.features import patch_positions, patchify
from slotcourse.models.slots import BroadcastDecoder, PatchEncoder, SoftKMeans

VARIANTS = ("L1", "L2", "L3", "L4")


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "L1"
    num_slots: int = 11
    hidden: int = 64
    slot_dim: int = 64
    decoder_hidden: int = 128
    gru_hidden: int = 64
    iters_first: int = 3  # soft K-means iterations on a fresh init
    iters_next: int = 2  # iterations when warm-started from the previous frame
    mask_ratio: float = 0.0  # fraction of frames masked for the BiGRU during training
    slot_penalty: float = 0.0  # weight on mean squared cosine similarity between slots

    @property
    def recurrent(self) -> bool:
        return self.variant in ("L2", "L3", "L4")

    @property
    def temporal(self) -> bool:
        return self.variant in ("L3", "L4")

    @classmethod
    def for_variant(cls, variant: str, **overrides: object) -> ModelConfig:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        defaults: dict[str, object] = {"variant": variant}
        if variant in ("L3", "L4"):
            defaults["mask_ratio"] = 0.3
        if variant == "L4":
            defaults["slot_penalty"] = 0.1
        cfg = cls(**{**defaults, **{k: v for k, v in overrides.items() if v is not None}})
        if not cfg.temporal and cfg.mask_ratio > 0:
            cfg = replace(cfg, mask_ratio=0.0)
        if cfg.variant != "L4" and cfg.slot_penalty > 0:
            cfg = replace(cfg, slot_penalty=0.0)
        return cfg

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Output(NamedTuple):
    recon: torch.Tensor  # [B, T, N, 12]
    target: torch.Tensor  # [B, T, N, 12]
    masks: torch.Tensor  # [B, T, N, K] assignment of keys to the final slots (the segmentation)
    slots: torch.Tensor  # [B, T, K, D] slots fed to the decoder
    assign: torch.Tensor  # [B, T, N, K] soft K-means assignments (before any temporal model)


class MaskedBiGRU(nn.Module):
    """Bidirectional GRU along each slot's track over time, with BERT-style frame masking."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gru = nn.GRU(dim, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, dim)
        self.mask_token = nn.Parameter(torch.zeros(dim))

    def forward(self, slots: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        """slots [B, T, K, D]; frame_mask [B, T] bool (True = hidden from the model)."""
        b, t, k, d = slots.shape
        x = slots
        if frame_mask is not None:
            x = torch.where(frame_mask[:, :, None, None], self.mask_token.expand_as(slots), slots)
        seq = self.norm(x).permute(0, 2, 1, 3).reshape(b * k, t, d)
        h, _ = self.gru(seq)
        delta = self.out(h).reshape(b, k, t, d).permute(0, 2, 1, 3)
        return x + delta


def random_frame_mask(b: int, t: int, ratio: float, device: torch.device) -> torch.Tensor:
    """Mask round(ratio * t) random frames per clip, always leaving at least one visible."""
    n_mask = min(int(round(ratio * t)), t - 1)
    if n_mask <= 0:
        return torch.zeros(b, t, dtype=torch.bool, device=device)
    ranks = torch.rand(b, t, device=device).argsort(dim=1).argsort(dim=1)
    return ranks < n_mask


def slot_similarity_penalty(slots: torch.Tensor) -> torch.Tensor:
    """Mean squared cosine similarity between distinct slots of the same frame."""
    z = nn.functional.normalize(slots, dim=-1)
    sim = z @ z.transpose(-1, -2)  # [..., K, K]
    k = sim.shape[-1]
    off = ~torch.eye(k, dtype=torch.bool, device=slots.device)
    return sim[..., off].pow(2).mean()


class SlotVideoModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = PatchEncoder(FEATURE_DIM, cfg.hidden, cfg.slot_dim)
        self.grouping = SoftKMeans(cfg.slot_dim, cfg.num_slots)
        self.decoder = BroadcastDecoder(cfg.slot_dim, cfg.decoder_hidden, NUM_PATCHES, PIXEL_DIM)
        self.temporal = MaskedBiGRU(cfg.slot_dim, cfg.gru_hidden) if cfg.temporal else None
        self.register_buffer("positions", torch.from_numpy(patch_positions()), persistent=False)

    def features(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """video [B, T, 64, 64, 3] in [0, 1] -> (inputs [B, T, N, 14], pixel targets [B, T, N, 12])."""
        pix = patchify(video)
        pos = self.positions.expand(*pix.shape[:-1], 2)
        return torch.cat([pix * 2 - 1, pos * 2 - 1], dim=-1), pix

    def group(self, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """keys [B, T, N, D] -> (slots [B, T, K, D], soft assignments [B, T, N, K])."""
        b, t, n, d = keys.shape
        if not self.cfg.recurrent:
            flat = keys.reshape(b * t, n, d)
            slots, attn = self.grouping(flat, self.grouping.init_slots(flat), self.cfg.iters_first)
            return slots.reshape(b, t, -1, d), attn.reshape(b, t, n, -1)
        slots, attn = self.grouping(keys[:, 0], self.grouping.init_slots(keys[:, 0]), self.cfg.iters_first)
        per_frame, per_attn = [slots], [attn]
        for i in range(1, t):
            slots, attn = self.grouping(keys[:, i], slots, self.cfg.iters_next)
            per_frame.append(slots)
            per_attn.append(attn)
        return torch.stack(per_frame, dim=1), torch.stack(per_attn, dim=1)

    def forward(self, video: torch.Tensor, frame_mask: torch.Tensor | None = None) -> Output:
        inputs, target = self.features(video)
        b, t, n, _ = inputs.shape
        keys = self.encoder(inputs)
        slots, assign = self.group(keys)
        if self.temporal is not None:
            slots = self.temporal(slots, frame_mask)
            # re-assign patches to the refined slots, which live in the same key space
            masks = self.grouping.assign(keys.reshape(b * t, n, -1), slots.reshape(b * t, *slots.shape[2:]))
        else:
            masks = assign.reshape(b * t, n, -1)
        rgb = self.decoder(slots.reshape(b * t, *slots.shape[2:]))  # [BT, K, N, 12]
        recon = torch.einsum("bnk,bknc->bnc", masks, rgb)
        return Output(
            recon=recon.reshape(b, t, n, -1),
            target=target,
            masks=masks.reshape(b, t, n, -1),
            slots=slots,
            assign=assign,
        )

    def loss(self, video: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Training loss: reconstruction MSE (+ slot penalty for L4), masking frames for L3/L4."""
        mask = None
        if self.temporal is not None and self.cfg.mask_ratio > 0:
            mask = random_frame_mask(video.shape[0], video.shape[1], self.cfg.mask_ratio, video.device)
        out = self(video, mask)
        rec = (out.recon - out.target).pow(2).mean()
        total = rec
        logs = {"recon": float(rec.detach())}
        if self.cfg.slot_penalty > 0:
            pen = slot_similarity_penalty(out.slots)
            total = total + self.cfg.slot_penalty * pen
            logs["slot_sim"] = float(pen.detach())
        if mask is not None and mask.any():
            logs["recon_masked"] = float((out.recon - out.target).pow(2)[mask].mean().detach())
        logs["loss"] = float(total.detach())
        return total, logs

    @torch.no_grad()
    def segment(self, video: torch.Tensor) -> torch.Tensor:
        """video [B, T, 64, 64, 3] in [0, 1] -> patch labels [B, T, N] (argmax assignment)."""
        return self(video).masks.argmax(dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
