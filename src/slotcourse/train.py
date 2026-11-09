"""Train one learned rung (L1-L4) with Adam on random 6-frame windows.

Usage:
    python -m slotcourse.train --variant L2 --seed 0
    python -m slotcourse.train --variant L4 --seed 1 --set train.steps=5000 --set model.slot_penalty=0.3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from slotcourse.config import RESULTS_DIR
from slotcourse.data.dataset import WindowDataset, load_split
from slotcourse.inference import pick_device, segment_videos
from slotcourse.metrics import score_video
from slotcourse.models.video import ModelConfig, SlotVideoModel, count_parameters
from slotcourse.settings import load_settings

log = logging.getLogger(__name__)


def run_name(variant: str, seed: int, tag: str = "") -> str:
    return f"{variant}{'-' + tag if tag else ''}_seed{seed}"


def peak_memory_gb(device: torch.device) -> float:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    return float("nan")


def quick_val(model: SlotVideoModel, settings: dict[str, Any], device: torch.device) -> dict[str, float]:
    val = load_split("val", limit=settings["train"]["val_videos"])
    video, seg = val.get(np.arange(len(val)))
    labels = segment_videos(model, video, device, settings["eval"]["batch_videos"])
    scores = [score_video(g, p) for g, p in zip(seg, labels)]
    return {k: float(np.nanmean([s[k] for s in scores])) for k in scores[0]}


def train(variant: str, seed: int, settings: dict[str, Any], out_dir: Path) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device()
    tcfg = settings["train"]
    mcfg = ModelConfig.for_variant(variant, **settings["model"])
    model = SlotVideoModel(mcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])

    data = WindowDataset(load_split("train"), window=tcfg["window"], seed=seed)
    sampler = RandomSampler(data, replacement=True, num_samples=tcfg["steps"] * tcfg["batch_size"])
    loader = DataLoader(data, batch_size=tcfg["batch_size"], sampler=sampler, num_workers=0, drop_last=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"model": mcfg.to_dict(), "settings": settings}, indent=2))
    log.info("%s seed %d: %d params, %d train videos, device %s", variant, seed, count_parameters(model), len(data), device)

    rows: list[dict[str, float]] = []
    peak, start = 0.0, time.time()
    model.train()
    for step, clip in enumerate(loader, start=1):
        loss, logs = model.loss(clip.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        opt.step()
        if not np.isfinite(logs["loss"]):
            raise FloatingPointError(f"non-finite loss at step {step}: {logs}")
        if step % tcfg["log_every"] == 0 or step == tcfg["steps"]:
            peak = max(peak, peak_memory_gb(device))
            row = {"step": step, "minutes": (time.time() - start) / 60, **logs}
            if step % tcfg["val_every"] == 0 or step == tcfg["steps"]:
                model.eval()
                row.update({f"val_{k}": v for k, v in quick_val(model, settings, device).items()})
                model.train()
            rows.append(row)
            log.info("step %d %s", step, {k: round(v, 4) for k, v in row.items() if k != "step"})

    minutes = (time.time() - start) / 60
    torch.save(model.state_dict(), out_dir / "model.pt")
    _write_log(rows, out_dir / "log.csv")
    meta = {"variant": variant, "seed": seed, "params": count_parameters(model), "train_minutes": minutes,
            "peak_memory_gb": peak, "device": str(device), "train_videos": len(data)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("done in %.1f min -> %s", minutes, out_dir)
    return out_dir


def _write_log(rows: list[dict[str, float]], path: Path) -> None:
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "step", k))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_model(run_dir: Path, device: torch.device) -> SlotVideoModel:
    cfg = json.loads((run_dir / "config.json").read_text())["model"]
    model = SlotVideoModel(ModelConfig(**cfg)).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    return model.eval()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["L1", "L2", "L3", "L4"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="suffix for the run name (e.g. tuning runs)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = p.parse_args()
    settings = load_settings(overrides=args.overrides)
    train(args.variant, args.seed, settings, RESULTS_DIR / "runs" / run_name(args.variant, args.seed, args.tag))


if __name__ == "__main__":
    main()
