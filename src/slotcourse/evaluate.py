"""Score one method on a split at every evaluation length; saves per-video scores.

Usage:
    python -m slotcourse.evaluate --method C1               # clustering baselines: C1, C2, C3
    python -m slotcourse.evaluate --method L2 --seed 0      # learned rungs, needs results/runs/L2_seed0
    python -m slotcourse.evaluate --method C3 --split val   # model selection uses the val split

Each run writes results/scores/<split>/<name>.npz (per-video arrays named <metric>_L<len>)
and <name>.json (cost and config). Causal methods are run once on 24 frames and sliced;
the BiGRU rungs (L3, L4) see the future, so they are re-run on each prefix length.
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from slotcourse.config import RESULTS_DIR
from slotcourse.data.dataset import load_split
from slotcourse.inference import cluster_videos, fit_pca, pick_device, segment_videos
from slotcourse.metrics import METRICS, score_video
from slotcourse.settings import load_settings

log = logging.getLogger(__name__)

CLUSTERING = ("C1", "C2", "C3")
LEARNED = ("L1", "L2", "L3", "L4")
NON_CAUSAL = ("L3", "L4")
SAVED_LABEL_VIDEOS = 8


def _max_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6  # bytes on macOS, KiB on Linux


def make_segmenter(method: str, seed: int, settings: dict[str, Any], run_dir: Path | None):
    """Return (fn: uint8 videos -> pixel labels, metadata dict)."""
    if method in CLUSTERING:
        ccfg = settings["clustering"]
        pca = fit_pca(load_split("train").video, ccfg, seed=seed)
        meta = {"params": 0, "train_minutes": 0.0, "pca_explained": float(pca.explained_variance_ratio.sum()),
                "clustering": ccfg}
        return (lambda v: cluster_videos(method, v, pca, ccfg, seed=seed)), meta
    from slotcourse.train import load_model, run_name  # local import keeps C-only runs light

    run_dir = run_dir or RESULTS_DIR / "runs" / run_name(method, seed)
    if not (run_dir / "model.pt").exists():
        raise FileNotFoundError(f"no trained model at {run_dir}; run `python -m slotcourse.train --variant {method}`")
    device = pick_device()
    model = load_model(run_dir, device)
    meta = json.loads((run_dir / "meta.json").read_text())
    batch = settings["eval"]["batch_videos"]
    return (lambda v: segment_videos(model, v, device, batch)), meta


def evaluate(method: str, seed: int, split: str, settings: dict[str, Any], run_dir: Path | None = None,
             limit: int | None = None, name: str | None = None) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    data = load_split(split, limit=limit)
    videos, segs = data.get(np.arange(len(data)))
    segment, meta = make_segmenter(method, seed, settings, run_dir)
    lengths = sorted(settings["eval"]["lengths"])
    full = max(lengths)

    start = time.time()
    full_labels = segment(videos[:, :full])
    seg_minutes = (time.time() - start) / 60
    scores: dict[str, np.ndarray] = {}
    for length in lengths:
        if method in NON_CAUSAL and length != full:
            labels = segment(videos[:, :length])
        else:
            labels = full_labels[:, :length]
        per_video = [score_video(g, p) for g, p in zip(segs[:, :length], labels)]
        for m in METRICS:
            scores[f"{m}_L{length}"] = np.asarray([s[m] for s in per_video], dtype=np.float64)

    name = name or (method if method in CLUSTERING else f"{method}_seed{seed}")
    out_dir = RESULTS_DIR / "scores" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{name}.npz", **scores, video_index=data.indices)
    np.save(out_dir / f"{name}_labels.npy", full_labels[:SAVED_LABEL_VIDEOS])
    info = {
        "method": method, "seed": seed, "split": split, "num_videos": len(data),
        "segment_minutes_24f": seg_minutes, "eval_minutes": (time.time() - start) / 60,
        "peak_rss_gb": _max_rss_gb(), **meta,
        "means": {k: float(np.nanmean(v)) for k, v in scores.items()},
    }
    (out_dir / f"{name}.json").write_text(json.dumps(info, indent=2, default=str))
    log.info("%s on %s (%d videos): %s", name, split, len(data),
             {k: round(100 * v, 1) for k, v in info["means"].items() if k.endswith(f"L{full}")})
    return scores


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=CLUSTERING + LEARNED, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="score only the first N videos (debugging)")
    p.add_argument("--name", default=None, help="output name (default: method[_seedN])")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = p.parse_args()
    settings = load_settings(overrides=args.overrides)
    evaluate(args.method, args.seed, args.split, settings, args.run_dir, args.limit, args.name)


if __name__ == "__main__":
    main()
