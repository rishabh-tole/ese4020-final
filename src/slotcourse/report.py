"""Build the results table, rung-difference table, length plot and mask overlays.

Usage:
    python -m slotcourse.report            # reads results/scores/test, writes results/report/
    python -m slotcourse.report --split val
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from slotcourse.bootstrap import bootstrap_ci, paired_diff_ci  # noqa: E402
from slotcourse.config import RESULTS_DIR  # noqa: E402
from slotcourse.data.dataset import load_split  # noqa: E402

log = logging.getLogger(__name__)

LADDER = ("C1", "C2", "C3", "L1", "L2", "L3", "L4")
RUNGS = (("C1", "C2"), ("C1", "C3"), ("C1", "L1"), ("L1", "L2"), ("L2", "L3"), ("L3", "L4"), ("C3", "L4"))
TABLE_METRICS = ("fg_ari", "mbo_f", "mbo_v", "fg_ari_v")
LENGTH_METHODS = ("C1", "C3", "L1", "L2", "L4")  # plotted when available


def load_scores(score_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[dict[str, Any]]]]:
    """Per method: per-video scores averaged over seeds, plus each run's metadata."""
    by_method: dict[str, list[dict[str, np.ndarray]]] = {}
    metas: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(score_dir.glob("*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text())
        method = meta["method"]
        if path.stem not in (method, f"{method}_seed{meta['seed']}"):
            continue  # skip tuning / ad-hoc runs saved under a custom name
        with np.load(path) as z:
            by_method.setdefault(method, []).append({k: z[k] for k in z.files if k != "video_index"})
        metas.setdefault(method, []).append(meta)
    averaged = {
        m: {k: np.nanmean(np.stack([r[k] for r in runs]), axis=0) for k in runs[0]} for m, runs in by_method.items()
    }
    return averaged, metas


def results_table(scores: dict[str, dict[str, np.ndarray]], metas: dict[str, list[dict[str, Any]]],
                  n_boot: int, length: int = 24) -> pd.DataFrame:
    rows = []
    for method in [m for m in LADDER if m in scores]:
        row: dict[str, Any] = {"method": method, "seeds": len(metas[method])}
        for metric in TABLE_METRICS:
            row[metric] = bootstrap_ci(scores[method][f"{metric}_L{length}"], n_boot=n_boot).fmt()
        runs = metas[method]
        row["params"] = int(runs[0].get("params", 0))
        row["minutes_per_run"] = round(float(np.mean([r.get("train_minutes", 0) + r["eval_minutes"] for r in runs])), 1)
        row["peak_mem_gb"] = round(float(np.max([r.get("peak_memory_gb", r.get("peak_rss_gb", np.nan)) for r in runs])), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def rung_table(scores: dict[str, dict[str, np.ndarray]], n_boot: int, length: int = 24) -> pd.DataFrame:
    rows = []
    for a, b in RUNGS:
        if a not in scores or b not in scores:
            continue
        row: dict[str, Any] = {"comparison": f"{a} -> {b}"}
        for metric in TABLE_METRICS:
            ci = paired_diff_ci(scores[a][f"{metric}_L{length}"], scores[b][f"{metric}_L{length}"], n_boot=n_boot)
            row[metric] = ci.fmt() + (" *" if ci.excludes_zero() else "")
        rows.append(row)
    return pd.DataFrame(rows)


def length_plot(scores: dict[str, dict[str, np.ndarray]], lengths: list[int], out: Path, n_boot: int) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for method in [m for m in LENGTH_METHODS if m in scores]:
        cis = [bootstrap_ci(scores[method][f"mbo_v_L{n}"], n_boot=n_boot) for n in lengths]
        means = [100 * c.mean for c in cis]
        err = [[100 * (c.mean - c.lo) for c in cis], [100 * (c.hi - c.mean) for c in cis]]
        ax.errorbar(lengths, means, yerr=err, marker="o", capsize=3, label=method)
    ax.set_xlabel("evaluation length (frames)")
    ax.set_ylabel("mBO-V (95% CI)")
    ax.set_xticks(lengths)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def overlays(score_dir: Path, split: str, out: Path, methods: tuple[str, ...] = LADDER,
             videos: tuple[int, ...] = (0, 1, 2), frames: tuple[int, ...] = (0, 8, 16, 23)) -> None:
    data = load_split(split)
    available = [m for m in methods if (score_dir / f"{m if m.startswith('C') else m + '_seed0'}_labels.npy").exists()]
    rows = ["input", "GT", *available]
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(len(rows) * len(videos), len(frames), figsize=(1.6 * len(frames), 1.6 * len(rows) * len(videos)))
    for vi, v in enumerate(videos):
        video, seg = data.get(v)
        for ri, row in enumerate(rows):
            if row == "input":
                imgs = video
            elif row == "GT":
                imgs = cmap(seg % 20)
            else:
                name = row if row.startswith("C") else f"{row}_seed0"
                lab = np.load(score_dir / f"{name}_labels.npy")[v]
                imgs = 0.5 * video / 255.0 + 0.5 * cmap(lab % 20)[..., :3]
            for fi, f in enumerate(frames):
                ax = axes[vi * len(rows) + ri, fi]
                ax.imshow(imgs[f])
                ax.set_xticks([])
                ax.set_yticks([])
                if fi == 0:
                    ax.set_ylabel(f"v{v} {row}", fontsize=7)
                if vi == 0 and ri == 0:
                    ax.set_title(f"t={f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()
    score_dir = RESULTS_DIR / "scores" / args.split
    out_dir = RESULTS_DIR / "report" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    scores, metas = load_scores(score_dir)
    if not scores:
        raise SystemExit(f"no scores in {score_dir}; run `python -m slotcourse.evaluate` first")
    lengths = sorted({int(k.rsplit("_L", 1)[1]) for s in scores.values() for k in s})
    table = results_table(scores, metas, args.bootstrap, max(lengths))
    rungs = rung_table(scores, args.bootstrap, max(lengths))
    table.to_csv(out_dir / "table.csv", index=False)
    rungs.to_csv(out_dir / "rungs.csv", index=False)
    length_plot(scores, lengths, out_dir / "mbo_v_vs_length.png", args.bootstrap)
    overlays(score_dir, args.split, out_dir / "overlays.png")
    n_videos = len(next(iter(scores.values()))[f"mbo_v_L{max(lengths)}"])
    md = [
        f"# Results ({args.split}, {n_videos} videos, {max(lengths)} frames)\n",
        "Scores x100, mean [95% bootstrap CI over videos, "
        f"{args.bootstrap} resamples]. Learned rungs average per-video scores over seeds.\n",
        table.to_markdown(index=False),
        "\n## Rung-to-rung differences (paired bootstrap; * = CI excludes zero)\n",
        rungs.to_markdown(index=False),
        "\n![mBO-V vs length](mbo_v_vs_length.png)\n\n![overlays](overlays.png)\n",
    ]
    (out_dir / "report.md").write_text("\n".join(md))
    log.info("wrote %s", out_dir / "report.md")
    print(table.to_string(index=False))  # noqa: T201 - CLI output
    print(rungs.to_string(index=False))  # noqa: T201


if __name__ == "__main__":
    main()
