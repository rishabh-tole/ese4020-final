"""Validation-set model selection (Module 3). Never touches the test split.

Usage:
    # K, PCA size and position weight for C1-C3 (shared features), scored by C1 mBO-F
    python -m slotcourse.tune clustering --videos 100

    # one learned hyperparameter at a time, scored by mBO-V on 24-frame val clips
    python -m slotcourse.tune learned --variant L3 --param model.mask_ratio --values 0.15 0.3 0.5
    python -m slotcourse.tune learned --variant L4 --param model.slot_penalty --values 0.01 0.1 1.0
"""

from __future__ import annotations

import argparse
import itertools
import logging

import numpy as np
import pandas as pd

from slotcourse.config import RESULTS_DIR
from slotcourse.evaluate import evaluate
from slotcourse.settings import load_settings
from slotcourse.train import run_name, train

log = logging.getLogger(__name__)


def tune_clustering(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    grid = itertools.product(args.k, args.pca, args.pos_weight)
    for k, pca, pw in grid:
        overrides = [f"clustering.num_clusters={k}", f"clustering.pca_components={pca}",
                     f"clustering.pos_weight={pw}", "eval.lengths=[24]"]
        settings = load_settings(overrides=args.overrides + overrides)
        name = f"tune_{args.method}_k{k}_p{pca}_w{pw}"
        scores = evaluate(args.method, 0, "val", settings, limit=args.videos, name=name)
        rows.append({"k": k, "pca": pca, "pos_weight": pw, **{m: float(np.nanmean(v)) for m, v in scores.items()}})
        log.info("%s", rows[-1])
    return pd.DataFrame(rows)


def tune_learned(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for value in args.values:
        overrides = args.overrides + [f"{args.param}={value}", f"train.steps={args.steps}"]
        settings = load_settings(overrides=overrides)
        tag = f"tune-{args.param.split('.')[-1]}{value}"
        run_dir = train(args.variant, args.seed, settings, RESULTS_DIR / "runs" / run_name(args.variant, args.seed, tag))
        eval_settings = load_settings(overrides=overrides + ["eval.lengths=[24]"])
        scores = evaluate(args.variant, args.seed, "val", eval_settings, run_dir=run_dir, limit=args.videos,
                          name=f"{args.variant}_{tag}_seed{args.seed}")
        rows.append({args.param: value, **{m: float(np.nanmean(v)) for m, v in scores.items()}})
        log.info("%s", rows[-1])
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clustering")
    c.add_argument("--method", choices=["C1", "C3"], default="C1")
    c.add_argument("--k", type=int, nargs="+", default=[6, 8, 11, 14])
    c.add_argument("--pca", type=int, nargs="+", default=[3, 6, 10, 14])
    c.add_argument("--pos-weight", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    c.add_argument("--metric", default="mbo_f_L24")
    learned = sub.add_parser("learned")
    learned.add_argument("--variant", choices=["L1", "L2", "L3", "L4"], required=True)
    learned.add_argument("--param", required=True, help="e.g. model.mask_ratio")
    learned.add_argument("--values", type=float, nargs="+", required=True)
    learned.add_argument("--steps", type=int, default=1500)
    learned.add_argument("--seed", type=int, default=0)
    learned.add_argument("--metric", default="mbo_v_L24")
    for sp in (c, learned):
        sp.add_argument("--videos", type=int, default=100, help="val videos to score")
        sp.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = p.parse_args()

    table = tune_clustering(args) if args.cmd == "clustering" else tune_learned(args)
    out_dir = RESULTS_DIR / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = "clustering" if args.cmd == "clustering" else f"{args.variant}_{args.param.split('.')[-1]}"
    table = table.sort_values(args.metric, ascending=False)
    table.to_csv(out_dir / f"{label}.csv", index=False)
    print(table.to_string(index=False))  # noqa: T201 - CLI output
    print(f"\nbest by {args.metric}:\n{table.iloc[0].to_string()}")  # noqa: T201


if __name__ == "__main__":
    main()
