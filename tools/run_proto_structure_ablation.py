"""Run the missing K and prototype-aggregation ablations for CIFAR-100 MPKD."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/cifar100/mpkd/res32x4_res8x4.yaml")
AGGREGATIONS = ("logmeanexp", "mean", "max")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run missing K and prototype-aggregation ablations sequentially."
    )
    parser.add_argument("--mode", choices=("k", "aggregation", "all"), default="all")
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 2, 4, 5],
        help="K values still missing from the paper; K=3 is already recorded.",
    )
    parser.add_argument(
        "--aggregations",
        nargs="+",
        choices=AGGREGATIONS,
        default=["mean", "max"],
        help="Aggregation variants still missing; Log-Mean-Exp is already recorded.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path):
    with path.open("r", encoding="utf-8-sig") as stream:
        return yaml.safe_load(stream)


def completed(worklog: Path):
    if not worklog.is_file():
        return False
    return re.search(
        r"(?m)^best_acc\s+[0-9]+(?:\.[0-9]+)?\s*$",
        worklog.read_text(encoding="utf-8", errors="ignore"),
    ) is not None


def build_runs(config, args):
    experiment = config["EXPERIMENT"]
    base_name = experiment.get("NAME") or experiment["TAG"]
    field = "NAME" if experiment.get("NAME") else "TAG"
    runs = []
    if args.mode in {"k", "all"}:
        for k in args.k_values:
            if k < 1:
                raise ValueError("All K values must be at least 1.")
            suffix = f"structure_k{k}"
            runs.append((suffix, {"MPKD.PROTO_K": str(k)}))
    if args.mode in {"aggregation", "all"}:
        for aggregation in args.aggregations:
            suffix = f"structure_agg_{aggregation}"
            runs.append((suffix, {"MPKD.PROTO_AGGREGATION": aggregation}))
    return [
        {
            "suffix": suffix,
            "tag": f"{base_name},{suffix}",
            "field": field,
            "overrides": overrides,
        }
        for suffix, overrides in runs
    ]


def main():
    args = parse_args()
    config_path = args.cfg if args.cfg.is_absolute() else ROOT / args.cfg
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = load_config(config_path)
    relative_config = config_path.relative_to(ROOT)
    project = config["EXPERIMENT"]["PROJECT"]
    runs = build_runs(config, args)

    print(f"Config: {relative_config}")
    print(f"Seed: {args.seed}")
    print(f"Runs: {len(runs)}")
    for index, run in enumerate(runs, start=1):
        worklog = ROOT / "output" / project / run["tag"] / "worklog.txt"
        command = [
            sys.executable,
            "tools/train_proto.py",
            "--cfg",
            str(relative_config),
            "--seed",
            str(args.seed),
            f"EXPERIMENT.{run['field']}",
            run["tag"],
        ]
        for key, value in run["overrides"].items():
            command.extend([key, value])

        print(f"\n[{index}/{len(runs)}] {run['suffix']}")
        print(f"output: {worklog.relative_to(ROOT)}")
        print("command:", subprocess.list2cmdline(command))
        if completed(worklog) and not args.force:
            print("status: skipped (completed; use --force to rerun)")
            continue
        if args.dry_run:
            print("status: dry-run")
            continue

        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(
                f"{run['suffix']} failed with exit code {result.returncode}."
            )


if __name__ == "__main__":
    main()
