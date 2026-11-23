"""Run the KD/DKD/NKD + MPKD compatibility matrix on CIFAR-100."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "cifar100" / "mpkd_compatibility"
ARCHITECTURES = {
    "res56_res20": "res56_res20.yaml",
    "res110_res32": "res110_res32.yaml",
    "res32x4_res8x4": "res32x4_res8x4.yaml",
    "wrn40_2_wrn16_2": "wrn40_2_wrn16_2.yaml",
    "wrn40_2_wrn40_1": "wrn40_2_wrn40_1.yaml",
    "vgg13_vgg8": "vgg13_vgg8.yaml",
}
METHODS = ("KD", "DKD", "NKD")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the 6-architecture KD/DKD/NKD + MPKD matrix."
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=list(ARCHITECTURES),
        default=list(ARCHITECTURES),
    )
    parser.add_argument(
        "--methods", nargs="+", choices=list(METHODS), default=list(METHODS)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--tag-suffix",
        default="",
        help="Append an isolated tag segment, e.g. tune1, to preserve old runs.",
    )
    return parser.parse_args()


def load_yaml(path):
    import yaml

    with path.open("r", encoding="utf-8-sig") as stream:
        return yaml.safe_load(stream)


def make_tag(architecture, method, tag_suffix=""):
    tag = f"mpkd,{method.lower()}"
    if tag_suffix:
        tag += f",{tag_suffix}"
    return f"{tag},{architecture}"


def worklog_path(config, architecture, method, tag_suffix=""):
    tag = make_tag(architecture, method, tag_suffix)
    return ROOT / "output" / config["EXPERIMENT"]["PROJECT"] / tag / "worklog.txt"


def completed(path):
    if not path.is_file():
        return False
    return re.search(
        r"(?m)^best_acc\s+[0-9]+(?:\.[0-9]+)?\s*$",
        path.read_text(encoding="utf-8", errors="ignore"),
    ) is not None


def build_command(config_path, architecture, method, seed, epochs, tag_suffix=""):
    tag = make_tag(architecture, method, tag_suffix)
    command = [
        sys.executable,
        "tools/train_proto.py",
        "--cfg",
        str(config_path.relative_to(ROOT)),
        "--seed",
        str(seed),
        "EXPERIMENT.TAG",
        tag,
        "EXPERIMENT.METHOD",
        f"{method}-MPKD",
        "EXPERIMENT.BASELINE",
        method,
        "MPKD.BASE_METHOD",
        method,
    ]
    if epochs is not None:
        command.extend(["SOLVER.EPOCHS", str(epochs)])
    return command


def main():
    args = parse_args()
    jobs = []
    for architecture in args.architectures:
        config_path = CONFIG_DIR / ARCHITECTURES[architecture]
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = load_yaml(config_path)
        for method in args.methods:
            jobs.append((architecture, method, config_path, config))

    print(f"Jobs: {len(jobs)} | seed={args.seed}")
    for index, (architecture, method, config_path, config) in enumerate(jobs, 1):
        output = worklog_path(config, architecture, method, args.tag_suffix)
        command = build_command(
            config_path,
            architecture,
            method,
            args.seed,
            args.epochs,
            args.tag_suffix,
        )
        print(f"\n[{index}/{len(jobs)}] {method} + MPKD | {architecture}")
        print(f"output: {output.relative_to(ROOT)}")
        print("command:", subprocess.list2cmdline(command))

        if completed(output) and not args.force:
            print("status: skipped (best_acc already exists)")
            continue
        if args.dry_run:
            print("status: dry-run")
            continue

        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(
                f"{method} + MPKD on {architecture} failed "
                f"with exit code {result.returncode}."
            )
        if not completed(output):
            raise RuntimeError(f"Run ended without best_acc: {output}")
        print("status: completed")


if __name__ == "__main__":
    main()
