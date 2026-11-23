"""Run the controlled CIFAR-100 MPKD ablations sequentially."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "experiments" / "log.md"
LOG_START = "<!-- CIFAR100_ABLATION_RESULTS_START -->"
LOG_END = "<!-- CIFAR100_ABLATION_RESULTS_END -->"
DEFAULT_CONFIG = Path("configs/cifar100/mpkd/res32x4_res8x4.yaml")
MODE_GROUPS = {
    "ae": ["A", "B", "C", "D", "E"],
    "gates": [
        "gate_no_uncertainty",
        "gate_no_agreement",
        "gate_no_positive_gain",
        "gate_no_confidence",
    ],
}
MODE_TAGS = {
    "A": "ablation_a",
    "B": "ablation_b",
    "C": "ablation_c",
    "D": "ablation_d",
    "E": "ablation_e",
    "gate_no_uncertainty": "gate_no_uncertainty",
    "gate_no_agreement": "gate_no_agreement",
    "gate_no_positive_gain": "gate_no_positive_gain",
    "gate_no_confidence": "gate_no_confidence",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run A-E or gate ablations one at a time on a single GPU."
    )
    parser.add_argument("--phase", choices=["ae", "gates"], default="ae")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(MODE_TAGS),
        help="Optional explicit mode list; overrides --phase.",
    )
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when the isolated worklog already contains best_acc.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and output paths without starting training.",
    )
    return parser.parse_args()


def load_config(path: Path):
    with path.open("r", encoding="utf-8-sig") as stream:
        return yaml.safe_load(stream)


def output_path(config, mode):
    experiment = config["EXPERIMENT"]
    base_name = experiment.get("NAME") or experiment["TAG"]
    run_name = f"{base_name},{MODE_TAGS[mode]}"
    return ROOT / "output" / experiment["PROJECT"] / run_name / "worklog.txt"


def has_completed_result(worklog: Path):
    if not worklog.is_file():
        return False
    text = worklog.read_text(encoding="utf-8", errors="ignore")
    # Do not promote legacy worklogs that predate the current metadata format.
    if "prototype_aggregation:" not in text:
        return False
    return re.search(r"(?m)^best_acc\s+[0-9]+(?:\.[0-9]+)?\s*$", text) is not None


def parse_completed_result(worklog: Path, mode: str):
    text = worklog.read_text(encoding="utf-8", errors="ignore")
    metadata_index = text.rfind(f"ablation_mode: {mode}")
    if metadata_index >= 0:
        text = text[metadata_index:]

    best_matches = re.findall(
        r"(?m)^best_acc\s+([0-9]+(?:\.[0-9]+)?)\s*$", text
    )
    epoch_matches = re.findall(
        r"epoch:\s*(\d+)(?:(?!epoch:).)*?test_acc:\s*"
        r"([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.DOTALL,
    )
    if not best_matches or not epoch_matches:
        raise RuntimeError(f"Incomplete ablation worklog: {worklog}")

    best_acc = float(best_matches[-1])
    epochs = [(int(epoch), float(acc)) for epoch, acc in epoch_matches]
    best_epoch, _ = max(epochs, key=lambda item: item[1])
    final_epoch, final_acc = epochs[-1]
    return {
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "final_acc": final_acc,
    }


def update_markdown_row(markdown: str, mode: str, result: dict):
    pattern = re.compile(rf"(?m)^\|\s*{re.escape(mode)}\s*\|.*$")
    match = pattern.search(markdown)
    if match is None:
        raise RuntimeError(f"Result table row not found for mode {mode}")

    cells = [cell.strip() for cell in match.group(0).split("|")]
    if len(cells) != 9:
        raise RuntimeError(f"Unexpected result row format for mode {mode}")
    cells[4] = f"{result['best_acc']:.2f}%"
    cells[5] = str(result["best_epoch"])
    cells[6] = f"{result['final_acc']:.2f}%"
    cells[7] = "已完成"
    replacement = "| " + " | ".join(cells[1:-1]) + " |"
    return markdown[:match.start()] + replacement + markdown[match.end():]


def record_completed_result(worklog: Path, mode: str):
    result = parse_completed_result(worklog, mode)
    markdown = LOG_PATH.read_text(encoding="utf-8")
    if LOG_START not in markdown or LOG_END not in markdown:
        raise RuntimeError("log.md 缺少消融结果标记")
    markdown = update_markdown_row(markdown, mode, result)
    LOG_PATH.write_text(markdown, encoding="utf-8")
    print(
        f"recorded: best={result['best_acc']:.2f}% at epoch "
        f"{result['best_epoch']}, final={result['final_acc']:.2f}%"
    )


def main():
    args = parse_args()
    config_path = args.cfg if args.cfg.is_absolute() else ROOT / args.cfg
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    config = load_config(config_path)
    modes = args.modes or MODE_GROUPS[args.phase]
    relative_config = config_path.relative_to(ROOT)

    print(f"Ablation config: {relative_config}")
    print(f"Seed: {args.seed}")
    print(f"Modes: {', '.join(modes)}")

    for index, mode in enumerate(modes, start=1):
        worklog = output_path(config, mode)
        command = [
            sys.executable,
            "tools/train_proto.py",
            "--cfg",
            str(relative_config),
            "--seed",
            str(args.seed),
            "--ablation-mode",
            mode,
        ]
        print(f"\n[{index}/{len(modes)}] mode={mode}")
        print(f"output: {worklog.relative_to(ROOT)}")
        print("command:", subprocess.list2cmdline(command))

        if has_completed_result(worklog) and not args.force:
            if not args.dry_run:
                record_completed_result(worklog, mode)
            print("status: skipped (completed best_acc found; use --force to rerun)")
            continue
        if args.dry_run:
            print("status: dry-run")
            continue

        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(
                f"Ablation {mode} failed with exit code {result.returncode}."
            )
        if not has_completed_result(worklog):
            raise RuntimeError(
                f"Ablation {mode} ended without a best_acc record: {worklog}"
            )
        record_completed_result(worklog, mode)
        print("status: completed")


if __name__ == "__main__":
    main()
