"""Draw the paper-ready two-panel MPKD robustness figure from CSV results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_summary(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["augmentation"] not in {"RandAugment", "clean"}:
                continue
            # Put the clean evaluation point before the perturbation levels.
            row_n = 0 if row["augmentation"] == "clean" else int(row["n"])
            rows.append(
                {
                    "model": row["model"],
                    "n": row_n,
                    "top1": float(row["top1_mean"]),
                    "top1_std": float(row["top1_std"]),
                    "top5": float(row["top5_mean"]),
                    "top5_std": float(row["top5_std"]),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Plot MPKD robustness curves")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_summary(args.summary)
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["model"]][row["n"]] = row
    n_values = sorted({row["n"] for row in rows})
    if not n_values or not {"CRLD", "MPKD"}.issubset(grouped):
        raise RuntimeError("The summary must contain CRLD and MPKD RandAugment rows.")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.unicode_minus": False,
        }
    )
    colors = {"CRLD": "#3B6BB5", "MPKD": "#D97706"}
    markers = {"CRLD": "o", "MPKD": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharex=True)
    for ax, metric, ylabel, title in (
        (axes[0], "top1", "Top-1 accuracy (%)", "(a) Top-1"),
        (axes[1], "top5", "Top-5 accuracy (%)", "(b) Top-5"),
    ):
        for model in ("CRLD", "MPKD"):
            means = np.array([grouped[model][n][metric] for n in n_values])
            stds = np.array([grouped[model][n][metric + "_std"] for n in n_values])
            ax.plot(
                n_values,
                means,
                color=colors[model],
                marker=markers[model],
                markersize=4.5,
                linewidth=1.6,
                label=model,
                zorder=3,
            )
            ax.fill_between(
                n_values,
                means - stds,
                means + stds,
                color=colors[model],
                alpha=0.13,
                linewidth=0,
                zorder=1,
            )
        ax.set_title(title, pad=7)
        ax.set_xlabel("Strong-view perturbation")
        ax.set_ylabel(ylabel)
        ax.set_xticks(n_values)
        ax.set_xticklabels(["Clean" if n == 0 else str(n) for n in n_values])
        ax.grid(axis="y", color="#D9DEE7", linewidth=0.65, alpha=0.8)
        ax.tick_params(axis="both", labelsize=9, length=3)
        ax.legend(frameon=False, loc="best", ncol=2, handlelength=2.2)

    fig.tight_layout(w_pad=1.8, rect=(0, 0, 1, 0.98))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
