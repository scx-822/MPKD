"""Generate paper figures for MPKD prototype construction and target repair."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from mdistiller.dataset.cifar100 import (  # noqa: E402
    get_cifar100_dataloaders_strong,
    get_data_folder,
    get_cifar100_train_transform_strong,
)
from mdistiller.models import cifar_model_dict  # noqa: E402
from train_proto import (  # noqa: E402
    build_multicenter_prototypes,
    extract_teacher_logits_and_feature,
    mine_reliable_teacher_features,
    set_training_seed,
)


def load_teacher(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model = cifar_model_dict["resnet32x4"][0](num_classes=100)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def prototype_distribution(features, prototypes, temperature):
    classes, k_num, feature_dim = prototypes.shape
    features = F.normalize(features, dim=1)
    prototypes = F.normalize(prototypes, dim=-1)
    similarity = torch.matmul(
        features, prototypes.view(classes * k_num, feature_dim).t()
    ).view(-1, classes, k_num)
    logits = torch.logsumexp(similarity / temperature, dim=2) - math.log(k_num)
    return F.softmax(logits, dim=1)


def gated_reconstruction(prob_teacher, prob_proto, weak_pred, uncertainty, scale):
    anchor = weak_pred.view(-1, 1)
    agreement = prob_proto.argmax(dim=1).eq(weak_pred).float().view(-1, 1)
    teacher_anchor = prob_teacher.gather(1, anchor)
    proto_anchor = prob_proto.gather(1, anchor)
    positive_gain = ((proto_anchor - teacher_anchor).clamp_min(0.0) / (proto_anchor + 1e-8)).clamp_(0.0, 1.0)
    classes = prob_proto.size(1)
    entropy = -torch.sum(prob_proto * torch.log(prob_proto + 1e-8), dim=1, keepdim=True)
    proto_confidence = (1.0 - entropy / math.log(classes)).clamp_(0.0, 1.0)
    fusion_weight = (scale * uncertainty.view(-1, 1) * agreement * positive_gain * proto_confidence).clamp_(0.0, 1.0)
    fused = (1.0 - fusion_weight) * prob_teacher + fusion_weight * prob_proto
    return fused, fusion_weight


def make_prototypes(teacher, batch_size, workers, device, tau_w, k_num):
    train_loader, _, _ = get_cifar100_dataloaders_strong(
        batch_size=batch_size, val_batch_size=batch_size, num_workers=workers
    )
    features_by_class = mine_reliable_teacher_features(
        teacher,
        train_loader,
        num_classes=100,
        device=device,
        tau_w=tau_w,
    )
    feature_dim = next(iter(features_by_class.values()))[0].numel()
    prototypes = build_multicenter_prototypes(
        features_by_class,
        num_classes=100,
        K=k_num,
        device=device,
        feature_dim=feature_dim,
    )
    return features_by_class, prototypes


def plot_feature_distribution(features_by_class, prototypes, output_path, k_num):
    selected = sorted(features_by_class, key=lambda c: len(features_by_class[c]), reverse=True)[:8]
    points = []
    labels = []
    prototype_flags = []
    for class_id in selected:
        samples = torch.stack(features_by_class[class_id][:120]).numpy()
        points.append(samples)
        labels.extend([class_id] * len(samples))
        prototype_points = prototypes[class_id].detach().cpu().numpy()
        points.append(prototype_points)
        labels.extend([class_id] * k_num)
        prototype_flags.extend([False] * len(samples) + [True] * k_num)
    points = np.concatenate(points, axis=0)
    embedding = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=42,
        max_iter=1000,
    ).fit_transform(points)
    labels = np.asarray(labels)
    prototype_flags = np.asarray(prototype_flags)

    class_names = datasets.CIFAR100(
        root=get_data_folder(), train=False, download=False
    ).classes
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 6.0), dpi=260)
    axes = axes.ravel()
    feature_color = "#4C78A8"
    prototype_color = "#E68632"
    for ax, class_id in zip(axes, selected):
        mask = (labels == class_id) & ~prototype_flags
        proto_mask = (labels == class_id) & prototype_flags
        class_points = embedding[mask]
        proto_points = embedding[proto_mask]
        ax.scatter(
            class_points[:, 0], class_points[:, 1],
            s=17, alpha=0.48, color=feature_color, edgecolors="none", zorder=2,
        )
        ax.scatter(
            proto_points[:, 0], proto_points[:, 1],
            s=135, marker="*", color=prototype_color,
            edgecolors="white", linewidths=1.0, zorder=4,
        )
        bounds = np.concatenate((class_points, proto_points), axis=0)
        low = bounds.min(axis=0)
        high = bounds.max(axis=0)
        padding = np.maximum((high - low) * 0.16, 1.0)
        ax.set_xlim(low[0] - padding[0], high[0] + padding[0])
        ax.set_ylim(low[1] - padding[1], high[1] + padding[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"Class {class_id}: {class_names[class_id].replace('_', ' ')}",
            fontsize=9.5, fontweight="semibold", pad=5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#D9DEE5")
            spine.set_linewidth(0.8)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", color=feature_color,
                   markeredgecolor="none", markersize=6.5,
                   label="Reliable teacher features"),
            Line2D([0], [0], marker="*", linestyle="none", color=prototype_color,
                   markeredgecolor="white", markeredgewidth=0.8, markersize=11,
                   label=f"Sub-prototypes (K={k_num})"),
        ],
        frameon=False, ncol=2, fontsize=9, loc="upper center",
        bbox_to_anchor=(0.5, 0.99), columnspacing=2.0,
    )
    fig.supxlabel("Local view of the shared t-SNE embedding", fontsize=9, y=0.015)
    fig.tight_layout(rect=(0.01, 0.045, 0.99, 0.92), w_pad=0.8, h_pad=1.0)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def evaluate_target_reconstruction(teacher, prototypes, batch_size, workers, device, temperature, proto_temperature, fusion_scale):
    dataset = datasets.CIFAR100(
        root=get_data_folder(),
        train=False,
        transform=get_cifar100_train_transform_strong(),
        download=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    sums = {key: 0.0 for key in ("raw_true_prob", "fused_true_prob", "raw_wrong_true_prob", "fused_wrong_true_prob", "lambda")}
    counts = {key: 0 for key in ("total", "raw_wrong", "raw_right")}
    outcomes = {key: 0 for key in ("raw_correct", "proto_correct", "fused_correct", "rescued", "harmed")}

    for images, targets in loader:
        weak, strong = images[0].to(device), images[1].to(device)
        targets = targets.to(device)
        logits_w, _ = extract_teacher_logits_and_feature(teacher(weak))
        logits_s, features_s = extract_teacher_logits_and_feature(teacher(strong))
        raw = F.softmax(logits_s / temperature, dim=1)
        proto = prototype_distribution(features_s, prototypes, proto_temperature)
        weak_pred = logits_w.argmax(dim=1)
        hard_prob = F.softmax(logits_s, dim=1)
        entropy = -torch.sum(hard_prob * torch.log(hard_prob + 1e-8), dim=1)
        uncertainty = entropy / math.log(raw.size(1))
        fused, fusion_weight = gated_reconstruction(raw, proto, weak_pred, uncertainty, fusion_scale)

        raw_pred = raw.argmax(dim=1)
        proto_pred = proto.argmax(dim=1)
        fused_pred = fused.argmax(dim=1)
        raw_correct = raw_pred.eq(targets)
        fused_correct = fused_pred.eq(targets)
        raw_wrong = ~raw_correct
        raw_right = raw_correct
        outcomes["raw_correct"] += raw_correct.sum().item()
        outcomes["proto_correct"] += proto_pred.eq(targets).sum().item()
        outcomes["fused_correct"] += fused_correct.sum().item()
        outcomes["rescued"] += (raw_wrong & fused_correct).sum().item()
        outcomes["harmed"] += (raw_right & ~fused_correct).sum().item()
        sums["raw_true_prob"] += raw.gather(1, targets[:, None]).sum().item()
        sums["fused_true_prob"] += fused.gather(1, targets[:, None]).sum().item()
        sums["lambda"] += fusion_weight.sum().item()
        counts["total"] += targets.numel()
        counts["raw_wrong"] += raw_wrong.sum().item()
        counts["raw_right"] += raw_right.sum().item()
        if raw_wrong.any():
            sums["raw_wrong_true_prob"] += raw[raw_wrong].gather(1, targets[raw_wrong, None]).sum().item()
            sums["fused_wrong_true_prob"] += fused[raw_wrong].gather(1, targets[raw_wrong, None]).sum().item()

    total = counts["total"]
    wrong = counts["raw_wrong"]
    rows = [
        ("Raw strong-view teacher target Top-1", 100.0 * outcomes["raw_correct"] / total),
        ("Prototype distribution Top-1", 100.0 * outcomes["proto_correct"] / total),
        ("Gated reconstructed target Top-1", 100.0 * outcomes["fused_correct"] / total),
        ("Raw target true-class probability", 100.0 * sums["raw_true_prob"] / total),
        ("Reconstructed target true-class probability", 100.0 * sums["fused_true_prob"] / total),
        ("Rescue rate on raw-wrong samples", 100.0 * outcomes["rescued"] / wrong if wrong else 0.0),
        ("Harm rate on raw-correct samples", 100.0 * outcomes["harmed"] / counts["raw_right"] if counts["raw_right"] else 0.0),
        ("Mean fusion weight", sums["lambda"] / total),
    ]
    return rows


def plot_target_metrics(rows, output_path):
    values = dict(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=180)
    labels = ["Raw teacher", "Prototype", "Reconstructed"]
    top1 = [values["Raw strong-view teacher target Top-1"], values["Prototype distribution Top-1"], values["Gated reconstructed target Top-1"]]
    axes[0].bar(labels, top1, color=["#4C78A8", "#72B7B2", "#F58518"])
    axes[0].set_ylabel("Target Top-1 (%)")
    axes[0].set_title("Target correctness")
    axes[0].set_ylim(max(0, min(top1) - 3), max(top1) + 3)
    axes[0].tick_params(axis="x", rotation=18)
    prob = [values["Raw target true-class probability"], values["Reconstructed target true-class probability"]]
    axes[1].bar(["Raw teacher", "Reconstructed"], prob, color=["#4C78A8", "#F58518"])
    axes[1].set_ylabel("Mean true-class probability (%)")
    axes[1].set_title("Target confidence")
    axes[1].set_ylim(max(0, min(prob) - 3), min(100, max(prob) + 3))
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Quality of strong-view teacher target reconstruction", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_ablation(output_path):
    labels = ["CRLD", "+ prototype reconstruction", "+ reliability gate"]
    values = [74.42, 74.99, 75.21]
    gains = [value - values[0] for value in values]
    colors = ["#697887", "#31988B", "#E68632"]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), dpi=260,
                            gridspec_kw={"width_ratios": [1.12, 1]})
    ax_accuracy, ax_gain = axes
    bars = ax_accuracy.barh(y, values, color=colors, height=0.58)
    ax_accuracy.set_yticks(y, labels)
    ax_accuracy.invert_yaxis()
    ax_accuracy.set_xlim(0, 80)
    ax_accuracy.set_xticks(np.arange(0, 81, 20))
    ax_accuracy.set_xlabel("Top-1 accuracy (%)")
    ax_accuracy.set_title("(a) Absolute performance", loc="left", fontsize=10.5, pad=10)
    for bar, value in zip(bars, values):
        ax_accuracy.text(value + 0.8, bar.get_y() + bar.get_height() / 2,
                         f"{value:.2f}", va="center", fontsize=9.5, fontweight="semibold")

    gain_bars = ax_gain.barh(y, gains, color=colors, height=0.58)
    ax_gain.set_yticks(y, ["CRLD baseline", "After prototype reconstruction", "After reliability gate"])
    ax_gain.invert_yaxis()
    ax_gain.set_xlim(0, 1.0)
    ax_gain.set_xticks(np.arange(0, 1.01, 0.2))
    ax_gain.set_xlabel("Cumulative gain over CRLD (percentage points)")
    ax_gain.set_title("(b) Improvement over baseline", loc="left", fontsize=10.5, pad=10)
    for index, (bar, gain) in enumerate(zip(gain_bars, gains)):
        label = "0.00" if index == 0 else f"+{gain:.2f}"
        x_pos = max(gain, 0.0) + 0.025
        ax_gain.text(x_pos, bar.get_y() + bar.get_height() / 2,
                     label, va="center", fontsize=9.5, fontweight="semibold")
    for ax in axes:
        ax.grid(axis="x", color="#D9DEE5", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=8.8, pad=7)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.tight_layout(pad=1.2, w_pad=2.0)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "download_ckpts/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/analysis")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau-w", type=float, default=0.8)
    parser.add_argument("--proto-temperature", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--fusion-scale", type=float, default=0.5)
    parser.add_argument("--ablation-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.ablation_only:
        plot_reconstruction_ablation(args.output_dir / "mpkd_target_reconstruction_ablation.png")
        return
    set_training_seed(args.seed)
    device = torch.device(args.device)
    teacher = load_teacher(args.checkpoint, device)
    features_by_class, prototypes = make_prototypes(
        teacher, args.batch_size, args.workers, device, args.tau_w, args.k
    )
    plot_feature_distribution(
        features_by_class,
        prototypes,
        args.output_dir / "mpkd_prototype_feature_distribution.png",
        args.k,
    )
    rows = evaluate_target_reconstruction(
        teacher,
        prototypes,
        args.batch_size,
        args.workers,
        device,
        args.temperature,
        args.proto_temperature,
        args.fusion_scale,
    )
    with (args.output_dir / "mpkd_target_reconstruction.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)
    plot_target_metrics(rows, args.output_dir / "mpkd_target_reconstruction.png")
    print("Saved analysis to", args.output_dir)
    for metric, value in rows:
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
