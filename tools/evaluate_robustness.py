"""Evaluate CRLD and MPKD under paired strong-view perturbations.

The evaluator uses the same deterministic augmented image for both models at
each (n, repeat, sample) tuple. It produces raw CSV files and two paper-ready
figures for Section 4.4.3.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from mdistiller.dataset.cifar100 import (
    RandAugment,
    get_cifar100_test_transform,
    get_data_folder,
)
from mdistiller.models import cifar_model_dict


DEFAULT_BASELINE = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "cifar100_baselines"
    / "crld_proto,res32x4,res8x4,ablation_a"
    / "student_best"
)
DEFAULT_MPKD = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "cifar100_baselines"
    / "crld_proto,res32x4,res8x4"
    / "student_best"
)
DEFAULT_TEACHER = (
    Path(__file__).resolve().parents[1]
    / "download_ckpts"
    / "cifar_teachers"
    / "resnet32x4_vanilla"
    / "ckpt_epoch_240.pth"
)


def apply_seeded_transform(transform, image, seed):
    seed = int(seed) % (2**32 - 1)
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        torch.set_rng_state(generator.get_state())
        return transform(image)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


class PairedStrongCIFAR100(Dataset):
    """CIFAR-100 test set with index-seeded strong-view augmentation."""

    def __init__(self, root, n, repeat, seed):
        self.base = datasets.CIFAR100(
            root=root, train=False, transform=None, download=False
        )
        self.n = int(n)
        self.repeat = int(repeat)
        self.seed = int(seed)
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )
        self.transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                RandAugment(self.n, 10),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image = Image.fromarray(self.base.data[index])
        seed = self.seed + self.repeat * 1000003 + self.n * 10007 + index
        image = apply_seeded_transform(self.transform, image, seed)
        return image, self.base.targets[index]


class ReliabilityCIFAR100(Dataset):
    """Paired weak/strong views used for the qualitative reliability figure."""

    def __init__(self, root, indices, n, seed):
        self.base = datasets.CIFAR100(
            root=root, train=True, transform=None, download=False
        )
        self.indices = [int(index) for index in indices]
        self.n = int(n)
        self.seed = int(seed)
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )
        self.weak_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.strong_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                RandAugment(self.n, 10),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        image = Image.fromarray(self.base.data[index])
        seed = self.seed + position * 10007 + index
        weak = apply_seeded_transform(self.weak_transform, image.copy(), seed + 13)
        strong = apply_seeded_transform(self.strong_transform, image.copy(), seed + 29)
        return weak, strong, self.base.targets[index], index


def parse_int_csv(value):
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def load_student(checkpoint, device):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint}")

    state = dict(state)
    prefixes = ("module.student.", "student.", "module.")
    if not any(key.startswith("conv1.") for key in state):
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in state):
                state = {key[len(prefix):]: value for key, value in state.items()}
                break

    model = cifar_model_dict["resnet8x4"][0](num_classes=100)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match resnet8x4: "
            f"missing={incompatible.missing_keys[:3]}, "
            f"unexpected={incompatible.unexpected_keys[:3]}"
        )
    model.eval().to(device)
    return model


def load_teacher(checkpoint, device):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported teacher checkpoint format: {checkpoint}")
    model = cifar_model_dict["resnet32x4"][0](num_classes=100)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Teacher checkpoint does not match resnet32x4: "
            f"missing={incompatible.missing_keys[:3]}, "
            f"unexpected={incompatible.unexpected_keys[:3]}"
        )
    model.eval().to(device)
    return model


def model_logits(model, images):
    output = model(images)
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output


@torch.inference_mode()
def evaluate_loader(model, loader, device):
    correct1 = 0
    correct5 = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model_logits(model, images)
        topk = logits.topk(5, dim=1).indices
        correct1 += topk[:, :1].eq(targets.unsqueeze(1)).sum().item()
        correct5 += topk.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.size(0)
    return 100.0 * correct1 / total, 100.0 * correct5 / total


def make_loader(dataset, batch_size, workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )


def evaluate_models(args, output_dir):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    data_root = get_data_folder()
    clean_dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        transform=get_cifar100_test_transform(),
        download=False,
    )
    clean_loader = make_loader(clean_dataset, args.batch_size, args.workers, device)

    model_paths = {"CRLD": Path(args.baseline_checkpoint), "MPKD": Path(args.mpkd_checkpoint)}
    models = {}
    for name, path in model_paths.items():
        print(f"Loading {name}: {path}", flush=True)
        models[name] = load_student(path, device)

    rows = []
    for name, model in models.items():
        top1, top5 = evaluate_loader(model, clean_loader, device)
        rows.append(
            {
                "model": name,
                "augmentation": "clean",
                "n": 0,
                "repeat": 0,
                "top1": top1,
                "top5": top5,
            }
        )
        print(f"{name:5s} clean      top1={top1:.2f} top5={top5:.2f}", flush=True)

    for n in args.n_values:
        for repeat in range(1, args.repeats + 1):
            dataset = PairedStrongCIFAR100(data_root, n, repeat, args.seed)
            loader = make_loader(dataset, args.batch_size, args.workers, device)
            for name, model in models.items():
                top1, top5 = evaluate_loader(model, loader, device)
                rows.append(
                    {
                        "model": name,
                        "augmentation": "RandAugment",
                        "n": n,
                        "repeat": repeat,
                        "top1": top1,
                        "top5": top5,
                    }
                )
                print(
                    f"{name:5s} n={n} repeat={repeat} "
                    f"top1={top1:.2f} top5={top5:.2f}",
                    flush=True,
                )

    raw_path = output_dir / "robustness_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "augmentation", "n", "repeat", "top1", "top5"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_path = output_dir / "robustness_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "augmentation", "n", "top1_mean", "top1_std", "top5_mean", "top5_std"],
        )
        writer.writeheader()
        writer.writerows(summary)

    plot_extreme(summary, output_dir / "mpkd_robustness_extreme.png", max(args.n_values))
    plot_sensitivity(summary, output_dir / "mpkd_robustness_sensitivity.png", args.n_values)
    candidate_count = min(args.stability_candidates, len(clean_dataset))
    candidate_indices = np.random.default_rng(args.seed + 404).choice(
        len(clean_dataset), candidate_count, replace=False
    )
    teacher = load_teacher(args.teacher_checkpoint, device)
    reliability_dataset = ReliabilityCIFAR100(
        data_root, candidate_indices, args.stability_n, args.seed + 505
    )
    reliability_loader = make_loader(
        reliability_dataset, args.batch_size, args.workers, device
    )
    reliability_records = collect_reliability_samples(
        teacher, reliability_loader, device
    )
    plot_reliability_samples(
        reliability_records,
        output_dir / "mpkd_reliability_samples.png",
        count=min(8, args.stability_count),
    )
    print(f"Saved raw results: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved figures: {output_dir / 'mpkd_robustness_extreme.png'}")
    print(f"               {output_dir / 'mpkd_robustness_sensitivity.png'}")
    print(f"               {output_dir / 'mpkd_reliability_samples.png'}")


def summarize(rows):
    grouped = defaultdict(lambda: {"top1": [], "top5": []})
    for row in rows:
        key = (row["model"], row["augmentation"], row["n"])
        grouped[key]["top1"].append(row["top1"])
        grouped[key]["top5"].append(row["top5"])
    result = []
    for (model, augmentation, n), values in sorted(grouped.items()):
        result.append(
            {
                "model": model,
                "augmentation": augmentation,
                "n": n,
                "top1_mean": float(np.mean(values["top1"])),
                "top1_std": float(np.std(values["top1"])),
                "top5_mean": float(np.mean(values["top5"])),
                "top5_std": float(np.std(values["top5"])),
            }
        )
    return result


def lookup(summary, model, augmentation, n):
    for row in summary:
        if row["model"] == model and row["augmentation"] == augmentation and row["n"] == n:
            return row
    raise KeyError((model, augmentation, n))


def configure_plot():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_extreme(summary, path, extreme_n):
    configure_plot()
    models = ["CRLD", "MPKD"]
    labels = ["Clean", f"RandAugment (n={extreme_n})"]
    values = np.array(
        [
            [lookup(summary, model, "clean", 0)["top1_mean"], lookup(summary, model, "RandAugment", extreme_n)["top1_mean"]]
            for model in models
        ]
    )
    errors = np.array(
        [
            [lookup(summary, model, "clean", 0)["top1_std"], lookup(summary, model, "RandAugment", extreme_n)["top1_std"]]
            for model in models
        ]
    )
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bars1 = ax.bar(x - width / 2, values[0], width, yerr=errors[0], capsize=3, label="CRLD", color="#4C78A8")
    bars2 = ax.bar(x + width / 2, values[1], width, yerr=errors[1], capsize=3, label="MPKD", color="#F58518")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Robustness under extreme strong-view perturbation")
    ax.set_ylim(max(0, values.min() - 8), min(100, values.max() + 5))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(summary, path, n_values):
    configure_plot()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    colors = {"CRLD": "#4C78A8", "MPKD": "#F58518"}
    markers = {"CRLD": "o", "MPKD": "s"}
    for model in ("CRLD", "MPKD"):
        means = np.array([lookup(summary, model, "RandAugment", n)["top1_mean"] for n in n_values])
        stds = np.array([lookup(summary, model, "RandAugment", n)["top1_std"] for n in n_values])
        ax.plot(n_values, means, marker=markers[model], linewidth=2, label=model, color=colors[model])
        ax.fill_between(n_values, means - stds, means + stds, color=colors[model], alpha=0.14, linewidth=0)
    ax.set_xlabel("Number of RandAugment operations (n)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Sensitivity to strong-view augmentation strength")
    ax.set_xticks(n_values)
    all_values = [lookup(summary, model, "RandAugment", n)["top1_mean"] for model in ("CRLD", "MPKD") for n in n_values]
    ax.set_ylim(max(0, min(all_values) - 4), min(100, max(all_values) + 4))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def collect_reliability_samples(model, loader, device):
    records = []
    for weak, strong, targets, indices in loader:
        weak = weak.to(device, non_blocking=True)
        strong = strong.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits_w = model_logits(model, weak)
        logits_s = model_logits(model, strong)
        probs_w = logits_w.softmax(dim=1)
        probs_s = logits_s.softmax(dim=1)
        target_prob = 0.5 * (
            probs_w.gather(1, targets.unsqueeze(1)).squeeze(1)
            + probs_s.gather(1, targets.unsqueeze(1)).squeeze(1)
        )
        pred_w = logits_w.argmax(dim=1)
        pred_s = logits_s.argmax(dim=1)
        for position in range(targets.size(0)):
            records.append(
                {
                    "index": int(indices[position]),
                    "target": int(targets[position]),
                    "pred_w": int(pred_w[position]),
                    "pred_s": int(pred_s[position]),
                    "score": float(target_prob[position]),
                    "image": strong[position].cpu(),
                }
            )
    return records


def tensor_to_image(image):
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
    image = (image * std + mean).clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def plot_reliability_samples(records, path, count=8):
    if len(records) < count:
        raise RuntimeError("Not enough reliability samples to draw the figure")
    reliable = [
        row
        for row in records
        if row["pred_w"] == row["target"] and row["pred_s"] == row["target"]
    ]
    high_pool = reliable if len(reliable) >= count else records
    high = sorted(high_pool, key=lambda row: row["score"], reverse=True)[:count]
    low = sorted(records, key=lambda row: row["score"])[:count]

    configure_plot()
    fig, axes = plt.subplots(2, count, figsize=(16, 4.8))
    for row_index, selected in enumerate((high, low)):
        for column, item in enumerate(selected):
            axes[row_index, column].imshow(tensor_to_image(item["image"]))
            axes[row_index, column].set_title(
                f"y={item['target']}  s={item['score']:.2f}\n"
                f"pw={item['pred_w']}  ps={item['pred_s']}",
                fontsize=8,
            )
            axes[row_index, column].axis("off")
    axes[0, 0].set_ylabel("High reliability", fontsize=10)
    axes[1, 0].set_ylabel("Low reliability", fontsize=10)
    fig.suptitle("Strong-view reliability diagnosis by the frozen teacher", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate paired CIFAR-100 robustness curves.")
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--mpkd-checkpoint", type=Path, default=DEFAULT_MPKD)
    parser.add_argument("--output-dir", type=Path, default=root / "experiments" / "robustness")
    parser.add_argument("--n-values", type=parse_int_csv, default=[1, 2, 3, 4, 5])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--stability-n", type=int, default=2)
    parser.add_argument("--stability-candidates", type=int, default=256)
    parser.add_argument("--stability-count", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if not args.n_values:
        parser.error("--n-values cannot be empty")
    if args.stability_n < 1:
        parser.error("--stability-n must be positive")
    if args.stability_count < 1:
        parser.error("--stability-count must be positive")
    if args.stability_candidates < args.stability_count:
        parser.error("--stability-candidates must cover --stability-count")
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluate_models(args, args.output_dir.resolve())


if __name__ == "__main__":
    main()
