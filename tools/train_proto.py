import argparse
import copy
import csv
import glob
import math
import os
import random

import shutil
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mdistiller.dataset import get_dataset_strong
from mdistiller.dataset.cifar100 import get_cifar100_test_transform
from mdistiller.dataset.tinyimagenet200 import (
    get_tinyimagenet200_test_transform,
    get_tinyimagenet200_test_transform_384,
)
from mdistiller.distillers import distiller_dict
from mdistiller.distillers._base import Distiller
from mdistiller.engine import trainer_dict
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.models import (
    cifar_model_dict,
    imagenet_model_dict,
    tinyimagenet200_model_dict,
)

# cuDNN benchmarking may select a convolution algorithm with a large
# transient workspace on Windows, which can fail as HOST_ALLOCATION_FAILED
# after a long run. Keep the stable setting by default and allow an explicit
# opt-in for machines with enough memory.
_cudnn_bool = lambda name, default=False: os.environ.get(
    name, "1" if default else "0"
).strip().lower() in {"1", "true", "yes", "on"}
cudnn.benchmark = _cudnn_bool("CRLD_CUDNN_BENCHMARK")
cudnn.deterministic = _cudnn_bool("CRLD_CUDNN_DETERMINISTIC")


DEFAULT_TRAINING_SEED = 42
METHOD_DISPLAY_NAME = "CRLD-Proto"
BASELINE_DISPLAY_NAME = "CRLD"
PROTO_DISTILLER_TYPES = {"CRLD_Proto", "MPKD"}


ABLATION_MODE_SPECS = {
    "A": {
        "tag": "ablation_a",
        "description": "Complete CRLD without prototype correction.",
        "PROTO_ENABLED": False,
        "PROTO_FUSION_MODE": "fixed",
        "PROTO_GATE_UNCERTAINTY": False,
        "PROTO_GATE_AGREEMENT": False,
        "PROTO_GATE_POSITIVE_GAIN": False,
        "PROTO_GATE_CONFIDENCE": False,
    },
    "B": {
        "tag": "ablation_b",
        "description": (
            "CRLD plus K-Means multi-center prototypes, Log-Mean-Exp, and "
            "unconditional fixed-weight fusion."
        ),
        "PROTO_ENABLED": True,
        "PROTO_FUSION_MODE": "fixed",
        "PROTO_GATE_UNCERTAINTY": False,
        "PROTO_GATE_AGREEMENT": False,
        "PROTO_GATE_POSITIVE_GAIN": False,
        "PROTO_GATE_CONFIDENCE": False,
    },
    "C": {
        "tag": "ablation_c",
        "description": "B plus teacher strong-view uncertainty gating.",
        "PROTO_ENABLED": True,
        "PROTO_FUSION_MODE": "gated",
        "PROTO_GATE_UNCERTAINTY": True,
        "PROTO_GATE_AGREEMENT": False,
        "PROTO_GATE_POSITIVE_GAIN": False,
        "PROTO_GATE_CONFIDENCE": False,
    },
    "D": {
        "tag": "ablation_d",
        "description": "C plus class-agreement and positive-gain gating.",
        "PROTO_ENABLED": True,
        "PROTO_FUSION_MODE": "gated",
        "PROTO_GATE_UNCERTAINTY": True,
        "PROTO_GATE_AGREEMENT": True,
        "PROTO_GATE_POSITIVE_GAIN": True,
        "PROTO_GATE_CONFIDENCE": False,
    },
    "E": {
        "tag": "ablation_e",
        "description": "Complete CRLD-Proto with all reliability gates.",
        "PROTO_ENABLED": True,
        "PROTO_FUSION_MODE": "gated",
        "PROTO_GATE_UNCERTAINTY": True,
        "PROTO_GATE_AGREEMENT": True,
        "PROTO_GATE_POSITIVE_GAIN": True,
        "PROTO_GATE_CONFIDENCE": True,
    },
    "gate_no_uncertainty": {
        "tag": "gate_no_uncertainty",
        "description": "Complete CRLD-Proto without uncertainty gating.",
        "PROTO_ENABLED": True,
        "PROTO_FUSION_MODE": "gated",
        "PROTO_GATE_UNCERTAINTY": False,
        "PROTO_GATE_AGREEMENT": True,
        "PROTO_GATE_POSITIVE_GAIN": True,
        "PROTO_GATE_CONFIDENCE": True,
    },
    "gate_no_agreement": {
        "tag": "gate_no_agreement",
        "description": "Complete CRLD-Proto without class-agreement gating.",
        "PROTO_ENABLED": True,
        "PROTO_FUSION_MODE": "gated",
        "PROTO_GATE_UNCERTAINTY": True,
        "PROTO_GATE_AGREEMENT": False,
        "PROTO_GATE_POSITIVE_GAIN": True,
        "PROTO_GATE_CONFIDENCE": True,
    },
    "gate_no_positive_gain": {
        "tag": "gate_no_positive_gain",
        "description": "Complete CRLD-Proto without positive-gain gating.",
        "PROTO_ENABLED": True,
        "PROTO_FUSION_MODE": "gated",
        "PROTO_GATE_UNCERTAINTY": True,
        "PROTO_GATE_AGREEMENT": True,
        "PROTO_GATE_POSITIVE_GAIN": False,
        "PROTO_GATE_CONFIDENCE": True,
    },
    "gate_no_confidence": {
        "tag": "gate_no_confidence",
        "description": "Complete CRLD-Proto without prototype-confidence gating.",
        "PROTO_ENABLED": True,
        "PROTO_FUSION_MODE": "gated",
        "PROTO_GATE_UNCERTAINTY": True,
        "PROTO_GATE_AGREEMENT": True,
        "PROTO_GATE_POSITIVE_GAIN": True,
        "PROTO_GATE_CONFIDENCE": False,
    },
}


def apply_ablation_mode(config, mode):
    """Apply one controlled ablation and isolate its output directory."""
    if mode is None:
        return

    spec = ABLATION_MODE_SPECS[mode]
    for key, value in spec.items():
        if key not in {"tag", "description"}:
            setattr(config.CRLD, key, value)

    experiment_field = "NAME" if config.EXPERIMENT.NAME else "TAG"
    base_name = getattr(config.EXPERIMENT, experiment_field)
    suffix = spec["tag"]
    if not base_name.endswith("," + suffix):
        setattr(config.EXPERIMENT, experiment_field, f"{base_name},{suffix}")


def validate_ablation_context(config, mode):
    """Reject accidental ablations under a different experimental setting."""
    if mode is None:
        return

    expected = {
        "DATASET.TYPE": (config.DATASET.TYPE, "cifar100"),
        "DISTILLER.TYPE": (config.DISTILLER.TYPE, "CRLD_Proto"),
        "CRLD.PROTO_K": (int(config.CRLD.PROTO_K), 3),
    }
    mismatches = [
        f"{name}={actual!r} (expected {required!r})"
        for name, (actual, required) in expected.items()
        if actual != required
    ]
    if mismatches:
        raise ValueError(
            "CIFAR-100 CRLD-Proto ablations require CRLD_Proto and K=3. "
            "Invalid setting(s): " + "; ".join(mismatches)
        )


def set_training_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


# ==============================================================================
# 1. Checkpoint cleanup
# ==============================================================================
def cleanup_useless_checkpoints(exp_dir):
    print(log_msg("Cleaning up extra checkpoints...", "INFO"))
    garbage_patterns = ['epoch_*', 'student_[0-9]*', 'latest*', 'student_latest*', 'best*']
    deleted_count = 0
    for pattern in garbage_patterns:
        for filepath in glob.glob(os.path.join(exp_dir, pattern)):
            if 'student_best' in os.path.basename(filepath):
                continue
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                deleted_count += 1
            except OSError:
                continue
    print(log_msg(f"Cleanup finished. Removed {deleted_count} files.", "INFO"))


# ==============================================================================
# 2. Feature extraction hook
# ==============================================================================
class FeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.feature = None

    def hook_fn(self, module, input, output):
        if isinstance(output, (tuple, list)):
            output = output[0] if output else None
        self.feature = (
            output.flatten(1) if torch.is_tensor(output) else None
        )

    def clear(self):
        self.feature = None

    def close(self):
        self.hook.remove()


def register_avgpool_hook(model):
    hook = None
    for name, module in model.named_modules():
        if "avg_pool" in name or "global_pool" in name or "avgpool" in name:
            hook = FeatureHook(module)
            break
    if hook is None:
        modules = list(model.children())
        for i, module in enumerate(modules):
            if isinstance(module, nn.Linear) and i > 0:
                hook = FeatureHook(modules[i - 1])
                break
    return hook


def split_logits_and_pooled_feature(output):
    """Read logits and pooled features without shared hook state."""
    if isinstance(output, (tuple, list)):
        logits = output[0]
        feature_info = output[1] if len(output) > 1 else None
    else:
        logits = output
        feature_info = None

    pooled_feature = None
    if isinstance(feature_info, dict):
        pooled_feature = feature_info.get("pooled_feat")
    elif torch.is_tensor(feature_info):
        pooled_feature = feature_info
    return logits, pooled_feature


def extract_teacher_logits_and_feature(output, hook=None):
    """Prefer a model-returned pooled feature, with a legacy hook fallback."""
    logits, feature = split_logits_and_pooled_feature(output)
    if feature is None and hook is not None:
        feature = hook.feature
    if feature is None:
        raise RuntimeError(
            "Teacher output contains no pooled feature and the fallback "
            "feature hook is empty."
        )
    return logits, feature.detach().flatten(1)


# ==============================================================================
# 3. Multi-center prototypes: reliable feature mining + K-Means
# ==============================================================================
def mine_reliable_teacher_features(
    teacher,
    train_loader,
    num_classes,
    device='cuda',
    tau_w=0.8,
    max_features_per_class=None,
    storage_dtype=torch.float32,
):
    teacher.eval()
    hook = register_avgpool_hook(teacher)
    max_entropy = math.log(num_classes)

    print(log_msg(f"Phase 1: Mining high-purity features (tau_w={tau_w})...", "INFO"))
    class_features = {c: [] for c in range(num_classes)}

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Feature Mining"):
            image_weak = batch[0][0] if isinstance(batch[0], (list, tuple)) else batch[0]
            target = batch[1].to(device)
            image_weak = image_weak.to(device)

            teacher_output = teacher(image_weak)
            logits, features = extract_teacher_logits_and_feature(
                teacher_output, hook
            )
            # Keep the offline prototype cache on CPU. This is essential for
            # ImageNet, where storing every 2048-D feature on the GPU is not
            # feasible.
            features_norm = F.normalize(features, dim=1).cpu().to(storage_dtype)
            if hook is not None:
                hook.clear()

            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            conf_w = 1.0 - (entropy / max_entropy)
            preds = torch.argmax(probs, dim=1)

            for i in range(image_weak.size(0)):
                c = target[i].item()
                if conf_w[i].item() >= tau_w and preds[i].item() == c:
                    if (
                        max_features_per_class is None
                        or len(class_features[c]) < max_features_per_class
                    ):
                        class_features[c].append(features_norm[i].clone())
    if hook is not None:
        hook.close()

    selected_counts = {c: len(class_features[c]) for c in range(num_classes)}
    empty_classes = [c for c, count in selected_counts.items() if count == 0]
    if empty_classes:
        raise RuntimeError(
            "No reliable weak-view features were selected for classes: "
            f"{empty_classes}. Lower PROTO_TAU_W only after checking the "
            "class-wise selection statistics."
        )

    min_count = min(selected_counts.values())
    max_count = max(selected_counts.values())
    mean_count = sum(selected_counts.values()) / num_classes
    print(log_msg(
        "Reliable feature counts per class: "
        f"min={min_count}, mean={mean_count:.1f}, max={max_count}",
        "INFO",
    ))

    return class_features


def build_multicenter_prototypes(
    class_features,
    num_classes,
    K=3,
    device='cuda',
    feature_dim=128,
):
    selected_counts = {c: len(class_features[c]) for c in range(num_classes)}
    empty_classes = [c for c, count in selected_counts.items() if count == 0]
    if empty_classes:
        raise RuntimeError(
            "No reliable weak-view features were selected for classes: "
            f"{empty_classes}. Lower PROTO_TAU_W only after checking the "
            "class-wise selection statistics."
        )

    print(log_msg(f"Phase 2: K-Means initialization (K={K}) for each class...", "INFO"))
    prototypes_nk = torch.zeros(
        num_classes, K, feature_dim, device=device, dtype=torch.float32
    )
    for c in range(num_classes):
        feats_c = class_features[c]
        if len(feats_c) < K:
            raise RuntimeError(
                f"Class {c} has only {len(feats_c)} reliable features, "
                f"but K={K} sub-prototypes are required."
            )

        X = torch.stack(feats_c).to(device=device, dtype=torch.float32)
        N_c = X.size(0)

        # Use cosine K-Means centers to represent the class's modes.
        indices = torch.randperm(N_c, device=device)[:K]
        centers = X[indices].clone()
        for _ in range(10):
            sim_matrix = torch.matmul(X, centers.t())
            cluster_assignments = torch.argmax(sim_matrix, dim=1)
            new_centers = torch.zeros_like(centers)
            for k in range(K):
                mask = (cluster_assignments == k)
                if mask.sum() > 0:
                    new_centers[k] = F.normalize(X[mask].mean(dim=0), dim=0)
                else:
                    new_centers[k] = X[torch.randint(0, N_c, (1,))].squeeze(0)
            centers = new_centers
        prototypes_nk[c] = centers

    print(log_msg("K-Means prototypes ready; prototypes are fixed during student training.", "INFO"))
    return F.normalize(prototypes_nk, dim=-1)


def precompute_multicenter_prototypes(
    teacher,
    train_loader,
    num_classes,
    K=3,
    device='cuda',
    feature_dim=128,
    tau_w=0.8,
    max_features_per_class=None,
    storage_dtype=torch.float32,
):
    class_features = mine_reliable_teacher_features(
        teacher=teacher,
        train_loader=train_loader,
        num_classes=num_classes,
        device=device,
        tau_w=tau_w,
        max_features_per_class=max_features_per_class,
        storage_dtype=storage_dtype,
    )
    return build_multicenter_prototypes(
        class_features=class_features,
        num_classes=num_classes,
        K=K,
        device=device,
        feature_dim=feature_dim,
    )


# ==============================================================================
# 4. Evaluate N*K multi-center prototypes
# ==============================================================================
def evaluate_prototypes_nk(teacher, val_loader, prototypes_nk, proto_temp=0.5, device='cuda',
                           view='clean', repeats=1, return_metrics=False):
    if view not in {'clean', 'weak', 'strong'}:
        raise ValueError(f"Unsupported prototype evaluation view: {view}")
    if repeats < 1:
        raise ValueError("Prototype evaluation repeats must be at least 1.")

    teacher.eval()
    hook = register_avgpool_hook(teacher)
    correct, teacher_correct, total = 0, 0, 0
    teacher_wrong, rescued = 0, 0
    teacher_right, harmed = 0, 0

    num_classes, k_num, feat_dim = prototypes_nk.shape
    prototypes_norm = F.normalize(prototypes_nk, dim=-1)

    print(log_msg(
        "Evaluating K-Means prototypes with Log-Mean-Exp "
        f"(K={k_num}, tau_p={proto_temp}, view={view}, repeats={repeats})...",
        "INFO",
    ))

    with torch.no_grad():
        for repeat_index in range(repeats):
            progress_desc = f"Prototype NPC {view} {repeat_index + 1}/{repeats}"
            for batch in tqdm(val_loader, desc=progress_desc):
                image_data = batch[0]
                if isinstance(image_data, (list, tuple)):
                    view_index = 1 if view == 'strong' else 0
                    images = image_data[view_index]
                else:
                    images = image_data
                targets = batch[1].to(device)
                images = images.to(device)

                teacher_output = teacher(images)
                logits_teacher, features = extract_teacher_logits_and_feature(
                    teacher_output, hook
                )
                features_norm = F.normalize(features, dim=1)
                if hook is not None:
                    hook.clear()

                sim = torch.matmul(
                    features_norm,
                    prototypes_norm.view(num_classes * k_num, feat_dim).t(),
                ).view(-1, num_classes, k_num)
                class_logits = torch.logsumexp(sim / proto_temp, dim=2) - math.log(k_num)
                preds = torch.argmax(class_logits, dim=1)
                teacher_preds = torch.argmax(logits_teacher, dim=1)

                proto_correct_mask = preds.eq(targets)
                teacher_correct_mask = teacher_preds.eq(targets)
                correct += proto_correct_mask.sum().item()
                teacher_correct += teacher_correct_mask.sum().item()
                teacher_wrong += (~teacher_correct_mask).sum().item()
                rescued += ((~teacher_correct_mask) & proto_correct_mask).sum().item()
                teacher_right += teacher_correct_mask.sum().item()
                harmed += (teacher_correct_mask & (~proto_correct_mask)).sum().item()
                total += targets.size(0)

    if hook is not None:
        hook.close()
    proto_acc = 100. * correct / total
    teacher_acc = 100. * teacher_correct / total
    rescue_rate = 100. * rescued / teacher_wrong if teacher_wrong else 0.0
    harm_rate = 100. * harmed / teacher_right if teacher_right else 0.0
    print(log_msg("=" * 60, "INFO"))
    print(log_msg(
        f"Prototype (K={k_num}, view={view}) Top-1 Accuracy: {proto_acc:.2f}%",
        "INFO",
    ))
    if return_metrics:
        print(log_msg(
            f"Teacher Accuracy: {teacher_acc:.2f}% | Rescue Rate: {rescue_rate:.2f}% | "
            f"Harm Rate: {harm_rate:.2f}%",
            "INFO",
        ))
    print(log_msg("=" * 60, "INFO"))

    metrics = {
        'proto_acc': proto_acc,
        'teacher_acc': teacher_acc,
        'rescue_rate': rescue_rate,
        'harm_rate': harm_rate,
        'samples': total,
    }
    return metrics if return_metrics else proto_acc


# ==============================================================================
# 5. Fast prototype-parameter scan
# ==============================================================================
def get_proto_scan_clean_transform(dataset_type):
    if dataset_type == "cifar100":
        return get_cifar100_test_transform()
    if dataset_type == "tinyimagenet200":
        return get_tinyimagenet200_test_transform(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    if dataset_type == "tinyimagenet200_384":
        return get_tinyimagenet200_test_transform_384(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    raise ValueError(
        "Prototype parameter scan supports CIFAR-100 and Tiny-ImageNet only; "
        f"received DATASET.TYPE={dataset_type!r}."
    )


def create_proto_scan_loaders(train_loader, dataset_type, num_classes,
                              split_seed, val_per_class=50):
    dataset = train_loader.dataset
    if not hasattr(dataset, 'targets'):
        raise ValueError(
            "Prototype parameter scan requires a training dataset with class targets."
        )

    targets = np.asarray(dataset.targets)
    rng = np.random.RandomState(split_seed)
    prototype_indices = []
    validation_indices = []

    for class_index in range(num_classes):
        class_indices = np.flatnonzero(targets == class_index)
        if len(class_indices) <= val_per_class:
            raise ValueError(
                f"Class {class_index} has {len(class_indices)} samples, which is not enough "
                f"for val_per_class={val_per_class}."
            )
        rng.shuffle(class_indices)
        validation_indices.extend(class_indices[:val_per_class].tolist())
        prototype_indices.extend(class_indices[val_per_class:].tolist())

    # Shallow copies share image metadata/storage but use independent transforms.
    prototype_dataset = copy.copy(dataset)
    clean_validation_dataset = copy.copy(dataset)
    clean_validation_dataset.transform = get_proto_scan_clean_transform(dataset_type)
    strong_validation_dataset = copy.copy(dataset)

    common_loader_args = {
        'batch_size': train_loader.batch_size,
        'shuffle': False,
        'num_workers': min(int(getattr(train_loader, 'num_workers', 0)), 8),
        'pin_memory': getattr(train_loader, 'pin_memory', False),
    }
    prototype_loader = DataLoader(
        Subset(prototype_dataset, prototype_indices),
        **common_loader_args,
    )
    clean_validation_loader = DataLoader(
        Subset(clean_validation_dataset, validation_indices),
        **common_loader_args,
    )
    strong_validation_loader = DataLoader(
        Subset(strong_validation_dataset, validation_indices),
        **common_loader_args,
    )

    print(log_msg(
        "Prototype scan split: "
        f"prototype_train={len(prototype_indices)}, validation={len(validation_indices)}, "
        f"validation_per_class={val_per_class}, split_seed={split_seed}",
        "INFO",
    ))
    return prototype_loader, clean_validation_loader, strong_validation_loader
def create_proto_scan_output_paths(experiment_name):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "output", experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"prototype_parameter_scan_{timestamp}"
    return (
        os.path.join(output_dir, f"{stem}.csv"),
        os.path.join(output_dir, f"{stem}.md"),
    )


def select_proto_scan_candidate(rows):
    if not rows:
        raise RuntimeError("Prototype parameter scan produced no valid candidates.")

    best_strong_acc = max(row['strong_acc'] for row in rows)
    # Strong-view prototype accuracy is the primary proxy. Within 0.10 points,
    # prefer a correction with more rescue potential and less harm potential.
    near_best = [
        row for row in rows
        if row['strong_acc'] >= best_strong_acc - 0.10
    ]
    return max(
        near_best,
        key=lambda row: (
            row['rescue_rate'] - row['harm_rate'],
            row['clean_acc'],
            -abs(row['proto_temp'] - 0.5),
            -abs(row['tau_w'] - 0.8),
        ),
    )


def write_proto_scan_summary(markdown_path, rows, settings, selected):
    with open(markdown_path, 'w', encoding='utf-8') as file:
        file.write("# 原型参数快速扫描\n\n")
        file.write(f"- 创建时间：{datetime.now().isoformat(timespec='seconds')}\n")
        file.write(f"- 子原型数量：$K={settings['K']}$\n")
        file.write(
            f"- 原型构建集：每类保留 {settings['val_per_class']} 张，"
            f"共 {settings['val_per_class'] * settings['num_classes']} 张分层验证图像\n"
        )
        file.write("- 原型构建：类内 K-Means（10 次更新）\n")
        file.write(f"- 强视图重复次数：{settings['strong_repeats']}\n")
        file.write(f"- 随机种子：{settings['seed']}\n\n")

        file.write("## 候选结果\n\n")
        file.write(
            "| $\\tau_w$ | $\\tau_p$ | 每类可靠样本最小/均值 | "
            "干净视图原型准确率 | 强视图原型准确率 | 强视图教师准确率 | "
            "Rescue Rate | Harm Rate | 耗时（秒） |\n"
        )
        file.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            file.write(
                f"| {row['tau_w']:.2f} | {row['proto_temp']:.2f} | "
                f"{row['min_reliable_count']}/{row['mean_reliable_count']:.1f} | "
                f"{row['clean_acc']:.2f} | {row['strong_acc']:.2f} | "
                f"{row['teacher_strong_acc']:.2f} | {row['rescue_rate']:.2f} | "
                f"{row['harm_rate']:.2f} | {row['elapsed_seconds']:.1f} |\n"
            )

        file.write("\n## 推荐候选\n\n")
        file.write(
            f"- `PROTO_TAU_W: {selected['tau_w']:.2f}`\n"
            f"- `PROTO_TEMP: {selected['proto_temp']:.2f}`\n"
            f"- `PROTO_K: {settings['K']}`\n"
        )
        file.write(
            "\n筛选规则：先取强视图原型准确率距最优不超过 0.10 个百分点的候选，"
            "再优先选择 $\\mathrm{RescueRate}-\\mathrm{HarmRate}$ 更高者，"
            "随后比较干净视图原型准确率。\n\n"
        )
        file.write(
            "> 这是原型空间的快速代理筛选，不等价于完整 240 epoch 学生训练的最终准确率。"
            "最终配置仍需通过完整 CRLD-Proto 训练确认。\n"
        )


def run_proto_parameter_scan(teacher, train_loader, num_classes, feature_dim, K,
                             proto_temps, tau_values, seed, split_seed,
                             val_per_class, strong_repeats, dataset_type,
                             experiment_name, device='cuda'):
    if dataset_type not in {"cifar100", "tinyimagenet200", "tinyimagenet200_384"}:
        raise ValueError(
            "The fast prototype parameter scan supports CIFAR-100 and "
            f"Tiny-ImageNet only; received {dataset_type!r}."
        )
    if not proto_temps or not tau_values:
        raise ValueError("At least one prototype temperature and one reliability threshold are required.")

    previous_benchmark = cudnn.benchmark
    previous_deterministic = cudnn.deterministic
    cudnn.benchmark = False
    cudnn.deterministic = True

    try:
        set_training_seed(split_seed)
        prototype_loader, clean_loader, strong_loader = create_proto_scan_loaders(
            train_loader=train_loader,
            dataset_type=dataset_type,
            num_classes=num_classes,
            split_seed=split_seed,
            val_per_class=val_per_class,
        )
        csv_path, markdown_path = create_proto_scan_output_paths(experiment_name)
        fieldnames = [
            'tau_w', 'proto_temp', 'min_reliable_count', 'mean_reliable_count',
            'clean_acc', 'strong_acc', 'teacher_strong_acc', 'rescue_rate',
            'harm_rate', 'elapsed_seconds',
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as file:
            csv.DictWriter(file, fieldnames=fieldnames).writeheader()

        print(log_msg(f"Prototype parameter scan CSV: {csv_path}", "INFO"))
        rows = []
        evaluation_seed = split_seed + 100000

        for tau_w in tau_values:
            print(log_msg("=" * 72, "INFO"))
            print(log_msg(
                f"Prototype parameter scan: tau_w={tau_w:.2f}, K={K}",
                "INFO",
            ))
            start_time = time.time()
            set_training_seed(seed)
            try:
                class_features = mine_reliable_teacher_features(
                    teacher=teacher,
                    train_loader=prototype_loader,
                    num_classes=num_classes,
                    device=device,
                    tau_w=tau_w,
                )
                counts = [len(features) for features in class_features.values()]
                prototypes_nk = build_multicenter_prototypes(
                    class_features=class_features,
                    num_classes=num_classes,
                    K=K,
                    device=device,
                    feature_dim=feature_dim,
                )
            except RuntimeError as error:
                print(log_msg(
                    f"Skipping tau_w={tau_w:.2f}: {error}",
                    "INFO",
                ))
                continue

            for proto_temp in proto_temps:
                set_training_seed(evaluation_seed)
                clean_metrics = evaluate_prototypes_nk(
                    teacher=teacher,
                    val_loader=clean_loader,
                    prototypes_nk=prototypes_nk,
                    proto_temp=proto_temp,
                    device=device,
                    view='clean',
                    repeats=1,
                    return_metrics=True,
                )
                set_training_seed(evaluation_seed)
                strong_metrics = evaluate_prototypes_nk(
                    teacher=teacher,
                    val_loader=strong_loader,
                    prototypes_nk=prototypes_nk,
                    proto_temp=proto_temp,
                    device=device,
                    view='strong',
                    repeats=strong_repeats,
                    return_metrics=True,
                )
                row = {
                    'tau_w': tau_w,
                    'proto_temp': proto_temp,
                    'min_reliable_count': min(counts),
                    'mean_reliable_count': sum(counts) / len(counts),
                    'clean_acc': clean_metrics['proto_acc'],
                    'strong_acc': strong_metrics['proto_acc'],
                    'teacher_strong_acc': strong_metrics['teacher_acc'],
                    'rescue_rate': strong_metrics['rescue_rate'],
                    'harm_rate': strong_metrics['harm_rate'],
                    'elapsed_seconds': time.time() - start_time,
                }
                rows.append(row)
                with open(csv_path, 'a', newline='', encoding='utf-8') as file:
                    csv.DictWriter(file, fieldnames=fieldnames).writerow(row)

            del prototypes_nk
            torch.cuda.empty_cache()

        selected = select_proto_scan_candidate(rows)
        settings = {
            'K': K,
            'num_classes': num_classes,
            'seed': seed,
            'split_seed': split_seed,
            'val_per_class': val_per_class,
            'strong_repeats': strong_repeats,
        }
        write_proto_scan_summary(markdown_path, rows, settings, selected)
        print(log_msg("=" * 72, "INFO"))
        print(log_msg(
            "Prototype parameter scan recommendation: "
            f"PROTO_TAU_W={selected['tau_w']:.2f}, "
            f"PROTO_TEMP={selected['proto_temp']:.2f}, K={K}",
            "INFO",
        ))
        print(log_msg(f"Prototype parameter scan Markdown: {markdown_path}", "INFO"))
        return selected
    finally:
        cudnn.benchmark = previous_benchmark
        cudnn.deterministic = previous_deterministic


# ==============================================================================
# 6. Complete CRLD-Proto: reliability-gated prototype probability fusion
# ==============================================================================
def positive_gain_proto_fusion(
    prob_t_s,
    prob_proto,
    weak_pred,
    uncertainty,
    fusion_scale=0.5,
    fusion_mode="gated",
    use_uncertainty=True,
    use_agreement=True,
    use_positive_gain=True,
    use_proto_confidence=True,
):
    """Fuse prototype evidence using the selected controlled-ablation gates."""
    if fusion_mode not in {"fixed", "gated"}:
        raise ValueError(f"Unsupported prototype fusion mode: {fusion_mode}")

    weak_index = weak_pred.view(-1, 1)
    proto_pred = prob_proto.argmax(dim=1)
    agreement = proto_pred.eq(weak_pred).float().view(-1, 1)

    teacher_anchor_prob = prob_t_s.gather(1, weak_index)
    proto_anchor_prob = prob_proto.gather(1, weak_index)
    positive_gain = (
        (proto_anchor_prob - teacher_anchor_prob).clamp_min(0.0)
        / (proto_anchor_prob + 1e-8)
    ).clamp_(0.0, 1.0)

    num_classes = prob_proto.size(1)
    proto_entropy = -torch.sum(
        prob_proto * torch.log(prob_proto + 1e-8), dim=1, keepdim=True
    )
    proto_confidence = (
        1.0 - proto_entropy / math.log(num_classes)
    ).clamp_(0.0, 1.0)

    if fusion_mode == "fixed":
        lambda_eff = torch.full_like(
            uncertainty.view(-1, 1), fusion_scale
        )
    else:
        lambda_eff = torch.full_like(
            uncertainty.view(-1, 1), fusion_scale
        )
        if use_uncertainty:
            lambda_eff = lambda_eff * uncertainty.view(-1, 1)
        if use_agreement:
            lambda_eff = lambda_eff * agreement
        if use_positive_gain:
            lambda_eff = lambda_eff * positive_gain
        if use_proto_confidence:
            lambda_eff = lambda_eff * proto_confidence
        lambda_eff = lambda_eff.clamp_(0.0, 1.0)

    fused = (
        (1.0 - lambda_eff) * prob_t_s
        + lambda_eff * prob_proto
    )
    return fused


class CRLD_Proto(Distiller):
    """Complete CRLD with prototype-corrected teacher strong-view targets."""

    def __init__(self, student, teacher, cfg, precomputed_prototypes=None):
        super(CRLD_Proto, self).__init__(student, teacher)

        # Keep the original CRLD values as defaults, while allowing controlled
        # command-line overrides for architecture-specific ablations.
        self.temperature = float(getattr(cfg.CRLD, 'TEMPERATURE', 4.0))
        self.ce_loss_weight = float(getattr(cfg.CRLD, 'CE_WEIGHT', 1.0))
        self.wv_loss_weight = float(getattr(cfg.CRLD, 'WV_WEIGHT', 2.0))
        self.cv_loss_weight = float(getattr(cfg.CRLD, 'CV_WEIGHT', 0.5))
        # Keep the default unchanged, but allow controlled architecture-specific
        # ablations to reduce the influence of K-Means-only prototypes.
        self.proto_fusion_scale = float(
            getattr(cfg.CRLD, 'PROTO_FUSION_SCALE', 0.5)
        )
        if not 0.0 < self.proto_fusion_scale <= 1.0:
            raise ValueError("PROTO_FUSION_SCALE must be in (0, 1].")
        self.tau_w = float(getattr(cfg.CRLD, 'TAU_W', 0.9))
        self.tau_s = float(getattr(cfg.CRLD, 'TAU_S', 0.2))

        if self.temperature <= 0.0:
            raise ValueError("CRLD temperature must be positive.")
        if min(self.ce_loss_weight, self.wv_loss_weight, self.cv_loss_weight) < 0.0:
            raise ValueError("CRLD loss weights must be non-negative.")

        self.proto_temp = float(getattr(cfg.CRLD, 'PROTO_TEMP', 0.5))
        if self.proto_temp <= 0.0:
            raise ValueError("PROTO_TEMP must be positive.")

        self.proto_enabled = bool(getattr(cfg.CRLD, 'PROTO_ENABLED', True))
        self.proto_fusion_mode = str(
            getattr(cfg.CRLD, 'PROTO_FUSION_MODE', 'gated')
        ).lower()
        self.proto_gate_uncertainty = bool(
            getattr(cfg.CRLD, 'PROTO_GATE_UNCERTAINTY', True)
        )
        self.proto_gate_agreement = bool(
            getattr(cfg.CRLD, 'PROTO_GATE_AGREEMENT', True)
        )
        self.proto_gate_positive_gain = bool(
            getattr(cfg.CRLD, 'PROTO_GATE_POSITIVE_GAIN', True)
        )
        self.proto_gate_confidence = bool(
            getattr(cfg.CRLD, 'PROTO_GATE_CONFIDENCE', True)
        )
        if self.proto_fusion_mode not in {'fixed', 'gated'}:
            raise ValueError(
                "PROTO_FUSION_MODE must be either 'fixed' or 'gated'."
            )

        self.student.cuda()
        self.teacher.cuda()

        if self.proto_enabled:
            if precomputed_prototypes is None:
                raise ValueError(
                    "Prototype probability fusion requires precomputed prototypes."
                )
            self.max_entropy = math.log(precomputed_prototypes.size(0))
            self.register_buffer("prototypes", precomputed_prototypes.clone())
            self.prototypes = F.normalize(self.prototypes, dim=-1)
        elif precomputed_prototypes is not None:
            raise ValueError(
                "PROTO_ENABLED is false, but precomputed prototypes were supplied."
            )

    def get_learnable_parameters(self):
        return list(self.student.parameters())

    def get_extra_parameters(self):
        return 0

    def _prototype_logits(self, features):
        features = F.normalize(features, dim=1)
        num_classes, k_num, feat_dim = self.prototypes.shape
        proto_flat = self.prototypes.view(num_classes * k_num, feat_dim)
        sim = torch.matmul(features, proto_flat.t()).view(-1, num_classes, k_num)
        scaled_sim = sim / self.proto_temp
        return torch.logsumexp(scaled_sim, dim=2) - math.log(k_num)

    def _soft_target_kd(self, logits_student, target_prob):
        log_prob = F.log_softmax(logits_student / self.temperature, dim=1)
        return F.kl_div(log_prob, target_prob, reduction='none').sum(1) * (self.temperature ** 2)

    def forward_train(self, image, target, epoch=0, **kwargs):
        image_weak, image_strong = image

        # Teacher inference and entropy-based strong-view unreliability.
        with torch.no_grad():
            teacher_out_w = self.teacher(image_weak)
            logits_t_w, _ = split_logits_and_pooled_feature(teacher_out_w)

            teacher_out_s = self.teacher(image_strong)
            logits_t_s, f_s_t = split_logits_and_pooled_feature(teacher_out_s)
            if self.proto_enabled:
                if f_s_t is None:
                    raise RuntimeError(
                        "Teacher output does not contain feats['pooled_feat']."
                    )
                f_s_t = f_s_t.detach().flatten(1)

            probs_t_w = F.softmax(logits_t_w.detach(), dim=1)
            probs_t_s = F.softmax(logits_t_s.detach(), dim=1)
            conf_w = probs_t_w.max(dim=1).values
            conf_s = probs_t_s.max(dim=1).values
            uncertainty_t_s = None
            if self.proto_enabled:
                entropy_t_s = -torch.sum(
                    probs_t_s * torch.log(probs_t_s + 1e-8), dim=1
                )
                uncertainty_t_s = entropy_t_s / self.max_entropy

        mask_w = conf_w.ge(self.tau_w).float()
        mask_s = conf_s.ge(self.tau_s).float()

        logits_s_w = self.student(image_weak)
        if isinstance(logits_s_w, (list, tuple)):
            logits_s_w = logits_s_w[0]
        logits_s_s = self.student(image_strong)
        if isinstance(logits_s_s, (list, tuple)):
            logits_s_s = logits_s_s[0]

        with torch.no_grad():
            prob_t_w_soft = F.softmax(logits_t_w.detach() / self.temperature, dim=1)
            prob_t_s_soft = F.softmax(logits_t_s.detach() / self.temperature, dim=1)
            prob_t_s_fused = prob_t_s_soft
            if self.proto_enabled:
                prob_sim = F.softmax(self._prototype_logits(f_s_t), dim=1)
                weak_teacher_pred = probs_t_w.argmax(dim=1)
                prob_t_s_fused = positive_gain_proto_fusion(
                    prob_t_s_soft,
                    prob_sim,
                    weak_teacher_pred,
                    uncertainty_t_s,
                    fusion_scale=self.proto_fusion_scale,
                    fusion_mode=self.proto_fusion_mode,
                    use_uncertainty=self.proto_gate_uncertainty,
                    use_agreement=self.proto_gate_agreement,
                    use_positive_gain=self.proto_gate_positive_gain,
                    use_proto_confidence=self.proto_gate_confidence,
                )
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_s_w, target) + F.cross_entropy(logits_s_s, target))

        kd_sw_to_tw = self._soft_target_kd(logits_s_w, prob_t_w_soft)
        kd_ss_to_fused = self._soft_target_kd(logits_s_s, prob_t_s_fused)
        loss_kd_wv = self.wv_loss_weight * ((kd_sw_to_tw + kd_ss_to_fused) * mask_w).mean()

        # Preserve the original CRLD cross-view term. The raw strong-view
        # teacher target is retained here; prototype fusion only replaces the
        # strong-view same-view supervision target.
        kd_ss_to_tw = self._soft_target_kd(logits_s_s, prob_t_w_soft)
        kd_sw_to_ts = self._soft_target_kd(logits_s_w, prob_t_s_soft)
        loss_kd_cv = self.cv_loss_weight * (
            (kd_ss_to_tw + kd_sw_to_ts) * mask_s
        ).mean()

        return logits_s_w, {
            "loss_ce": loss_ce,
            "loss_kd_wv": loss_kd_wv,
            "loss_kd_cv": loss_kd_cv,
        }

    def forward(self, *args, **kwargs):
        if 'image_w' in kwargs and 'image_s' in kwargs and 'target' in kwargs:
            image = (kwargs['image_w'], kwargs['image_s'])
            target = kwargs['target']
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['image_w', 'image_s', 'target']}
            return self.forward_train(image, target, **filtered_kwargs)
        elif 'image' in kwargs:
            with torch.no_grad():
                logits = self.student(kwargs['image'])
                return logits[0] if isinstance(logits, (list, tuple)) else logits
        raise ValueError("Missing required inputs.")

distiller_dict["CRLD_Proto"] = CRLD_Proto


# ==============================================================================
# 7. Main entry
# ==============================================================================
def parse_positive_float_csv(value):
    try:
        parsed = [float(item.strip()) for item in value.split(',') if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated floating-point values, received: {value}"
        ) from error
    if not parsed or any(item <= 0.0 for item in parsed):
        raise argparse.ArgumentTypeError("All comma-separated values must be positive.")
    return list(dict.fromkeys(parsed))


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_proto_method_cfg(config):
    """Return the method-specific prototype configuration node."""
    if config.DISTILLER.TYPE == "MPKD":
        return config.MPKD
    if config.DISTILLER.TYPE == "CRLD_Proto":
        return config.CRLD
    raise ValueError(
        "tools/train_proto.py supports CRLD_Proto and MPKD; "
        "use tools/train.py for the original single-view KD/DKD/NKD runs."
    )


def append_method_metadata(
    experiment_name,
    cfg,
    training_seed,
    method_name,
    baseline_name,
    ablation_mode=None,
):
    """Append explicit method provenance before a prototype run."""
    log_dir = os.path.join(cfg.LOG.PREFIX, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    worklog_path = os.path.join(log_dir, "worklog.txt")
    method_cfg = get_proto_method_cfg(cfg)
    proto_enabled = bool(getattr(method_cfg, "PROTO_ENABLED", True))
    if cfg.DISTILLER.TYPE == "MPKD":
        relationship = (
            "MPKD keeps the selected base method's category-level loss and "
            "uses prototypes only for the teacher strong-view same-view target."
        )
    else:
        relationship = (
            "CRLD-Proto preserves complete CRLD and changes only the teacher "
            "strong-view same-view target through gated prototype correction."
        )
    if not proto_enabled:
        relationship = (
            "the selected two-view distillation method without prototype "
            "correction"
        )
    if ablation_mode is not None:
        ablation_description = ABLATION_MODE_SPECS[ablation_mode]["description"]
    elif cfg.DISTILLER.TYPE == "MPKD":
        ablation_description = "Formal MPKD compatibility run."
    else:
        ablation_description = "Formal complete CRLD-Proto run."
    with open(worklog_path, "a", encoding="utf-8") as writer:
        writer.writelines([
            "\n" + "=" * 48 + "\n",
            f"method: {method_name}\n",
            f"baseline: {baseline_name}\n",
            f"relationship: {relationship.rstrip('.')}.\n",
            f"ablation_mode: {ablation_mode or 'none'}\n",
            f"ablation_description: {ablation_description}\n",
            f"prototype_enabled: {proto_enabled}\n",
            f"prototype_fusion_mode: {getattr(method_cfg, 'PROTO_FUSION_MODE', 'gated')}\n",
            f"prototype_fusion_scale: {float(getattr(method_cfg, 'PROTO_FUSION_SCALE', 0.5)):.3f}\n",
            "prototype_gates: "
            f"uncertainty={bool(getattr(method_cfg, 'PROTO_GATE_UNCERTAINTY', True))}, "
            f"agreement={bool(getattr(method_cfg, 'PROTO_GATE_AGREEMENT', True))}, "
            f"positive_gain={bool(getattr(method_cfg, 'PROTO_GATE_POSITIVE_GAIN', True))}, "
            f"confidence={bool(getattr(method_cfg, 'PROTO_GATE_CONFIDENCE', True))}\n",
            f"base_method: {getattr(method_cfg, 'BASE_METHOD', 'CRLD')}\n",
            f"seed: {training_seed}\n",
            f"started_at: {datetime.now().isoformat(timespec='seconds')}\n",
            "=" * 48 + "\n",
        ])


def get_model_registry(dataset_type):
    if dataset_type == "imagenet":
        return imagenet_model_dict
    if dataset_type.startswith("tinyimagenet200"):
        return tinyimagenet200_model_dict
    if dataset_type == "cifar100":
        return cifar_model_dict
    raise ValueError(
        f"Unsupported dataset type for {METHOD_DISPLAY_NAME}: {dataset_type}"
    )


def load_state_dict_from_checkpoint(path):
    if not path:
        raise FileNotFoundError("No teacher checkpoint path was configured.")
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(_project_root(), path)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {path}. "
            "Run tools/prepare_experiments.py or provide the checkpoint explicitly."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }


def build_models_for_dataset(cfg, num_classes):
    dataset_type = cfg.DATASET.TYPE
    registry = get_model_registry(dataset_type)
    teacher_name = cfg.DISTILLER.TEACHER
    student_name = cfg.DISTILLER.STUDENT

    if teacher_name not in registry:
        raise KeyError(f"Unknown {dataset_type} teacher model: {teacher_name}")
    if student_name not in registry:
        raise KeyError(f"Unknown {dataset_type} student model: {student_name}")

    if dataset_type == "imagenet":
        teacher = registry[teacher_name](pretrained=False)
        checkpoint = getattr(cfg.DISTILLER, "TEACHER_CHECKPOINT", "")
        if not checkpoint:
            checkpoint = os.path.join(
                _project_root(),
                "download_ckpts",
                "imagenet_teachers",
                {
                    "ResNet34": os.path.join(
                        "ResNet34_vanilla", "ckpt_epoch_90.pth"
                    ),
                    "ResNet50": os.path.join(
                        "ResNet50_vanilla", "ckpt_epoch_90.pth"
                    ),
                }.get(teacher_name, ""),
            )
        teacher.load_state_dict(load_state_dict_from_checkpoint(checkpoint), strict=True)
        student = registry[student_name](pretrained=False)
        return teacher, student, checkpoint

    teacher_factory, checkpoint = registry[teacher_name]
    configured_checkpoint = getattr(cfg.DISTILLER, "TEACHER_CHECKPOINT", "")
    if configured_checkpoint:
        checkpoint = configured_checkpoint
    teacher = teacher_factory(num_classes=num_classes)
    teacher.load_state_dict(load_state_dict_from_checkpoint(checkpoint), strict=True)
    student_factory = registry[student_name][0]
    student = student_factory(num_classes=num_classes)
    return teacher, student, checkpoint


def input_size_for_dataset(dataset_type):
    if dataset_type == "imagenet":
        return 224
    if dataset_type.startswith("tinyimagenet200_384"):
        return 384
    if dataset_type.startswith("tinyimagenet200"):
        return 64
    return 32


def main(cfg, resume, proto_scan=False, proto_scan_temps=None,
         proto_scan_taus=None, proto_scan_seed=42, proto_scan_split_seed=2026,
         proto_scan_val_per_class=20, proto_scan_strong_repeats=1,
         training_seed=DEFAULT_TRAINING_SEED,
         ablation_mode=None):
    if cfg.DISTILLER.TYPE not in PROTO_DISTILLER_TYPES:
        raise ValueError(
            f"Unsupported distiller type {cfg.DISTILLER.TYPE!r} for "
            "tools/train_proto.py. Use CRLD_Proto or MPKD here; use "
            "tools/train.py for the original KD/DKD/NKD implementations."
        )
    set_training_seed(training_seed)
    experiment_name = cfg.EXPERIMENT.NAME or cfg.EXPERIMENT.TAG
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    method_name = cfg.EXPERIMENT.METHOD or (
        "MPKD" if cfg.DISTILLER.TYPE == "MPKD" else METHOD_DISPLAY_NAME
    )
    baseline_name = cfg.EXPERIMENT.BASELINE or (
        getattr(cfg.MPKD, "BASE_METHOD", "KD")
        if cfg.DISTILLER.TYPE == "MPKD"
        else BASELINE_DISPLAY_NAME
    )
    method_cfg = get_proto_method_cfg(cfg)

    cfg.defrost()
    cfg.LOG.SAVE_CHECKPOINT_FREQ = 9999
    cfg.freeze()

    show_cfg(cfg)
    if ablation_mode is not None:
        print(log_msg(
            f"Ablation mode {ablation_mode}: "
            f"{ABLATION_MODE_SPECS[ablation_mode]['description']}",
            "INFO",
        ))
    if cfg.DISTILLER.TYPE == "MPKD":
        method_description = (
            "MPKD keeps the selected KD/DKD/NKD category-level loss and "
            "applies prototype correction only to the teacher strong-view "
            "same-view target."
        )
    else:
        method_description = (
            "CRLD-Proto preserves complete CRLD and changes only the teacher "
            "strong-view same-view target through gated prototype correction."
        )
    print(log_msg(
        f"Method: {method_name}; baseline: {baseline_name}. " + method_description,
        "INFO",
    ))
    print(log_msg(f"Training seed: {training_seed}", "INFO"))
    train_loader, val_loader, _, num_classes = get_dataset_strong(cfg)

    print(log_msg("Loading models...", "INFO"))
    model_teacher, model_student, pretrain_model_path = build_models_for_dataset(
        cfg, num_classes
    )
    model_teacher = model_teacher.cuda()
    # Feature-dimension probing must not update the pretrained teacher's
    # BatchNorm running statistics before prototype mining and distillation.
    model_teacher.eval()

    proto_enabled = bool(getattr(method_cfg, 'PROTO_ENABLED', True))
    feat_dim = None
    if proto_enabled or proto_scan:
        dummy_size = input_size_for_dataset(cfg.DATASET.TYPE)
        with torch.no_grad():
            hook = register_avgpool_hook(model_teacher)
            probe_output = model_teacher(
                torch.zeros(1, 3, dummy_size, dummy_size).cuda()
            )
            _, probe_feature = extract_teacher_logits_and_feature(
                probe_output, hook
            )
            feat_dim = probe_feature.shape[1]
            if hook is not None:
                hook.close()

    tau_w = getattr(method_cfg, 'PROTO_TAU_W', 0.80)
    proto_temp = getattr(method_cfg, 'PROTO_TEMP', 0.5)
    K_subcenters = int(getattr(method_cfg, 'PROTO_K', 3))
    max_proto_samples = int(getattr(method_cfg, 'PROTO_MAX_SAMPLES_PER_CLASS', 0))
    max_proto_samples = max_proto_samples if max_proto_samples > 0 else None
    storage_dtype = (
        torch.float16 if cfg.DATASET.TYPE == 'imagenet' else torch.float32
    )
    if K_subcenters < 1:
        raise ValueError("PROTO_K must be at least 1.")

    if proto_scan:
        if not proto_enabled:
            raise ValueError("Prototype scanning requires PROTO_ENABLED=True.")
        print(log_msg(
            "Fast prototype parameter scan enabled. Student distillation will not start.",
            "INFO",
        ))
        run_proto_parameter_scan(
            teacher=model_teacher,
            train_loader=train_loader,
            num_classes=num_classes,
            feature_dim=feat_dim,
            K=K_subcenters,
            proto_temps=proto_scan_temps,
            tau_values=proto_scan_taus,
            seed=proto_scan_seed,
            split_seed=proto_scan_split_seed,
            val_per_class=proto_scan_val_per_class,
            strong_repeats=proto_scan_strong_repeats,
            dataset_type=cfg.DATASET.TYPE,
            experiment_name=experiment_name,
            device='cuda',
        )
        print(log_msg("Fast prototype parameter scan finished. Exiting before training.", "INFO"))
        return

    model_student = model_student.cuda()

    prototypes_nk = None
    if proto_enabled:
        training_rng_state = capture_rng_state()
        with torch.no_grad():
            prototypes_nk = precompute_multicenter_prototypes(
                teacher=model_teacher,
                train_loader=train_loader,
                num_classes=num_classes,
                K=K_subcenters,
                device='cuda',
                feature_dim=feat_dim,
                tau_w=tau_w,
                max_features_per_class=max_proto_samples,
                storage_dtype=storage_dtype,
            )

            evaluate_prototypes_nk(
                model_teacher,
                val_loader,
                prototypes_nk,
                proto_temp=proto_temp,
                device='cuda',
            )

        restore_rng_state(training_rng_state)
        if cfg.DISTILLER.TYPE == "MPKD":
            prototype_message = (
                "Selected KD/DKD/NKD loss retained; prototype correction changes "
                "only the teacher strong-view same-view target."
            )
        else:
            prototype_message = (
                "Complete CRLD structure retained; prototype correction changes "
                "only the teacher strong-view same-view target."
            )
        print(log_msg(
            prototype_message + " "
            f"Prototype construction=K-Means, K={K_subcenters}, "
            f"tau_p={proto_temp}, seed={training_seed}.",
            "INFO",
        ))
    else:
        print(log_msg(
            "Ablation A: prototype construction and correction are disabled; "
            "all complete CRLD losses remain unchanged.",
            "INFO",
        ))
    distiller = distiller_dict[cfg.DISTILLER.TYPE](
        model_student,
        model_teacher,
        cfg,
        precomputed_prototypes=prototypes_nk,
    )
    distiller = torch.nn.DataParallel(distiller.cuda())

    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    append_method_metadata(
        experiment_name,
        cfg,
        training_seed,
        method_name,
        baseline_name,
        ablation_mode=ablation_mode,
    )

    print(log_msg("Training started...", "INFO"))
    start_time = time.time()

    trainer.train(resume=resume)

    end_time = time.time()
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    print(log_msg(f"Total time: {hours}h {minutes}m {seconds}s", "INFO"))

    # Keep only the final student checkpoint and worklog under the actual
    # output prefix. The experiment name itself is relative to that prefix.
    cleanup_useless_checkpoints(os.path.join(cfg.LOG.PREFIX, experiment_name))


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser("CRLD-Proto/MPKD training")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument(
        "--ablation-mode",
        choices=list(ABLATION_MODE_SPECS),
        default=None,
        help=(
            "Run one isolated A-E or gate ablation. The output tag is changed "
            "automatically so formal results are not overwritten."
        ),
    )
    parser.add_argument(
        "--proto-scan",
        action="store_true",
        help="Quickly scan prototype temperature and reliability threshold; do not train a student.",
    )
    parser.add_argument(
        "--proto-scan-temps",
        type=parse_positive_float_csv,
        default=[0.35, 0.50, 0.65],
        metavar="T1,T2,...",
        help="Prototype temperatures used by --proto-scan.",
    )
    parser.add_argument(
        "--proto-scan-taus",
        type=parse_positive_float_csv,
        default=[0.75, 0.80, 0.85],
        metavar="W1,W2,...",
        help="Reliable-feature thresholds used by --proto-scan.",
    )
    parser.add_argument("--proto-scan-seed", type=int, default=42)
    parser.add_argument("--proto-scan-split-seed", type=int, default=2026)
    parser.add_argument("--proto-scan-val-per-class", type=int, default=20)
    parser.add_argument("--proto-scan-strong-repeats", type=int, default=1)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.proto_scan and args.resume:
        parser.error("--resume cannot be combined with a prototype-only diagnostic mode.")
    if args.proto_scan and args.ablation_mode is not None:
        parser.error("--proto-scan cannot be combined with --ablation-mode.")
    if args.proto_scan_val_per_class < 1:
        parser.error("--proto-scan-val-per-class must be at least 1.")
    if args.proto_scan_strong_repeats < 1:
        parser.error("--proto-scan-strong-repeats must be at least 1.")
    if any(tau > 1.0 for tau in args.proto_scan_taus):
        parser.error("--proto-scan-taus values must be in (0, 1].")

    cfg.defrost()
    cfg.set_new_allowed(True)
    if args.cfg:
        with open(args.cfg, 'r', encoding='utf-8') as f:
            cfg.merge_from_other_cfg(cfg.load_cfg(f))
    cfg.merge_from_list(args.opts)
    apply_ablation_mode(cfg, args.ablation_mode)
    validate_ablation_context(cfg, args.ablation_mode)
    cfg.freeze()
    main(
        cfg,
        args.resume,
        proto_scan=args.proto_scan,
        proto_scan_temps=args.proto_scan_temps,
        proto_scan_taus=args.proto_scan_taus,
        proto_scan_seed=args.proto_scan_seed,
        proto_scan_split_seed=args.proto_scan_split_seed,
        proto_scan_val_per_class=args.proto_scan_val_per_class,
        proto_scan_strong_repeats=args.proto_scan_strong_repeats,
        training_seed=args.seed,
        ablation_mode=args.ablation_mode,
    )
