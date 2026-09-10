"""按顺序运行未完成的无 GLVQ 对照实验，并记录成功结果。"""

import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "experiments" / "log.md"
DEFAULT_TRAINING_SEED = 42

GLVQ_MODE = "E_no_glvq"
GLVQ_TAG = "ablation_e_no_glvq"
GLVQ_LOG_MARKER = "<!-- GLVQ_RESULTS_AUTO -->"

# 已完成的两组无 GLVQ 对照不重复运行；这里保留其余十组架构。
EXPERIMENTS = [
    {
        "name": "ResNet110 -> ResNet32",
        "config": "configs/cifar100/crld_proto/res110_res32.yaml",
        "formal_best": 74.99,
        "baseline": 74.42,
    },
    {
        "name": "WRN40-2 -> WRN16-2",
        "config": "configs/cifar100/crld_proto/wrn40_2_wrn16_2.yaml",
        "formal_best": 76.97,
        "baseline": 76.45,
    },
    {
        "name": "WRN40-2 -> WRN40-1",
        "config": "configs/cifar100/crld_proto/wrn40_2_wrn40_1.yaml",
        "formal_best": 75.86,
        "baseline": 75.58,
    },
    {
        "name": "VGG13 -> VGG8",
        "config": "configs/cifar100/crld_proto/vgg13_vgg8.yaml",
        "formal_best": 75.57,
        "baseline": 75.27,
    },
    {
        "name": "ResNet32x4 -> ShuffleNetV2",
        "config": "configs/cifar100/crld_proto/res32x4_shuv2.yaml",
        "formal_best": 78.96,
        "baseline": 78.27,
    },
    {
        "name": "ResNet32x4 -> WRN16-2",
        "config": "configs/cifar100/crld_proto/res32x4_wrn_16_2.yaml",
        "formal_best": 77.27,
        "baseline": 76.92,
    },
    {
        "name": "ResNet50 -> MobileNetV2",
        "config": "configs/cifar100/crld_proto/res50_mv2.yaml",
        "formal_best": 71.68,
        "baseline": 71.36,
    },
    {
        "name": "VGG13 -> MobileNetV2",
        "config": "configs/cifar100/crld_proto/vgg13_mv2.yaml",
        "formal_best": 71.00,
        "baseline": 70.39,
    },
    {
        "name": "WRN40-2 -> MobileNetV2",
        "config": "configs/cifar100/crld_proto/wrn40_2_mv2.yaml",
        "formal_best": 71.00,
        "baseline": 70.37,
    },
    {
        "name": "WRN40-2 -> ResNet8x4",
        "config": "configs/cifar100/crld_proto/wrn40_2_res8x4.yaml",
        "formal_best": 77.53,
        "baseline": 77.28,
    },
]


def load_experiment_config(config_path):
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def effective_experiment_name(config, mode=None):
    experiment = config["EXPERIMENT"]
    field = "NAME" if experiment.get("NAME") else "TAG"
    name = experiment[field]
    if mode is not None and not name.endswith("," + GLVQ_TAG):
        name = f"{name},{GLVQ_TAG}"
    return name


def worklog_path_for(config, mode=None):
    experiment = config["EXPERIMENT"]
    return (
        ROOT
        / "output"
        / experiment["PROJECT"]
        / effective_experiment_name(config, mode)
        / "worklog.txt"
    )


def read_new_best_acc(worklog_path, start_offset):
    if not worklog_path.exists():
        return None

    file_size = worklog_path.stat().st_size
    if file_size < start_offset:
        start_offset = 0

    with worklog_path.open("rb") as file:
        file.seek(start_offset)
        appended_text = file.read().decode("utf-8", errors="ignore")

    matches = re.findall(r"best_acc\s+([0-9]+(?:\.[0-9]+)?)", appended_text)
    return float(matches[-1]) if matches else None


def initialize_log_section():
    log_text = LOG_PATH.read_text(encoding="utf-8")
    if "## 6. 结果汇总" not in log_text:
        raise RuntimeError("log.md 缺少统一结果汇总章节，未启动批量训练。")
    if GLVQ_LOG_MARKER not in log_text:
        raise RuntimeError("log.md 缺少 GLVQ 结果写入标记，未启动批量训练。")


def append_glvq_result(experiment, config, best_acc, training_seed):
    output_name = effective_experiment_name(config, GLVQ_MODE)
    log_text = LOG_PATH.read_text(encoding="utf-8")
    if output_name in log_text:
        print(f"{experiment['name']} 已有 GLVQ 对照记录，跳过重复写入。")
        return

    formal_best = experiment["formal_best"]
    baseline = experiment["baseline"]
    delta = best_acc - baseline
    row = (
        f"| {experiment['name']} | {baseline:.2f}% | {formal_best:.2f}% | "
        f"**{best_acc:.2f}%** | — | **{best_acc:.2f}%（{delta:+.2f}）** | "
        "新增 E（scale=0.5） |\n"
    )
    LOG_PATH.write_text(
        log_text.replace(GLVQ_LOG_MARKER, row + GLVQ_LOG_MARKER, 1),
        encoding="utf-8",
    )
def main():
    initialize_log_section()
    print("开始批量运行其余无 GLVQ 对照，结果将写入 log.md 第 6 节...")

    for index, experiment in enumerate(EXPERIMENTS, start=1):
        model_name = experiment["name"]
        relative_config_path = experiment["config"]
        training_seed = DEFAULT_TRAINING_SEED
        config_path = ROOT / relative_config_path
        config = load_experiment_config(config_path)
        worklog_path = worklog_path_for(config, GLVQ_MODE)
        start_offset = worklog_path.stat().st_size if worklog_path.exists() else 0
        command = [
            sys.executable,
            "tools/train_proto.py",
            "--cfg",
            str(relative_config_path),
            "--seed",
            str(training_seed),
            "--ablation-mode",
            GLVQ_MODE,
        ]
        print(f"正在运行第 {index} 个实验：{model_name}")
        print("命令：", " ".join(command))
        try:
            result = subprocess.run(command, cwd=ROOT)
        except KeyboardInterrupt:
            print("训练被手动终止，YAML 未被修改。")
            raise

        if result.returncode != 0:
            print(
                f"第 {index} 个实验失败，退出码：{result.returncode}；"
                "本次结果未写入 log.md。"
            )
            continue

        best_acc = read_new_best_acc(worklog_path, start_offset)
        if best_acc is None:
            print(f"第 {index} 个实验未找到新增 best_acc，本次结果未写入 log.md。")
            continue

        append_glvq_result(experiment, config, best_acc, training_seed)
        print(
            f"第 {index} 个实验完成，Best Acc={best_acc:.2f}%，"
            "已写入统一结果汇总。"
        )

        if index < len(EXPERIMENTS):
            print("暂停 10 秒后运行下一组...")
            time.sleep(10)

    print("批量运行结束。")


if __name__ == "__main__":
    main()
