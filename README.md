# CRLD-Proto / MPKD

本项目基于 `mdistiller`，实现并复现知识蒸馏、跨视图蒸馏以及多中心原型目标修正实验。

## 项目简介

- `CRLD`：原始论文中的 Cross-View Consistency Regularisation 基线。
- `CRLD-Proto`：在完整 CRLD 损失上增加多中心原型修正，只替换教师强视图同视图蒸馏目标。
- `MPKD`：将同一原型修正模块适配到 `KD`、`DKD` 和 `NKD` 的兼容入口。
- 原始 `KD`、`DKD`、`NKD`、`CRLD` 训练入口均保留，互不覆盖。

当前正式原型构建使用可靠弱视图教师特征和类内 K-Means 多中心，不使用 GLVQ。论文草稿和逐轮实验记录位于 `experiments/`。

## 目录结构

```text
configs/                 训练配置
mdistiller/              数据集、模型、蒸馏器和训练引擎
tools/train.py           原始单视图蒸馏训练入口
tools/train_proto.py     CRLD-Proto 与 MPKD 训练入口
tools/run_*.py           批量实验编排脚本
tools/eval.py            单模型评估
tools/prepare_experiments.py  数据和教师权重检查
experiments/             协议、结果、日志和论文材料
data/                    数据集（本地，不提交）
download_ckpts/          教师权重（本地，不提交）
output/                  训练输出（本地，不提交）
```

## 环境准备

```powershell
pip install -r requirements.txt
pip install -e .
python tools/prepare_experiments.py
```

脚本会检查 CIFAR-100、Tiny-ImageNet 和教师权重状态。ImageNet 原始数据受许可限制，需要用户自行准备。目录应为：

```text
data/imagenet/train/<class>/*.JPEG
data/imagenet/val/<class>/*.JPEG
```

资源清单会写入 `experiments/resources_manifest.json`。数据集和权重不会被提交到 Git。

## 常用运行命令

### 原始蒸馏入口

```powershell
# KD、DKD、NKD 或 CRLD 使用对应 YAML
python tools/train.py --cfg configs/cifar100/kd/res32x4_res8x4.yaml
python tools/train.py --cfg configs/cifar100/dkd/res32x4_res8x4.yaml
python tools/train.py --cfg configs/cifar100/nkd/res32x4_res8x4.yaml
python tools/train.py --cfg configs/cifar100/crld/res32x4_res8x4.yaml
```

### CRLD-Proto

```powershell
python tools/train_proto.py `
  --cfg configs/cifar100/crld_proto/res32x4_res8x4.yaml `
  --seed 42
```

其他教师-学生架构只需替换 `--cfg` 路径，例如 `res56_res20.yaml` 或 `vgg13_vgg8.yaml`。

### CIFAR-100 架构批量运行

```powershell
python tools/run_cifar100_architectures.py
```

该脚本保留原批量架构训练功能，正式入口名称已统一为英文；不再使用根目录下的中文脚本名。

### A-E 与门控消融

```powershell
# 先查看命令，不启动训练
python tools/run_cifar100_ablations.py --phase ae --dry-run

# A-E
python tools/run_cifar100_ablations.py --phase ae

# 门控因子消融
python tools/run_cifar100_ablations.py --phase gates
```

### MPKD 兼容实验

```powershell
# 查看 6 个架构 x 3 个方法的命令
python tools/run_mpkd_compatibility.py --dry-run

# 单组 1 epoch 冒烟测试
python tools/run_mpkd_compatibility.py `
  --architectures res32x4_res8x4 `
  --methods KD `
  --epochs 1

# 正式运行兼容矩阵
python tools/run_mpkd_compatibility.py --seed 42
```

### Tiny-ImageNet 与 ImageNet

```powershell
python tools/train_proto.py `
  --cfg configs/tinyimagenet200/crld_proto/res32x4_res8x4.yaml `
  --seed 42

python tools/train_proto.py `
  --cfg configs/imagenet/crld_proto/r34_r18.yaml `
  --seed 42
```

显存不足时，优先在 YAML 中降低 `SOLVER.BATCH_SIZE`，并使用梯度累积；不要改变 `LR_DECAY_STAGES`，除非实验设计明确要求。

### 评估、鲁棒性和论文导出

```powershell
python tools/eval.py --dataset cifar100 --model resnet8x4 --ckpt <checkpoint>
python tools/evaluate_robustness.py --help
python tools/export_paper_docx.py --help
```

## 输出与记录

训练输出默认位于：

```text
output/<project>/<tag>/
```

每个实验目录中的 `worklog.txt` 是准确率和最佳轮次的原始依据。实验资料索引见 `experiments/README.md`，结构化结果见 `experiments/RESULTS.md`，详细历史记录见 `experiments/log.md`，实验协议和文件说明见 `experiments/PROTOCOL.md`。

`output/_archive/` 只存放 1 epoch 冒烟和未采用的调参候选，不参与正式结果汇总。

## 论文材料

- `experiments/MPKD论文草稿.md`：中文论文草稿。
- `experiments/MPKD论文草稿.docx`：Word 版本。
- `experiments/CRLD-Proto写作结构分析.md`：写作结构参考。
- `experiments/robustness/`：鲁棒性分析数据和图表。

## 引用

本项目基于以下工作和代码框架：

```bibtex
@inproceedings{crld,
  author = {Weijia Zhang and Dongnan Liu and Weidong Cai and Chao Ma},
  title = {Cross-View Consistency Regularisation for Knowledge Distillation},
  booktitle = {ACM MM},
  year = {2024}
}
```
