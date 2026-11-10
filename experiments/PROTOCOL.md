# 实验协议与文件说明

本文件集中记录实验设置、运行入口、结果口径和资料目录。结果、状态和实验日志统一保存在 `log.md`。

## 1. 方法关系

| 方法 | 定位 | 入口 |
| --- | --- | --- |
| CRLD | 原始论文基线 | `tools/train.py` + CRLD 配置 |
| CRLD-Proto | 本项目主方法 | `tools/train_proto.py` + `CRLD_Proto` |
| MPKD | KD/DKD/NKD 的原型兼容适配器 | `tools/train_proto.py` + `MPKD` |
| KD、DKD、NKD | 原始类别级蒸馏方法 | `tools/train.py` |

CRLD-Proto 保留完整 CRLD 的监督、弱视图同视图和两项交叉视图约束，仅对教师强视图同视图目标执行原型修正。教师和原型在学生训练阶段冻结，原型不参与反向传播。当前正式原型构建为可靠弱视图特征筛选、类内 K-Means 多中心、L2 归一化和固定原型。

## 2. 固定设置

- 数据集：CIFAR-100；跨数据集实验另见 Tiny-ImageNet 和 ImageNet 配置。
- 默认随机种子：42。
- CIFAR-100 主实验：240 epochs，架构相关的学习率和 CRLD 权重读取对应 YAML。
- 原型数量：`PROTO_K=3`；除非做明确的 K 消融，不改变该值。
- 结果依据：每个输出目录中的 `worklog.txt`；只填写实际存在 `best_acc` 的本地结果。
- 文献数值与本地训练数值必须分开标记，未运行实验写为“未跑”。

## 3. 目录和命名规范

### 3.1 配置

配置按数据集和方法分组：

```text
configs/cifar100/crld/        原始 CRLD
configs/cifar100/crld_proto/  CRLD-Proto
configs/cifar100/mpkd/        MPKD 兼容实验
configs/tinyimagenet200/      Tiny-ImageNet-200
configs/imagenet/              ImageNet
```

架构文件使用 `teacher_student.yaml` 命名，例如 `res32x4_res8x4.yaml`、`res56_res20.yaml`。

### 3.2 工具

| 文件 | 用途 |
| --- | --- |
| `tools/train.py` | 原始蒸馏训练 |
| `tools/train_proto.py` | CRLD-Proto 和 MPKD 训练 |
| `tools/run_cifar100_architectures.py` | CIFAR-100 多架构批量运行 |
| `tools/run_cifar100_ablations.py` | A-E 与门控消融 |
| `tools/run_mpkd_compatibility.py` | KD/DKD/NKD 与 MPKD 矩阵 |
| `tools/prepare_experiments.py` | 数据、权重和资源清单检查 |
| `tools/eval.py` | checkpoint 评估 |
| `tools/evaluate_robustness.py` | 强视图鲁棒性评估和制图 |
| `tools/export_paper_docx.py` | Markdown 论文导出 Word |
| `tools/train_crld.py`、`tools/train_mlld.py` | 历史专用训练入口，保留兼容性 |

根目录不再放置批量实验脚本。旧的中文文件名入口已迁移为 `tools/run_cifar100_architectures.py`。

### 3.3 输出

训练输出统一使用：

```text
output/<project>/<method>,<teacher>,<student>[,<tag>]/
```

`worklog.txt` 是原始指标，`student_best` 和 `student_latest` 是模型权重，`train.events/` 是 TensorBoard 日志。一次性冒烟和已废弃调参结果放入 `output/_archive/`，不混入正式结果表。

## 4. CIFAR-100 消融

默认配置为 `configs/cifar100/crld_proto/res32x4_res8x4.yaml`、seed 42、240 epochs、K=3。

### 4.1 A-E

| 模式 | 说明 |
| --- | --- |
| A | 完整 CRLD，无原型修正 |
| B | A + 可靠弱视图多中心 K-Means 原型 + 固定融合 |
| C | B + 强视图不确定度权重 |
| D | C + 类别一致性和锚定类别正向增益 |
| E | D + 原型分布置信度，作为完整门控版本 |

运行：

```powershell
python tools/run_cifar100_ablations.py --phase ae --dry-run
python tools/run_cifar100_ablations.py --phase ae
```

### 4.2 门控消融

以 E 为完整参考，每次只关闭一个门控因子：

```powershell
python tools/run_cifar100_ablations.py --phase gates --dry-run
python tools/run_cifar100_ablations.py --phase gates
```

运行器会读取独立 worklog，直接更新 `log.md` 的消融表。

## 5. MPKD 兼容矩阵

兼容矩阵包含六组 CIFAR-100 架构：

1. ResNet56 -> ResNet20
2. ResNet110 -> ResNet32
3. ResNet32x4 -> ResNet8x4
4. WRN40-2 -> WRN16-2
5. WRN40-2 -> WRN40-1
6. VGG13 -> VGG8

`MPKD.BASE_METHOD` 可选 `KD`、`DKD`、`NKD`。适配器保留所选方法的类别级损失，只在统一双视图框架中修正教师强视图同视图目标；不能把两个完整总损失直接相加。

```powershell
python tools/run_mpkd_compatibility.py --dry-run
python tools/run_mpkd_compatibility.py --architectures res32x4_res8x4 --methods KD --epochs 1
python tools/run_mpkd_compatibility.py --seed 42
```

新增组合没有本地 worklog 前，结果表必须写“未跑”，不能用其他论文数值代填。

正式运行前至少检查：

1. 一个 batch 的前向和反向能够完成；
2. 学生参数有梯度，教师参数和原型没有梯度；
3. 关闭原型时，基础方法的损失与原实现一致；
4. 1 epoch 冒烟能够写出日志和 `best_acc`；
5. 正式结果记录 seed、最佳 epoch、Top-1、Top-5 和结果来源。

若只验证基础方法在双视图框架中的行为，可关闭原型修正：

```powershell
python tools/train_proto.py --cfg configs/cifar100/mpkd/res32x4_res8x4.yaml --seed 42 EXPERIMENT.TAG smoke,dkd,res32x4,res8x4 MPKD.BASE_METHOD DKD MPKD.PROTO_ENABLED false
```

## 6. 跨数据集运行

### Tiny-ImageNet-200

数据目录为 `data/tiny-imagenet-200`，教师权重应与 200 类输出匹配。先使用 1 epoch 检查数据、特征提取、原型构建、前向、反向和验证，再恢复正式 epoch：

```powershell
python tools/train_proto.py --cfg configs/tinyimagenet200/crld_proto/res32x4_res8x4.yaml --seed 42
```

### ImageNet

ImageNet 只使用用户合法获取并整理好的 ImageFolder 目录：

```text
data/imagenet/train/<class>/*.JPEG
data/imagenet/val/<class>/*.JPEG
```

运行示例：

```powershell
python tools/train_proto.py --cfg configs/imagenet/crld_proto/r34_r18.yaml --seed 42
```

ImageNet 原始数据不由本项目自动下载；教师权重必须与配置中的架构和类别数一致。

## 7. 结果记录规则

每次正式实验至少记录：

- 教师和学生架构；
- 配置文件路径；
- 方法、seed 和 epoch；
- Best Top-1、Best Top-5、最佳 epoch；
- 输出目录和教师权重来源；
- 结果属于文献、历史审计还是本地复现。

不要把 1 epoch 冒烟结果写入正式主结果，不要用“最终 epoch”替代“Best Acc”，也不要把不同架构的参数覆盖写入同一个 YAML。

## 8. 资源检查

```powershell
python tools/prepare_experiments.py
```

检查结果写入 `resources_manifest.json`。数据集、checkpoint、训练输出和 TensorBoard 文件均被 `.gitignore` 忽略。
