# 实验结果汇总

本文件是结构化结果入口。每个数值都应能在对应的 `worklog.txt` 中核验；“文献结果”和“本地实验”分开标记，未完成实验不填估计值。

## 1. CIFAR-100 消融结果

固定配置：ResNet32x4 -> ResNet8x4、seed 42、240 epochs、K=3，配置文件为 `configs/cifar100/crld_proto/res32x4_res8x4.yaml`。

<!-- ABLATION_RESULTS_START -->
### 1.1 A-E 递进消融

| 模式 | 关键设置 | 输出目录后缀 | Best Acc | Best Epoch | 最终 Acc | 状态 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A | 完整 CRLD，无原型 | ablation_a | 77.29% | 190 | 76.95% | 已完成 |
| B | K-Means 原型 + 固定融合 | ablation_b | 77.72% | 224 | 77.63% | 已完成 |
| C | B + GLVQ（历史审计） | ablation_c | 77.41% | 211 | 77.29% | 已完成 |
| D | C + 不确定度权重（历史审计） | ablation_d | 77.58% | 216 | 77.33% | 已完成 |
| E | 完整 CRLD-Proto | ablation_e | 77.04% | 228 | 76.89% | 已完成 |

### 1.2 门控因子消融

| 模式 | 删除的唯一因子 | 输出目录后缀 | Best Acc | Best Epoch | 最终 Acc | 状态 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| gate_no_uncertainty | 强视图不确定度 | gate_no_uncertainty | 77.40% | 225 | 77.28% | 已完成 |
| gate_no_agreement | 类别一致性 | gate_no_agreement | 77.19% | 236 | 77.07% | 已完成 |
| gate_no_positive_gain | 锚定类正向增益 | gate_no_positive_gain | 77.31% | 236 | 77.00% | 已完成 |
| gate_no_confidence | 原型置信度 | gate_no_confidence | 77.22% | 200 | 77.00% | 已完成 |

### 1.3 无 GLVQ 对照

以下记录只用于判断 GLVQ 的影响，不改变 CRLD-Proto 的方法定义。

| 模式 | 关键设置 | 输出目录后缀 | Best Acc | Best Epoch | 最终 Acc | 状态 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A | 完整 CRLD，无原型 | ablation_a（复用） | 77.29% | 190 | 76.95% | 已完成 |
| B | K-Means 原型 + 固定融合 | ablation_b（复用） | 77.72% | 224 | 77.63% | 已完成 |
| D_no_glvq | K-Means 原型 + 不确定度门控，无 GLVQ | ablation_d_no_glvq | 77.53% | 215 | 77.25% | 已完成 |
| E_no_glvq | K-Means 原型 + 全部门控，无 GLVQ | ablation_e_no_glvq | 77.44% | 228 | 77.24% | 已完成 |
<!-- ABLATION_RESULTS_END -->

消融运行器会更新上面的消融表，并同步 `experiments/log.md` 的自动汇总区。早期含 GLVQ 的行保留为历史审计，当前代码路径不再执行 GLVQ。

## 2. CIFAR-100 主架构结果

下表汇总历史上已记录的完整 CRLD-Proto 结果；基线列是对应架构的 CRLD 参考值，结果来源和逐轮信息以 `experiments/log.md` 为准。

| 教师 -> 学生 | CRLD 基线 | CRLD-Proto Best Acc | 差值 | 配置 |
| --- | ---: | ---: | ---: | --- |
| ResNet56 -> ResNet20 | 72.10% | 72.36% | +0.26 | `configs/cifar100/crld_proto/res56_res20.yaml` |
| ResNet110 -> ResNet32 | 74.42% | 74.99% | +0.57 | `configs/cifar100/crld_proto/res110_res32.yaml` |
| ResNet32x4 -> ResNet8x4 | 77.60% | 77.33% | -0.27 | `configs/cifar100/crld_proto/res32x4_res8x4.yaml` |
| WRN40-2 -> WRN16-2 | 76.45% | 76.97% | +0.52 | `configs/cifar100/crld_proto/wrn40_2_wrn16_2.yaml` |
| WRN40-2 -> WRN40-1 | 75.58% | 75.86% | +0.28 | `configs/cifar100/crld_proto/wrn40_2_wrn40_1.yaml` |
| VGG13 -> VGG8 | 75.27% | 75.57% | +0.30 | `configs/cifar100/crld_proto/vgg13_vgg8.yaml` |
| ResNet32x4 -> ShuffleNetV2 | 78.27% | 78.96% | +0.69 | `configs/cifar100/crld_proto/res32x4_shuv2.yaml` |
| ResNet32x4 -> WRN16-2 | 76.92% | 77.27% | +0.35 | `configs/cifar100/crld_proto/res32x4_wrn_16_2.yaml` |
| ResNet50 -> MobileNetV2 | 71.36% | 71.68% | +0.32 | `configs/cifar100/crld_proto/res50_mv2.yaml` |
| VGG13 -> MobileNetV2 | 70.39% | 71.00% | +0.61 | `configs/cifar100/crld_proto/vgg13_mv2.yaml` |
| WRN40-2 -> MobileNetV2 | 70.37% | 71.00% | +0.63 | `configs/cifar100/crld_proto/wrn40_2_mv2.yaml` |
| WRN40-2 -> ResNet8x4 | 77.28% | 77.53% | +0.25 | `configs/cifar100/crld_proto/wrn40_2_res8x4.yaml` |

这张表只表示已记录的 CRLD-Proto 主结果，不把一次性调参候选或 1 epoch 冒烟结果混入其中。无 GLVQ 的独立结果及其调参记录保留在 `experiments/log.md`。

## 3. MPKD 兼容性矩阵

兼容性实验采用对应蒸馏基线加 MPKD，按架构使用 YAML 配置、seed 42、240 epochs。未完成的运行不作为完整结果；暂停中的准确率仅表示当前检查点之前的最佳验证结果。

| 方法 | R56/R20 | R110/R32 | R32x4/R8x4 | W40-2/W16-2 | W40-2/W40-1 | V13/V8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KD | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 |
| KD + CRLD | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 |
| KD + MPKD | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 |
| DKD | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 |
| DKD + CRLD | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 |
| DKD + MPKD | 71.04%* | 73.56%* | 77.31%* | 76.24%* | 暂停（105/240） | 未跑 |
| NKD | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 |
| NKD + CRLD | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 |
| NKD + MPKD | 72.85%* | 74.66%* | 78.58%* | 暂停（83/240，65.35%） | 未跑 | 未跑 |

配置位于 `configs/cifar100/mpkd/`，运行器为 `tools/run_mpkd_compatibility.py`。

`*` 表示本地完整 240-epoch 兼容性结果，来源为对应架构的 `worklog.txt`。NKD + MPKD 的 WRN40-2/WRN16-2 当前完成 83/240 epochs，65.35% 是中途最佳验证 Top-1，不与完整结果作最终比较；WRN40-2/WRN40-1 与 VGG13/VGG8 尚未开始。

## 4. 跨数据集结果

| 数据集 | 教师 -> 学生 | 配置 | 状态 |
| --- | --- | --- | --- |
| Tiny-ImageNet-200 | ResNet32x4 -> ResNet8x4 | `configs/tinyimagenet200/crld_proto/res32x4_res8x4.yaml` | 结果以对应 worklog 为准 |
| ImageNet | ResNet34 -> ResNet18 | `configs/imagenet/crld_proto/r34_r18.yaml` | 需单独核验数据和教师权重 |

## 5. 结果来源约定

- **本地实验**：对应输出目录存在完整 `worklog.txt` 和 `best_acc`。
- **文献/基线**：来自论文或项目提供的参考值，不等同于本地复现。
- **历史审计**：曾经运行过但不属于当前正式实现的分支，例如含 GLVQ 的早期原型优化。
- **未跑**：没有完整本地结果，不填写估计值。
