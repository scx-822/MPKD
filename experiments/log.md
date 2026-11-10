# CRLD-Proto 实验日志

> 更新时间：2026-10-17
> 记录范围：CIFAR-100 正式结果、消融与筛选审计、MPKD 兼容性结果，以及 Tiny-ImageNet 结果。
> 结果口径：仅记录已完成训练并产生 `best_acc` 的运行；中途检查点和未完成实验不进入正式表。

## 1. 实验目标

CRLD-Proto 在完整 CRLD 双视图蒸馏目标上，对教师强干扰视图的同视图蒸馏目标进行多中心原型重构。其余监督损失、弱干扰同视图蒸馏和两项交叉视图蒸馏保持不变。

离线阶段从冻结教师的弱干扰视图特征中筛选可靠样本，按类别执行球面 K-Means，得到每类 `K=3` 个固定子原型。在线阶段使用教师强干扰视图特征计算原型类别分布，再通过可靠性门控得到样本级融合权重，仅重构强干扰同视图目标。教师和原型均不参与学生训练阶段的反向传播。

## 2. 统一设置

- 数据集：CIFAR-100。
- 学生训练：240 epochs，SGD，momentum `0.9`，weight decay `5e-4`，学习率在第 150、180、210 个 epoch 衰减为 `0.1` 倍。
- 批量大小：`64`；随机种子：`42`。
- 原型数量：`PROTO_K=3`；类内更新最多 10 轮；聚合方式：`logmeanexp`。
- 原型参数：`PROTO_TEMP`、`PROTO_TAU_W` 和 `PROTO_FUSION_SCALE` 按教师-学生架构读取对应 YAML。
- 结果指标：以验证集 `best_acc` 为正式 Top-1 结果，基线值采用对应架构的 CRLD 参考结果。

当前正式结果采用当前的多中心原型实现；早期版本输出仍保留在 `output/`，但不纳入正式主表或论文结论。

## 3. 当前正式结果

表中 `CRLD` 依次表示 `TEMPERATURE / CE_WEIGHT / WV_WEIGHT / CV_WEIGHT`；`门控` 表示 `TAU_W / TAU_S`；`原型` 表示 `PROTO_TAU_W / PROTO_TEMP / PROTO_K`。

| 教师 -> 学生 | CRLD 基线 | CRLD 参数 | 门控 | 原型 | 融合比例 | Best Acc | Δ | 配置 |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| ResNet56 -> ResNet20 | 72.10% | `3/1/1/0.25` | `0.8/0.4` | `0.75/0.35/3` | 0.50 | **72.63%** | +0.53 | [`res56_res20.yaml`](../configs/cifar100/crld_proto/res56_res20.yaml) |
| ResNet110 -> ResNet32 | 74.42% | `3/1/1/0.25` | `0.8/0.5` | `0.75/0.50/3` | 0.50 | **75.21%** | +0.79 | [`res110_res32.yaml`](../configs/cifar100/crld_proto/res110_res32.yaml) |
| ResNet32x4 -> ResNet8x4 | 77.60% | `4/1/2/0.5` | `0.9/0.2` | `0.80/0.15/3` | 0.50 | **77.44%** | -0.16 | [`res32x4_res8x4.yaml`](../configs/cifar100/crld_proto/res32x4_res8x4.yaml) |
| WRN40-2 -> WRN16-2 | 76.45% | `3/1/1/0.1` | `0.8/0.6` | `0.80/0.50/3` | 0.25 | **77.36%** | +0.91 | [`wrn40_2_wrn16_2.yaml`](../configs/cifar100/crld_proto/wrn40_2_wrn16_2.yaml) |
| WRN40-2 -> WRN40-1 | 75.58% | `4/1/1/0.1` | `0.8/0.6` | `0.85/0.02/3` | 0.50 | **76.05%** | +0.47 | [`wrn40_2_wrn40_1.yaml`](../configs/cifar100/crld_proto/wrn40_2_wrn40_1.yaml) |
| VGG13 -> VGG8 | 75.27% | `4/1/2/0.5` | `0.8/0.2` | `0.80/0.20/3` | 0.25 | **75.59%** | +0.32 | [`vgg13_vgg8.yaml`](../configs/cifar100/crld_proto/vgg13_vgg8.yaml) |
| ResNet32x4 -> ShuffleNetV2 | 78.27% | `4/1/2/0.5` | `0.9/0.3` | `0.80/0.10/3` | 0.50 | **79.38%** | +1.11 | [`res32x4_shuv2.yaml`](../configs/cifar100/crld_proto/res32x4_shuv2.yaml) |
| ResNet32x4 -> WRN16-2 | 76.92% | `4/1/2/0.5` | `0.9/0.3` | `0.80/0.15/3` | 0.25 | **77.36%** | +0.44 | [`res32x4_wrn_16_2.yaml`](../configs/cifar100/crld_proto/res32x4_wrn_16_2.yaml) |
| ResNet50 -> MobileNetV2 | 71.36% | `3/1/1/0.25` | `0.8/0.5` | `0.75/0.50/3` | 0.50 | **71.39%** | +0.03 | [`res50_mv2.yaml`](../configs/cifar100/crld_proto/res50_mv2.yaml) |
| VGG13 -> MobileNetV2 | 70.39% | `4/1/2/0.5` | `0.8/0.2` | `0.80/0.20/3` | 0.25 | **71.67%** | +1.28 | [`vgg13_mv2.yaml`](../configs/cifar100/crld_proto/vgg13_mv2.yaml) |
| WRN40-2 -> MobileNetV2 | 70.37% | `3/1/1/0.1` | `0.8/0.65` | `0.80/0.35/3` | 0.25 | **71.24%** | +0.87 | [`wrn40_2_mv2.yaml`](../configs/cifar100/crld_proto/wrn40_2_mv2.yaml) |
| WRN40-2 -> ResNet8x4 | 77.28% | `4/1/1/0.1` | `0.8/0.65` | `0.80/0.80/3` | 0.25 | **77.54%** | +0.26 | [`wrn40_2_res8x4.yaml`](../configs/cifar100/crld_proto/wrn40_2_res8x4.yaml) |

## 4. 结果记录规则

1. 每个架构只读取自己的 YAML，不使用批量脚本的统一参数覆盖。
2. 只有完整训练并产生 `best_acc` 的运行进入正式结果表。
3. `worklog.txt` 保存逐轮原始指标，`experiments/log.md` 保存核验后的汇总结果。
4. 训练输出、教师权重和原始实验目录均保留，不在整理日志时删除。

## 5. 运行命令

单模型：

```powershell
python tools/train_proto.py --cfg configs/cifar100/crld_proto/res110_res32.yaml --seed 42
```

批量运行：

```powershell
python tools/run_cifar100_architectures.py
```

## 6. CIFAR-100 消融与筛选审计

正式消融采用当前的多中心原型实现；A、B、D、E 是当前保留的控制链。所有数值均来自独立 `worklog.txt`，指标为验证集 Top-1。

### 6.1 当前正式消融：A、B、D、E

| 模式 | 关键设置 | 输出目录后缀 | Best Acc | Best Epoch | 最终 Acc | 状态 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A | 完整 CRLD，无原型修正 | `ablation_a` | **77.29%** | 190 | 76.95% | 已完成 |
| B | 多中心原型 + 固定融合 | `ablation_b` | **77.72%** | 224 | 77.63% | 已完成 |
| D | 多中心原型 + 不确定度门控 | `ablation_d_no_glvq` | **77.53%** | 215 | 77.25% | 已完成 |
| E | 多中心原型 + 完整可靠性门控 | `ablation_e_no_glvq` | **77.44%** | 228 | 77.24% | 已完成 |

`C` 没有单独的当前版本正式运行，因此不填入正式消融链。

### 6.2 历史 A-E 与单门控结果

以下结果来自早期实现，仅用于审计已有输出，不作为当前方法的正式结论。

| 实验 | Best Acc | Best Epoch | 最终 Acc | 输出目录 |
| --- | ---: | ---: | ---: | --- |
| A | 77.29% | 190 | 76.95% | `ablation_a` |
| B | 77.72% | 224 | 77.63% | `ablation_b` |
| C | 77.41% | 211 | 77.29% | `ablation_c` |
| D | 77.58% | 216 | 77.33% | `ablation_d` |
| E | 77.04% | 228 | 76.89% | `ablation_e` |
| 去除不确定度门控 | 77.40% | 225 | 77.28% | `gate_no_uncertainty` |
| 去除一致性门控 | 77.19% | 236 | 77.07% | `gate_no_agreement` |
| 去除正向增益门控 | 77.31% | 236 | 77.00% | `gate_no_positive_gain` |
| 去除原型置信度门控 | 77.22% | 200 | 77.00% | `gate_no_confidence` |

### 6.3 教师筛选控制实验

“仅教师预测正确”表示关闭置信度筛选、保留正确性条件；“无筛选”表示原型按真实类别接收全部训练样本。未完成的运行不填入结果。

| 教师 -> 学生 | 仅教师预测正确 | 无筛选 | 备注 |
| --- | ---: | ---: | --- |
| ResNet56 -> ResNet20 | 72.29% | 72.42% | 已完成 |
| ResNet110 -> ResNet32 | 75.05% | 75.12% | 已完成 |
| ResNet32x4 -> ResNet8x4 | 未跑 | 77.46% | 无筛选已完成 |
| ResNet32x4 -> ShuffleNetV2 | 78.88% | 未跑 | 仅正确已完成 |
| ResNet32x4 -> WRN16-2 | 77.03% | 未跑 | 仅正确已完成 |
| ResNet50 -> MobileNetV2 | 71.51% | 未跑 | 仅正确已完成 |
| VGG13 -> MobileNetV2 | 69.47% | 未跑 | 仅正确已完成 |
| VGG13 -> VGG8 | 未完成 | 未跑 | 仅运行至第 2 轮 |
| WRN40-2 -> WRN16-2 | 未跑 | 运行中 | 第 37/240 轮，无 `best_acc` |

## 7. MPKD 兼容性结果

兼容性实验采用对应蒸馏方法加 MPKD，seed 42、240 epochs。下表记录所有已经完成的本地运行；DKD 的 `tune1/tune2` 是参数候选，`最佳` 取已完成候选中的最高值。

### 7.1 DKD + MPKD

| 教师 -> 学生 | 初始 | tune1 | tune2 | 当前最佳 |
| --- | ---: | ---: | ---: | ---: |
| ResNet56 -> ResNet20 | 71.04% | 71.81% | — | **71.81%** |
| ResNet110 -> ResNet32 | 73.56% | 74.79% | — | **74.79%** |
| ResNet32x4 -> ResNet8x4 | 77.31% | 77.43% | — | **77.43%** |
| WRN40-2 -> WRN16-2 | 76.24% | 76.09% | 75.92% | **76.24%** |
| WRN40-2 -> WRN40-1 | 75.75% | 75.76% | — | **75.76%** |
| VGG13 -> VGG8 | 75.20% | 75.65% | 75.39% | **75.65%** |

### 7.2 NKD + MPKD

| 教师 -> 学生 | Best Acc | Best Epoch | 状态 |
| --- | ---: | ---: | --- |
| ResNet56 -> ResNet20 | **72.85%** | 232 | 已完成 |
| ResNet110 -> ResNet32 | **74.66%** | 237 | 已完成 |
| ResNet32x4 -> ResNet8x4 | **78.58%** | 225 | 已完成 |
| WRN40-2 -> WRN16-2 | **77.10%** | 229 | 已完成 |
| WRN40-2 -> WRN40-1 | **75.99%** | 228 | 已完成 |
| VGG13 -> VGG8 | **75.66%** | 219 | 已完成 |

KD、KD+CRLD、KD+MPKD、DKD、DKD+CRLD、NKD 和 NKD+CRLD 的完整统一对照实验目前没有本地完整 `worklog.txt`，仍标记为“未跑”，不使用文献数值代填。配置位于 `configs/cifar100/mpkd/`，运行器为 `tools/run_mpkd_compatibility.py`。

## 8. 跨数据集结果

| 数据集 | 教师 -> 学生 | 配置 | Best Top-1 | Best Top-5 | Best Epoch | 状态 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Tiny-ImageNet-200 教师 | ResNet32x4 | `configs/tinyimagenet200/teacher/res32x4.yaml` | **65.31%** | 85.24% | 190 | 已完成 |
| Tiny-ImageNet-200 | ResNet32x4 -> ResNet8x4 | `configs/tinyimagenet200/crld_proto/res32x4_res8x4.yaml` | **62.42%** | 83.80% | 228 | 已完成 |
| ImageNet-1K | ResNet34 -> ResNet18 | `configs/imagenet/crld_proto/r34_r18.yaml` | — | — | — | 尚无本地完整结果 |
| ImageNet-1K | ResNet50 -> MobileNetV1 | `configs/imagenet/crld_proto/r50_mv1.yaml` | — | — | — | 尚无本地完整结果 |

Tiny-ImageNet 的同一 `worklog.txt` 中保留过两次完整运行，最终正式值取两次运行的最高 `best_acc`。ImageNet 配置已准备好，但当前 `output/` 下没有对应完整训练日志。

## 9. 原型筛选保留率审计

以下为 `PROTO_TAU_W=0.80` 的训练集扫描结果。`教师准确率` 是教师在训练集上的准确率，`保留率` 是同时满足教师预测正确和置信度阈值的样本比例。

| 教师 -> 学生 | 教师准确率 | 保留率 | 保留样本数 |
| --- | ---: | ---: | ---: |
| ResNet56 -> ResNet20 | 97.786% | 92.358% | 46179/50000 |
| ResNet110 -> ResNet32 | 99.554% | 98.250% | 49125/50000 |
| ResNet32x4 -> ResNet8x4 | 99.978% | 99.918% | 49959/50000 |
| ResNet32x4 -> ShuffleNetV2 | 99.978% | 99.916% | 49958/50000 |
| ResNet32x4 -> WRN16-2 | 99.974% | 99.920% | 49960/50000 |
| ResNet50 -> MobileNetV2 | 99.966% | 99.930% | 49965/50000 |
| VGG13 -> MobileNetV2 | 99.966% | 99.844% | 49922/50000 |
| VGG13 -> VGG8 | 99.948% | 99.856% | 49928/50000 |
| WRN40-2 -> MobileNetV2 | 99.884% | 99.286% | 49643/50000 |
| WRN40-2 -> ResNet8x4 | 99.894% | 99.282% | 49641/50000 |
| WRN40-2 -> WRN16-2 | 99.906% | 99.292% | 49646/50000 |
| WRN40-2 -> WRN40-1 | 99.890% | 99.250% | 49625/50000 |

该扫描说明多数架构的双重筛选保留率接近 100%，但 ResNet56 -> ResNet20 的保留率相对较低，因此筛选消融仍有必要单独报告。

## 10. 当前运行状态

| 实验 | 当前进度 | 结果状态 | 输出目录 |
| --- | ---: | --- | --- |
| 无筛选：WRN40-2 -> WRN16-2 | 37/240 epochs | 运行中，尚无 `best_acc` | `output/cifar100_baselines/crld_proto,wrn_40_2,wrn_16_2,no_filter` |

## 11. 结果来源约定

- **本地实验**：对应输出目录存在完整 `worklog.txt` 和 `best_acc`。
- **文献/基线**：来自论文或项目提供的参考值，不等同于本地复现。
- **历史探索**：曾经运行过但不属于当前正式结果的早期候选配置。
- **未跑**：没有完整本地结果，不填写估计值。
