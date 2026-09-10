# CIFAR-100 消融结果

固定设置：ResNet32x4 -> ResNet8x4、seed 42、240 epochs、K=3；配置文件为 configs/cifar100/crld_proto/res32x4_res8x4.yaml。

## A–E 递进消融

| 模式 | 关键设置 | 输出目录后缀 | Best Acc | Best Epoch | 最终 Acc | 状态 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A | 完整 CRLD，无原型 | ablation_a | 77.29% | 190 | 76.95% | 已完成 |
| B | K-Means 原型 + 固定融合 | ablation_b | 77.72% | 224 | 77.63% | 已完成 |
| C | B + GLVQ | ablation_c | 77.41% | 211 | 77.29% | 已完成 |
| D | C + 不确定度权重 | ablation_d | 77.58% | 216 | 77.33% | 已完成 |
| E | 完整 CRLD-Proto | ablation_e | 77.04% | 228 | 76.89% | 已完成 |

## 门控因子消融

| 模式 | 删除的唯一因子 | 输出目录后缀 | Best Acc | Best Epoch | 最终 Acc | 状态 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| gate_no_uncertainty | 强视图不确定度 | gate_no_uncertainty | 77.40% | 225 | 77.28% | 已完成 |
| gate_no_agreement | 类别一致性 | gate_no_agreement | 77.19% | 236 | 77.07% | 已完成 |
| gate_no_positive_gain | 锚定类正向增益 | gate_no_positive_gain | 77.31% | 236 | 77.00% | 已完成 |
| gate_no_confidence | 原型置信度 | gate_no_confidence | 77.22% | 200 | 77.00% | 已完成 |

## A、B、D、E（关闭 GLVQ）

该组复用同一配置和 `seed=42` 下已经完成的 A、B 结果，仅新增运行关闭 GLVQ 的 D、E，避免覆盖原有含 GLVQ 记录。

| 模式 | 关键设置 | 输出目录后缀 | Best Acc | Best Epoch | 最终 Acc | 状态 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A | 完整 CRLD，无原型 | ablation_a（复用） | 77.29% | 190 | 76.95% | 已完成 |
| B | K-Means 原型 + 固定融合 | ablation_b（复用） | 77.72% | 224 | 77.63% | 已完成 |
| D_no_glvq | K-Means 原型 + 不确定度门控，无 GLVQ | ablation_d_no_glvq | 77.53% | 215 | 77.25% | 已完成 |
| E_no_glvq | K-Means 原型 + 全部门控，无 GLVQ | ablation_e_no_glvq | 77.44% | 228 | 77.24% | 已完成 |

只有独立输出目录中出现完整 best_acc 后，才把“未跑”替换为本地结果。

