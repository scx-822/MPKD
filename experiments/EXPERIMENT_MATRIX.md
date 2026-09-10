# CRLD-Proto 实验矩阵

## 结果来源

| 标记 | 含义 |
| --- | --- |
| 文献结果 | 来自原论文或官方 benchmark，不是本地重新训练 |
| 本地实验 | 本仓库实际运行且存在对应 worklog.txt |
| 未跑 | 当前没有可核验的本地结果，不填写估计值 |

## CIFAR-100 A–E 消融

固定配置：ResNet32x4 -> ResNet8x4、seed 42、240 epochs、K=3。

| 模式 | CRLD | K-Means 多中心 | GLVQ | 不确定度 | 一致性 | 正向增益 | 原型置信度 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 是 | 否 | 否 | 否 | 否 | 否 | 否 | 未跑 |
| B | 是 | 是 | 否 | 否 | 否 | 否 | 否 | 未跑 |
| C | 是 | 是 | 是 | 否 | 否 | 否 | 否 | 未跑 |
| D | 是 | 是 | 是 | 是 | 否 | 否 | 否 | 未跑 |
| E | 是 | 是 | 是 | 是 | 是 | 是 | 是 | 未跑 |

## 门控因子消融

| 模式 | 删除的唯一因子 | 完整参考 | 状态 |
| --- | --- | --- | --- |
| gate_no_uncertainty | 强视图不确定度 | E | 未跑 |
| gate_no_agreement | 类别一致性 | E | 未跑 |
| gate_no_positive_gain | 锚定类正向增益 | E | 未跑 |
| gate_no_confidence | 原型置信度 | E | 未跑 |

## CIFAR-100 主实验

| 教师 -> 学生 | CRLD-Proto 配置 | CRLD 基线来源 |
| --- | --- | --- |
| ResNet56 -> ResNet20 | configs/cifar100/crld_proto/res56_res20.yaml | 文献结果 |
| ResNet110 -> ResNet32 | configs/cifar100/crld_proto/res110_res32.yaml | 文献结果 |
| ResNet32x4 -> ResNet8x4 | configs/cifar100/crld_proto/res32x4_res8x4.yaml | 文献结果 |
| WRN40-2 -> WRN16-2 | configs/cifar100/crld_proto/wrn40_2_wrn16_2.yaml | 文献结果 |
| WRN40-2 -> WRN40-1 | configs/cifar100/crld_proto/wrn40_2_wrn40_1.yaml | 文献结果 |
| VGG13 -> VGG8 | configs/cifar100/crld_proto/vgg13_vgg8.yaml | 文献结果 |
| ResNet32x4 -> ShuffleNetV2 | configs/cifar100/crld_proto/res32x4_shuv2.yaml | 文献结果 |
| ResNet32x4 -> WRN16-2 | configs/cifar100/crld_proto/res32x4_wrn_16_2.yaml | 文献结果 |
| WRN40-2 -> ResNet8x4 | configs/cifar100/crld_proto/wrn40_2_res8x4.yaml | 文献结果 |
| WRN40-2 -> MobileNetV2 | configs/cifar100/crld_proto/wrn40_2_mv2.yaml | 文献结果 |
| VGG13 -> MobileNetV2 | configs/cifar100/crld_proto/vgg13_mv2.yaml | 文献结果 |
| ResNet50 -> MobileNetV2 | configs/cifar100/crld_proto/res50_mv2.yaml | 文献结果 |

## 跨方法兼容性

兼容性实验使用 ResNet56/20、ResNet110/32、ResNet32x4/8x4、WRN40-2/16-2、WRN40-2/40-1 和 VGG13/8 六组架构，测试 KD、DKD、NKD 与 CRLD/CRLD-Proto 的组合。当前新增组合均为“未跑”，不得填入参考模板的 RVKD 结果。

## Tiny-ImageNet-200

| 教师 -> 学生 | 配置 | CRLD 基线来源 | 状态 |
| --- | --- | --- | --- |
| ResNet32x4 -> ResNet8x4 | configs/tinyimagenet200/crld_proto/res32x4_res8x4.yaml | 未跑 | 教师权重已就绪，等待冒烟测试 |

## 记录要求

- 每个本地实验至少记录 Top-1、Top-5、best epoch、seed、配置路径和结果来源。
- worklog.txt 是本地准确率的原始依据；无日志的数值不得作为本地结果。
- A–E 和门控消融不得同时修改温度、K、优化器或数据增强。
