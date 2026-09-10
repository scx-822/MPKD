# CRLD-Proto 实验执行顺序

## 文件职责

- `RUN_ORDER.md`：唯一的人工执行顺序和阶段说明。
- `EXPERIMENT_MATRIX.md`：实验架构与方法矩阵，不记录逐次运行结果。
- `ABLATION_RESULTS.md`：A-E 与门控消融的结构化结果表，由运行器自动更新。
- `COMPATIBILITY_PROTOCOL.md`：KD、DKD、NKD 兼容性实验协议和待填结果表。
- `resources_manifest.json`：`tools/prepare_experiments.py` 生成的资源状态清单。
- `log.md`：完整实验日志和结果汇总；不作为运行配置读取。
- `CRLD-Proto论文草稿.docx`、`CRLD-Proto写作结构分析.md`：论文材料，不参与训练。

三个实验工具保持独立入口：`export_paper_docx.py` 负责 Markdown 到 Word 导出，
`prepare_experiments.py` 负责数据与教师权重准备，`run_cifar100_ablations.py` 负责消融训练编排和结果同步。
它们职责不同，不合并为单一脚本。

CRLD 是原始文献基线；CRLD-Proto 是本文方法。CRLD-Proto 保留完整 CRLD 的双视图监督、同视图蒸馏和交叉视图蒸馏，仅修正教师强视图的同视图蒸馏目标。所有未实际运行的本地实验统一标记为“未跑”，不得使用参考模板中的 RVKD 数值。

## 结果来源约定

- CRLD、KD、DKD 和 NKD 的公开结果可以作为文献对照，但必须注明来源。
- CRLD-Proto、A–E 消融、门控消融及新增组合只采用本仓库实际生成的 worklog.txt。
- 当前不重复运行全部 CRLD 主结果。A 仅作为同代码、同随机种子的必要消融起点，不等同于复现所有 CRLD 架构。
- 消融固定 CIFAR-100、ResNet32x4 -> ResNet8x4、240 epochs、seed 42、K=3，其余参数读取主 YAML。

## 阶段 1：CIFAR-100 A–E 递进消融

五组实验按模块阶段递进。E 一次加入完整安全门控模块中的三个协同因子；这些因子的独立作用在阶段 2 中再逐项验证：

| 模式 | 保留或新增组件 | 原型优化 | 融合权重 |
| --- | --- | --- | --- |
| A | 完整 CRLD，无原型修正 | 无 | 无 |
| B | A + 弱视图可靠样本、多中心原型、Log-Mean-Exp 类别分布 | K-Means | 固定 0.5，无条件融合 |
| C | B + 判别式原型优化 | K-Means + GLVQ | 固定 0.5，无条件融合 |
| D | C + 教师强视图不确定度 | K-Means + GLVQ | 0.5 × uncertainty |
| E | D + 类别一致性、锚定类正向增益、原型置信度 | K-Means + GLVQ | 完整可靠性门控，即正式 CRLD-Proto |

这里的 Log-Mean-Exp 是带 1/K 归一化的 Log-Sum-Exp。A–E 都保留完整 CRLD 损失，差别只发生在教师强视图同视图目标及其原型构建流程。

先检查命令与输出目录：

    python tools/run_cifar100_ablations.py --phase ae --dry-run

按 A、B、C、D、E 顺序运行：

    python tools/run_cifar100_ablations.py --phase ae

只运行第一组 A：

    python tools/train_proto.py --cfg configs/cifar100/crld_proto/res32x4_res8x4.yaml --seed 42 --ablation-mode A

输出目录自动添加 ablation_a 至 ablation_e 后缀，不会覆盖正式 CRLD-Proto 的 worklog.txt。顺序运行器发现某目录已有完整 best_acc 时默认跳过；只有明确使用 --force 才会重跑。

## 阶段 2：门控因子消融

以阶段 1 的 E 为完整对照，每次只删除一个门控因子：

| 模式 | 唯一改动 |
| --- | --- |
| gate_no_uncertainty | 删除强视图不确定度因子 |
| gate_no_agreement | 删除弱视图预测与原型预测的类别一致性因子 |
| gate_no_positive_gain | 删除锚定类别正向增益因子 |
| gate_no_confidence | 删除原型分布置信度因子 |

    python tools/run_cifar100_ablations.py --phase gates --dry-run
    python tools/run_cifar100_ablations.py --phase gates

阶段 1 的 E 直接作为完整门控结果，不再重复训练。

## 阶段 3：整理 CIFAR-100 消融结果

从每个独立目录读取 best_acc、最佳 epoch、最终 Top-1/Top-5 和最终测试损失，填入 experiments/ABLATION_RESULTS.md，再汇总到 experiments/log.md。没有 best_acc 的运行仍标记为“未跑”或“未完成”。

## 阶段 4：现有 Logit 蒸馏方法兼容性

加入参考模板 Table 5 同类型的兼容性实验，但方法和结果必须换成本项目的 CRLD-Proto。固定六组同系列教师–学生架构，比较九行：

1. KD、KD + CRLD、KD + CRLD-Proto。
2. DKD、DKD + CRLD、DKD + CRLD-Proto。
3. NKD、NKD + CRLD、NKD + CRLD-Proto。

基础方法与 CRLD 的公开值可以作为文献结果；真正新增的组合必须本地运行。不得直接把 DKD/NKD 总损失与 CRLD-Proto 总损失相加，否则会重复计算监督和蒸馏项。正确做法是保留各基础方法的官方类别级蒸馏定义，在统一弱/强视图框架中，仅对教师强视图同视图目标应用原型修正。详细协议和未跑表见 experiments/COMPATIBILITY_PROTOCOL.md。

该阶段计算量明显高于 A–E。先完成阶段 1 和阶段 2，再实现并验证 DKD/NKD 的概率目标适配器；当前不能用尚未实现的组合命令替代真实实验。

## 阶段 5：Tiny-ImageNet-200 准备与冒烟测试

数据目录为 data/tiny-imagenet-200。正式训练前还必须具备匹配的 200 类 ResNet32x4 教师权重：

    download_ckpts/tinyimagenet_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth

当前已使用 configs/tinyimagenet200/teacher/res32x4.yaml 完成 240 epochs 本地教师训练，并将最佳权重规范化到上述默认路径。正式实验前先做 1 epoch 冒烟测试，确认 200 类输出、原型构建、前向、反向、验证和日志均正常；冒烟结果不写入论文。

## 阶段 6：Tiny-ImageNet 正式训练

冒烟测试通过后恢复 240 epochs，运行正式 CRLD-Proto，并记录 Top-1、Top-5、最佳 epoch、配置、教师权重来源和结果来源。

## 阶段 7：后续实验

Tiny-ImageNet 结果稳定后再决定 ImageNet。兼容性组合已纳入论文实验矩阵，但应在核心 A–E 与门控消融完成后执行，不阻塞当前第一阶段。

## 资源检查

    python tools/prepare_experiments.py
