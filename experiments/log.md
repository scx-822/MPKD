# CRLD-Proto 实验日志

> 更新时间：2026-08-29
> 记录范围：汇总完整 CRLD 结构上的 CRLD-Proto 主结果、无 GLVQ 对照，以及无 GLVQ 分支的融合权重调参结果。
> 方法关系：CRLD 是原始文献基线；CRLD-Proto 是本文在完整 CRLD 上加入多中心原型修正后的方法。

## 1. 实验目标

### 1.1 问题定义

CRLD 同时使用弱增强视图和强增强视图进行知识蒸馏。教师在弱视图上的预测通常可靠，但强增强可能破坏图像语义，使教师强视图预测出现类别偏移。如果直接把不可靠的教师强视图分布用于蒸馏，错误监督会通过同视图和交叉视图一致性约束传递给学生。

CRLD-Proto 的目标是在不削弱完整 CRLD 的前提下，利用全局类别原型修正教师强视图目标：

1. 从教师弱视图中筛选预测正确且置信度高的样本，构建可靠的类别原型。
2. 用教师强视图特征与多子原型的关系生成原型类别分布。
3. 仅在原型证据可靠且能改善弱视图锚定类别时进行融合，避免错误原型放大噪声。
4. 在同视图蒸馏中，仅将强视图蒸馏项由 $D_T\!\left(\widetilde{p}_{s,i}^{T},z_{s,i}^{S}\right)$ 替换为 $D_T\!\left(\widehat{p}_{s,i}^{T},z_{s,i}^{S}\right)$，其余三个 CRLD 蒸馏方向保持不变。
5. 在不同教师-学生架构上分别调参，并与该架构对应的 CRLD 基线比较。

### 1.2 核心假设

弱视图教师高置信度正确样本能够提供相对稳定的全局类别结构；当强视图教师预测不确定时，该结构可以作为辅助先验。但原型不是无条件正确，因此在线融合必须同时检查类别一致性、锚定类别增益和原型置信度。

## 2. 方法与主要步骤

### 2.1 双视图教师预测

对样本 $i$，弱视图和强视图分别记为 $x_{w,i}$ 与 $x_{s,i}$。教师输出 logits：

$$
z_{w,i}^{T}=T_{\theta}(x_{w,i}),
\qquad
z_{s,i}^{T}=T_{\theta}(x_{s,i}).
$$

用于置信度门控的原始概率为：

$$
p_{v,i}^{T}=\operatorname{Softmax}(z_{v,i}^{T}),
\qquad v\in\{w,s\}.
$$

用于知识蒸馏的温度概率为：

$$
\widetilde{p}_{v,i}^{T}
=
\operatorname{Softmax}\left(\frac{z_{v,i}^{T}}{T}\right).
$$

### 2.2 离线构建多子原型

首先提取教师弱视图特征 $f_{w,i}^{T}$，并进行 L2 归一化。弱视图熵置信度定义为：

$$
\operatorname{Conf}_{w,i}^{T}
=
1-\frac{H(p_{w,i}^{T})}{\log C},
\qquad
H(p)=-\sum_{c=1}^{C}p_c\log p_c.
$$

仅保留预测正确且置信度达到阈值 $\tau_w^{p}$ 的样本：

$$
m_i^{p}
=
\mathbb{I}\!\left(
\arg\max_c p_{w,i,c}^{T}=y_i
\right)
\mathbb{I}\!\left(
\operatorname{Conf}_{w,i}^{T}\geq\tau_w^{p}
\right).
$$

原型构建流程：

1. 按真实类别收集满足 $m_i^{p}=1$ 的弱视图教师特征。
2. 每个类别执行 10 次 K-Means 更新，初始化 $K=3$ 个子原型。
3. 使用 Smooth GLVQ 进行 100 轮判别式优化。
4. GLVQ 前 70% 轮次保持学习率 $0.01$，后 30% 使用余弦退火降至 $0.001$。
5. Soft-min 参数固定为 $\gamma=10$，每次原型更新后重新进行 L2 归一化。
6. 原型在学生训练开始前固定，不参与学生反向传播。

GLVQ 使用余弦距离 $d=1-\cos(\cdot,\cdot)$，分别计算平滑正类距离与负类距离：

$$
d_i^{+}
=
-\frac{1}{\gamma}
\log\sum_{k\in\mathcal{P}(y_i)}
\exp(-\gamma d_{i,k}),
\qquad
d_i^{-}
=
-\frac{1}{\gamma}
\log\sum_{k\notin\mathcal{P}(y_i)}
\exp(-\gamma d_{i,k}).
$$

相对距离和优化目标为：

$$
\mu_i
=
\frac{d_i^{+}-d_i^{-}}
{d_i^{+}+d_i^{-}+\epsilon},
\qquad
\mathcal{L}_{\mathrm{GLVQ}}
=
\frac{1}{N}\sum_{i=1}^{N}\sigma(10\mu_i).
$$

### 2.3 强视图原型分布

教师强视图特征与第 $c$ 类第 $k$ 个子原型的余弦相似度为：

$$
s_{i,c,k}
=
\cos\left(f_{s,i}^{T},p_{c,k}\right).
$$

每一类的多个子原型通过 Log-Mean-Exp 进行软汇总：

$$
g_{i,c}
=
\log\left[
\frac{1}{K}
\sum_{k=1}^{K}
\exp\left(
\frac{s_{i,c,k}}{\tau_p}
\right)
\right].
$$

原型类别分布为：

$$
p_{\mathrm{sim},i}
=
\operatorname{Softmax}(g_i).
$$

这里使用的是余弦相似度，数值越大表示越相似；代码没有把余弦距离与余弦相似度的方向混淆。

### 2.4 可靠性门控的概率融合

弱视图教师预测类别作为在线锚定类别：

$$
\hat{y}_{w,i}^{T}
=
\arg\max_c p_{w,i,c}^{T}.
$$

教师强视图不确定度为：

$$
u_i
=
\frac{H(p_{s,i}^{T})}{\log C}.
$$

再计算三个原型可靠性因子：

$$
a_i
=
\mathbb{I}\!\left(
\arg\max_c p_{\mathrm{sim},i,c}
=
\hat{y}_{w,i}^{T}
\right),
$$

$$
r_i
=
\frac{
\left[
p_{\mathrm{sim},i,\hat{y}_{w,i}^{T}}
-
\widetilde{p}_{s,i,\hat{y}_{w,i}^{T}}^{T}
\right]_{+}
}{
p_{\mathrm{sim},i,\hat{y}_{w,i}^{T}}+\epsilon
},
$$

$$
q_i
=
1-\frac{H(p_{\mathrm{sim},i})}{\log C}.
$$

最终有效融合权重为：

$$
\lambda_i^{\mathrm{eff}}
=
\operatorname{clip}
\left(
0.5\,u_i a_i r_i q_i,\,
0,\,
1
\right).
$$

修正后的教师强视图目标为：

$$
\widehat{p}_{s,i}^{T}
=
\left(1-\lambda_i^{\mathrm{eff}}\right)
\widetilde{p}_{s,i}^{T}
+
\lambda_i^{\mathrm{eff}}p_{\mathrm{sim},i}.
$$

只有原型类别与弱视图教师一致、原型能提高锚定类别概率且原型分布本身较确定时，融合权重才大于零。在线融合不直接使用真实标签；真实标签只用于原型样本筛选和学生监督交叉熵。

### 2.5 保留完整 CRLD 损失

弱视图与强视图置信度掩码为：

$$
m_{w,i}
=
\mathbb{I}\!\left(
\max_c p_{w,i,c}^{T}\geq\tau_w
\right),
\qquad
m_{s,i}
=
\mathbb{I}\!\left(
\max_c p_{s,i,c}^{T}\geq\tau_s
\right).
$$

监督损失同时约束学生弱视图和强视图：

$$
\mathcal{L}_{\mathrm{CE}}
=
\operatorname{CE}(z_w^S,y)
+
\operatorname{CE}(z_s^S,y).
$$

同视图蒸馏为：

$$
\mathcal{L}_{\mathrm{WV}}
=
\frac{1}{N}\sum_i m_{w,i}
\left[
D_T\!\left(\widetilde{p}_{w,i}^{T},z_{w,i}^{S}\right)
+
D_T\!\left(\widehat{p}_{s,i}^{T},z_{s,i}^{S}\right)
\right].
$$

交叉视图蒸馏仍使用原始教师目标：

$$
\mathcal{L}_{\mathrm{CV}}
=
\frac{1}{N}\sum_i m_{s,i}
\left[
D_T\!\left(\widetilde{p}_{w,i}^{T},z_{s,i}^{S}\right)
+
D_T\!\left(\widetilde{p}_{s,i}^{T},z_{w,i}^{S}\right)
\right],
$$

其中：

$$
D_T(q,z^S)
=
T^2
\operatorname{KL}
\left(
q
\;\middle\|\;
\operatorname{Softmax}\left(\frac{z^S}{T}\right)
\right).
$$

总损失为：

$$
\mathcal{L}_{\mathrm{total}}
=
\alpha\mathcal{L}_{\mathrm{CE}}
+
\beta_w\mathcal{L}_{\mathrm{WV}}
+
\beta_c\mathcal{L}_{\mathrm{CV}}.
$$

因此，原型只把同视图强视图蒸馏项中的教师目标由 $\widetilde{p}_{s,i}^{T}$ 修正为 $\widehat{p}_{s,i}^{T}$，学生预测 $z_{s,i}^{S}$ 以及其余三个 CRLD 蒸馏方向均保持不变；该方法没有新增独立原型损失，也没有删除完整 CRLD 的任何损失项。

## 3. 统一实验协议

| 项目           | 设置                     |
| -------------- | ------------------------ |
| 数据集         | CIFAR-100                |
| 训练轮数       | 240                      |
| 优化器         | SGD                      |
| 初始学习率     | 0.05                     |
| Momentum       | 0.9                      |
| 权重衰减       | 默认 0.0005，各对照沿用对应正式配置 |
| 学习率衰减轮次 | 150、180、210            |
| 学习率衰减率   | 0.1                      |
| 随机种子       | 42                       |
| 子原型数量     | $K=3$                     |
| 评价指标       | 训练过程中的`best_acc` |
| 教师状态       | 参数冻结并始终保持 `eval()`，固定 BatchNorm 运行统计量 |

### 3.1 教师 BatchNorm 修复说明

原始 CRLD 在教师前向时使用 `torch.no_grad()`，优化器只更新学生参数；同时，`Distiller.train()` 会在学生训练期间强制执行 `teacher.eval()`。因此，教师权重和 BatchNorm 运行统计量都应保持固定。

此前的 `train_proto.py` 在教师加载后，为探测池化特征维度执行了一次全零张量前向，但该次前向发生在教师仍处于训练模式时。虽然没有梯度反向传播，BatchNorm 的 `running_mean`、`running_var` 和 `num_batches_tracked` 仍会被更新，导致后续弱视图样本筛选、原型构建、强视图预测和 KD 目标均基于被轻微改变的教师。

历史最佳检查点与原始预训练教师的逐项核验结果如下：

| 教师-学生组合 | 发生差异的 BN 状态项 / 可比状态项 |
| ------------- | ---------------------------------: |
| ResNet56 -> ResNet20 | 114 / 114 |
| ResNet110 -> ResNet32 | 218 / 222 |
| ResNet32x4 -> ResNet8x4 | 68 / 68 |
| WRN40-2 -> WRN16-2 | 74 / 74 |
| WRN40-2 -> WRN40-1 | 74 / 74 |
| VGG13 -> VGG8 | 20 / 20 |

各网络中代表性 `running_mean` 的修复前比例均约为原值的 `0.9`，与 PyTorch 默认 BatchNorm 动量 `0.1` 下的一次零输入更新完全一致。现已在教师加载到 GPU 后立即执行 `model_teacher.eval()`，再进行特征维度探测；静态验证中，探测前后教师 BN 状态变化数量为 `0`。

该修改属于实现纠错，不是新增方法模块，也不作为论文创新点。完整 CRLD 损失、原型构建、Log-Mean-Exp 原型分布和可靠性门控融合均未改变。修复使 CRLD-Proto 与原始 CRLD 遵循相同的冻结教师规范，因此提升了基线比较的公平性。

### 3.2 原型准确率口径

普通训练启动时打印的原型准确率，是固定教师特征与离线原型在 CIFAR-100 官方测试集干净视图上的诊断准确率，不参与学生反向传播。类别预测使用与训练融合一致的 Log-Mean-Exp 多子原型评分，而不是旧版“最近单个子原型”规则，因此不能与旧口径直接横向比较。`--proto-scan` 输出的训练集分层留出结果仅用于参数诊断，也不能与官方测试集结果混用。原型干净视图准确率升高不等价于强视图纠偏能力提高，最终判断仍以强视图诊断和学生 `best_acc` 为准。

正式结果只保留完成 240 epoch 且产生 `best_acc` 的运行；早期探索、重复运行、未刷新历史最好值的候选配置和中断运行不再写入本汇总。教师 BN 修复后的结果作为当前正式结果起点。无 GLVQ 的融合权重调参结果单独标注，不覆盖含 GLVQ 主结果。

结果表采用以下紧凑记法：

- `CRLD`：`T / CE_WEIGHT / WV_WEIGHT / CV_WEIGHT`
- `门控`：`TAU_W / TAU_S`
- `原型`：`PROTO_TAU_W / PROTO_TEMP / K`
- 非默认求解器参数会直接写在实验名称中，例如 `WD=0.00075`
- `Δ`：`Best Acc - 对应模型的 CRLD 基线`，单位为百分点

第 6 节将完整 CRLD-Proto 主结果、无 GLVQ 对照和本次无 GLVQ 调参结果放在同一张表中；原始逐轮记录仍保存在对应输出目录的 `worklog.txt` 中。

## 4. 分模型实验结果

### 4.1 ResNet56 -> ResNet20

参考 CRLD 基线：**72.10%**

| 正式配置 | CRLD | 门控 | 原型 | Best Acc | Δ（百分点） |
| -------- | ---- | ---- | ---- | -------: | -----------: |
| BN 修复后正式结果 | `3.0/1.0/1.0/0.25` | `0.8/0.4` | `0.75/0.35/3` | **72.36%** | **+0.26** |

该结果完成 240 epoch 训练，最佳值出现在 epoch 224，超过对应 CRLD 基线 `0.26` 个百分点。

### 4.2 ResNet110 -> ResNet32

参考 CRLD 基线：**74.42%**

| 正式配置 | CRLD | 门控 | 原型 | Best Acc | Δ（百分点） |
| -------- | ---- | ---- | ---- | -------: | -----------: |
| BN 修复后正式结果 | `3.0/1.0/1.0/0.25` | `0.8/0.5` | `0.75/0.50/3` | **74.99%** | **+0.57** |

该结果完成 240 epoch 训练，最佳值出现在 epoch 221，超过对应 CRLD 基线 `0.57` 个百分点。

### 4.3 ResNet32x4 -> ResNet8x4

参考 CRLD 基线：**77.60%**

| 正式配置 | CRLD | 门控 | 原型 | Best Acc | Δ（百分点） |
| -------- | ---- | ---- | ---- | -------: | -----------: |
| BN 修复后正式结果 | `4.0/1.0/2.0/0.5` | `0.9/0.2` | `0.80/0.15/3` | **77.33%** | **-0.27** |

该结果是当前正式最好，仍低于对应 CRLD 基线 `0.27` 个百分点；后续调参记录不在本汇总中展开。

### 4.4 WRN40-2 -> WRN16-2

参考 CRLD 基线：**76.45%**

| 正式配置 | CRLD | 门控 | 原型 | Best Acc | Δ（百分点） |
| -------- | ---- | ---- | ---- | -------: | -----------: |
| BN 修复后正式结果 | `3.0/1.0/1.0/0.1` | `0.8/0.6` | `0.80/0.50/3` | **76.97%** | **+0.52** |

该结果超过对应 CRLD 基线 `0.52` 个百分点，正式配置固定为 `PROTO_TEMP=0.50`。

### 4.5 WRN40-2 -> WRN40-1

参考 CRLD 基线：**75.58%**

| 正式配置 | CRLD | 门控 | 原型 | Best Acc | Δ（百分点） |
| -------- | ---- | ---- | ---- | -------: | -----------: |
| BN 修复后正式结果 | `4.0/1.0/1.0/0.1` | `0.8/0.6` | `0.85/0.02/3` | **75.86%** | **+0.28** |

该结果完成 240 epoch 训练，超过对应 CRLD 基线 `0.28` 个百分点。

### 4.6 VGG13 -> VGG8

参考 CRLD 基线：**75.27%**

| 正式配置 | CRLD | 门控 | 原型 | Best Acc | Δ（百分点） |
| -------- | ---- | ---- | ---- | -------: | -----------: |
| BN 修复后正式结果 | `4.0/1.0/2.0/0.5` | `0.8/0.2` | `0.80/0.20/3` | **75.57%** | **+0.30** |

该结果完成 240 epoch 训练，超过对应 CRLD 基线 `0.30` 个百分点。

### 4.7 ResNet32x4 -> ShuffleNetV2

参考 CRLD 基线：**78.27%**

| 正式配置 | CRLD | 门控 | 原型 | Best Acc | Δ（百分点） |
| -------- | ---- | ---- | ---- | -------: | -----------: |
| 正式最佳结果 | `4.0/1.0/2.0/0.5` | `0.9/0.3` | `0.80/0.10/3` | **78.96%** | **+0.69** |

该结果超过对应 CRLD 基线 `0.69` 个百分点，正式配置固定为 `PROTO_TEMP=0.10`。

### 4.8 ResNet32x4 -> WRN16-2

参考 CRLD 基线：**76.92%**

| 正式配置 | CRLD | 门控 | 原型 | Best Acc | Δ（百分点） |
| -------- | ---- | ---- | ---- | -------: | -----------: |
| 正式最佳结果 | `4.0/1.0/2.0/0.5` | `0.9/0.3` | `0.80/0.15/3` | **77.27%** | **+0.35** |

该结果超过对应 CRLD 基线 `0.35` 个百分点，正式配置固定为 `PROTO_TEMP=0.15`。

### 4.9 ResNet50 -> MobileNetV2

参考 CRLD 基线：**71.36%**

| 正式配置 | CRLD | 门控 | 原型 | Best Acc | Δ（百分点） |
| -------- | ---- | ---- | ---- | -------: | -----------: |
| 正式最佳结果 | `3.0/1.0/1.0/0.25` | `0.8/0.5` | `0.75/0.50/3` | **71.68%** | **+0.32** |

该结果超过对应 CRLD 基线 `0.32` 个百分点，正式配置固定为 `PROTO_TAU_W=0.75`。

### 4.10 VGG13 -> MobileNetV2

参考 CRLD 基线：**70.39%**

| 正式配置 | CRLD | 门控 | 原型 | Best Acc | Δ（百分点） |
| -------- | ---- | ---- | ---- | -------: | -----------: |
| 正式最佳结果 | `4.0/1.0/2.0/0.5` | `0.8/0.2` | `0.80/0.20/3` | **71.00%** | **+0.61** |

该结果超过对应 CRLD 基线 `0.61` 个百分点，正式配置固定为 `PROTO_TEMP=0.20`。

### 4.11 WRN40-2 -> MobileNetV2

参考 CRLD 基线：**70.37%**

| 正式配置 | CRLD | 门控 | 原型 | Best Acc | Δ（百分点） |
| -------- | ---- | ---- | ---- | -------: | -----------: |
| 正式最佳结果 | `3.0/1.0/1.0/0.1` | `0.8/0.65` | `0.80/0.35/3` | **71.00%** | **+0.63** |

该结果超过对应 CRLD 基线 `0.63` 个百分点。

### 4.12 WRN40-2 -> ResNet8x4

参考 CRLD 基线：**77.28%**

| 正式配置 | CRLD | 门控 | 原型 | Best Acc | Δ（百分点） |
| -------- | ---- | ---- | ---- | -------: | -----------: |
| 正式最佳结果 | `4.0/1.0/1.0/0.1` | `0.8/0.65` | `0.80/0.80/3` | **77.53%** | **+0.25** |

该结果超过对应 CRLD 基线 `0.25` 个百分点，正式配置固定为 `PROTO_TEMP=0.80`。

## 5. 结果管理与复现

### 5.1 记录规则

> 本日志中的 CRLD 基线数值采用参考论文或官方 benchmark 的公开结果，不作为本地重新训练结果。完整 CRLD-Proto 主结果、GLVQ 对照及无 GLVQ 调参结果统一列于第 6 节。

1. 各模型只和对应架构的 CRLD 基线比较。
2. 同一配置的重复运行聚合在同一模型小节，不再创建重复的批次章节。
3. 候选配置必须完成训练并产生 `best_acc` 后，才能进入正式结果表。
4. `worklog.txt` 是逐轮训练原始记录，`log.md` 是经过结构核验后的汇总记录。
5. 批量脚本不使用统一的 `CRLD.*` 覆盖项，每个模型读取自己的 YAML；候选未刷新历史最好时，只恢复本轮涉及的参数标量。
6. GLVQ 对照统一使用 `seed=42`，仅改变 `CRLD.PROTO_USE_GLVQ`；本轮无 GLVQ 调参仅将 `PROTO_FUSION_SCALE` 从 `0.5` 改为 `0.25`，其余设置不变；中断训练或未产生 `best_acc` 的运行不进入汇总表。

### 5.2 输出目录

```text
output/cifar100_baselines/crld_proto,res56,res20
output/cifar100_baselines/crld_proto,res110,res32
output/cifar100_baselines/crld_proto,res32x4,res8x4
output/cifar100_baselines/crld_proto,wrn_40_2,wrn_16_2
output/cifar100_baselines/crld_proto,wrn_40_2,wrn_40_1
output/cifar100_baselines/crld_proto,vgg13,vgg8
output/cifar100_baselines/crld_proto,res32x4,shuv2
output/cifar100_baselines/crld_proto,res32x4,wrn_16_2
output/cifar100_baselines/crld_proto,res50,mv2
output/cifar100_baselines/crld_proto,vgg13,mv2
output/cifar100_baselines/crld_proto,wrn40_2,mv2
output/cifar100_baselines/crld_proto,wrn_40_2,res8x4
```

### 5.3 运行命令

单模型训练：

```powershell
python tools/train_proto.py --cfg configs/cifar100/crld_proto/res110_res32.yaml --seed 42
```

按 `批量运行.py` 中启用的列表顺序训练：

```powershell
python 批量运行.py
```

## 6. 结果汇总

为避免主结果和消融结果分散，下面按教师-学生架构合并展示。所有行均使用对应架构的 CRLD 基线进行比较，表中准确率均为完成 240 epochs 后记录的 `best_acc`。

- `含 GLVQ 主结果`：完整 CRLD-Proto 结构，包含 GLVQ 原型优化。
- `无 GLVQ（E）`：仅关闭 GLVQ，保留多中心 K-Means 原型、完整 CRLD 损失和全部可靠性门控；默认 `PROTO_FUSION_SCALE=0.5`。
- `无 GLVQ 调参`：仅将 `PROTO_FUSION_SCALE` 改为 `0.25`，其余设置不变；本轮只对六组下降架构进行验证。
- `无 GLVQ 当前最佳`：无 GLVQ 分支中已完成实验的最高有效结果；括号内为相对 CRLD 基线的差值。

| 模型架构 | CRLD 基线 | 含 GLVQ 主结果 | 无 GLVQ（E） | 无 GLVQ 调参（scale=0.25） | 无 GLVQ 当前最佳（Δ） | 当前处理 |
| -------- | --------: | ------------: | -----------: | -------------------------: | ---------------------: | -------- |
| ResNet56 -> ResNet20 | 72.10% | 72.36% | **72.63%** | — | **72.63%（+0.53）** | 保留 E（scale=0.5） |
| ResNet110 -> ResNet32 | 74.42% | 74.99% | **75.21%** | — | **75.21%（+0.79）** | 保留 E（scale=0.5） |
| ResNet32x4 -> ResNet8x4 | 77.60% | 77.33% | **77.44%** | — | **77.44%（-0.16）** | 当前仍未超过基线 |
| WRN40-2 -> WRN16-2 | 76.45% | 76.97% | 77.00% | **77.36%** | **77.36%（+0.91）** | 保留 scale=0.25 |
| WRN40-2 -> WRN40-1 | 75.58% | 75.86% | **76.05%** | — | **76.05%（+0.47）** | 保留 E（scale=0.5） |
| VGG13 -> VGG8 | 75.27% | 75.57% | 75.41% | **75.59%** | **75.59%（+0.32）** | 保留 scale=0.25 |
| ResNet32x4 -> ShuffleNetV2 | 78.27% | 78.96% | **79.38%** | — | **79.38%（+1.11）** | 保留 E（scale=0.5） |
| ResNet32x4 -> WRN16-2 | 76.92% | 77.27% | 76.91% | **77.36%** | **77.36%（+0.44）** | 保留 scale=0.25 |
| ResNet50 -> MobileNetV2 | 71.36% | 71.68% | **71.39%** | 71.30%（回退） | **71.39%（+0.03）** | 回退至 E（scale=0.5） |
| VGG13 -> MobileNetV2 | 70.39% | 71.00% | 70.79% | **71.67%** | **71.67%（+1.28）** | 保留 scale=0.25 |
| WRN40-2 -> MobileNetV2 | 70.37% | 71.00% | 70.85% | **71.24%** | **71.24%（+0.87）** | 保留 scale=0.25 |
| WRN40-2 -> ResNet8x4 | 77.28% | 77.53% | 77.20% | **77.54%** | **77.54%（+0.26）** | 保留 scale=0.25 |
<!-- GLVQ_RESULTS_AUTO -->

### 6.1 结果判断

完整 CRLD-Proto 主结果中，12 组架构有 11 组超过对应 CRLD 基线；未超过的仍是 ResNet32x4 -> ResNet8x4。无 GLVQ 分支的当前最佳结果同样有 11 组超过基线，其中五组通过降低融合强度至 `PROTO_FUSION_SCALE=0.25` 得到提升。

本轮六组调参结果中，ResNet32x4 -> WRN16-2、VGG13 -> VGG8、VGG13 -> MobileNetV2、WRN40-2 -> MobileNetV2 和 WRN40-2 -> ResNet8x4 保留 `0.25`；ResNet50 -> MobileNetV2 的 `71.30%` 低于原无 GLVQ 结果 `71.39%`，因此回退到 `PROTO_FUSION_SCALE=0.5`。这说明融合强度对教师-学生架构具有依赖性，不能将 `0.25` 无条件推广到所有模型。

该表中的无 GLVQ 结果用于独立评估 GLVQ 的贡献，不替换“含 GLVQ 主结果”的 CRLD-Proto 正式方法定义。原始逐轮记录仍保存在各输出目录的 `worklog.txt` 中。

### 6.2 本轮无 GLVQ 调参输出目录

```text
无 GLVQ（E，PROTO_FUSION_SCALE=0.5）
ResNet56 -> ResNet20: output/cifar100_baselines/crld_proto,res56,res20,e_no_glvq
ResNet32x4 -> ResNet8x4: output/cifar100_baselines/crld_proto,res32x4,res8x4,ablation_e_no_glvq
ResNet110 -> ResNet32: output/cifar100_baselines/crld_proto,res110,res32,ablation_e_no_glvq
WRN40-2 -> WRN16-2: output/cifar100_baselines/crld_proto,wrn_40_2,wrn_16_2,ablation_e_no_glvq
WRN40-2 -> WRN40-1: output/cifar100_baselines/crld_proto,wrn_40_2,wrn_40_1,ablation_e_no_glvq
VGG13 -> VGG8: output/cifar100_baselines/crld_proto,vgg13,vgg8,ablation_e_no_glvq
ResNet32x4 -> ShuffleNetV2: output/cifar100_baselines/crld_proto,res32x4,shuv2,ablation_e_no_glvq
ResNet32x4 -> WRN16-2: output/cifar100_baselines/crld_proto,res32x4,wrn_16_2,ablation_e_no_glvq
ResNet50 -> MobileNetV2: output/cifar100_baselines/crld_proto,res50,mv2,ablation_e_no_glvq
VGG13 -> MobileNetV2: output/cifar100_baselines/crld_proto,vgg13,mv2,ablation_e_no_glvq
WRN40-2 -> MobileNetV2: output/cifar100_baselines/crld_proto,wrn40_2,mv2,ablation_e_no_glvq
WRN40-2 -> ResNet8x4: output/cifar100_baselines/crld_proto,wrn_40_2,res8x4,ablation_e_no_glvq

无 GLVQ 调参（PROTO_FUSION_SCALE=0.25）
output/cifar100_baselines/crld_proto,tune_noglvq,res32x4_wrn_16_2,f025,ablation_e_no_glvq
output/cifar100_baselines/crld_proto,tune_noglvq,vgg13_vgg8,f025,ablation_e_no_glvq
output/cifar100_baselines/crld_proto,tune_noglvq,vgg13_mv2,f025,ablation_e_no_glvq
output/cifar100_baselines/crld_proto,tune_noglvq,wrn40_2_mv2,f025,ablation_e_no_glvq
output/cifar100_baselines/crld_proto,tune_noglvq,wrn40_2_res8x4,f025,ablation_e_no_glvq
output/cifar100_baselines/crld_proto,tune_noglvq,res50_mv2,f025,ablation_e_no_glvq
```

其中最后一项为已完成但未采用的候选结果，保留目录仅用于审计和复现；其余五组调参结果已采用，E 对照目录仍按各架构原有的 `ablation_e_no_glvq` 或 `e_no_glvq` 目录保存。
