# 暂停运行记录

更新时间：2026-10-23

## 当前存档：DKD + MPKD 与 NKD + MPKD（ResNet56 -> ResNet20）

两组实验均使用 seed 42、计划 240 epochs，以及更新后的配置 `PROTO_TAU_W=0.6`、`PROTO_FUSION_SCALE=0.5`。DKD 已完成；NKD 在第 14 轮完整检查点处暂停。暂停时未保存的训练状态不保留，续跑从 `latest` 中的第 14 轮恢复。

| 实验 | 状态 | 已保存进度 | Best Top-1 | 输出目录 |
| --- | --- | ---: | ---: | --- |
| DKD + MPKD，ResNet56 -> ResNet20 | 已完成 | 240 / 240 | 72.22%（第 240 轮） | `output/cifar100_compatibility/mpkd,dkd,proto_tau06,res56_res20` |
| NKD + MPKD，ResNet56 -> ResNet20 | 已暂停 | 14 / 240 | 54.13%（第 12 轮） | `output/cifar100_compatibility/mpkd,nkd,proto_tau06,res56_res20` |

从 NKD + MPKD 的第 14 轮检查点继续：

```powershell
$env:CRLD_CUDNN_BENCHMARK="0"
& "D:\softwares\python\python.exe" tools/train_proto.py --cfg configs/cifar100/mpkd_compatibility/res56_res20.yaml --resume --seed 42 EXPERIMENT.TAG mpkd,nkd,proto_tau06,res56_res20 EXPERIMENT.METHOD NKD-MPKD EXPERIMENT.BASELINE NKD MPKD.BASE_METHOD NKD
```

已保留上述输出目录中的 `latest`、`best`、`student_latest`、`student_best` 和 `worklog.txt`；不要清理或改名，以便从当前进度续训。

## NKD + MPKD（seed 42）

本轮六组实验均采用 NKD + MPKD、seed 42 和 240 epochs；模型及其参数读取各自 YAML。WRN40-2 -> WRN16-2 已从第 49 轮检查点续训至第 83 轮后暂停。检查点按完整 epoch 保存，表中准确率为验证集最佳 Top-1。

| 架构 | 状态 | 已保存轮次 | Best Top-1 | 输出目录 |
| --- | --- | ---: | ---: | --- |
| ResNet56 -> ResNet20 | 已完成 | 240 / 240 | 72.85% | `output/cifar100_compatibility/mpkd,nkd,nkd1,res56_res20` |
| ResNet110 -> ResNet32 | 已完成 | 240 / 240 | 74.66% | `output/cifar100_compatibility/mpkd,nkd,nkd1,res110_res32` |
| ResNet32x4 -> ResNet8x4 | 已完成 | 240 / 240 | 78.58% | `output/cifar100_compatibility/mpkd,nkd,nkd1,res32x4_res8x4` |
| WRN40-2 -> WRN16-2 | 已暂停 | 83 / 240 | 65.35%（best 为第62轮） | `output/cifar100_compatibility/mpkd,nkd,nkd1,wrn40_2_wrn16_2` |
| WRN40-2 -> WRN40-1 | 未开始 | 0 / 240 | — | `output/cifar100_compatibility/mpkd,nkd,nkd1,wrn40_2_wrn40_1` |
| VGG13 -> VGG8 | 未开始 | 0 / 240 | — | `output/cifar100_compatibility/mpkd,nkd,nkd1,vgg13_vgg8` |

从 WRN40-2 -> WRN16-2 的第 83 轮检查点继续：

```powershell
$env:CRLD_CUDNN_BENCHMARK="0"
python tools/train_proto.py --cfg configs/cifar100/mpkd_compatibility/wrn40_2_wrn16_2.yaml --resume --seed 42 EXPERIMENT.TAG mpkd,nkd,nkd1,wrn40_2_wrn16_2 EXPERIMENT.METHOD NKD-MPKD EXPERIMENT.BASELINE NKD MPKD.BASE_METHOD NKD
```

该组完成后运行剩余两组：

```powershell
python tools/run_mpkd_compatibility.py --architectures wrn40_2_wrn40_1 vgg13_vgg8 --methods NKD --seed 42 --tag-suffix nkd1
```

保留以上输出目录中的 `latest`、`best`、`student_latest`、`student_best` 和 `worklog.txt`；不要清理这些检查点。

## DKD + MPKD（历史暂停快照）

以下内容保留 2026-09-18 的 DKD + MPKD 暂停状态，供追溯历史进度；当前运行状态见上方 NKD + MPKD 表。暂停不等于删除，续训使用原输出目录中的检查点和 `worklog.txt`。

| 实验 | 状态 | 已保存进度 | Best Acc | 输出目录 |
| --- | --- | ---: | ---: | --- |
| DKD + MPKD，ResNet56 -> ResNet20 | 已完成 | 240 / 240 | 71.04% | `output/cifar100_compatibility/mpkd,dkd,res56_res20` |
| DKD + MPKD，ResNet110 -> ResNet32 | 已完成 | 240 / 240 | 73.56% | `output/cifar100_compatibility/mpkd,dkd,res110_res32` |
| DKD + MPKD，ResNet32x4 -> ResNet8x4 | 已完成 | 240 / 240 | 77.31% | `output/cifar100_compatibility/mpkd,dkd,res32x4_res8x4` |
| DKD + MPKD，WRN40-2 -> WRN16-2 | 已完成 | 240 / 240 | 76.24% | `output/cifar100_compatibility/mpkd,dkd,wrn40_2_wrn16_2` |
| DKD + MPKD，WRN40-2 -> WRN40-1 | 已暂停 | 105 / 240 | 待完成 | `output/cifar100_compatibility/mpkd,dkd,wrn40_2_wrn40_1` |
| DKD + MPKD，VGG13 -> VGG8 | 未开始 | 0 / 240 | 待完成 | `output/cifar100_compatibility/mpkd,dkd,vgg13_vgg8` |

## 续训命令

ResNet110 -> ResNet32、ResNet32x4 -> ResNet8x4 和 WRN40-2 -> WRN16-2 已完成。当前续训目标是 WRN40-2 -> WRN40-1 的第 105 轮检查点。

从 WRN40-2 -> WRN40-1 的 `latest` 继续：

```powershell
python tools/train_proto.py --cfg configs/cifar100/mpkd_compatibility/wrn40_2_wrn40_1.yaml --resume --seed 42 EXPERIMENT.TAG mpkd,dkd,wrn40_2_wrn40_1 EXPERIMENT.METHOD DKD-MPKD EXPERIMENT.BASELINE DKD MPKD.BASE_METHOD DKD
```

不要删除或改名 `output/cifar100_compatibility/mpkd,dkd,wrn40_2_wrn40_1/`，否则 `--resume` 无法读取当前检查点。

第五组完成后运行第六组：

```powershell
python tools/run_mpkd_compatibility.py --architectures vgg13_vgg8 --methods DKD --seed 42
```

## 2026-10-24 暂停：真实标签锚点对照

本组为 `ResNet56 -> ResNet20` 的原型锚点对照实验，使用 `seed 42`、完整门控和
`PROTO_ANCHOR=ground_truth`，计划训练 240 个 epoch。训练进程已在第 199 轮检查点
保存后停止，检查点和日志均已保留。

| 实验 | 状态 | 已保存轮次 | Best Top-1 | Best Top-5 | 输出目录 |
| --- | --- | ---: | ---: | ---: | --- |
| 真实标签锚点，ResNet56 -> ResNet20 | 已暂停 | 199 / 240 | 72.11%（第190轮） | 92.90%（第197轮） | `output/cifar100_baselines/crld_proto,res56,res20,anchor_ground_truth` |

从第 199 轮检查点继续：

```powershell
$env:CRLD_CUDNN_BENCHMARK="0"
& "D:\softwares\python\python.exe" tools/train_proto.py --cfg configs/cifar100/mpkd/res56_res20.yaml --resume --seed 42 EXPERIMENT.TAG crld_proto,res56,res20,anchor_ground_truth MPKD.PROTO_ANCHOR ground_truth
```

已保留该目录中的 `latest`、`best`、`student_latest`、`student_best` 和 `worklog.txt`，
不要删除或改名，以便后续续训。

## 2026-10-27 完成：K=2 原型数量实验

本组为 `ResNet32x4 -> ResNet8x4` 的 `K=2` 原型数量实验，使用 `seed 42`，计划训练
240 个 epoch。实验已从第 96 轮检查点续跑并完成。

| 实验 | 状态 | 完成轮次 | 最佳 Top-1 | 最佳 Top-5 | 第 240 轮 Top-1 / Top-5 | 输出目录 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `K=2`, ResNet32x4 -> ResNet8x4 | 已完成 | 240 / 240 | 77.05%（第228轮） | 94.78%（第217轮） | 76.90% / 94.53% | `output/cifar100_baselines/crld_proto,res32x4,res8x4,structure_k2` |

输出目录保留 `student_best`、`worklog.txt` 和 TensorBoard 事件日志；训练进程已结束。
