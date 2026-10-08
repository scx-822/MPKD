# 暂停运行记录

更新时间：2026-09-24

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
python tools/train_proto.py --cfg configs/cifar100/mpkd/wrn40_2_wrn16_2.yaml --resume --seed 42 EXPERIMENT.TAG mpkd,nkd,nkd1,wrn40_2_wrn16_2 EXPERIMENT.METHOD NKD-MPKD EXPERIMENT.BASELINE NKD MPKD.BASE_METHOD NKD
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
python tools/train_proto.py --cfg configs/cifar100/mpkd/wrn40_2_wrn40_1.yaml --resume --seed 42 EXPERIMENT.TAG mpkd,dkd,wrn40_2_wrn40_1 EXPERIMENT.METHOD DKD-MPKD EXPERIMENT.BASELINE DKD MPKD.BASE_METHOD DKD
```

不要删除或改名 `output/cifar100_compatibility/mpkd,dkd,wrn40_2_wrn40_1/`，否则 `--resume` 无法读取当前检查点。

第五组完成后运行第六组：

```powershell
python tools/run_mpkd_compatibility.py --architectures vgg13_vgg8 --methods DKD --seed 42
```
