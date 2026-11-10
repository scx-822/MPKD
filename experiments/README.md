# 实验资料索引

这里集中放置实验结果、复现协议、逐轮日志和论文材料。训练代码仍在仓库根目录的 `tools/`，数据、权重和输出不放入本目录。

## 先看什么

| 需求 | 文件 |
| --- | --- |
| 了解如何运行实验 | `PROTOCOL.md` |
| 查看结果、状态和逐轮历史 | `log.md` |
| 查看暂停实验和续训命令 | `PAUSED_RUNS.md` |
| 查看资源是否齐全 | `resources_manifest.json` |
| 查看鲁棒性图表 | `robustness/` |

## 结果口径

- `log.md` 统一保存可核验的结构化结果、实验背景、参数解释、历史尝试和输出目录。
- 1 epoch 冒烟、失败运行和未采用的候选参数不进入正式主结果表。
- 文献基线、本地复现、历史审计和未跑实验必须明确区分。

## 文件命名

- 协议和结果使用英文大写文件名，便于检索和脚本引用。
- 论文材料保留中文文件名，避免破坏已有引用。
- 配置使用 `teacher_student.yaml` 格式。
- 输出目录使用 `<method>,<teacher>,<student>[,<tag>]` 格式。

旧版的 `RUN_ORDER.md`、`EXPERIMENT_MATRIX.md`、`ABLATION_RESULTS.md` 和 `COMPATIBILITY_PROTOCOL.md` 已合并到 `PROTOCOL.md` 与 `log.md`，不再维护重复版本。
