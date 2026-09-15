# 文档中心

项目分为预测研究和历史证据审计两部分。第一次阅读建议按“项目状态 → 综合报告 → 结果总览 → 复现指南”的顺序进入。

| 主题 | 内容 |
|---|---|
| 项目概况 | [当前状态](PROJECT_STATUS.md)、[研究综合报告](reports/PROJECT_REPORT.md) |
| 安装与使用 | [快速开始](guides/quickstart.md)、[复现指南](guides/reproducibility.md)、[数据准备](guides/data.md) |
| 代码与维护 | [架构和命令](guides/architecture.md)、[贡献指南](../CONTRIBUTING.md)、[维护规则](maintenance/README.md)、[代码与流程精简](maintenance/DEFENSIVE_CODE_REVIEW.md) |
| 科学问题 | [研究地图](research/README.md)、[下一步路线](research/roadmap.md)、[比较合同](research/evaluation.md)、[基线登记](research/baselines.md) |
| 实验结果 | [结果总览](../results/README.md)、[研究与证据登记](../results/studies.json)、[分类报告](reports/README.md) |
| 理论与协议 | [理论目录](theory/README.md)、[阶段协议目录](protocols/README.md) |
| 评审与历史 | [评审记录](reviews/README.md)、[历史资料](history/README.md)、[论文版本](../paper/README.md) |
| 许可与披露 | [数据许可](DATA_LICENSES.md)、[模型身份](MODEL_IDENTITY.md)、[公开边界](PRIVACY_BOUNDARY.md)、[署名](PUBLIC_ATTRIBUTION.md)、[AI 使用](AI_ASSISTANCE.md) |

[完整资料目录](catalog.md)覆盖仓库中的全部 Markdown 和 PDF；[证据文件清单](../results/inventory.csv)逐项登记公开汇总的所属研究、文件大小与哈希。两者均可通过 `python scripts/repository.py build` 重建。

日期报告是对应阶段的记录；其中的“当前”“下一步”和执行限制适用于报告写作时点。现在的主线以[项目状态](PROJECT_STATUS.md)为准。旧路径迁移关系见[路径映射](maintenance/path-map.json)，历史版本可通过记录中的原提交访问。
