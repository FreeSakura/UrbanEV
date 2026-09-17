# UrbanEV Forecast

**完整论文工作稿已形成。** [英文正文、完整证明与中文摘要](paper/persistent_events/manuscript.md) · [Word构建与端到端复现](paper/persistent_events/README.md)。核心为尖锐结构定理A，B为有经典来源的几何推论；完整cover-LP和ILP已匹配全部8,352个冻结自然比较。以下AP2叙述保留阶段背景，昌平案例现在由完整cover-LP解释。

[![CI](https://github.com/FreeSakura/UrbanEV/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FreeSakura/UrbanEV/actions/workflows/ci.yml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Docs: CC BY 4.0](https://img.shields.io/badge/docs-CC_BY_4.0-green.svg)](licenses/CC-BY-4.0.txt)

本仓库保存预测研究的代码、数学推导、实验协议和公开证据。当前研究**共享缺测下持续事件预测的模型比较**；此前围绕 **275 个区域的小时充电占用率**开展的 UrbanEV 预测研究及历史审计论文也在此维护。

本项目由 FreeSakura 维护，独立于 [UrbanEV 数据集官方项目](https://github.com/IntelligentSystemsLab/UrbanEV)。

## 当前研究：共享缺测下的模型比较

当重叠预警窗口共享缺失观测时，不同窗口的未知标签不能任意组合。研究比较独立标签界、两两一致性约束和完整轨迹优化，确定已有观测是否足以判定两个固定预测器的优劣。

**AP2 已完成（2026-09-17）。** 固定 AP1 模型、零新拟合，在预留年份评价 4,176 个自然比较：新增 17 个周级与 1 个站点年度判向；北京 K24 首次出现 5 个周级新增判断。1 个昌平 K72 案例确实需要高阶一致性，已用独立 MILP 核对。AP1/AP2 剩余的 154/141 个未定比较均有相反胜负的合法补全见证。功率与跨站全年聚合仍无新增判向。

- [最新结构理论](docs/theory/PERSISTENT_EVENT_COVER_STRUCTURE_20260917.md) · [完整证明包](PROOF_PACKAGE.md)：长窗口覆盖完整性、两两充分性几何条件，以及覆盖 LP 的边界。
- [AP2 完整报告](docs/reports/audit/SHARED_MISSING_EVENTS_AP2_REPORT.md) · [最新主张与后续任务](docs/research/SHARED_MISSING_EVENTS_AP2_CLAIMS.md) · [方法/结果初稿](docs/research/SHARED_MISSING_EVENTS_AP2_METHOD_RESULTS_DRAFT.md)
- [AP1 完整报告](docs/reports/audit/SHARED_MISSING_EVENTS_AP1_REPORT.md) · [AP1 历史主张](docs/research/SHARED_MISSING_EVENTS_AP1_CLAIMS_LOCK.md)
- [AP1 理论与边界](docs/theory/SHARED_MISSING_EVENTS_AP1_THEORY.md) · [公开数值证据](artifacts/summaries/shared_missing_events_ap1)
- [AP0 首轮报告](docs/reports/audit/SHARED_MISSING_EVENTS_AP0_REPORT.md) · [精确比较推导](docs/theory/SHARED_MISSING_EVENTS_COMPRESSION.md)

当前支持有限面板的比较证书与适用边界研究，尚未建立普遍模型选择收益或 JCR Q2 投稿成熟度。下一步进行贡献核对与写稿，不自动追加年份、模型或阈值；旧六折不是新方向的前置条件。

## 从这里开始

| 你希望了解 | 入口 |
|---|---|
| 安装并跑通一个例子 | [快速开始](docs/guides/quickstart.md) |
| 当前完成了什么、尚缺什么 | [项目状态](docs/PROJECT_STATUS.md) |
| 查看数值、原始汇总及证据来源 | [结果总览](results/README.md) |
| 理解全部研究方向 | [研究地图](docs/research/README.md) |
| 阅读报告、理论和论文 | [文档中心](docs/README.md) · [完整资料目录](docs/catalog.md) |
| 重现结果或继续开发 | [复现指南](docs/guides/reproducibility.md) · [贡献指南](CONTRIBUTING.md) |

## 原 UrbanEV 预测研究结果

已完成的统一**开发集比较**包含两种 ridge 和五种神经配置；在该比较的四个视野上，RIDGE_OD 的平均 RMSE 为 **0.111146795**，MAE 为 **0.068494595**，低于其余核心系统。数据来自一个已曝光的开发窗口，信息轨道和训练方式有差异，适用范围见[可重建结果表](results/README.md#开发集核心比较)。

完整六折比较已登记 **72 次 ridge 拟合、432 次神经训练**及两个本地基础模型。当前主分支的公开材料包含注册清单，尚无该轮完整执行和统一评分回执；计划工作量不能当成完成成绩。研究尚未建立 SOTA 证据。[状态依据](docs/PROJECT_STATUS.md)

历史负结果、合成验证及旧审计结果分别保留，按各自协议解释。完整研究叙述见[综合报告](docs/reports/PROJECT_REPORT.md)。

## 快速验证

Python 3.10 或更高版本。在仓库根目录执行：

```bash
python -m pip install -e ".[test,evidence]"
python scripts/repository.py check
python -m pytest
```

运行不依赖真实数据的 CPU 预测示例：

```bash
python -m pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test,research]"
python -m urbanev_forecast smoke --model innovation_attention --epochs 2 --output local-data/smoke
```

`smoke` 只生成合成序列，用于检查训练流程。输出目录可预先创建，已有结果文件不会被覆盖。Windows 虚拟环境、真实数据准备及各复现层级见[快速开始](docs/guides/quickstart.md)。

日常迭代采用[轻量开发流程](CONTRIBUTING.md)，代码身份变化会记录在评价结果中，不再因无关源码改动阻塞 checkpoint 评价。

## 仓库结构

```text
src/               预测、共享缺测评价与历史审计 Python 包
scripts/           实验、资料索引、论文构建与工件校验入口
tests/             评价、模型、复现与仓库维护测试
configs/           固定的协议、模型配置和环境记录
results/           综合结果、研究登记表和可重建资料清单
artifacts/         原始公开汇总、执行回执与完整性清单
docs/              指南、研究主线、分类报告、理论和历史资料
paper/             历史论文、补充材料、理论 PDF 与可编辑源码
models/            带许可证及来源记录的模型源码快照
release-assets/    外部 Release 工件的说明和校验元数据
```

数值证据保存在 `artifacts/summaries/`，结果总览由这些文件生成；报告按主题归档。[架构说明](docs/guides/architecture.md)解释各模块和入口之间的关系。

## 数据、许可证与引用

真实数据需按上游条款自行取得。仓库提供代码、协议、汇总和来源哈希；数据、目标数组和权重保存在本地。参见[数据指南](docs/guides/data.md)、[数据许可](docs/DATA_LICENSES.md)及[公开边界](docs/PRIVACY_BOUNDARY.md)。

原创代码采用 MIT，原创文档和论文采用 CC BY 4.0；第三方源码沿用各自许可证。引用软件使用 [CITATION.cff](CITATION.cff)，引用历史审计论文使用[论文引用信息](docs/history/AUDIT_CITATION.cff)。
