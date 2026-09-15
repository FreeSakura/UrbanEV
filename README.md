# UrbanEV Forecast

[![CI](https://github.com/FreeSakura/UrbanEV/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FreeSakura/UrbanEV/actions/workflows/ci.yml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Docs: CC BY 4.0](https://img.shields.io/badge/docs-CC_BY_4.0-green.svg)](licenses/CC-BY-4.0.txt)

城市公共充电占用预测研究项目，围绕 **275 个区域的小时占用率**，提供预测模型、评价合同、实验报告、公开结果与历史审计论文。

本项目由 FreeSakura 维护，独立于 [UrbanEV 数据集官方项目](https://github.com/IntelligentSystemsLab/UrbanEV)。主要问题是：滞后观测和长历史带来多少预测信息，动态结构能否在同信息强基线之上产生可重复的收益？

## 从这里开始

| 你希望了解 | 入口 |
|---|---|
| 安装并跑通一个例子 | [快速开始](docs/guides/quickstart.md) |
| 当前完成了什么、尚缺什么 | [项目状态](docs/PROJECT_STATUS.md) |
| 查看数值、原始汇总及证据来源 | [结果总览](results/README.md) |
| 理解全部研究方向 | [研究地图](docs/research/README.md) |
| 阅读报告、理论和论文 | [文档中心](docs/README.md) · [完整资料目录](docs/catalog.md) |
| 重现结果或继续开发 | [复现指南](docs/guides/reproducibility.md) · [贡献指南](CONTRIBUTING.md) |

## 当前结果

已完成的统一**开发集比较**包含两种 ridge 和五种神经配置；在该比较的四个视野上，RIDGE_OD 的平均 RMSE 为 **0.111146795**，MAE 为 **0.068494595**，低于其余核心系统。数据来自一个已曝光的开发窗口，信息轨道和训练方式有差异，适用范围见[可重建结果表](results/README.md#开发集核心比较)。

完整六折比较已登记 **72 次 ridge 拟合、432 次神经训练**及两个本地基础模型。当前主分支的公开材料包含注册清单，尚无该轮完整执行和统一评分回执；计划工作量不能当成完成成绩。研究尚未建立 SOTA 证据。[状态依据](docs/PROJECT_STATUS.md)

历史负结果、合成验证及旧审计结果分别保留，按各自协议解释。完整研究叙述见[综合报告](docs/reports/PROJECT_REPORT.md)。

## 快速验证

Python 3.10 或更高版本。在仓库根目录执行：

```bash
python -m pip install -e ".[test]"
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
src/               预测与历史审计 Python 包
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
