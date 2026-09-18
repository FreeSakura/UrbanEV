# Persistent Event Evaluation

[![CI](https://github.com/FreeSakura/UrbanEV/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FreeSakura/UrbanEV/actions/workflows/ci.yml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Docs: CC BY 4.0](https://img.shields.io/badge/docs-CC_BY_4.0-green.svg)](licenses/CC-BY-4.0.txt)

**部分观测下持续事件标签的结构与预测评价。** 当前成果为一项尖锐结构定理、一个有经典来源的几何判据，以及两轮冻结实证。英文完整稿已形成，含完整证明附录和中文摘要。

**[Word 全稿](paper/persistent_events/WORD_MANUSCRIPT.docx) · [英文稿源](paper/persistent_events/manuscript.md) · [复现说明](paper/persistent_events/README.md) · [结果与证据](results/README.md)**

Research code for *Persistent Event Evaluation under Partial Observations — A Sharp Structural Regime for Overlapping Labels*. Start with the [English manuscript and reproduction guide](paper/persistent_events/README.md).

## 当前研究

长度为 $K$ 的滑动窗口包含连续 $L$ 个1时，事件标签为正。重叠窗口共享未知输入，因此标签不能独立补全。核心结论是：当 $K\geq2L-1$ 时，对任意固定部分二元观测，确定标签、一元蕴含和两点覆盖约束完整描述所有可实现的二元标签；这一统一阈值是尖锐的。

- **核心定理 A：** 输出像的精确结构及长短窗口分界。
- **几何判据 B：** 基于 A 与经典并封闭集合族的原子/主状态理论，直接从观测检查一元蕴含是否充分。
- **冻结实证：** 检查结构差异何时产生风险端点差异、何时进一步改变模型判断。

两点覆盖涉及三个标签变量，不能与两两蕴含混称。完整二元描述不等于连续 LP 的一般整数性；不主张新通用 DP 或新的多面体原理。[完整证明](PROOF_PACKAGE.md) · [贡献边界](docs/reviews/PERSISTENT_EVENT_FROZEN_CONTRIBUTION_20260917.md)

## 已有结果

| 固定范围 | 数量与结论 |
|---|---|
| 全部自然评价 | 2,784 个面板，8,352 个模型比较 |
| 含模糊标签 | 470 个面板，1,410 个比较 |
| 一元蕴含结构 | 323 个面板精确，147 个不完整 |
| 超出一元蕴含的差异 | 110 个比较端点更紧，1 个增加严格判向 |
| 完整 cover-LP / cover-ILP | 在全部冻结自然目标上与轨迹 DP 端点数值一致 |
| 剩余未定比较 | 295 个均有严格相反胜负的合法补全见证 |

昌平案例由完整 cover-LP 恢复严格判断，不能再据此声称必须使用轨迹 DP。当前目标上的数值一致不推出 cover-LP 对任意权重都精确。周级、年度及同一面板内的模型比较也不是独立重复。[从冻结文件生成的结果表](results/README.md) · [完整 cover 回执](artifacts/summaries/persistent_event_complete_covers_20260917/STRUCTURE_VALIDATION.json)

## 五分钟检查

Python 3.10 或更高版本，在仓库根目录执行；不需要原始数据、PyTorch 或 GPU。

```bash
python -m pip install -e ".[test,evidence]"
python scripts/repository.py check
python scripts/repository.py verify-manifest
python scripts/research/validate_persistent_event_covers.py --selfcheck --output local-data/cover-check
```

请使用新的输出目录。该检查核对构造实例和求解实现，不生成新的真实预测成绩。

| 任务 | 入口 |
|---|---|
| 从头安装环境 | [快速开始](docs/guides/quickstart.md) |
| 下载 UCI 数据并重建 AP1/AP2 | [完整复现步骤](paper/persistent_events/README.md#rebuild-the-two-fixed-evaluations-from-public-data) |
| 用已有预测缓存核对风险界 | [缓存与重新训练的区别](docs/guides/reproducibility.md) |
| 重建图表和可编辑 Word | [稿件构建](paper/persistent_events/README.md#regenerate-manuscript-figures-and-word) |
| 定位主张与证据 | [证据映射](paper/persistent_events/EVIDENCE_MAP.md) |
| 查看当前状态与历史阶段 | [项目状态](docs/PROJECT_STATUS.md) · [研究地图](docs/research/README.md) |

## 代码与资料

```text
paper/persistent_events/  当前完整稿、Word、图表和复现说明
src/urbanev_audit/        事件几何、约束层与轨迹优化，也保留旧审计模块
scripts/research/        当前事件复现与历史研究入口
artifacts/summaries/     冻结公开数值、见证与执行回执
results/                 自动生成的当前结果和全部研究登记
docs/                    使用指南、当前贡献与阶段历史
src/urbanev_forecast/    历史 UrbanEV 预测工程
```

[文档中心](docs/README.md) · [脚本索引](scripts/README.md) · [完整资料目录](docs/catalog.md) · [贡献与 CI 规则](CONTRIBUTING.md)

## 历史 UrbanEV 工作

仓库保留原充电占用预测、六折协议和审计论文，原数据结果与路径继续有效。它们不构成当前持续事件论文的附加模型对照，也不作为新读者的安装前置任务。[历史预测使用说明](docs/guides/forecasting.md) · [历史研究地图](docs/research/README.md#历史充电预测研究) · [论文归档](paper/README.md#历史审计论文与理论报告)

仓库由 FreeSakura 维护，独立于 [UrbanEV 数据集官方项目](https://github.com/IntelligentSystemsLab/UrbanEV)。为保持已有链接和 Python 导入兼容，仓库地址与包名 `urbanev-forecast` 不变。

## 数据与引用

原始数据、预测缓存与模型在本地生成；公开仓库提供代码、固定配置、汇总和来源记录。[数据许可](docs/DATA_LICENSES.md) · [公开边界](docs/PRIVACY_BOUNDARY.md)

原创代码采用 MIT，原创文档和论文采用 CC BY 4.0；第三方材料保留原许可。引用研究软件使用 [CITATION.cff](CITATION.cff)。当前论文仍是匿名完整工作稿，没有虚构期刊、作者身份或 DOI；历史审计论文另用[对应引用记录](docs/history/AUDIT_CITATION.cff)。
