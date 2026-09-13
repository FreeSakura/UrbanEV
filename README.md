# UrbanEV Forecast

[![CI](https://github.com/FreeSakura/UrbanEV/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FreeSakura/UrbanEV/actions/workflows/ci.yml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

研究城市公共充电设施的多变量占用预测：从可复现强基线出发，探索滞后观测、驻留动态与简洁预测状态。主要研究对象为UrbanEV的275区域小时数据。本仓库不是[UrbanEV官方仓库](https://github.com/IntelligentSystemsLab/UrbanEV)。

**当前SOTA状态：证据尚未建立。** 原因是尚缺完整、最新且同协议的比较，不是未达到自行设置的1%提升。自2026-09-13本轮整合起，旧1%改善、MAE零恶化及单元损害门仅用于解释历史实验，不作为新的统一研究门或SOTA定义。已有NO_GO保留为原协议结论，不等于“该信息无效”或“国际认定失败”。

## 研究入口

| 内容 | 入口 |
|---|---|
| 如何判断SOTA与如何报告研究价值 | [比较与声明规则](docs/research/sota/SOTA_COMPARISON_STANDARD.md) |
| 已有什么结果，哪些还不能下结论 | [研究成果地图](docs/research/sota/RESEARCH_INDEX.md) |
| 下一步研究和当前执行范围 | [研究路线](docs/research/sota/RESEARCH_ROADMAP.md) |
| 官方锚点、现代强基线与待核论文 | [基线登记](docs/research/sota/BASELINES.md) |

## 本轮推进：先对齐真实比较对象

[官方代码核查与已执行探针](docs/research/sota/BENCHMARK_ALIGNMENT_REPORT_20260913.md)发现：上游最终指标取第H小时末点，旧开发协议评分未来1至H全部点；启动脚本历史12小时，旧研究168小时；数据归档还经历了占用比例到占用桩数的更新。不能直接把这些数字排成同一排行榜。

新增评分接口强制声明`terminal_H`或`path_1_to_H`及raw/clip规则；已计算24个日历/原点合同单元，并完成42个已曝光原点的真实开发桥接。H3/H12共同支持上，Chronos原生raw末点RMSE为0.089446，全路径为0.081448；last/day基线在两种口径下的RMSE排名发生反转。这是同口径基线研究，尚非官方完整测试或SOTA；零新增拟合/基础推理，旧候选未重选，保护数据未打开。

## 当前科学问题

本轮[短摘要与条件弛豫真实实验](docs/research/sota/SHORT_STATE_RELAXATION_REPORT_20260913.md)已完成8次神经训练、2次ridge拟合：主候选未优于同信息直接模型；长历史DIRECT/RIDGE分别较短历史降低四H末点RMSE约11.21%/15.62%。长ridge接近native的H3/H12 RMSE，但MAE更差。**保留长历史研究线索，不主张单模弛豫优势或SOTA**；全部结果返回人工审核，没有恢复旧收益门。

**容量约束、隐藏驻留阶段和滞后累计观测下，什么简洁摘要足以预测未来占用？** [理论报告](docs/research/sota/COARSE_MEMORY_THEORY_REPORT_20260913.md)整合物理记忆、经济选择及交通调度思想。[合成机制检查](docs/research/sota/COARSE_MEMORY_IDENTIFIABILITY_REPORT_20260913.md)显示，在一个固定慢动态人工系统中，完整历史对纯占用历史的RMSE增量为1.6010%，对短双观测摘要仅0.2468%，且等于已知参数HMM过滤。它支持问题建模，尚未建立新方法优势或UrbanEV成绩。

研究继续围绕可比基线、预测状态及实际增量展开。每轮完成后返回人工审核；定时任务暂停，不自动进入下一轮。


## 开始研究

以下训练示例使用既有开发入口与path_1_to_H口径，不是已对齐官方的SOTA复现命令。新比较规则和执行范围以研究入口为准。

在仓库根目录，使用Python 3.10或更高版本：

```bash
python -m pip install -e .[test,research]
python -m urbanev_forecast smoke --model innovation_attention --epochs 2 --output local-data/smoke
pytest
```

`smoke`生成合成周期序列并执行训练/验证，用于检查代码，不产生真实UrbanEV成绩。无需下载数据或大模型。

取得数据授权后，先准备第一折训练/验证前缀：

```bash
python -m urbanev_forecast.prepare --data-root local-data/UrbanEV --hours 648 --output local-data/urbanev-rates.csv
python -m urbanev_forecast train --csv local-data/urbanev-rates.csv --model seasonal_linear --fold 1 --horizon 3 --epochs 30 --output local-data/seasonal-f1-h3-s42
python -m urbanev_forecast train --csv local-data/urbanev-rates.csv --model timexer --fold 1 --horizon 3 --epochs 30 --output local-data/timexer-f1-h3-s42
```

`prepare`将占用数量除以对应区域容量；输入应为上游`occupancy.csv`和`inf.csv`。`train`只读取选定折的训练/验证前缀，保存最佳验证checkpoint、配置、数据与代码哈希。输出目录必须为空，目标与checkpoint仅留在本地。可用`--device cuda`启用现有GPU。

在方法和协议冻结后，单独使用`test`评价完整测试前缀：

```bash
python -m urbanev_forecast test --csv local-data/urbanev-rates-complete.csv --checkpoint local-data/seasonal-f1-h3-s42/checkpoint.pt --output local-data/seasonal-f1-h3-s42-test
```

该步骤要求CSV覆盖所选折的完整时间范围，并核对训练/验证前缀及当前代码与checkpoint一致。测试输出不用于选模型。详见[路线与运行顺序](docs/research/sota/RESEARCH_ROADMAP.md)。

## 代码与研究资产

| 位置 | 用途 |
|---|---|
| `src/urbanev_forecast/` | 预测模型、数据窗口、训练/测试入口 |
| `configs/research/` | 新研究协议 |
| `docs/research/sota/` | 研究路线、基线清单、实验与评审 |
| `models/timexer/` | 保留原许可证和溯源的TimeXer实现 |
| `artifacts/summaries/` | 公开汇总结果及历史证据 |
| `src/urbanev_audit/` | 数据身份、指标、隐私和工件验证工具 |
| `paper/` | 理论报告及历史审计论文 |

## 历史成果与公开边界

[原审计项目入口](docs/history/AUDIT_PROJECT_README.md)、[历史论文](paper/main/UrbanEV_Evidence_Audit_Main.pdf)及其[引用信息](docs/history/AUDIT_CITATION.cff)完整保留。历史负结果不会因项目转型被改判。旧`python -m urbanev_audit`命令继续可用。

仓库公开代码、协议、汇总与理论，不上传原始目标、缺失mask、私有路径或模型checkpoint。当前署名使用FreeSakura；旧提交和标签仍可能包含历史署名，详见[公开署名与隐私范围](docs/PUBLIC_ATTRIBUTION.md)。Paris formal/protected材料不属于本轮标准UrbanEV预测研究的输入。

原创代码采用MIT许可证；论文和原创文档采用CC BY 4.0；第三方源码与数据遵循各自许可证。运行环境依赖不会自动下载基础模型权重。引用本研究软件请使用[CITATION.cff](CITATION.cff)。
