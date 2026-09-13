# UrbanEV Forecast

[![CI](https://github.com/FreeSakura/UrbanEV/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FreeSakura/UrbanEV/actions/workflows/ci.yml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

**面向 UrbanEV 标准多变量预测的研究项目，目标是突破可复现强基线与 SOTA。** 主任务是275个区域的小时占用率预测，视野为3、6、9、12小时，以RMSE为主指标、MAE共同报告。当前尚未取得SOTA结果。

这不是UrbanEV数据集官方仓库。原始数据与官方方法见 [IntelligentSystemsLab/UrbanEV](https://github.com/IntelligentSystemsLab/UrbanEV)。本项目从证据审计转向预测方法研究：新模型、强基线、实验和结果是主线，既有审计工具承担验证工作。

## 动态电量条件增量（2026-09-13）

[新电量实验](docs/research/sota/VOLUME_COMPOSITION_INCREMENT_REPORT_20260913.md)已执行：以总时长和历史静态功率关系为条件，比较区域匹配的动态电量差与重复/错位对照。5个普通ridge在固定网格中全部回退α=0，35个正步长点均使宏MAE恶化，状态为**VOLUME_INCREMENT_CALIBRATION_NO_GO**，未进入1392评价。动态差11列均保留非零训练尺度，不能把本次失败解释为电量完全冗余或所有表示无效。未使用旧门控方向，也未切换价格、天气或11kW版本。

## 有限步长实测（2026-09-13）

[双风险有限步长实验](docs/research/sota/DUAL_RISK_FINITE_STEP_REPORT_20260913.md)已按新协议执行一次：720拟合4模型，1056从固定八点网格选步；主候选α=4，相对最佳非时长对照的RMSE改善仅0.185943%，MAE也未满足全部对照要求，同信息硬门控更好。状态为**FINITE_STEP_CALIBRATION_NO_GO**，1392没有数值解码。误差穿越抵消了约72%的局部MAE收益。旧方向探针正线索不改写，旧NO_GO不改判；无SOTA，未追加步长或打开保留数据。

## 信息有效性与下一研究路线（2026-09-13）

**没有证明“除时长之外的其他信息均无效”。** [信息证据地图](docs/research/sota/INFORMATION_EVIDENCE_MAP_20260913.md)区分基础输入、历史探索、特定方法失败与当前条件增量未验证；本轮完成8类信息、21字段的元数据登记，未读取新时序行或开展新拟合。历史存在价格/天气实验，不等于已在当前强基线下排除其价值。

[进一步理论研究](docs/research/sota/INFORMATION_VALUE_AND_NEXT_ROUTE_20260913.md)明确理想信息价值与算法成绩的区别，并推导有限步长的误差穿越增损项。研究顺序为：本次登记完成后，先冻结现有双风险方向的有限步长验证，再安排公平的其他信息筛选。时长曾作为优先线索，其他信息不被预先排除；后续一次有限步长阶段已执行并在校准门停止。随后已执行上述动态电量条件增量实验；价格、天气等筛选仍未执行。

## 文献驱动的新研究（2026-09-13）

[期刊阅读笔记](docs/research/sota/JOURNAL_READING_NOTES_20260913.md)与[理论研究报告](docs/research/sota/JOURNAL_METHOD_REPORT_20260913.md)将主问题改为：额外时长信息能否产生RMSE–MAE共同改进。一次冻结的[条件双风险方向探针](docs/research/sota/DUAL_RISK_PROBE_REPORT_20260913.md)已完成：8次小型线性拟合、36份既有数组、零新增基础推理；时长表示在两个前向开发窗口均给出共同下降方向，达到机制设计准入。**这不是≥1%有限步长改善或SOTA，旧V1/V2 NO_GO不变。** 尚未训练新点预测方法或打开保留数据；定时任务继续暂停。

## 阶段性收尾（2026-09-10）

**9月10日阶段已结束，自动跟进已暂停。** [阶段总结](docs/research/sota/PHASE_CLOSEOUT_20260910.md)集中记录各方向成果、失败原因、最终数值与恢复条件。当时完成阶段收尾；9月13日按用户要求另起文献研究与有界探针，完整SOTA比较尚未建立。

## 已完成的执行结果

[连续步长V2两视野校准](docs/research/sota/CONTINUOUS_V2_CALIBRATION_2H_REPORT.md)已按限定授权实际执行一次：正交时长α≈0.09796，宏RMSE改善0.3896%，MAE不高于native，但低于1%信息门；结构门也未通过。全部计算有效，阶段为**CALIBRATION_2H_INFORMATION_NO_GO**。没有补H6/H9或打开尾部、第三折验证/测试。

[连续步长V2阶段注册](docs/research/sota/CONTINUOUS_STEP_V2_REGISTRATION.md)已固定时间划分、七系统、主候选和推进门，并完成缓存与索引核验。求解器已通过代码验收并完成一次限定H3/H12连续校准；尾部评分和新增基础推理未执行，原V1失败结论不变。

[残差信息筛选报告](docs/research/sota/RESIDUAL_INFORMATION_REPORT.md)：更深的[理论推导](docs/research/sota/RESIDUAL_DERIVATION_PACKAGE.md)已转为实验，完成三个训练内滚动窗口。正交时长修正在α=0.5时RMSE改善1.48%，但MAE恶化1.28%；全部系统在注册网格下回退α=0，信息门与结构门均未通过。第三折验证段未打开。后验导数揭示更小步长可能有局部MAE改善，该线索随后进入上述V2连续校准，仍未通过门，不改判旧门。

[第二折信息线索复核](docs/research/sota/SIGNAL_REPLICATION_REPORT.md)：简单相关分组未复现首折增益，RMSE比全区域Chronos差0.42%。滞后时长在ridge中改善1.86%，但直接添加至Chronos仅改善0.082%，未过1%门；两类预测器联合复现未成立。随后完成上述残差信息V1与连续步长V2实验。

[双方向试验报告](docs/research/sota/DUAL_DIRECTION_PILOT_REPORT.md)：区域残差引导交换未通过，RMSE比最佳相关分组对照差1.73%；观测算子的额外优势也未成立。追加优化后，直接加入滞后时长相对占用输入的RMSE/MAE下降2.86%/3.54%，仅保留为开发线索。原始门、后验诊断和失败原因分别保留，尚无独立确认或SOTA结果。

[基础模型校准开发报告](docs/research/sota/FM_CALIBRATION_REPORT.md)：Chronos-2与TimesFM-3均完成275区域联合推理及H3/H12训练/验证实验。单纯形头在两个骨干上均为**NO_GO**；相对原始Q0.5的RMSE改善约2.18%/2.37%，但没有建立相对同信息ridge的实质优势，MAE也上升。停止扩建该候选，保留校准与ridge为强对照。完整测试与SOTA仍未建立。

## 最新选题判断（2026-09-10）

[文献与研究方向报告](docs/research/sota/RESEARCH_DIRECTIONS_20260910.md)提出的两个方向已完成首折试验、第二折信息复核及残差修正训练内筛选。当前难点是把RMSE收益转化为满足MAE约束的可外推改进；已有失败门保持不变。

## 当前研究主线

- [研究目标与路线](docs/research/sota/RESEARCH_ROADMAP.md)：标准预测任务、现代强基线和可否定的机制研究。
- [当前完整方案与评审](docs/research/sota/FINAL_PROPOSAL.md)：五轮自动评审后转入实验，尚未达到成熟SOTA方法阈值。
- [实验协议](configs/research/URBANEV_SOTA_V1.json)：任务与划分口径、种子和推进门槛。
- [基线清单与结果状态](docs/research/sota/BASELINES.md)：区分文献结果、历史工件、尚未运行的候选，避免跨协议混排。
- [理论积累](docs/research/PAIRED_AUDIT_V3_REPORT.md)：V1–V3关于风险、事件与缺失评价的研究；这部分不是标准预测SOTA成绩。

[有界分位点预测适配](docs/research/sota/QUANTILE_MEAN_THEORY.md)的数值核心与开发实验已完成；[理论反馈](docs/research/sota/FM_CALIBRATION_THEORY_FEEDBACK.md)解释稳健中点和RMSE–MAE权衡的边界。本候选没有通过原定推进门。

首轮实现提供季节线性、季节MLP、创新注意力和原始水平注意力探针，以及现有TimeXer源代码适配。探针用于验证机制，尚未确立方法优势或新颖性。Chronos-2和TimesFM-3已接通本地冻结推理与分位缓存；首折开发对照完成，六折完整基线与测试比较尚未完成。

## 开始研究

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
