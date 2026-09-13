# 研究成果地图

更新：2026-09-13。这里按科学问题组织成果；历史数值、原协议判定和SOTA证据分列。旧NO_GO不被删除，也不再被当成全局科学失败或SOTA裁决。新的比较依据见[SOTA比较规则](SOTA_COMPARISON_STANDARD.md)。

## 一、方法与理论积累

| 方向 | 已获得的证据 | 原协议结论 | 当前研究含义与SOTA范围 |
|---|---|---|---|
| 分位输出适配 | 两骨干共享单纯形头相对原生Q0.5有RMSE开发收益，但同信息ridge更强且MAE有代价 | 首折双骨干NO_GO | 留下均值/中位数语义与同信息基线；尚无同协议最新方法比较。[报告](FM_CALIBRATION_REPORT.md) |
| 跨区结构 | 首折相关分组有线索；第二折未复现，残差引导交换不及对照 | 双方向与复核未通过 | 特定分组方法缺乏稳定优势，不证明空间信息无效。[初试](DUAL_DIRECTION_PILOT_REPORT.md)、[复核](SIGNAL_REPLICATION_REPORT.md) |
| 观测算子 | 额外结构未优于直接时长输入；直接时长开发收益存在 | 结构优势未成立 | 不支持把潜状态解释当成额外贡献，保留输入线索。[报告](DUAL_DIRECTION_PILOT_REPORT.md) |
| 时长残差与连续步长 | 部分RMSE改善伴MAE代价；V2两视野合法非零修正改善RMSE0.3896%且不高于native MAE | 旧内部1%/结构门NO_GO | 低于1%并非国际失败依据；范围仍限开发窗口，不能补写SOTA。[V1](RESIDUAL_INFORMATION_REPORT.md)、[V2](CONTINUOUS_V2_CALIBRATION_2H_REPORT.md) |
| 双风险局部方向 | 两窗口出现局部共同下降线索 | 探针准入 | 局部导数不保证有限收益，不能作真实预测SOTA比较。[探针](DUAL_RISK_PROBE_REPORT_20260913.md) |
| 双风险有限步长 | 对强非时长对照RMSE改善0.185943%，同信息硬门更好 | 有限步长NO_GO | 保留权衡和强对照结论；不证明新门控优于经典同信息方法。[有限步长](DUAL_RISK_FINITE_STEP_REPORT_20260913.md) |
| 动态电量条件增量 | 动态列确有变化；固定ridge正网格点均恶化MAE，主候选网格最佳RMSE改善0.444636% | 校准NO_GO，未开1392评价 | 保留这一表示的负结果，不推断电量或其他变量普遍无效。[报告](VOLUME_COMPOSITION_INCREMENT_REPORT_20260913.md) |
| 粗粒化记忆与交通启发 | 固定人工系统下完整历史对I2改善1.6010%，对I3仅0.2468%；同信息HMM一致 | 合成机制检查完成 | 是观测充分性与时间尺度线索，非真实预测优势；没有新模型SOTA证据。[理论](COARSE_MEMORY_THEORY_REPORT_20260913.md)、[结果](COARSE_MEMORY_IDENTIFIABILITY_REPORT_20260913.md) |

**这些行没有共同的实验总体，不能相互按百分比排行。** 已有结果多数来自开发/校准范围，完整最新基线比较仍缺。停止一轮试验可以是预算决策；SOTA判断必须另给可比证据。旧claim继续消费状态，不以本次调整为由重跑或改写旧网格。对应[机器可读八方向映射](../../../artifacts/summaries/benchmark_alignment_20260913/research_results.csv)分别记录原门、证据和SOTA适用范围。

## 二、当前统一入口

[底座正增损稳定化](REFERENCE_HARM_STABILITY_REPORT_20260913.md)完成六方案、三个新种子、18次神经训练。主方法相对原MSE平均RMSE/MAE改善0.0834%/0.6728%，但仍差于ridge底座，成绩标准差增加；普通MAE混合与通用MLP控制限制了专用稳定化解释。**减少增损的局部作用存在，整体稳定性主张未成立，无SOTA声明。**

[条件O–D乘性交互真实实验](CONDITIONAL_OD_INTERACTION_REPORT_20260913.md)完成冻结ridge底座上的四个等参数模型、八次GPU训练。主模型两种子平均四H末点RMSE较底座降低0.410%，但MAE增加0.922%，种子结果分化；平均RMSE低于三个神经对照。**保留交互结构的有限开发线索，不宣称稳定优势或SOTA。**

最新[时长历史有序正则化](DURATION_LAG_REGULARIZATION_REPORT_20260913.md)：以同信息普通收缩和错序同谱控制检验二阶平滑。13组拟合完成，未支持真实时间邻接的特定预测优势；惩罚生效不等于机制成立，长D增量旧结论保持。

最新[长历史来源拆分](LONG_HISTORY_SOURCE_REPORT_20260913.md)：8个固定ridge消融揭示长D在给定长O后仍有开发增量，且超过本轮重复表示控制。没有将系数、有效自由度或重复窗口误当作总体信息或独立确认；旧锚点重建不算新性能发现。

最新[短摘要与条件弛豫真实模型研究](SHORT_STATE_RELAXATION_REPORT_20260913.md)：六模型族、10次拟合完成，单模短OD弛豫优势未成立；长历史在直接MLP和ridge中有开发收益，但没有确认SOTA。它是粗粒化问题进入真实参数学习后的新阶段，不回写前述人工机制结果。

- [比较口径的实质缺口与本轮探针](BENCHMARK_ALIGNMENT_REPORT_20260913.md)：末点/全路径、上下文、原点、数据版本。
- [研究路线](RESEARCH_ROADMAP.md)：当前范围、下一科学问题与前瞻协议。
- [基线登记](BASELINES.md)：上游复现锚点、现代基线与近期相关论文。
- [预测输出语义](POINT_FORECAST_CONTRACT.md)：接口名称不自动等于条件均值。

`FINAL_PROPOSAL.md`、`EXPERIMENT_PLAN.md`及各日期协议是对应历史阶段的设计记录，不能从文件名推断它们仍是当前总方案。完整原文留在原路径以保持引用和复现关系。

## 三、工程与论文资产

预测代码在`src/urbanev_forecast/`；公开配置在`configs/research/`；经脱敏的数值与回执在`artifacts/summaries/`。历史审计工具在`src/urbanev_audit/`，历史论文在`paper/`，它们是支撑积累，不是当前标准占用预测的SOTA论文。源码、模型权重、数据版本与训练预算分开登记。

[信息地图](INFORMATION_EVIDENCE_MAP_20260913.md)的21字段/8类信息登记与[历史审计成果](../../history/AUDIT_PROJECT_README.md)单列为支撑资产。没有证明其他信息均无效，也不把事件、覆盖或审计指标当作占用预测RMSE。

每轮完成后由人工审核决定下一轮，不启动定时或自动实验。公开只包含代码、协议、推导与汇总；原始序列、私有预测和权重不上传。
