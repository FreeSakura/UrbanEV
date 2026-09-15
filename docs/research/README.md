# 研究地图

研究目标是解释并改善城市充电占用预测。区分三个问题：额外观测是否有信息、怎样表示历史、具体模型是否优于同信息控制。下表按问题组织，不把不同实验的改善幅度混成排行。

| 研究问题 | 已有认识 | 主要记录 |
|---|---|---|
| 如何比较模型 | 单位、原点、末点/path 和训练选择必须对齐 | [评价合同](evaluation.md)、[比较桥接](../reports/benchmark/BENCHMARK_ALIGNMENT_REPORT_20260913.md) |
| 强参考是什么 | 统一开发比较中普通 RIDGE_OD 是重要参考；完整六折成绩待补 | [结果总览](../../results/README.md)、[基线登记](baselines.md) |
| 更长历史是否有用 | 长 O 与长 D 在指定开发设置有增量；短摘要未表现出充分性 | [长历史来源](../reports/forecasting/LONG_HISTORY_SOURCE_REPORT_20260913.md)、[短摘要实验](../reports/forecasting/SHORT_STATE_RELAXATION_REPORT_20260913.md) |
| 交互与动态结构是否必要 | 存在有限 RMSE 线索，但跨种子稳定性、MAE 和普通控制限制机制主张 | [条件交互](../reports/forecasting/CONDITIONAL_OD_INTERACTION_REPORT_20260913.md)、[稳定化](../reports/forecasting/REFERENCE_HARM_STABILITY_REPORT_20260913.md) |
| 纠错能否迁移 | 前向方案选择零修正；普通线性纠错有局部收益，专用交互未形成稳定优势 | [前向迁移](../reports/forecasting/FORWARD_RESIDUAL_TRANSFER_REPORT_20260913.md)、[固定底座](../reports/forecasting/FIXED_REFERENCE_CORRECTION_REPORT_20260913.md) |
| 输出语义与风险权衡 | 分位输出适配和连续步长有局部收益及代价，同信息控制仍重要 | [校准报告](../reports/calibration/)、[输出语义](../theory/POINT_FORECAST_CONTRACT.md) |
| 空间、时长、电量信息 | 一些初步信号未复现；特定结构的失败不证明变量普遍无效 | [信息研究](../reports/information/) |
| 理论如何连接数据 | 可观测性和粗粒化机制给出假设与边界；合成证明不等于真实预测优势 | [理论目录](../theory/README.md)、[人工机制检查](../reports/mechanisms/COARSE_MEMORY_IDENTIFIABILITY_REPORT_20260913.md) |
| 调用前成本决策 | 仅完成惰性分支与成本计数合成检查 | [条件调用准备](../reviews/ADAPTIVE_INVOCATION_REVIEW_20260914.md) |

全部证据目录、阶段状态、配置与对应报告见[机器可读登记](../../results/studies.json)；逐项文件见[证据清单](../../results/inventory.csv)。

历史 NO_GO 表示对应版本内部规则下的阶段决定。自 2026-09-13 起，旧 1% 改善、MAE 零恶化等不再作为普遍晋级门；旧实验结果和决定仍保留，不追改为新赢家。

当前总状态见[项目状态](../PROJECT_STATUS.md)，后续工作见[研究路线](roadmap.md)。旧 `FINAL_PROPOSAL` 等文件已归入[历史方案](../history/proposals/)，文件名中的“最终”只指当时阶段。
