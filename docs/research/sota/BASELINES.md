# 基线登记：来源、复现与可比性

更新：2026-09-13。本表是基线登记，**不是SOTA排行榜**。`paper_reported`表示文献值，`locally_reproduced`表示指定本地协议实际运行，`adapted`表示额外训练/后处理，`not_run`表示尚未运行；可比性另列，不能由名称或论文地位替代。

## 官方锚点与最新相关工作

| 方法或来源 | 身份 | 可比性状态 | 下一处理 |
|---|---|---|---|
| UrbanEV论文Table 3：TimeXer及统计/深度基线 | peer_reviewed / paper_reported | 本地尚未完整复现，末点/全路径、历史和原点差异已核实 | 作为原始基准锚点，不视为永久SOTA |
| Time Series Foundation Models as Strong Baselines in Transportation Forecasting，arXiv:2602.24238v2 | preprint / paper_reported；包含UrbanEV与Chronos-2 | 上下文168；忙桩数量与本项目占用率需区分，完整协议待桥接 | 纳入现代基础模型证据，不直接混排 |
| DyConfuse-Net，Electric Power Systems Research 2026 | peer_reviewed / paper_reported | unresolved：本轮只有出版商摘要及元数据，未复现完整设置 | 必须跟进，不能忽略也不能按摘要小数值直接排名 |
| TriCast，Pattern Recognition Letters 2026 | peer_reviewed / paper_reported | mismatch_documented：预览写247区、5分钟、2022年6—7月 | 独立相关任务，非本项目275区小时结果 |
| Urban-CSTPNet，Electronics 2026 | peer_reviewed / paper_reported | unresolved：概率目标及数据/视野需逐项核查 | 保留待核，不当作已经可比 |

来源：[UrbanEV论文](https://doi.org/10.1038/s41597-025-04874-4)、[官方代码固定版本](https://github.com/IntelligentSystemsLab/UrbanEV/tree/44f2aa0c8d89f192bce00bafb0def74a21b39c68)、[交通基础模型预印本](https://arxiv.org/abs/2602.24238v2)、[DyConfuse-Net](https://doi.org/10.1016/j.epsr.2026.112765)、[TriCast](https://doi.org/10.1016/j.patrec.2026.04.028)、[Urban-CSTPNet](https://doi.org/10.3390/electronics15153297)。截至检索日未建立完整维护的可比排行榜；本表也不声称穷尽全部文献。

官方论文TimeXer表3的文献RMSE为H3/H6/H9/H12：0.0832/0.0938/0.0989/0.0939，文中平均0.0924；对应MAE为0.0471/0.0566/0.0620/0.0589，平均0.0561。**不将这些文献值与本地局部开发分数直接比较**；也不从单节点或外生因素表挑更低数值拼成一行。

## 本地方法与运行状态

| 系统 | 本地身份与状态 | 比较范围 |
|---|---|---|
| Last/day/week | 确定性、无训练；本轮V2桥接按实际回执登记 | 42个已曝光开发原点，四视野，端点/全路径并列 |
| Chronos-2 native Q0.5 | 既有冻结预测，局部locally_reproduced | V2只复用H3/H12缓存；raw与clip明确分行，H6/H9不补推理 |
| TimesFM-3 native Q0.5 | 既有首折开发运行 | 本轮不读取其缓存；完整同协议测试缺失 |
| TimeXer本地适配 | 代码接通，历史紧凑实现另有结果 | 当前完整同协议复现not_run；不能把同名旧配置当新复现 |
| Seasonal Linear / MLP | 已实现、合成工程检查通过 | 本轮无拟合；完整真实比较not_run |
| Full-quantile ridge、bias/affine、simplex头 | adapted，历史开发已运行 | 同信息控制有效，但旧局部表不构成前沿榜 |
| HMM/状态空间/隐半马尔可夫动态模型 | 经典机制对照 | 已知人工参数HMM核对不等于真实训练；真实研究not_run |
| Innovation/Level Attention | 实现为机制探针 | 尚无成熟方法优势；不预设其新颖性 |

当前bridge表中的排序只适用于它列出的基线和共同支持，不代表覆盖上表全部强方法。现代方法的原生零样本输出与监督适配系统须分别标明预算；预测API名称不自动等于条件均值，见[输出语义](POINT_FORECAST_CONTRACT.md)。

## 比较与发表口径

SOTA声明遵循[统一比较规则](SOTA_COMPARISON_STANDARD.md)。不足1%不自动淘汰；RMSE优但MAE不优应报告权衡；无完整同协议基线与泛化证据则限定为开发结果。模型新意、实际价值与数值领先分开讨论。

历史数值与各阶段NO_GO集中见[成果地图](RESEARCH_INDEX.md)，旧原文及配置保留。后续每轮人工审核，不自动训练或扩展测试数据。
