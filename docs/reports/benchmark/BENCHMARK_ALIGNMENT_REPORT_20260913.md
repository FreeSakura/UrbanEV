# UrbanEV比较口径核查与本轮研究推进

> 阶段记录：下文保留原研究时点的结论和执行范围；当前公开状态见[项目状态](../../PROJECT_STATUS.md)。目录迁移不改变原结果。

日期：2026-09-13。公开上游代码锁定为`44f2aa0c8d89f192bce00bafb0def74a21b39c68`。先读取13份源码/脚本文本并完成无真实数据的日历/指标探针，随后按独立冻结V2执行一次真实开发桥接。**已产生同口径开发基线表，尚非新方法或官方测试成绩**；没有执行上游训练代码、拟合模型或调用基础模型。

## 1. 比较对象尚未对齐，不能用内部阈值代替

SOTA比较首先需要同一个预测对象。官方代码的测试评分与本项目旧开发协议存在以下差异；这些是指定源码版本的事实，不冒称已复现论文全部实验。

| 项目 | 上游指定代码 | 本项目既有开发协议 | 后续处理 |
|---|---|---|---|
| 目标时间 | Transformer `metric`取`pred[:,-1,:]`；classical创建单个第H小时目标 | 对未来1至H全部点评分 | 分开命名`terminal_H`与`path_1_to_H`；禁止直接排名 |
| 历史长度 | classical默认12；Transformer运行脚本显式12，覆盖CLI默认96 | 168小时 | 锁定实际调用参数；同历史对照才能归因方法差异 |
| 验证/测试原点 | Transformer允许前段上下文，末一合法窗口纳入 | 类似目标完整包含划分的规则 | 仍须逐原点、区域顺序匹配，不仅比较折名 |
| classical原点 | 每个划分内独立构窗，舍弃前12小时作为上下文；range未含最后一个合法窗 | 可借用划分前历史 | 复现上游与统一比较需要分别标注 |
| 折边界取整 | classical与Transformer算法略有差异 | 跟随Transformer式边界 | 不把“80/10/10”文字当成完全相同索引 |
| 输出裁剪 | 读取到的上游RMSE/MAE评分未裁剪 | 以clip到[0,1]作为主报告 | 新接口强制声明raw或clip；训练/后处理都是方法组成 |
| 训练预算 | 所读Transformer exp.sh中EPOCH=1，classical exp.sh中20 | 旧研究各自预算 | 当前脚本不是论文最佳实验身份；不能只击败一轮训练便声称击败最佳TimeXer |

证据：[上游末点指标](https://github.com/IntelligentSystemsLab/UrbanEV/blob/44f2aa0c8d89f192bce00bafb0def74a21b39c68/code-transformer/utils/metrics.py#L60)、[classical构窗与划分](https://github.com/IntelligentSystemsLab/UrbanEV/blob/44f2aa0c8d89f192bce00bafb0def74a21b39c68/code/utils.py#L160)、[Transformer划分](https://github.com/IntelligentSystemsLab/UrbanEV/blob/44f2aa0c8d89f192bce00bafb0def74a21b39c68/code-transformer/data_provider/data_loader.py#L62)、[实际启动脚本](https://github.com/IntelligentSystemsLab/UrbanEV/blob/44f2aa0c8d89f192bce00bafb0def74a21b39c68/code-transformer/exp.sh)。

上游Transformer训练损失可覆盖全部H步，而最终测试指标取末点；训练目标和报告指标必须分别记录。这里没有将上游实现差异判作论文不可信，也没有修改其代码或把修订版冒充原版。

## 2. 数据版本不能由文件名确定

Dryad列出2025-03-17、2025-04-25、2025-09-24、2026-02-04版本；变更记录明确将小时`occupancy.csv`由比例改为占用桩数。其部分变量说明及GitHub README仍写占用率，代码则按容量作除法。因此，前瞻协议要记录归档版本、文件哈希、原始单位和转换操作，防止对已经是比例的数据重复除容量。本轮没有重新下载归档或读取数据来证明本地内容与某个归档逐字节相同。[Dryad版本与变更记录](https://datadryad.org/dataset/doi:10.5061/dryad.np5hqc04z)

## 3. 已执行的可比性探针

新增[显式评分接口](../../../src/urbanev_forecast/benchmark_metrics.py)，调用时必须写目标范围和后处理；保持旧训练入口与冻结结果不变。输入统一为原点×视野×区域，异常形状、空数组和非有限值拒绝评分。五项针对性测试通过。

用两个固定说明性预测A=(0.6,0.6,0)、B=(0.3,0.3,0.3)，真值全零，得到：

| 评分范围 | A的RMSE | B的RMSE | 较低者 |
|---|---:|---:|---|
| 第3小时末点 | 0 | 0.3 | A |
| 第1至3小时全路径 | 0.489898 | 0.3 | B |

这只证明不同评分对象可以反转排名，**没有重算或挑选本项目候选成绩**。

另依据公开索引表达式和完整小时日历，计算6折×4视野的原点范围。例如第1折H3，classical验证末点为索引590至646，共57个；Transformer为578至647，共70个。第3折的classical验证结束索引为1965（不含），Transformer为1966（不含）。以上是日历/源码推算，不代表读取了这些受保护时段，更不证明数据无缺时。

[完整探针结果](../../../artifacts/summaries/benchmark_alignment_20260913/contract_probe.json)、[源码身份](../../../artifacts/summaries/benchmark_alignment_20260913/upstream_source_manifest.json)、[可复现入口](../../../scripts/research/probe_benchmark_alignment.py)。所有原始训练结果和旧claim保持不变。

## 4. 最新可比方法的研究入口

检索到与UrbanEV相关的2026工作，必须追踪，不能仍把2025表格当成当前完整前沿；也不能只按论文题名或小数值将它们排到同一表中。

| 来源 | 本轮可核查范围 | 比较处理 |
|---|---|---|
| UrbanEV，Scientific Data 2025 | 原论文与指定官方代码；6个扩展月折和3/6/9/12小时任务 | 复现锚点，非永久SOTA榜首 |
| DyConfuse-Net，Electric Power Systems Research 2026 | 出版商摘要及Crossref；摘要报告UrbanEV结果 | 完整划分、粒度、目标和实现待核；列待核候选，不宣称已超过它 |
| TriCast，Pattern Recognition Letters 2026 | 出版商预览写247区域、5分钟、2022-06-19至07-18 | 与本项目275区域小时任务不匹配，不直接混排；保留其方法研究价值 |
| Urban-CSTPNet，Electronics 2026 | 正式发表元数据核验；已有研究登记为概率目标，完整对齐未建立 | 维持待核，不能由文章标题推定同协议 |

来源：[UrbanEV](https://doi.org/10.1038/s41597-025-04874-4)、[DyConfuse-Net](https://doi.org/10.1016/j.epsr.2026.112765)、[TriCast出版商方法预览](https://www.sciencedirect.com/science/article/pii/S0167865526001510)、[Urban-CSTPNet](https://doi.org/10.3390/electronics15153297)。元数据登记见[近期论文记录](../../../artifacts/summaries/benchmark_alignment_20260913/recent_paper_metadata.json)。本轮检索不声称穷尽全部文献，未获得全文的方法不记为已复现。

## 5. 一次真实开发桥接结果

配置[V2](../../../configs/research/URBANEV_COMPARISON_V2.json)于提交`2154cb8f5904fcfdc0629fe243ed406f50c51d86`冻结后执行一次，状态为**COMPARABILITY_BRIDGE_COMPLETE_REVIEW_REQUIRED**。42原点为cuts720/1056/1392各14个，步长12小时；这是已曝光开发集合，不是官方六折测试或独立盲测。

仅读取occupancy[0,1560)、duration[0,1547)、静态容量所需三列及12份既有native/truth缓存。零拟合、零基础推理、零alpha搜索、零旧候选重选。1392用于本次新的基线桥接，不恢复旧实验的条件评价。实际耗时22.88秒，峰值内存未测量。

下表为**同一H3/H12共同支持、三个窗口共六单元等权平均**，不能与四视野平均或论文表3直接比较。

| 系统 | 末点RMSE | 末点MAE | 全路径RMSE | 全路径MAE |
|---|---:|---:|---:|---:|
| Last | 0.129825 | 0.076593 | 0.110484 | 0.055823 |
| Day | 0.122793 | 0.074500 | 0.123167 | 0.074843 |
| Week | 0.141872 | 0.093798 | 0.142680 | 0.094840 |
| Chronos原生raw Q0.5 | 0.089446 | 0.051398 | 0.081448 | 0.043156 |
| 同一Q0.5＋clip[0,1] | 0.089440 | 0.051377 | 0.081440 | 0.043121 |

这里出现真实的口径排名变化：**末点RMSE是Day优于Last，全路径RMSE则是Last优于Day**。同一个模型的两列也不能称为“改善”，因为评分对象变了。Chronos在这张有限开发表中误差较低，但对照只有三个确定性基线，不能据此声称SOTA。裁剪的微小变化属于明确后处理，不是新模型贡献。

在另列的四视野确定性基线表中，Day末点宏RMSE为0.121678，低于Last的0.123562；但Day末点MAE为0.074347，高于Last的0.071390。V2保留这一权衡，不因MAE稍差自动判NO_GO。没有对这些开发点估计进行显著性声明，窗口和区域不能视作独立随机重复。

有效评分96行，native H6/H9缺失状态24行。没有用H12切片冒充对应原生调用。源目标与缓存truth、原native历史path评分、独立直接归约均经过核对，详见[检查结果](../../../artifacts/summaries/comparability_bridge_v2/alignment_checks.json)；一致性不是新的独立成绩。

私有输入包已生成：短双观测42×275×3、长占用与长duration各42×168×275、H12标签42×12×275及原点/容量/日历。duration额外滞后1小时，不把快照减累计量解释为队列或会话年龄。公开仅含[包哈希与布局](../../../artifacts/summaries/comparability_bridge_v2/private_package_manifest.json)，原始数据包未上传。

[逐单元比较表](../../../artifacts/summaries/comparability_bridge_v2/comparison_table.csv) · [共同支持汇总](../../../artifacts/summaries/comparability_bridge_v2/summary.json) · [执行回执](../../../artifacts/summaries/comparability_bridge_v2/protocol_receipt.json)

## 6. 对后续研究的直接影响

应先建立可复现的同协议比较，再研究容量约束与滞后双观测摘要是否改善预测。新研究可以主张特定指标的改善，不再要求其超过自设1%或每个次指标都同步改善；但必须说明真实比较范围、训练预算与不确定性。仅满足旧内部预算门，也不能代替这条证据链。

本轮已经把比较对象落到真实开发基线与输入包。下一问题是短摘要/长历史及动态结构是否在指定预测时距提供超出同信息普通模型的增量，且应与现代强基线对齐。真实训练与保留测试没有自动开启；完整后续决定以[统一研究路线](../../research/roadmap.md)为准。定时保持暂停，每轮返回人工审核，未使用额度重置卡。
