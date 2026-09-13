# 从权威期刊学习问题驱动的方法设计

检索与核验日期：2026-09-13。重点是学习作者如何构造科学问题，不按期刊名称推断方法在UrbanEV上有效。本表覆盖能源、统计与预测领域的权威期刊；没有为“顶刊”提供统一排名，也不把预印本或会议混入期刊成果。本文是定向阅读，不是穷尽式系统综述。

## 来源、版本与阅读范围

下列8篇论文及1篇勘误均完成Crossref身份核验；出版社或作者原文用于核对方法。技能批量核验工具缺失，曾输出UNVERIFIED降级记录，随后使用独立的Crossref DOI查询逐项核验；未将缺失工具伪记为通过。核验元数据见[来源清单](../../../artifacts/summaries/journal_methods_20260913/source_verification.json)。摘要核验不等于全文复现。

| 文献 | 正式出版与版本 | 本轮阅读深度 | 身份状态 |
|---|---|---|---|
| Powell、Cezar、Min、Azevedo、Rajagopal：Charging infrastructure access and operation to reduce the grid impacts of deep electric vehicle adoption | Nature Energy 7, 932–945 (2022)，[原文](https://www.nature.com/articles/s41560-022-01105-7) | 出版社正文、Methods中的行为条件依赖和情景设定 | verified / Crossref |
| Zhang等：Behavioral uncertainty in EV charging drives heterogeneous grid load variability under climate goals | Nature Communications (2026)，[原文](https://www.nature.com/articles/s41467-025-66796-4) | 出版社检索可见Methods，式(2)–(6)；直接打开遇重定向限制 | verified / Crossref |
| Li、Zhang、Doel、Ross、Piggott：Deep learning predicts real-world electric vehicle direct current charging profiles and durations | Nature Communications (2025)，[原文](https://www.nature.com/articles/s41467-025-65970-y) | 出版社Results、Methods、损失式(4)–(6)和时长式(7) | verified / Crossref |
| Shang、Li、Li、Li：Explainable spatiotemporal multi-task learning for electric vehicle charging demand prediction | Applied Energy 384, 125460 (2025)，[原文](https://www.sciencedirect.com/science/article/pii/S0306261925001904) | 出版社摘要、引言与贡献预览；没有完整复核全部实验 | verified / Crossref |
| Lim、Arık、Loeff、Pfister：Temporal Fusion Transformers for interpretable multi-horizon time series forecasting | International Journal of Forecasting 37(4), 1748–1764 (2021)，[期刊](https://doi.org/10.1016/j.ijforecast.2021.03.012) | 期刊摘要及[作者预印本](https://arxiv.org/html/1912.09363)的方法、输入分类和§6.6消融；不假定版本逐字相同 | verified / Crossref |
| Gneiting：Making and Evaluating Point Forecasts | JASA 106(494), 746–762 (2011)，[期刊](https://doi.org/10.1198/jasa.2011.r10138) | 出版社摘要及[作者预印本](https://arxiv.org/abs/0912.0902)的决策论定义；不将PDF内部日期当期刊出版年 | verified / Crossref |
| Rothenhäusler、Meinshausen、Bühlmann、Peters：Anchor Regression: Heterogeneous Data Meet Causality | JRSS-B 83(2), 215–246 (2021)，[原文](https://academic.oup.com/jrsssb/article/83/2/215/7056043) | 作者公开期刊PDF§1–2、出版社§6局限；另核对[勘误](https://academic.oup.com/jrsssb/article/83/5/1071/7056048) | verified / Crossref；勘误也已核验 |
| Wickramasuriya、Athanasopoulos、Hyndman：Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization | JASA 114(526), 804–819 (2019)，[作者页面](https://robjhyndman.com/publications/mint/) | 作者摘要、书目及作者教材中的闭式公式；未重跑原论文 | verified / Crossref |

Zhang等论文DOI包含2025，但正式在线日期为2026-01-06。MinT于2018年在线、2019年编入卷期。来源清单分别保存online/print日期，避免把DOI年份或首次在线年份与卷期混用。

## 问题如何变成方法

### 1. 先拆解需求的生成过程：Powell等与Zhang等

Powell等面对的矛盾是：直接外推早期车主充电习惯，难以代表未来不同充电可及性的群体。作者先建立“地区与可及性→行为群体→充电会话→系统负荷”的条件依赖，再改变基础设施和行为情景，比较电网影响。这是机制化情景建模，不是小时占用预测SOTA。[Nature Energy原文](https://www.nature.com/articles/s41560-022-01105-7)

Zhang等将行为变化拆为群体比例与群体内部特征分布：`P(s,G)=P(G)P(s|G)`；情景中改变群体比例，保持组内分布。这种“指定什么会变、什么暂时不变”的做法比笼统声称模型适应漂移更明确。其组内稳定是假设，UrbanEV没有对应用户会话特征，不能直接复制群体识别。[Nature Communications原文](https://www.nature.com/articles/s41467-025-66796-4)

**对本项目的启发是待检验推断：** 跨折收益改变可能来自状态组成改变，也可能来自同一状态下的条件关系改变。两者需分开观察；当前没有证据确认哪一种主导。按未来误差分组只能解释已有损失，不能据此部署“有利时才修正”的规则。

### 2. 先找可以预测、又能推导目标的中间对象：Li等

真实充电曲线受车型、初始SoC和环境影响，固定理想曲线难以描述。作者从部分功率–SoC曲线预测后续曲线，再通过容量与功率的积分得到时长，结合异常识别与量化不确定性。关键是把目标分解为可学习对象及明确的物理映射；实验进一步比较已知前缀长度和不同测试集。[原文式(7)](https://www.nature.com/articles/s41467-025-65970-y)

**迁移边界：** 本项目的区域累计时长不是单车剩余服务时间，缺少会话SoC和功率轨迹。不能把“占用快照−归一化时长”直接命名为到达压力或真实潜状态。旧观测算子路线失败后，只有新的可识别变量或独立观测才能支持更强物理主张。

### 3. 先确定信息如何可用，再安排模块与对照：Shang等与TFT

Shang等将区域占用、充电量和时长视为相关任务，通过时间GraphSAGE与共享表示联合预测，另用遮蔽与Shapley分析贡献。它是本项目直接相关的先例，说明“多图＋多任务＋解释性”本身不是我们的新增贡献。其全文实验尚未在本轮完整核查，不引用无法核对的改善百分比。[Applied Energy预览](https://www.sciencedirect.com/science/article/pii/S0306261925001904)

TFT从静态、未来已知、历史观测三类输入出发设计模块。在变量选择消融中，把样本相关权重换成固定可训练系数，同时保留变量非线性处理。这种对照有助于区分“条件选择有效”与“额外非线性处理有效”，值得移植到未来候选的实验设计；不等于再造一个TFT就有新颖性。[作者预印本§6.6](https://arxiv.org/html/1912.09363)

### 4. 先明确估计对象，再解释指标冲突：Gneiting

作者把预测任务形式化为统计泛函与一致评分函数的匹配问题：平方损失对应条件均值，绝对损失对应条件中位数。方法研究因此应先问预测什么，再问用什么网络。加权损失会改变所求对象，不能只因指标改善就忽略其含义。[JASA原文](https://doi.org/10.1198/jasa.2011.r10138)

**对本项目的启发是待检验推断：** 原生Q0.5可能已接近中位数，而残差平方损失修正更接近均值；这可以解释部分冲突，但我们尚未证明基础预测是真实条件中位数。RMSE与MAE共同改善是否可行，取决于基础误差、可用信息及修正方向，不能靠无限缩小步长保证。

### 5. 先定义允许的分布变化，再推导鲁棒目标：Anchor Regression

作者指出OLS可能过度依赖训练分布，完全因果不变性又可能太保守，于是明确一类由外生anchor诱导的线性位移，再导出带投影惩罚的回归目标。保证只覆盖规定的位移集合；时间块可以提供环境划分，但不会自动满足线性和外生假设。勘误修正了Proposition 1与Theorem 3的排版连接，引用时应同时保留。[原文](https://academic.oup.com/jrsssb/article/83/2/215/7056043)与[勘误](https://academic.oup.com/jrsssb/article/83/5/1071/7056048)

**迁移边界：** 可以借鉴“先定义哪类漂移需要保护”的思路，不能将一个时间块惩罚直接命名为因果稳健预测，也不能把线性MSE保证外推到非线性、裁剪后的宏RMSE和零MAE恶化约束。

### 6. 先发现不可识别对象，再换成可估计对象：MinT

MinT指出旧协调误差协方差在实践中的识别困难，转而利用预测误差协方差，在无偏假设与线性加总约束下求最小迹协调。这条创新路径是“找到不能估的量→重写目标→给闭式解→模拟与实证检验”，并非给已有优化器增加精度。[作者原文说明](https://robjhyndman.com/publications/mint/)

**迁移边界：** 区域占用数量在覆盖一致时具有加总关系，占用率则需要容量权重；时长与占用没有自动成立的线性加总等式。若上层预测只是底层预测的机械求和，没有额外独立信息，就不能假设协调会带来收益。此方向仅作为理论思维参照，本轮不新增协调模型。

## 对研究方式的具体改变

新候选应先说明：它解释哪一项已有失败；哪个量在预测时确实可见；哪个命题在何种假设下成立；哪个同信息对照能否定机制。理论恒等式、依赖假设的保证和待检验研究猜想分别标记。本文不把任何文献的作者结论换算为本项目增益，也不把当前NO_GO改判。

## 另列：近期直接竞争工作，不以期刊等级排除

以下两篇补充工作均已Crossref核验，不计入前述8篇方法论精读。IEEE Access不在本文“顶刊方法论”定位中，但与候选高度重叠的先例不能因此忽略。

| 工作 | 已核对的机制 | 对我们的新颖性约束 |
|---|---|---|
| Waqar、Kim、Byun，CUP-EV，Information Sciences (2026)，[出版社](https://www.sciencedirect.com/science/article/abs/pii/S0020025526006122)；verified / Crossref | 线性分支、残差MLP、可学习混合、分位预测与成本评价；出版社Methods与数据说明可见 | 轻量残差头与概率校准已有先例；其目标为小时电量，数据为Jiaxing/Palo Alto，不能拿其结果当275区占用榜。卷期为2026年10月，检索时正文已可见，首次在线日期未确认。 |
| Hwang、Suh，PAR-LLM，IEEE Access 14, 52360–52370 (2026)，[作者机构摘要](https://research.knu.ac.kr/en/publications/par-llm-peak-aware-residual-llm-over-anchor-for-ev-charging-stati/)；verified / Crossref | 数值LSTM基础预测、历史波动风险定位、选择性LLM残差修正、差异相关混合与回退保护 | “识别高风险时段再修正”已经存在。摘要报告120区、三个视野，与本项目协议不同。本轮未获取完整IEEE正文，尚不能确认其全部理论或对照范围；不可断言它缺少某个未读机制。 |

因此，未来条件修正候选必须明确超出一般门控、风险定位与残差混合的贡献，并在相同信息和预算下比较。不应以“没有LLM”或“加了安全约束”作为独立新颖性论据。
