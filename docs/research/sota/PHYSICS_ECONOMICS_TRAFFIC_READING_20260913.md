# 物理学、经济学与交通调度：来源与迁移边界

日期：2026-09-13。研究对象仍为 UrbanEV 小时占用率预测；本次阅读没有产生新的真实数据预测成绩。书目元数据通过直接 Crossref DOI 接口核验，阅读层级另记；元数据核验不等于全文精读或方法复现。检索时使用的 ARIS 专用验证工具不可用，因此不声称该工具通过。以下为执行侧来源核查，路线结论见后续整合报告。

## 十篇核心来源

| 来源 | 学科与阅读层级 | 可学习的机制 | 迁移条件与风险 |
|---|---|---|---|
| Zwanzig (1961), *Memory Effects in Irreversible Thermodynamics*, Physical Review 124:983–992. [原文入口](https://journals.aps.org/pr/abstract/10.1103/PhysRev.124.983) | 统计物理；官方摘要，正文需订阅 | 消去微观变量后，宏观演化可能包含过去状态的记忆 | 不能从该结论推出某个短记忆模型胜过现有168小时历史预测器；没有复核全文证明 |
| Mori (1965), *Transport, Collective Motion, and Brownian Motion*, Progress of Theoretical Physics 33:423–455. [原文](https://doi.org/10.1143/PTP.33.423) | 理论物理；官方摘要 | 投影后的随机力、记忆和宏观动力学共同出现 | EV不是热平衡分子体系，涨落耗散关系不能直接充当充电系统定律 |
| Lei, Baker & Li (2016), *Data-driven parameterization of the generalized Langevin equation*, PNAS 113:14183–14188. [正文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5167214/) | 综合期刊中的计算物理；方法公式与假设段 | 在拉普拉斯域参数化记忆核，并用辅助状态实现有限维动态系统 | 原文从平衡时间序列统计估计核，并假定平均力已知；UrbanEV没有对应物理力和温度。可借鉴状态实现，不能宣称识别了物理核 |
| Little (1961), *A Proof for the Queuing Formula: L=λW*, Operations Research 9:383–387. [原文](https://doi.org/10.1287/opre.9.3.383) | 运筹学；原始摘要与证明前提 | 系统内人数、到达率和逗留时间的长期平均关系 | 不是逐小时反演公式；累计桩时不等于单会话平均逗留时间，忙碌人数也不自动包含等待者 |
| Eick, Massey & Whitt (1993), *The Physics of the M_t/G/∞ Queue*, Operations Research 41:731–742. [作者论文](https://www.columbia.edu/~ww2040/physics.pdf)；[DOI](https://doi.org/10.1287/opre.41.4.731) | 运筹学，不能因标题叫Physics而归为物理顶刊；摘要及作者论文局部 | 非平稳到达经过服务时间分布形成滞后的在服人数 | 无限服务台假设需与有限充电容量区分；只见在服人数时，到达和服务寿命不能自动联合识别 |
| Naor (1969), *The Regulation of Queue Size by Levying Tolls*, Econometrica 37:15–24. [原文](https://www.jstor.org/stable/1909200) | 经济学；期刊目录与正文开篇条件 | 个体加入队列的私人收益与社会拥堵成本不同；收费可改变进入行为 | 必须有可拒绝进入、等待成本及服务模型。检索中的“The Optimization of Queueing Systems by Means of Toll Charges”不是本篇真实标题，不采用 |
| Berry, Levinsohn & Pakes (1995), *Automobile Prices in Market Equilibrium*, Econometrica 63:841–890. [期刊DOI](https://doi.org/10.2307/2171802)；[学术机构论文副本](https://www.its.caltech.edu/~mshum/gradio/papers/BerryLevinsohnPakes1995.pdf) | 经济学；引言、需求聚合、外部选项及识别假设段 | 从异质偏好聚合出替代关系，并处理价格与未观测质量相关的问题 | 占用率不是购买份额；缺市场规模、外部选项、选择集合与有效外生变异时，不能直接估需求弹性 |
| Daganzo (1994), *The cell transmission model: A dynamic representation of highway traffic consistent with the hydrodynamic theory*, Transportation Research B 28:269–287. [DOI](https://doi.org/10.1016/0191-2615(94)90002-7) | 交通运输；期刊元数据、作者机构1993年前身报告摘要 | 以守恒及发送/接收能力描述拥堵传播与消散 | 1993年报告与1994年期刊版本分开标注；区间地理邻近不能替代实际车辆流向，充电站不是道路元胞 |
| Varaiya (2013), *Max pressure control of a network of signalized intersections*, Transportation Research C 36:177–195. [DOI](https://doi.org/10.1016/j.trc.2013.08.014)；[作者上传正文](https://www.researchgate.net/publication/259138901_Max_pressure_control_of_a_network_of_signalized_intersections) | 交通控制；模型定义、守恒、主要稳定性结论及适用边界 | 利用相邻队列、转向比例与饱和流率选择服务动作，稳定可服务需求 | 原模型点队列无存储上限。它不需要外部平均需求预测，却仍需要队列与服务参数；控制稳定性不能转写为预测RMSE保证 |
| Powell, Cezar, Min, Azevedo & Rajagopal (2022), *Charging infrastructure access and operation to reduce the grid impacts of deep electric vehicle adoption*, Nature Energy 7:932–945. [原文](https://www.nature.com/articles/s41560-022-01105-7) | 能源系统；摘要与公开文章信息 | 把基础设施可达性和充电运行方式分开，研究二者如何共同塑造充电负荷 | 情景干预和电网影响不等于固定数据集预测成绩；充电能量/电力负荷与忙碌占用并非同一目标 |

Daganzo前身报告的明确来源为[UC Berkeley ITS 1993报告页](https://its.berkeley.edu/node/2283)。其摘要支持守恒交通演化的启发，不能用来声称已核对1994全文全部公式。对可读性受限的论文，本轮不声称全文精读。

## 将三条参照线用于同一个问题

物理学提醒我们检查状态描述是否充分：聚合后的历史依赖可能来自被省略的服务进程。经济学提醒我们检查观测过程：到达需求可能被价格、等待和容量共同筛选。交通调度提醒我们检查演化约束：服务完成和接收能力决定占用如何变化。这些是不同机制，不能凭同一列占用数据分别拟合潜变量后，就认定三套解释都成立。

尤其要区分三个量：正在占用资源的车辆数、等候进入的队列、尚未实现的潜在到达需求。UrbanEV忙碌桩数仅直接对应第一类近似观测。满载时观察到的平台既可能是需求稳定，也可能是容量截断；没有到达/拒绝/等待记录，不能判定后者已发生。

电动公交排程适合在已知线路、时刻表、车队和充电动作下研究如何安排服务。公共充电预测没有这些已知输入，故不能把公交排程优化结果作为当前预测基线。它更适合指导未来应采集什么信息，以及如何单独设计调度价值评价。

## 对既有实验的约束

动态电量条件增量实验的35个正步长均恶化宏MAE，主候选网格内最佳RMSE改善也仅0.444636%。这支持停止该次固定表示与线性修正，不能证明电量无用，也不能证明缺少物理结构就是唯一原因。记忆模型、服务年龄模型和拥堵选择模型仍需各自证伪，不能从失败直接跳到新机制为真。

一个可形成论文价值的方向应同时回答：新增的可预测对象是什么、能从何种观测中辨认、较经典同信息动态模型多了什么、何种实验会推翻主张。只加入守恒损失、换成图网络或把长历史称作记忆核，尚不足以构成创新。

## 阅读与书目记录

### 交通补充查新：不计为上述十篇深读来源

- Xiao, Lou & Frisby (2018), *How likely am I to find parking? – A practical model-based framework for predicting parking availability*, Transportation Research B 112:19–39，[作者机构记录](https://experts.azregents.edu/en/publications/how-likely-am-i-to-find-parking-a-practical-model-based-framework/)，DOI [10.1016/j.trb.2018.04.001](https://doi.org/10.1016/j.trb.2018.04.001)。本轮读到摘要：已有历史占用输入、有限容量M/M/C/C参数估计及多时点预测。因此“用排队模型预测占用”有直接先例，必须作为机制对照；本文并未证明任意非参数到达/服务过程都可识别。
- Ding, Huh & Rong (2024), *Feature-Based Inventory Control with Censored Demand*, Manufacturing & Service Operations Management 26:1157–1172，[原文](https://doi.org/10.1287/msom.2021.0135)。摘要层级：容量限制导致观测销量与潜在需求不同，方法评价是库存决策成本/遗憾。可借鉴观测截断问题，不能直接把销量截断模型等同占用观测。
- Kong等，*Data-driven optimization of hybrid EV charging infrastructure: From cross-city demand forecasting to dynamic allocation*，[出版商页面](https://www.sciencedirect.com/science/article/pii/S0306261926012201)，DOI [10.1016/j.apenergy.2026.128564](https://doi.org/10.1016/j.apenergy.2026.128564)。公开摘要已描述固定/移动充电站、预测与调度联合框架，是“预测接调度”的近邻工作。页面标注Applied Energy 425、**2026年12月卷期**，晚于本次检索日；Crossref DOI记录创建于2026-07-31，但未提供首次在线日期，不能把创建日期当首次发表日期。本轮只作已检索到的近邻提醒，未复现其性能，也未将其与275区标准协议混排。

10个DOI的标题、作者、期刊和出版日期均于2026-09-13通过直接Crossref核验。页码不完整的Crossref记录由期刊目录或原论文首页补充；没有把摘要可读误记为全文已读。机器可读登记随报告发布；本文件不包含原论文全文、私有时序数据或本地账号信息。
