# 候选 A：最近邻方法及新颖性核查

检索日期：2026-09-16。范围是缺失标签评价、共享缺测诱导的重叠标签、游程自动机与滑动窗口语言。以下为本轮读取的原始来源；没有将检索未命中解释为全球首创证明。

| 来源 | 已有内容 | 对本项目的约束 |
|---|---|---|
| Dervovic、Cashmore，AISTATS 2025，*Model Evaluation in the Dark* | 缺失评价标签的极端界、多重插补与性能分布；正文第3节明确讨论上下界 | 不能声称首次求缺标签指标界；本文需处理由同一底层时间序列诱导的联合标签约束 |
| Joshi、Tchamgoue、Fischmeister，SAC 2017，*Runtime Verification of LTL on Lossy Traces* | 缺失事件轨迹上的逻辑监测及可靠性判断 | 不完整轨迹的自动机处理并非新概念；需要区分逻辑满足性与时间变化权重的配对风险极值 |
| Fu，Statistica Sinica 1996，*Distribution Theory of Runs and Patterns Associated with a Sequence of Multi-state Trials* | 有限状态嵌入及前后向游程/模式计算 | 连续游程状态和前后向计算不是原创贡献 |
| Ganardi 等，FSTTCS 2016 / STACS 2018 的滑动窗口自动机研究 | 滑动窗口正则语言及状态/空间复杂度 | 新算法不能仅以“不存完整窗口”作为创新；需精确定位特定目标及已有一般工具的关系 |

原文入口：[AISTATS/PMLR](https://proceedings.mlr.press/v258/dervovic25a.html)、[AISTATS 全文第3节](https://arxiv.org/html/2504.18385v1#S3)、[Waterloo 作者组记录](https://uwaterloo.ca/embedded-software-group/references/runtime-verification-ltl-lossy-traces)、[Fu 原文](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A6n410.pdf)、[FSTTCS 2016 原文](https://drops.dagstuhl.de/storage/00lipics/lipics-vol065-fsttcs2016/LIPIcs.FSTTCS.2016.18/LIPIcs.FSTTCS.2016.18.pdf)、[STACS 2018](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.STACS.2018.31)。

## 当前判断

配对二元损失对事件标签的仿射化以及在自动机上运行最短/最长路径，本身是可预期的标准工具组合。压缩为末尾游程长度和最近事件年龄的构造，有清楚的充分性证明和相对本仓库 V3 的计算优势；这不等于已经证明一个独立算法学术贡献。

本轮没有核实到完整等价的“共享缺测＋重叠持续事件＋模型配对风险极值”的已发表实证方案，但检索覆盖有限，尤其缺少对一般加权自动机、部分观测时序监测和统计部分识别全部近邻的系统排除。因此当前新颖性状态为 **尚未建立充分依据**，而非“已确立首创”。

继续研究的价值需要来自：真实较强模型之间的判断改变；准确描述何时联合约束有用/无用；以及原指数实现无法有效处理的窗口下，可复现、可解释的精确比较。若实证只是微小界宽改善且决策不变，应降低论文主张，不能用更多模块掩盖。

## 数据来源核查

北京数据采用[UCI 501](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)，家庭功率采用[UCI 235](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)。发布页均说明存在自然缺失，并列 CC BY 4.0。北京 API 返回的 `has_missing_values=no` 与网页说明不同，故不将该元字段作为数据资格判据；实际前缀的时间格点和缺失量以本轮文件检查为准。

官方 DOI 分别为 10.24432/C5RK5G 和 10.24432/C58K54。PM2.5 高值持续事件不自动等于法定空气预警；家庭分钟平均功率高值不表示设备过载。固定阈值仅用于这次方法验证，没有根据结果重新选择。
