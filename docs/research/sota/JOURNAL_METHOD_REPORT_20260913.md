# UrbanEV：条件分布信息与RMSE–MAE共同改进

日期：2026-09-13。状态：**文献驱动的理论与冻结探针方案；不是SOTA成果声明。** 复杂理论与实验规格由研究协作侧提出，执行侧核对来源、整理公式与实现；本节保留真实数据运行前的方案，实际结果另见[执行报告](DUAL_RISK_PROBE_REPORT_20260913.md)。

优先研究额外信息能否改变两种损失共同约束下的可行风险前沿；不再细化同一残差方向的步长。容量加总预测协调保留为备选，本轮不并行实验。V2的0.3896%宏RMSE改善、0.06368%结构增益及两个NO_GO不变。

## 一、从失败现象提出问题

| 现有证据 | 下一科学问题 |
|---|---|
| 区域相关分组跨折不复现 | 关系是否只在某些预测前可识别的状态下有效？ |
| 时长在ridge有效，在Chronos直接增广中收益很小 | 时长是否提供基础预测未利用、且对所需损失有价值的信息？ |
| 大步长损害MAE，小步长仅有微小RMSE收益 | 条件均值误差与误差符号能否同时被预测，并转为足够大的共同改进？ |

小时累计桩时不能冒充会话剩余时长，图相关性不能冒充行为机制，Q0.5输出也不能冒充真实条件中位数。下面分别给出精确恒等式、条件命题、反例及尚待检验的工作假设。

## 二、期刊中的创新思路及迁移边界

完整问题→假设→机制→反证的阅读记录见[期刊阅读笔记](JOURNAL_READING_NOTES_20260913.md)，包含8篇重点参考、1篇勘误和2篇直接竞争工作。研究侧另补充以下5篇，执行侧均完成Crossref身份核验，阅读范围逐项注明；这些身份核验不等于全文实验复现。

| 论文与来源 | 学习的方法设计思路 | 阅读与适用边界 |
|---|---|---|
| Powell、Cezar、Rajagopal，Scalable probabilistic estimates of electric vehicle charging given observed driver behavior，Applied Energy 309:118382 (2022)，[出版社](https://www.sciencedirect.com/science/article/pii/S0306261921016214) | 用群体与会话概率结构表达行为异质性，再做规模化加总。 | verified / Crossref；出版社摘要与章节预览。个体行为变量不可直接从区域累计量恢复。 |
| Qu、Kuang、Wang、Li、You，A Physics-Informed and Attention-Based Graph Learning Approach for Regional Electric Vehicle Charging Demand Prediction，IEEE TITS 25(10):14284–14297 (2024)，[期刊](https://doi.org/10.1109/TITS.2024.3401850) | 将价格弹性先验作用于扰动样本，并设置无先验对照。 | verified / Crossref；研究侧实读[作者稿](https://arxiv.org/abs/2309.05259)，正式版差异待补读。作者稿步长是五分钟单位，不能与我们小时视野混排。 |
| Li、Knoop、van Lint，Multistep traffic forecasting by dynamic graph convolution: Interpretations of real-time spatial correlations，TRC 128:103185 (2021)，[作者机构](https://research.tudelft.nl/en/publications/multistep-traffic-forecasting-by-dynamic-graph-convolution-interp/) | 使感受野随当前交通状态变化，再检查与传播机制的一致性。 | verified / Crossref；作者机构摘要与正式发表记录。EV区域没有可直接移植的交通波规律。 |
| Ehm、Gneiting、Jordan、Krüger，Of Quantiles and Expectiles: Consistent Scoring Functions, Choquet Representations and Forecast Rankings，JRSS-B 78(3):505–562 (2016)，[作者机构](https://publikationen.bibliothek.kit.edu/1000055110) | 将汇总评分拆成基本评分函数的混合，以理解区间贡献。 | verified / Crossref；研究侧读作者稿表示定理，执行侧核对来源。不能据此按真实未来误差选择预测样本。 |
| Henzi、Ziegel、Gneiting，Isotonic Distributional Regression，JRSS-B 83(5):963–993 (2021)，[原文](https://doi.org/10.1111/rssb.12450) | 先规定协变量偏序，再估计条件CDF，由分布服务不同决策。 | verified / Crossref；研究侧读正式版核心命题，执行侧核对机构期刊PDF。duration单调性未经证明，不直接强加偏序；训练最优不等于时间外推保证。 |

另外，[Atanane、Mkhadri、Oualkacha的HQER预印本](https://arxiv.org/abs/2510.05268)已研究分位与期望分位损失混合（2025，arXiv身份已核对，仅摘要），不列作期刊证据。“L1＋L2”“风险定位＋残差修正”均已有先例。本项目的潜在贡献必须落在可识别、可估计且可外推的双风险收益上，而非模块改名。

## 三、优先候选：学习“能够兼容两种损失的信息”

### 1. 理论对象：不是 $D$ 与 $Y$ 是否相关，而是可行风险前沿是否移动

令 $\mathcal I_0$ 为基础信息集，$\mathcal I_1\supseteq\mathcal I_0$ 为加入合法滞后时长信息后的集合。基础点预测为 $p\in[0,1]$，可选预测$q$也限制在[0,1]。$A$表示按注册单元等权的总体MAE，$R_{\mathrm{macro}}$表示单元RMSE的等权平均。

定义：

$$
\mathcal V(\mathcal I)=
\inf_{\substack{q\ \mathcal I\text{-可测}\\
A(q)\le A(p)\\
\text{满足注册单元损害约束}}}
R_{\mathrm{macro}}(q).
$$

信息集扩大只会扩大可选预测函数集合，因此：

$$
\mathcal V(\mathcal I_1)\le \mathcal V(\mathcal I_0).
$$

**这是约束优化的直接结论，不是新定理。** 严格改善是否存在、能否被有限样本估计、能否跨时间保持，才是我们的研究问题。

它比“时长与残差线性相关”更贴近任务：某个信息即使有平方损失价值，也可能无法在 MAE 零恶化约束下带来足够收益。

### 2. 一个必须面对的不可能性边界

**命题：**若 $p$ 已是给定信息集下的唯一条件中位数，那么任何使用同一信息集、且总体 MAE 不高于 $p$ 的预测，都必须与 $p$ 几乎处处相同。

**推导：**条件中位数最小化条件绝对损失；各信息状态下的损失差非负。总体差不大于零，只能要求条件差几乎处处为零；唯一性继而要求预测不变。这是中位数 Bayes 性质的推论。[Gneiting，JASA 2011](https://doi.org/10.1198/jasa.2011.r10138)

**反例说明为什么平方收益不够：**

$$
Y\sim\operatorname{Bernoulli}(0.1),\qquad p=0.
$$

对 $q\in[0,1]$：

$$
A(q)=0.1+0.8q,
\qquad
S(q)=0.1-0.2q+q^2.
$$

小幅正修正改善平方损失，但任何正修正都恶化 MAE。

唯一性也不能省略：若 $P(Y=1)=0.5$，则整个 $[0,1]$ 都是中位数集合，移动到 $q=0.5$ 可以降低平方损失而保持 MAE。

**因此，真正的突破条件是：找到基础预测尚未利用的信息，使它在更细的信息条件下不再处于绝对损失最优位置。** 不能从“模型叫中位数预测器”推断这件事已经不可能，也不能从“duration 对 ridge 有用”推断它一定成立。

### 3. 核心可检验对象：均值误差、符号失衡和原子质量

对某一单元，定义：

$$
m=E[Y-p\mid\mathcal I],
$$

$$
s=P(Y<p\mid\mathcal I)-P(Y>p\mid\mathcal I),
$$

$$
\pi=P(Y=p\mid\mathcal I).
$$

对预测前可计算的方向 $h(\mathcal I)$，并要求在 $p=0,1$ 处朝可行区间内部移动：

$$
D A_p[h]=E[sh+\pi|h|],
$$

$$
D R_{\mathrm{macro},p}[h]
=-\frac1C\sum_c\frac{E[(Y-p)h\mid c]}{r_{0,c}},
$$

其中 $r_{0,c}>0$ 是单元 native RMSE。

这说明两种损失分别关注：

- **RMSE：**误差幅度与方向的乘积；
- **MAE：**误差符号，以及恰好零误差处被扰动产生的损失。

旧 ridge 主要学习第一类对象；仅残差化 $D$ 对 $X$，不会自动学到第二类对象。

### 4. 一个显式的共同下降充分条件

令：

$$
k=[-s\,\operatorname{sign}(m)-\pi]_+,
\qquad
h=mk.
$$

则：

$$
E[(Y-p)h\mid\mathcal I]=m^2k\ge0,
$$

$$
sh+\pi|h|=-|m|k^2\le0.
$$

若 $m\ne0,k>0$ 的事件具有正概率，则两种风险的一阶导数均严格改善。

这是本报告给出的**可直接验证的推导**，不是“已证明 UrbanEV 存在这种方向”。它本质上属于多目标共同下降构造；其中正部截断也可以被看作连续门控，**结构名称本身不构成创新**。

还有两个重要限制：

第一，真实 $m,s,\pi$ 不可见，必须由过去数据估计。估计方向在训练内兼容，不代表在后续窗口兼容。

第二，严格负的一阶导数只支持存在足够小的有效步长，**不保证能达到 1% RMSE 门**。本轮已经证明，不能把“小步长有效”包装成研究突破。

### 5. 有限步长应依赖分布，而不仅是零点导数

设真实条件 CDF 为 $F$，估计 $\widehat F$ 也是支撑在[0,1]上的合法CDF，且假设：

$$
\sup_u|F(u)-\widehat F(u)|\le\varepsilon.
$$

对于任意 $p,q\in[0,1]$，条件绝对风险差满足：

$$
A_F(q)-A_F(p)=\int_p^q(2F(u)-1)\,du.
$$

因而：

$$
\left|\Delta A_F-\Delta A_{\widehat F}\right|
\le2\varepsilon|q-p|.
$$

平方风险差也有同样的界，因为：

$$
\Delta S_F=(q-p)(q+p-2\mu_F),
\quad
|\mu_F-\mu_{\widehat F}|\le\varepsilon.
$$

这给出一个研究方向：**对预测移动区间内的条件分布误差设置风险余量，而非只把拟合好的平方残差方向统一缩小。**

但必须严格区分：上述界是条件命题；当前没有可用的、对 UrbanEV 时序依赖和漂移有效的 $\varepsilon$ 保证。IDR 的训练最优性或普通随机交叉验证不能自动提供这个保证。[IDR，JRSS-B 2021](https://doi.org/10.1111/rssb.12450)

### 6. 候选的可实现形态与真正的新颖性边界

方法层可采用：

> 预测前特征 → 小型条件分布／条件矩模型 → 在固定 MAE 与单元风险约束下产生点预测。

必须比较普通均值残差头、普通分布后处理和直接混合损失方法。不能只比较 native，然后把全部收益归给“新理论”。

**可能形成研究贡献的部分**是：在有限时序样本下，刻画并估计“额外信息带来的双风险可行收益”，给出估计误差与环境变化下的有效条件，并证明这些条件比简单残差相关更能预示外推成败。

**目前尚未完成的部分**是其有限样本保证、跨时间经验支持和 ≥1% 的真实收益。不得提前写成已有创新成果。

---

## 四、第二候选：容量加总约束下的独立聚合预测与协调

这是备选，不与优先候选同时实验。

设区域占用率为 $y_i$，容量为 $C_i$。城市加权占用率为：

$$
a=\frac{\sum_i C_i y_i}{\sum_i C_i}.
$$

将城市与区域序列写成 $\boldsymbol y=S\boldsymbol b$。在基础预测无偏、误差协方差 $W$ 正定等条件下，MinT 协调为：

$$
\widetilde{\boldsymbol y}
=
S(S^\top W^{-1}S)^{-1}S^\top W^{-1}
\widehat{\boldsymbol y}.
$$

这里的理论属于已有 MinT，不是我们的新定理。[MinT作者页面](https://robjhyndman.com/publications/mint/)

**为什么可能有用：**城市共同变化与区域特有变化不一定适合由同一个预测器估计；独立的聚合预测可能提供不同的估计误差结构，比任意 K32 分组更容易解释。

**立即成立的反例：**若城市预测只是现有区域预测按容量加总，整个预测已经协调，那么上述投影保持其不变。仅增加一个加总层不会凭空产生信息。

另外，容量权重错误、聚合预测偏差、短样本协方差估计不稳定，都可能恶化区域表现。逐区域裁剪还可能破坏加总关系。**因此，当前不应把“MinT＋残差头”包装成突破；必须先有独立聚合预测的互补误差证据。**

---

## 五、唯一下一步：共同下降方向的时间外推证伪探针

### 1. 实验要证伪的具体假设

任务标识：

```text
DUAL_RISK_INFORMATION_PROBE_V1_20260913
```

**工作假设 H：**在固定、低维、预测前可获得的表示中，时长相关预测差包含可识别的条件均值误差与绝对损失符号信息；利用这些信息构造的共同下降方向，在两个后续开发窗口仍然有效。

这比“增加时长能够改善 RMSE”更强，也更容易被否定。失败只否定本次表示与学习规则，不证明所有时长函数无效。

本轮**不重新启动旧固定输出归因任务，不同时开第二个实验**。

### 2. 输入：只使用既有数组，不读取新的原始时段

使用六个既有单元：

$$
\text{cut}\in\{720,1056,1392\},
\qquad H\in\{3,12\}.
$$

每个单元读取已保存的 truth，以及 V1 的以下五个 alpha1 预测缓存：

```text
native
occupancy
raw_duration
duplicate
permuted_orthogonal
```

共最多 **36 份数组**。这些 alpha1 数组只用于构造已固定模型的预测前表示，**不重新评分 alpha1、不搜索 alpha，也不恢复旧门**。先核对已有文件身份和哈希；缺失则停止，不重新拟合或推理补文件。

不用 V2 最终选中的 alpha 构造输入，因为该 alpha 已使用全部三个校准窗口；避免把这层后验选择带进前向探针。

定义：

$$
p=\operatorname{clip}(p_{\mathrm{native}},0,1),
\quad
d_O=p_{\mathrm{occupancy},1}-p_{\mathrm{native},1},
$$

$$
z_D=p_{\mathrm{raw},1}-p_{\mathrm{occupancy},1},
$$

并类似定义 $z_{\mathrm{dup}},z_{\mathrm{perm}}$。

**$z_D$ 只是两个冻结系统的预测差，不是已证明与所有占用信息独立的“纯时长创新”。** 它也可能包含联合 ridge 参数化的差异，这正是需要 duplicate 等对照的原因。

### 3. 固定特征、模型和时间划分

基础表示固定为：

$$
\phi_0=
(1,p,p^2,d_O,H/12,j/H,
\sin(2\pi o/24),\cos(2\pi o/24),
\mathbf1_{p_{\mathrm{raw}}\le0},
\mathbf1_{p_{\mathrm{raw}}\ge1}),
$$

其中 $o$ 是预测原点，$j=1,\ldots,H$。

只比较四种表示：

$$
\phi_0,\quad
(\phi_0,z_D),\quad
(\phi_0,z_{\mathrm{dup}}),\quad
(\phi_0,z_{\mathrm{perm}}).
$$

| 前向探针 | 拟合所用窗口 | 评价窗口 |
|---|---|---|
| A | `[720,888)` | `[1056,1224)` |
| B | `[720,888)` 与 `[1056,1224)` | `[1392,1560)` |

保留原点间隔 12 小时、每窗每视野 14 个原点。训练统计只使用相应较早窗口。

每种表示拟合一个四输出 ridge：

$$
Y-p,\quad
\mathbf1_{Y<p},\quad
\mathbf1_{Y=p},\quad
\mathbf1_{Y>p}.
$$

固定正则 `0.01`，截距不惩罚；输入按训练样本加权均值与标准差标准化，零尺度列置零。训练权重使每个“窗口×视野”单元等权，而不是使 H12 获得四倍权重。概率输出投影到三分类概率单纯形，仅用于保证合法概率，不训练额外组合权重。

总计 **8 次小型多输出线性拟合**，不训练深度网络、不新增 Chronos 推理。

这些窗口已经被前期研究使用过。因此，即使本探针内部遵守前向时间顺序，也只能称**后验设计的开发诊断**，不能称新的盲测或独立确认。

### 4. 不产生新步长，只检验方向是否真的共同下降

由模型预测得到 $\hat m,\hat s,\hat\pi$，固定：

$$
\hat h
=
\operatorname{clip}(\hat m,-1,1)
[-\hat s\,\operatorname{sign}(\hat m)-\hat\pi]_+.
$$

在 $p=0$ 时将负方向置零，在 $p=1$ 时将正方向置零。公式、阈值和表示均不根据评价结果修改。

在两个评价窗口分别计算实际：

$$
\widehat D_A
=
\operatorname{macromean}
\left[
\operatorname{sign}(p-Y)\hat h
+\mathbf1_{Y=p}|\hat h|
\right],
$$

$$
\widehat D_R
=
-\frac12\sum_{H\in\{3,12\}}
\frac{\operatorname{mean}[(Y-p)\hat h]}{r_{0,H}}.
$$

**只计算方向导数与条件矩预测误差，不生成 $\alpha$ 曲线，不输出新修正预测的“最佳 RMSE”。**

这里研究的是从已裁剪 native 出发的新方向，不能把它冒充 V2 的 `clip(raw_native + alpha * old_delta)`。未来采用哪种输出族，必须重新注册。

### 5. 固定诊断准入与停止规则

只有同时满足以下条件，才记为：

```text
MECHANISM_SUPPORTED_FOR_NEW_PROTOCOL_DESIGN
```

主表示 $(\phi_0,z_D)$ 在两个评价窗口均有非零方向，且各窗宏 $\widehat D_R<0,\widehat D_A<0$；同时，其归一化 RMSE 下降率

$$
-\widehat D_R/\sqrt{\operatorname{macromean}(\hat h^2)}
$$

在两窗均严格优于同窗中满足 MAE 下降条件的所有固定对照。没有兼容对照时以零为参考，零方向的效率记零。

否则记为：

```text
MECHANISM_NOT_SUPPORTED_IN_REGISTERED_REPRESENTATION
```

这是**机制筛选，不是新的 1% 效果门**。导数为零不算严格支持；不得随后更换截断阈值、特征、正则或标签划分挽救结论。数据失配等技术问题单独记为 `PROBE_BLOCKED`。

真实标签仅用于监督拟合和事后评价。**不得用“未来误差大”“这次修正有益”等标签分组直接决定部署时是否修正。**

### 6. 回传工件

新增一份冻结配置和四类结果即可：

| 工件 | 内容 |
|---|---|
| `probe_manifest.json` | 来源提交、36 文件身份、时间范围、固定特征与模型 |
| `moment_prediction_scores.csv` | 四表示、两评价窗、各视野的误差均值与概率预测误差 |
| `directional_risk_report.json` | 实际方向导数、方向范数、对照比较、原子质量与零方向比例 |
| `probe_receipt.json` | 实际读取范围、8 次拟合计数、零基础推理、技术检查与状态 |

先完成冻结和合成公式测试，再由执行侧运行；**本报告没有执行这些真实数据计算。**

---

## 六、原守门标准如何保留，新方案必须重新冻结什么

**V1、V2 的 NO_GO 永久按原协议保留。** 探针通过也不赋予 V2 晋级资格。

未来新方法需要重新注册的内容至少包括：

| 必须冻结的对象 | 原因 |
|---|---|
| 信息集与时间可得性 | duration 的发布与端点未知，不能把离线版本可读等同于线上可得 |
| 条件分布／条件矩的估计对象 | 不能在看到效果后从均值、符号概率、分位数之间择优 |
| 点预测生成规则 | 从 raw native 还是 clipped native 出发、是否裁剪，必须明确 |
| 训练目标与单元权重 | 宏 RMSE 不是 pooled MSE，不能训练或报告时偷换 |
| 强对照 | 包括同信息普通 ridge、普通分布后处理、直接混合损失或受约束预测器 |
| 参数与选择预算 | 特征、正则、分布形式、随机种子、停止条件均预先固定 |
| 新开发评价方案 | 当前三个窗口已曝光；尾部、第三折只能在新的明确授权下使用 |

实际方法晋级仍要求：**相对注册的最佳非时长对照，宏 RMSE 改善至少 1%，宏 MAE 不恶化，单元损害不超过原 1%，并保留跨窗口和负对照要求。** 新结构主张还必须优于适当的同信息普通方法，不能用“相对 native 有收益”替代结构创新证据。

---

## 最终研究选择

最值得借鉴的不是某篇论文的网络名称，而是三种研究动作：

**Gneiting/IDR：把预测目标从单一残差重新定义为决策相关的条件分布。**

**Anchor/MinT：把笼统的“稳健”“协调”变成有假设、有可识别对象、有失败边界的命题。**

**能源与交通论文：让结构先验对应真实可观测过程，并用针对性的反证检验它，而不是以注意力图或模块名称充当机制。**

因此，本轮优先路线固定为：**先证伪“额外表示能否预测 RMSE–MAE 共同下降方向”，再决定是否值得设计新的分布驱动方法。** 不再打磨原求解器，不并行重启区域分组、会话生存模型或旧 Router。

定时任务保持 **PAUSED**；开发尾部、第三折验证／测试和 Paris protected/formal 保持关闭；一次性校准不重跑，不使用额度重置卡。


## 执行口径补充（真实运行前冻结）

协议见[冻结配置](../../../configs/research/DUAL_RISK_INFORMATION_PROBE_V1.json)。边界指示中的raw指native原始预测，额外三个差值均减去occupancy的α=1预测。岭回归目标使用总权重为1的加权平方和；训练单元等权，截距不惩罚。日历先按24小时取模，0/12小时正弦取精确0，避免把三角舍入噪声标准化放大。零方差列置零；原子判断使用转换到float64后的精确相等。若native单元RMSE为0则本注册导数不适用，返回技术阻塞。以上口径不根据评价结果选择。
