# 基于底座增损的交互稳定化：理论与真实实验

本轮检验：训练时单独惩罚相对共同底座新增的绝对误差，能否稳定O–D交互收益，以及这种作用是否超过普通MAE混合、统一收缩和通用MLP。[上一轮](CONDITIONAL_OD_INTERACTION_REPORT_20260913.md)存在RMSE开发信号，但种子、视野分化且平均MAE升高。本轮不按已见DEV状态损失设置路由，也不靠挑选种子刷新成绩。

## 1. 理论对象与目标

共同冻结的长OD ridge为q⁰，网络修正为f，预测q=q⁰+f。以FIT样本内底座残差e=Y−q⁰定义

\[
d_A=|e-f|-|e|,\quad h=[d_A]_+,\quad b=[-d_A]_+,\quad
A(q)-A(q^0)=E[h]-E[b].
\]

平均MAE反映收益和增损的净值。主方法额外惩罚逐条目的正增损，而非平均净变化的正部。两者不等价：不同条目的收益可以抵消后者。

先计算e32=float32(Y64−q⁰64)，冻结

\[
s=\max\{\sqrt{\operatorname{mean}(\operatorname{float64}(e32)^2)},10^{-6}\},\quad
\kappa=0.5,\quad \tau=\operatorname{float32}(\kappa s).
\]

主目标为

\[
J_{POSREG}=E[(e-f)^2]+\kappa s E[h].
\]

均值覆盖所有行和12输出。s使两个项具有平方误差量纲，只从FIT计算一次；不搜索κ，不使用SELECT或DEV估计尺度。底座在同一FIT估计，残差是样本内残差，不是总体可靠性的无偏估计。

必要对照为

\[
J_{MIX}=E[(e-f)^2]+\kappa sE[|e-f|-|e|],\quad
J_{L1}=E[(e-f)^2]+\kappa sE|f|.
\]

MIX减去固定常数，不改变普通MSE＋MAE优化；L1惩罚全部修正。逐函数由三角不等式得到

\[
[|e-f|-|e|]_+\le |f|,\qquad J_{MIX}\le J_{POSREG}\le J_{L1}.
\]

这不是三种训练结果的性能排序。若同一经验样本上的固定函数确实满足J_POSREG(f)≤J_POSREG(0)，才有

\[
\kappa sE[h]\le E[e^2]-E[(e-f)^2].
\]

神经优化、权重衰减、梯度截断与SELECT选择都不保证这个前提，更不提供DEV安全保证。反例：Y服从Bernoulli(0.1)、q⁰=0，限制常数q∈[0,1]。E[h]=0.9q，MSE=0.1−0.2q+q²，本轮κ与s下最优q≈0.028849，MSE下降而MAE从0.1升至约0.123079。因此本方法不是MAE非劣算法。

## 2. 与已有思想的关系

[Safe Policy Improvement by Minimizing Robust Baseline Regret，NeurIPS 2016](https://papers.nips.cc/paper_files/paper/2016/hash/9a3d458322d70046f63dfd8b0153ece4-Abstract.html)研究相对基线的稳健遗憾，其MDP模型与误差保证不能转移给这里的预测惩罚。[Sagawa等的group DRO研究](https://arxiv.org/abs/1911.08731)提示训练最差组风险与泛化仍有差距。本轮不复现这些算法，也不将正部、残差训练或损失混合本身称作原创理论。可能的贡献在于这一目标与交互预测器的适配价值，必须通过强对照建立；不是穷尽查新。

## 3. 六方案的结构与目标对照

| 方案 | 结构 | 目标 | 要排除的解释 |
|---|---|---|---|
| PRODUCT_MSE | 原OD_PRODUCT | MSE | 原方法在新种子下表现 |
| PRODUCT_POSREG（主方法） | 原OD_PRODUCT | 正增损 | 主假说 |
| PRODUCT_MIX | 原OD_PRODUCT | MSE＋MAE | 普通损失混合已足够 |
| PRODUCT_L1 | 原OD_PRODUCT | MSE＋修正L1 | 只是缩小修正 |
| CONCAT_MSE | 原拼接MLP | MSE | 普通非线性已足够 |
| CONCAT_POSREG | 原拼接MLP | 正增损 | 目标是通用收益而非交互特有 |

各网络9,108参数，同一架构同一种子完全同初值、同数据顺序；不同结构不声称逐参数相同。全部共用一次新拟合的4,104系数长OD ridge与FIT变换，随后冻结。原模型源和旧实验保持原样。

固定三个新种子20260915/16/17，不与旧两个种子混成事后五种子赢家。18次训练，每次40epoch、batch4096、2,360步，共42,480步。复用原Xavier子种子、零输出层、AdamW、余弦学习率及全局梯度截断1；abs/relu用PyTorch默认零点次梯度。细节全部登记于[配置](../../../configs/research/REFERENCE_HARM_STABILITY_V1.json)。

FIT原点169…1044，240,900行；SELECT1056…1212和DEV1392…1548各14原点，间隔12小时。O168、额外滞后1小时的D168、五维相位/容量上下文与归一化原样继承。所有窗口已曝光，仅为新开发比较。O数值阶段上限1056/1224/1548/1560，D上限1043/1211/1547且SCORE不扩展。保护尾部、后续验证/测试及Paris保持关闭。

先完成全部18训练，才在SELECT从epoch0/5/10/20/40按raw四H末点宏RMSE分别选择，完全并列取较早者。MAE、增损和状态组不参与选择；epoch0明确记底座别名，不称稳定化成功。无重训、最好种子、集成或新损失权重搜索。

全部预测冻结后才扩展最终SCORE前缀及读取四个原有1392 native/truth缓存。native仅有原调用的H3/H12，不切片制造H6/H9。四H和共同支持分别报告raw/clip、末点/路径。正常有1,440个选择评分、16个底座参考评分、312个有效DEV评分及8个缺失状态。

## 4. 稳定性诊断与反证

对每个H的三个种子修正a_s=q_s−q⁰，定义

\[
T_H=\frac13\sum_sE[a_s^2],\quad
V_H=\frac19\sum_{s<t}E[(a_s-a_t)^2],\quad
\eta_H=V_H/T_H\ (T_H>0).
\]

V是以1/3归约的跨种子预测方差。共同缩放a_s→ca_s时T、V同比缩放，而η不变。T=0时η记null并标为全退回底座。因此方差下降不自动意味着更稳定的非零信息；η下降也不保证准确度。用配对预测差计算，不产生或评分种子集成。raw与clip各自减去同后处理的底座。

另外报告逐H风险分解H_A=E[h]、B_A=E[b]并核对MAE差=H_A−B_A。降低增损若损失更多收益，不能称风险权衡改善。每epoch日志是训练遍历中的批量均值，不是冻结参数下完整FIT目标或泛化界。

结构×目标差为I_R=[R(PRODUCT_MSE)−R(PRODUCT_POSREG)]−[R(CONCAT_MSE)−R(CONCAT_POSREG)]，仅是描述性作用差，不是因果识别。三种子的指标均值、范围、样本标准差(ddof=1)与每个种子方向都保留。

沿用p<0.5/≥0.5与24小时变化<0/≥0四组，只做冻结后的贡献诊断，不进入训练或路由。四H平均的组损失和除3,850还原宏MAE差或平均端点MSE差；不能用宏RMSE平方差替代。没有新的状态阈值、聚类或有利原点选择。

MIX/L1或CONCAT_POSREG同样好，将分别限制正增损必要性、非统一收缩解释及交互特有性。若DEV未保持SELECT收益，或改进仍靠局部状态补偿，需明确记录。旧1%和MAE零恶化门不恢复。

## 5. 执行结果

本节在冻结后唯一一次真实实验完成时填入。人工代数、梯度、结构、稳定性和GPU检查通过；没有将人工结果当作UrbanEV预测成绩。

本轮完成即返回人工审核，不启动定时或下一轮，不使用额度重置卡。原始数据、权重、预测和标签仅保留本地。SOTA仍依据[统一比较规则](SOTA_COMPARISON_STANDARD.md)，本轮不能替代完整基准与最新强基线比较。
