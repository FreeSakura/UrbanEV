> 历史研究报告（2026-09-05）的公开文本副本。工作记录保存在本地，并非本仓库全部提供。后续自然缺失与共享窗口研究见 [V3报告](../reports/audit/PAIRED_AUDIT_V3_REPORT.md)。V2逐窗界的总体求和通常只是有效松弛界；共享快照下的联合sharp界由V3给出。

# UrbanEV后续研究的统一理论报告

## 1. 研究目标、结论层级与问题重述

本报告从现有研究的正负结果出发，建立一条以**信息条件下的预测风险**为核心的理论路线。要解释的不是“如何再给模型加一个模块”，而是三个更基础的问题：额外机制究竟消除了哪一项误差；观测是否足以识别欲预测的对象；即使机制在总体上有价值，现有样本是否足以学习并验证它。

**推导状态：COHERENT AFTER REFRAMING / EXTRA ASSUMPTION。** 原来的宽泛目标“观测合同适配可以提高预测”不能无条件成立；本报告将其改写为带明确条件的风险分解、识别边界与后续方法设计。文中列出的数学命题在各自假设下证明；研究路线的实证有效性仍未建立。

本报告不把成熟的文稿结构等同于成熟的新算法。条件期望、线性回归、正交投影、Gaussian conditioning、run-length递推和部分识别都有既有理论基础。这里的工作是把它们连成适合当前EV数据问题的可检验框架，并明确哪些组合仍只是假设。参考文献与知识来源在第13节列出。

### 1.1 已有证据的理论位置

最近CPU试点读取2022年9月至11月的既有开发前缀，完成3目标×3折的9次线性拟合、8方法的72个比较单元。直接发布目标回归相对细粒度重聚合的逐折平均RMSE改善为：占用+0.0014%、时长−0.0148%、电量−0.0716%。所有结果都未通过后续适配器分诊门。原始未裁剪预测的线性重聚合最大数值差为$3.668\times10^{-11}$。

这些结果能支持“当前固定线性预测族没有显示新增适配收益”，不能证明“所有非线性模型均无用”。特别是，线性交换性质是数学上预期成立的实现核对，不应把它当成数据发现或新机制。

更早的固定融合内部提分、动态路由小信号、蒸馏与Paris教师门失败，以及B*不完整筛选，提示两个彼此不同的缺口：一是**误差中可由预测原点信息解释的部分可能很小**；二是**点预测目标可能没有覆盖真正关心的联合事件**。本报告分别推导，不能用第二项为第一项的失败改判。

### 1.2 推导类型

全文使用四类标记：**恒等式**表示精确代数；**命题**表示需假设并给出证明；**近似模型**表示为了可实现而增加的建模条件；**研究假设**表示未来必须用新证据检验的主张。反例中的离散分布和人工矩阵只用于数学检验，从未进入真实研究训练或替代真实标签。

## 2. 统一对象：观测、信息与损失

### 2.1 物理过程与发布观测分开

在区域$z$、连续时间$u$上，记$B_z(u)$为不可用/繁忙端口数，$A_z(u)$为实际主动充电端口数，$P_z(u)$为对主动端口定义的平均功率。瞬时理想语义下$0\le A_z(u)\le B_z(u)\le C_z$，其中$C_z>0$为固定容量。这一瞬时关系不能直接套到不同时间支持的已发布观测上。

设第$k$个区间为$I_k=[s_k,s_k+\Delta)$，$\Delta$以小时计，则理想观测可写为：

$$
O_{z,k}=B_z(s_k),\qquad
D_{z,k}=\int_{I_k}A_z(u)\,du,\qquad
V_{z,k}=\int_{I_k}A_z(u)P_z(u)\,du. \tag{1}
$$

其中$O$单位为桩，$D$为桩·小时，$V$为kWh。这里把$P$定义为主动端口平均功率，使$AP$在$A=0$时取零；并不从观测$V/D$反演瞬时功率。五分钟duration对应区间贡献，$D/\Delta$是平均主动暴露，不自动是瞬时整数计数。

把未来已定义好的细粒度观测堆成$X\in\mathbb R^d$，发布目标为$Y_c\in\mathbb R^q$。已知线性部分用$H_c\in\mathbb R^{q\times d}$表示：

$$Y_c=H_cX+\varepsilon_c.\tag{2}$$

$H_c$可以是快照选择、积分求和、固定容量归一化或固定空间聚合的块矩阵。$\varepsilon_c$统称发布层差异，不能未经证据拆成传感噪声、人工修正、缺失填补或真实动态误差。模型式(2)是观测层表述，不证明$X$等于完整可恢复的微观物理状态。

### 2.2 信息集必须按实际可见性定义

以$t$表示预测发出时刻，$\mathcal F_t$为届时可获得的历史数据、元数据、模型训练记录与固定参数生成的$\sigma$-代数。每条记录$j$有测量时间、区间结束时间以及发布时间$a_j$；只有$a_j\le t$的记录允许进入$\mathcal F_t$。区间量还必须完成测量。已训练模型的内部随机数与训练样本可并入信息集，未来评价标签不得并入。

本地UrbanEV发布时间未逐条给出，官方说明还包含后向填补。因此现有试点实际使用的是“已发布数据上的离线索引信息集”$\mathcal F_t^{\rm rel}$；不能假设$\mathcal F_t^{\rm rel}$就是真实在线信息集$\mathcal F_t^{\rm online}$。后向填补甚至可能使两者不满足所希望的因果包含关系。这是本报告应用到真实部署时的首要外部条件。

以下省略下标$t,c$，但每项命题都在一个固定预测目标、固定原点和明确的信息集上成立。一个公式成立，不能自动授权把同一测试区用于方法选择。

### 2.3 不变量与损失

统一不变量是：对一个已声明目标$T$、行动$a$和损失$\ell$，最小可达风险

$$\mathcal R^*_{\ell}(\mathcal F)=\inf_{a\;\mathcal F\text{-可测}}\mathbb E[\ell(T,a)].\tag{3}$$

点预测取$T=Y$与加权平方损失$\ell(Y,f)=\|Y-f\|_W^2$，其中$W\in\mathbb R^{q\times q}$为**固定、对称正定**矩阵，$\|v\|_W^2=v^\top Wv$。$W$可以包含事先固定的单位尺度；不能看到评价数据后改变$W$来制造优势。联合事件预测在第8节明确把目标换成事件$E$并使用Brier/决策损失；仍使用式(3)，但不是同一个原始RMSE任务。

基本可测设定A0：底层空间取标准Borel概率空间，$\mathcal F$由标准Borel随机元素$\mathsf I$生成，按零集等价解释。使用正规条件概率核$\Pi_\iota$，连续时间条件矩选取该同一核给出的联合可测版本；条件等式均按几乎处处解释。

基本假设A1：$Y,X$具有所需有限二阶矩，预测器也在$L^2$中。A2：线性算子在预测原点已知且有界；若随机，则$\mathcal F$可测。A3：对含连续积分的交换，随机过程联合可测且满足文中明确给出的绝对可积条件，连续时间公式对Lebesgue几乎所有时间版本积分。A4：有限样本定理单独声明独立性，不从“按天分块”自动推出。

| 符号 | 类型 | 固定含义 |
|---|---|---|
| $X,Y,H,\varepsilon$ | $\mathbb R^d,\mathbb R^q,\mathbb R^{q\times d},\mathbb R^q$ | 细粒度观测、发布目标、观测矩阵、发布差异 |
| $\mathcal F\subseteq\mathcal G$ | 信息$\sigma$-代数 | 原信息和增加合法观测后的信息 |
| $m,\mu,b$ | 条件均值向量 | $m=\mathbb E[Y\mid\mathcal F]$，$\mu=\mathbb E[X\mid\mathcal F]$，$b=\mathbb E[\varepsilon\mid\mathcal F]$ |
| $f_0,g$ | $\mathcal F$可测向量 | 冻结基线、附加修正 |
| $C,r_0$ | 约束矩阵与右端 | 线性约束$Cy=r_0$；不与容量$C_z$混用 |
| $K,L$ | $K\in\mathbb Z_{\ge0},L\in\mathbb Z_{\ge1}$ | 未来离散步数、事件所需连续步数 |
| $p,\underline p,\overline p$ | $[0,1]$标量 | 事件概率及识别上下界 |
| $Z_{\rm des},D_{\rm w},P_{\rm reg},G_\lambda$ | ridge设计/权重/正则/正规矩阵 | 仅用于T5，不与物理状态、概率分布或投影混用 |
| $P_C,Q_C$ | $\mathbb R^{q\times q}$ | 约束行空间投影与正交补，仅用于T6–T7 |
| $\mathsf K_{\rm fuse}$ | $\mathbb R^{q\times q}$ | 不加结构限制的线性融合矩阵，不与步数$K$混用 |

全局符号按此表使用；各命题内重新声明的辅助常数、目标矩阵或两世界常量只在该命题内有效。物理功率$P_z(u)$、概率分布$P_F$与投影$P_C$通过下标和类型区分。

## 3. 风险分解：新增机制究竟能赚到什么

### 命题T1：条件均值的风险正交分解

在A1下，令$m=\mathbb E[Y\mid\mathcal F]$。对任意$\mathcal F$可测$f\in L^2$，有

$$\mathcal R_W(f)=\mathbb E\|Y-m\|_W^2+\mathbb E\|m-f\|_W^2.\tag{4}$$

**证明。** 将$Y-f=(Y-m)+(m-f)$代入平方范数。交叉项绝对可积，由Cauchy–Schwarz和A1成立。再用条件期望塔式法则与$m-f$的$\mathcal F$可测性：

$$\mathbb E[(Y-m)^\top W(m-f)]=\mathbb E\{\mathbb E[Y-m\mid\mathcal F]^\top W(m-f)\}=0.\tag{5}$$

得到式(4)。由于$W\succ0$，第二项非负且仅当$f=m$几乎处处时为零。故$m$是平方风险下唯一的几乎处处Bayes解。证毕。

**解释。** 第一项是当前信息下不可约风险；第二项包括函数类逼近、有限样本估计和优化不足。换模型可能减少第二项，但不能据此声称增加了信息或改变了第一项。

### 命题T2：残差修正的精确增益条件

固定$f_0\in L^2(\mathcal F)$，令$r=Y-f_0$，$a=\mathbb E[r\mid\mathcal F]$。对$g\in L^2(\mathcal F)$定义增益$\Gamma(g)=\mathcal R_W(f_0)-\mathcal R_W(f_0+g)$。则

$$\begin{aligned}
\Gamma(g)&=2\mathbb E[r^\top Wg]-\mathbb E\|g\|_W^2\\
&=\mathbb E\|a\|_W^2-\mathbb E\|g-a\|_W^2.
\end{aligned}\tag{6}$$

因此，实际修正有正收益的充要条件是其估计/逼近误差小于可预测残差信号：$\mathbb E\|g-a\|_W^2<\mathbb E\|a\|_W^2$。

**证明。** 展开$\|r\|_W^2-\|r-g\|_W^2$得到第一行。因为$g$可测且二阶可积，$\mathbb E[r^\top Wg]=\mathbb E[a^\top Wg]$。利用$2a^\top Wg-\|g\|_W^2=\|a\|_W^2-\|a-g\|_W^2$得到第二行。若$a=0$，任意非零$g$只能增加风险；$g=0$保持风险。证毕。

对于带训练随机性的$g$，可先条件于训练信息证明，再对训练随机性取期望；但评价标签不能参与选择$g$。不能用评价误差训练一个事后oracle，再把式(6)当部署收益。

### 推论T2a：发布差异与基线误差不能混为一项

在式(2)、A1–A2且$\widehat\mu\in L^2(\mathcal F;\mathbb R^d)$下，取$f_0=H\widehat\mu$。有界$H$保证$f_0\in L^2(\mathcal F)$，从而满足T2条件。训练随机性如存在，已并入$\mathcal F$，不包含未来评价标签。此时

$$a=H(\mu-\widehat\mu)+b,\qquad b=\mathbb E[\varepsilon\mid\mathcal F].\tag{7}$$

这是线性条件期望直接代入的恒等式。只拟合$b$不保证改善实际$f_0$：当$H(\mu-\widehat\mu)$与$b$方向相反时，修正可能消除原本偶然抵消的误差。只有$\widehat\mu=\mu$的理想细粒度均值情形，修正$b$的最大可达收益才恰好为$\mathbb E\|b\|_W^2$。

**反例。** 一维$HX=1$、$\varepsilon=1$，故$Y=2$；选$\widehat\mu$使$f_0=2$。基线已零误差，真实$b=1$；加上“完全学对的发布差异”后预测变成3，平方误差为1。该例表明需要检验最终风险，不能仅凭发布残差可拟合就放行模块。

### 命题T3：新增信息的总体价值

设$\mathcal F\subseteq\mathcal G$，且$m_F=\mathbb E[Y\mid\mathcal F]$、$m_G=\mathbb E[Y\mid\mathcal G]$。则

$$\mathcal R_W^*(\mathcal F)-\mathcal R_W^*(\mathcal G)=\mathbb E\|m_G-m_F\|_W^2\ge0.\tag{8}$$

**证明。** 将命题T1用于信息集$\mathcal G$，并取$f=m_F$；$m_F$在$\mathcal G$下可测，故$\mathbb E\|Y-m_F\|_W^2=\mathbb E\|Y-m_G\|_W^2+\mathbb E\|m_G-m_F\|_W^2$。证毕。

若新表示$Z=\phi(\mathcal F)$完全由已有信息确定，则$\sigma(\mathcal F,Z)=\mathcal F$，式(8)为零。新特征工程、图层、teacher输出或更大模型仍可能降低有限模型类误差，但不能宣称它们增加了Bayes信息。蒸馏可以改变学习过程；它不自动提高由相同输入定义的总体信息上限。

**路线含义。** 后续研究必须说明收益来自新增合法信息、较好的逼近/估计，还是一个新的目标函数。若三者都没有变化，只重命名线性运算，就没有新的理论收益来源。


## 4. 已知线性合同：解析法的充分性与有限模型边界

### 命题T4：条件期望与已知线性观测交换

在A1、A2下，$Y=HX+\varepsilon$满足

$$m=H\mu+b,\qquad \mu=\mathbb E[X\mid\mathcal F],\quad b=\mathbb E[\varepsilon\mid\mathcal F].\tag{9}$$

若$b=0$，则$H\mu$是$Y$的平方风险Bayes预测。若进一步二阶矩存在，则

$$\operatorname{Cov}(Y\mid\mathcal F)=H\Sigma_XH^\top+\Sigma_\varepsilon+H\Sigma_{X\varepsilon}+\Sigma_{X\varepsilon}^\top H^\top,\tag{10}$$

其中$\Sigma_X=\operatorname{Cov}(X\mid\mathcal F)$，$\Sigma_\varepsilon=\operatorname{Cov}(\varepsilon\mid\mathcal F)$，$\Sigma_{X\varepsilon}=\mathbb E[(X-\mu)(\varepsilon-b)^\top\mid\mathcal F]$。

**证明。** 有界且$\mathcal F$可测的$H$可从条件期望中提出，线性性给出式(9)。将$Y-m=H(X-\mu)+(\varepsilon-b)$代入条件二阶中心矩并展开四项，得到式(10)。若$b=0$，由T1得到Bayes结论。证毕。

**边界。** 不允许把交叉协方差无依据设零；不允许让未知未来覆盖矩阵从条件期望中提出；不允许将“理想fine-first充分”说成“任何有限fine-first模型都不可能被改进”。即使确定的线性$H$已知，有限数据下直接低维目标拟合仍可能有更小的估计误差。

### 命题T5：固定设计ridge的线性等价

令$Z_{\rm des}\in\mathbb R^{n\times p}$为包含截距列的固定设计矩阵，$D_{\rm w}\succ0$为固定样本权重，$P_{\rm reg}\succeq0$为固定正则矩阵；假设$G_\lambda=Z_{\rm des}^\top D_{\rm w}Z_{\rm des}+\lambda P_{\rm reg}$可逆，$\lambda\ge0$。细目标矩阵$T_f\in\mathbb R^{n\times d}$，粗目标$T_c=T_fH^\top+E_{\rm rel}\in\mathbb R^{n\times q}$。最小化

$$\begin{aligned}J(B;T_f)&=\operatorname{tr}[(T_f-Z_{\rm des}B)^\top D_{\rm w}(T_f-Z_{\rm des}B)]\\&\quad+\lambda\operatorname{tr}(B^\top P_{\rm reg}B)\end{aligned}\tag{11}$$

得到$\widehat B(T_f)=G_\lambda^{-1}Z_{\rm des}^\top D_{\rm w}T_f$，并且

$$\widehat B(T_c)=\widehat B(T_f)H^\top+\widehat B(E_{\rm rel}).\tag{12}$$

对任意固定测试行$z_*^\top$，乘以$z_*^\top$后得到相同预测等式。$E_{\rm rel}=0$时，直接拟合粗目标与先拟合细目标后重聚合完全一致。

**证明。** 对$B$求导得到$\nabla_BJ=2Z_{\rm des}^\top D_{\rm w}(Z_{\rm des}B-T_f)+2\lambda P_{\rm reg}B$。矩阵$G_\lambda$半正定且可逆，因此正定，使目标严格凸。唯一驻点为$G_\lambda^{-1}Z_{\rm des}^\top D_{\rm w}T_f$。将$T_c=T_fH^\top+E_{\rm rel}$代入，使用结合律与分配律得到式(12)。若$G_\lambda$不可逆，本命题不直接适用；可另行固定相同的Moore–Penrose解规则，但不能任意选取不同极小值。证毕。

式(12)解释了本次试点中的线性误差，而不需要任何特殊EV机制。其成立条件包括**相同设计、样本、样本权重、正则和预处理**。分别调正则、使用不同标签权重、目标依赖的表征学习、不同训练子集或非线性裁剪，都可能破坏等式；这不自动构成新机制，需要再用T2分析实际风险。

若在聚合前逐点裁剪，通常$H\operatorname{clip}(\widehat X)\ne\operatorname{clip}(H\widehat X)$。必须在未经裁剪的输出上核验线性等价，并把支持投影对所有方法的影响分开。试点采用聚合后统一输出边界处理。

### 4.1 可识别“发布残差”所需的配对观测

若同一时间、同一区域有真实细粒度$X$和真实发布$Y$，且$H$已知，则$\varepsilon=Y-HX$在这些历史样本上可计算。若只有$Y$，却用$Y-H\widehat X$估计$\varepsilon$，其中混入了$H(X-\widehat X)$的预测误差。只有配对源数据或可辩护的误差模型，才可能把两个来源分开。

因此，新的版本/来源记录比新增“release encoder”更基础。静态两文件有差异，只能支持跨文件比较；没有版本生效时间，不支持部署时合同变更的因果归因。

## 5. 硬约束、软约束与“物理一致”的代价

本节先在欧氏风险下推导。若$W\succ0$固定，可做可逆变换$y'=W^{1/2}y$、$\widehat y'=W^{1/2}\widehat y$、$C'=CW^{-1/2}$后使用相同证明，再变换回来。不能同时在一个度量中投影、另一个度量中声称逐点不劣。

### 命题T6：错误等式约束的精确风险差

设$C\in\mathbb R^{r\times q}$满行秩，约束集合$\mathcal M=\{v:Cv=r_0\}$非空。定义

$$C^\dagger=C^\top(CC^\top)^{-1},\quad P_C=C^\dagger C,\quad Q_C=I-P_C,\quad n(y)=C^\dagger(Cy-r_0).\tag{13}$$

给定任意真实$y$与预测$\widehat y=y+e$，硬投影$\widetilde y=\widehat y-C^\dagger(C\widehat y-r_0)$满足

$$\|\widetilde y-y\|_2^2-\|\widehat y-y\|_2^2=\|n(y)\|_2^2-\|P_Ce\|_2^2.\tag{14}$$

**证明。** $P_C$对称幂等，是$C$行空间上的正交投影；$Q_C$投到其正交补。$n(y)$属于$C$行空间，故$Q_Ce\perp n(y)$。又$C^\dagger(C\widehat y-r_0)=n(y)+P_Ce$，因此$\widetilde y-y=Q_Ce-n(y)$。两次应用正交勾股分解：$\|Q_Ce-n(y)\|^2=\|Q_Ce\|^2+\|n(y)\|^2$，$\|e\|^2=\|Q_Ce\|^2+\|P_Ce\|^2$。相减得到式(14)。证毕。

**解释。** 若真实标签确实在约束面上，$n(y)=0$，硬投影逐点不增平方误差。若发布标签不在该面上，投影引入$\|n(y)\|^2$代价；它只有在移除的预测法向误差更大时才改善。公式既不是“物理约束永远有益”，也不是“物理约束永远有害”。

**反例。** 真实$y=1$，预测$\widehat y=1$，却强制$v=0$。原误差0，投影后误差1。反之$y=0$、$\widehat y=1$、约束$v=0$时，投影消除全部误差。这两个一维极端包含在同一公式内。

### 命题T7：软投影的最优强度依赖交叉项

设所有二阶矩有限，$d=C^\dagger(C\widehat Y-r_0)$，$e=\widehat Y-Y$，软修正$\widetilde Y_\alpha=\widehat Y-\alpha d$，$\alpha\in[0,1]$。令$u=\mathbb E[e^\top d]$、$v=\mathbb E\|d\|^2$，则风险改善为

$$G(\alpha)=2\alpha u-\alpha^2v,\qquad
\alpha^*=\operatorname{clip}_{[0,1]}(u/v)\quad(v>0).\tag{15}$$

若$v=0$，$d=0$几乎处处，所有$\alpha$产生相同预测，可约定$\alpha^*=0$。

**证明。** 展开$\|e\|^2-\|e-\alpha d\|^2$并取期望得到二次式。$v>0$时该式严格凹，无约束驻点为$u/v$，闭区间上的极大点为其截断。$v=0$的非负随机变量期望为零，故$d=0$几乎处处。证毕。

记$a=\mathbb E\|P_Ce\|^2$、$b_0=\mathbb E\|n(Y)\|^2$、$c_0=\mathbb E[(P_Ce)^\top n(Y)]$，则

$$u=a+c_0,\qquad v=a+b_0+2c_0.\tag{16}$$

只有另加$c_0=0$且$a+b_0>0$，才能写$\alpha^*=a/(a+b_0)$。不能从“约束违背很大”直接推出“应该强修正”，因为违背可能来自真实发布差异。当前数据中稀少的发布修正也不能在未来以真值掩码驱动$\alpha$。

### 5.1 随机预测协调不等于新增观测

设两个无偏、二阶可积预测$p_1,p_2\in\mathbb R^q$的误差为$e_1,e_2$，协方差为$\Sigma_1,\Sigma_2$，交叉协方差$C_{12}=\mathbb E[e_1e_2^\top]$。沿用第2节固定$W\succ0$，令$S=\Sigma_1+\Sigma_2-C_{12}-C_{12}^\top\succ0$。在$\mathsf K_{\rm fuse}$可取任意$\mathbb R^{q\times q}$矩阵、不附加稀疏/凸权重等结构限制时，考虑$p_1+\mathsf K_{\rm fuse}(p_2-p_1)$。展开矩阵二次风险得到唯一最优线性融合

$$\mathsf K_{\rm fuse}^*=(\Sigma_1-C_{12})S^{-1}.\tag{17}$$

**推导。** 写$d=e_2-e_1$，$\mathbb E[dd^\top]=S$，$\mathbb E[e_1d^\top]=C_{12}-\Sigma_1$。取$K_*=\mathsf K_{\rm fuse}^*$，把风险写成与矩阵无关的常数加上$\operatorname{tr}\{W(\mathsf K_{\rm fuse}-K_*)S(\mathsf K_{\rm fuse}-K_*)^\top\}$。由于$W,S\succ0$，该项非负且仅在$\mathsf K_{\rm fuse}=K_*$时为零，从而得到式(17)。对$S$奇异或附加结构约束的矩阵，不能直接声称此唯一解，需要另解相应问题。

这与预测组合和协调文献的基本联系一致[R3,R7]，不是新增信息定理。把第二个模型的未来预测当作独立真实测量，会忽略$C_{12}$并可能过度收缩不确定性。

## 6. 未知观测机制：必须先谈识别

### 命题T8：线性功能的可识别性由行空间决定

固定$L_0\in\mathbb R^{m\times d}$，只观察$y=L_0x$，允许$x\in\mathbb R^d$。线性功能$v^\top x$能由$y$唯一确定，当且仅当$v\in\operatorname{row}(L_0)$。

**证明。** 若$v=L_0^\top a$，则$v^\top x=a^\top y$。反过来，若$v$不在行空间中，则它在$\ker L_0$上的正交投影$h$非零，且$v^\top h=\|h\|^2>0$。$x$与$x+h$产生相同$y$，但目标功能不同，不能唯一确定。证毕。

若额外限制$x$属于某个物理可行集合，行空间条件仍充分，但不再对任意可行集合都必要；需要在该集合的观测纤维上重新判断。更一般地，功能$\psi(x)$可识别当且仅当它在每个非空纤维$\{x:L_0x=y\}$上为常数。

### 命题T9：观测等价世界给出的不可恢复下界

若两个数据生成世界在所有可用观测$O$上具有相同分布，而待恢复目标分别恒为$a$与$b$，对任意仅依赖$O$的估计器$\widehat\theta$，有

$$\max\{\mathbb E_a\|\widehat\theta-a\|_W^2,\mathbb E_b\|\widehat\theta-b\|_W^2\}\ge\tfrac14\|a-b\|_W^2.\tag{18}$$

**证明。** 两个世界中$\widehat\theta$同分布。对其共同分布取期望，并应用恒等式

$$\tfrac12(\|z-a\|_W^2+\|z-b\|_W^2)=\|z-(a+b)/2\|_W^2+\tfrac14\|a-b\|_W^2.\tag{19}$$

最大风险至少是两风险均值，而右边第一项非负。证毕。

**已验证反例。** 世界一$X=1,H=1$；世界二$X=2,H=1/2$，两者均观察到$Y=1$。若目标是$X$且容量允许两者，最坏平方恢复误差至少$1/4$。若目标只是在同样机制下预测未来同分布$Y$，这个反例并不禁止准确预测$Y$。不能把“微观分解不可识别”扩大成“发布目标不可预测”。

增加同时作用于**同一状态**的已知观测可以改善识别：若堆叠矩阵$L_0$满列秩，噪声为零时$x=(L_0^\top L_0)^{-1}L_0^\top y$；有噪声$y=L_0x+\eta$时，误差满足$\|\widehat x-x\|_2\le\|L_0^\dagger\|_2\|\eta\|_2$。满秩不等于稳定，最小奇异值很小会放大噪声。不同时间的观测不能直接堆叠为“同一$x$”，除非另外建立并验证动态状态转移模型。


## 7. 非线性联合预测：真正超出重聚合的部分

### 命题T10：非线性目标需要额外条件矩

已知线性$H$可以与条件期望交换，但对非线性函数$h$一般没有$\mathbb E[h(X)\mid\mathcal F]=h(\mathbb E[X\mid\mathcal F])$。以主动暴露$A$和功率$P$为例，在有限二阶矩下：

$$\mathbb E[AP\mid\mathcal F]=\mathbb E[A\mid\mathcal F]\mathbb E[P\mid\mathcal F]+\operatorname{Cov}(A,P\mid\mathcal F).\tag{20}$$

**证明。** 写$A=\mu_A+(A-\mu_A)$、$P=\mu_P+(P-\mu_P)$，展开四项。两个单中心项的条件期望为零，余项分别为均值乘积与协方差。乘积可积由Cauchy–Schwarz保证。证毕。

对连续能量，使用A0的正规条件核$\Pi_\iota$，假设$A(u),P(u)$联合可测，且$\mathbb E\int_I[A(u)^2+P(u)^2]du<\infty$。由$2|AP|\le A^2+P^2$有绝对可积性；对$\mathsf I$分布几乎所有条件值$\iota$，核积分$\int_I\int|A(u,\omega)P(u,\omega)|\Pi_\iota(d\omega)du$也有限。因此可对同一核应用Fubini，将条件期望与时间积分交换。条件均值和协方差均选择由该核给出的联合可测版本，对Lebesgue几乎所有$u$定义。由此几乎处处有：

$$\mathbb E[V_I\mid\mathcal F]=\int_I\big[\mu_A(u)\mu_P(u)+\kappa_{AP}(u)\big]du,\tag{21}$$

其中$\kappa_{AP}(u)=\operatorname{Cov}(A(u),P(u)\mid\mathcal F)$。如果均值部分已精确给出，忽略协方差造成的Bayes均值偏差是$\int_I\kappa_{AP}(u)du$；其平方才是相对于该特定plug-in均值的可消除风险，实际估计该协方差仍要付出T2中的估计代价。

**近似模型：二阶展开。** 设标量函数$h\in C^3$，其定义域是包含$X$和$\mu$的凸集，三阶导数算子范数在该集合上被确定常数$M$统一控制，且$X$有有限三阶矩。Taylor积分余项给出

$$\begin{aligned}\mathbb E[h(X)\mid\mathcal F]&=h(\mu)+\tfrac12\operatorname{tr}\{\nabla^2h(\mu)\Sigma_X\}+r_h,\\
|r_h|&\le\tfrac M6\mathbb E[\|X-\mu\|_2^3\mid\mathcal F].\end{aligned}\tag{22}$$

这里一阶项的条件期望为零，二次型的条件期望为trace项；余项绝对可积由三阶矩保证。这不是无条件小误差结论：只有当三阶中心矩及$M$的乘积足够小时，二阶近似才有精度意义。对阈值、max和持续事件等不光滑目标，不能直接使用式(22)。

### 7.1 为何此处仍不能宣布新方法

如果已直接观测并预测五分钟电量$E_k$，小时电量只是$\sum_kE_k$，仍回到线性情形。为了制造非线性而引入不可识别的$A,P$，并没有增加真实信息。只有在存在可识别的联合状态/功率观测、目标确实需要其依赖结构、且现有直接联合模型不能充分处理时，协方差路线才有独立意义。

非线性点预测协调、概率投影与UKF条件化已有直接近邻[R4,R5]；混合类型概率协调也已有[R6]。其限制可以启发研究，但“已知方法有局限”不等于“任意组合就具有创新”。需要明确证明或实证说明新的支持、依赖或信息条件。

### 7.2 混合支持的合法概率表达

一种候选模型把未来路径拆成离散端口状态$S$与连续区间量$U$，定义

$$q_\theta(s,u\mid\mathcal F)=q_\theta(s\mid\mathcal F)q_\theta(u\mid s,\mathcal F),\qquad Y=\mathcal O_c(s,u)+\varepsilon.\tag{23}$$

这是条件分解恒等式加上一个**待选模型类**，不是新定理。$q_\theta(s\mid\mathcal F)$应尊重可验证的整数状态支持；连续部分应尊重单位、非负性和观察区间。UrbanEV的清洗小数或平均暴露不能直接当成整数$S$。

若$Y=\mathcal O_c(s,u)$存在精确确定性关系，其联合分布可能位于低维流形上。此时对整个环境空间的Lebesgue密度做普通Gaussian NLL可能没有合法真值密度，不能把奇异分布写成通常的非退化多元密度。必须声明离散计数×连续部分的支配测度，或在真实发布噪声模型下获得适当密度；另一种途径是用定义在样本空间上的proper score评价情景。

若条件真分布$P_F$和候选$Q_F$对同一支配测度有密度$p_F,q_F$，$P_F\ll Q_F$，且以下熵与交叉熵均有限，则负对数损失的超额风险为

$$\mathcal R_{\log}(Q)-\mathcal R_{\log}(P)=\mathbb E\operatorname{KL}(P_F\Vert Q_F)\ge0.\tag{23a}$$

**推导。** 在给定$\mathcal F$后，将$-\int p_F\log q_F$减去$-\int p_F\log p_F$，得到KL积分。由$\log x\le x-1$在$p_F>0$的支持上应用于$q_F/p_F$，$\int p_F\log(q_F/p_F)\le\int_{p_F>0}q_F-1\le0$，故KL非负；再对$\mathcal F$取期望。有限性条件使相减不出现未定义的$\infty-\infty$。这是标准proper log-score关系，不是新的概率模型证明。

对有限第一矩的欧氏随机向量，常用energy score为

$$\operatorname{ES}(Q,y)=\mathbb E_{X\sim Q}\|X-y\|_2-\tfrac12\mathbb E_{X,X'\stackrel{\rm iid}{\sim}Q}\|X-X'\|_2.\tag{24}$$

它是既有proper scoring rule[R8]，不是本项目贡献。固定尺度变换后比较才有清楚单位含义；逐边际校准不等于联合分布正确，某个proper score的小改善也不自动证明下游事件足够可靠。

## 8. 持续不可用风险：一个有独立理论动机的新目标

本节明确改变目标，从原始均值预测转为未来路径事件。不能用本节指标把旧RMSE失败改写为成功。

设$K\in\mathbb Z_{\ge0}$为未来等长观察区间数、$L\in\mathbb Z_{\ge1}$为所需连续长度。不可用率为$O_k/C_z$，固定阈值$\tau\in(0,1]$，令$Z_k=\mathbf1\{O_k/C_z\ge\tau\}$。当$K<L$时约定下式索引集合为空、事件为0；一般定义

$$E_{K,L}=\mathbf1\left\{\exists j\in\{1,\ldots,K-L+1\}:\prod_{i=j}^{j+L-1}Z_i=1\right\}.\tag{25}$$

这是“未来窗口内至少出现一段连续高不可用”事件。它不等于真实排队人数、拒绝服务或用户等待时间；被占用、被动占用和故障状态可能都属于不可用。若想研究等待或收益，需要另外的目标观测与决策模型。

### 命题T11：边际分布不能确定持续事件概率

当$K=L=2$，存在三个联合分布，使每个$Z_k$都服从Bernoulli$(1/2)$，但$\Pr(E_{2,2}=1)$分别为$1/2,1/4,0$。

**证明。** 分布一以各$1/2$概率取$(0,0),(1,1)$；分布二四种二元组合各$1/4$；分布三以各$1/2$概率取$(1,0),(0,1)$。每个坐标的边际成功率均为$1/2$，而持续事件只在$(1,1)$上发生，得到三个概率。证毕。

这不是对某个模型的模拟成绩，而是一个精确构造反例：即使单时刻边际预测完美，仍不能唯一确定持续风险。因此，联合依赖的研究动机不依赖于“再降低一点平均RMSE”。然而，若采用强直接事件分类器就能解决问题，也没有必要添加复杂路径模型。

### 命题T12：run-length状态的命中概率递推

建立下述明确的概率模型。未命中事件时，$R_k\in\{0,\ldots,L-1\}$是未来前$k$步末尾的连续1长度。因为式(25)只计算**未来窗口内部**的连续段，取$R_0=0$；不能把过去已有连续段无说明地算入事件。

假设存在原点已固定的转移概率$p_{k,r}$，满足

$$\Pr(Z_{k+1}=1\mid\mathcal F,Z_{1:k},\text{此前未命中})=p_{k,R_k},\quad 0\le p_{k,r}\le1.\tag{26}$$

这是条件run-length Markov假设，而非由观测数据自动推出的物理定律。设$q_{k,r}=\Pr(R_k=r,\text{此前未命中}\mid\mathcal F)$，则

$$\begin{aligned}
q_{0,0}&=1,\quad q_{0,r}=0\ (r>0),\\
q_{k+1,0}&=\sum_{r=0}^{L-1}q_{k,r}(1-p_{k,r}),\\
q_{k+1,r+1}&=q_{k,r}p_{k,r},\quad 0\le r<L-1,\\
\Pr(E_{K,L}=1\mid\mathcal F)&=1-\sum_{r=0}^{L-1}q_{K,r}.
\end{aligned}\tag{27}$$

**证明。** 对$k$归纳。初始无未来记录且未命中，故初值成立。给定未命中状态$r$，下一步为0时连续长度归零；下一步为1且$r<L-1$时长度加一；若$r=L-1$且下一步为1，则进入吸收的命中事件，不再计入$q$。由条件全概率公式得到前两条转移。未命中状态互斥且穷尽补事件，故最后一式成立。证毕。

当$L=1$时只有状态0，事件概率为$1-\prod_{k=0}^{K-1}(1-p_{k,0})$。若$K<L$则无法在未来窗口内凑足长度，定义事件概率为0。$p=0$或1、$K=0$等边界可以直接从递推验证。算法复杂度为$O(KL)$时间、$O(L)$内存，其中$K,L$是可变参数，转移概率计算成本不含在此界内。

run-length的有限Markov嵌入是经典方法[R10]，不能作为新算法。潜在贡献只能来自EV特有的可识别状态、延迟/缺失条件及相对于强事件分类器的独立价值。

### 8.1 最小可实现的候选模型

可以先采用$\operatorname{logit}p_{k,r}=\beta^\top u_t+\gamma_k+\eta_r$，其中$u_t$仅含原点可见的不可用状态、历史持续性、容量与日历；$\gamma_k$为视野参数，$\eta_r$为run-length参数。这是一个**研究候选**，尚无真实训练结果。不能将未来观测协变量直接代入$p_{k,r}$；若使用随机未来协变量，需要对其预测分布积分，或明确标注为plug-in近似。

至少比较三类强对照：直接logistic/树事件分类器；独立边际概率或一阶状态转移模型；上述run-length模型。若只增加$\eta_r$便有收益，应通过相同输入、容量与时间预算的控制证明持续依赖的必要性。普通半Markov持续时间建模也不是新概念。

对真正满足更新/持续时间模型的停留时长$T$，生存函数$S(a)=\Pr(T>a)$给出$\Pr(T>a+\Delta\mid T>a)=S(a+\Delta)/S(a)$，仅在$S(a)>0$时定义。把区域阈值过程当成会话更新过程是额外假设；不可由“看起来持续很久”直接推出。

## 9. 缺失观测下的事件部分识别

### 命题T13：单调持续事件的可达上下界

对未来每一格，观测可能明确给出$Z_k=0$、$Z_k=1$或未知。定义$z_k^-$为已知值，未知取0；$z_k^+$为已知值，未知取1。将式(25)看作单调布尔函数$\phi$，令$E^- =\phi(z^-)$、$E^+=\phi(z^+)$。在没有额外跨时约束、且任意未知位的0/1补全都被允许时：

$$E^-\le E_{K,L}\le E^+,\qquad
\mathbb E[E^-\mid\mathcal F]\le p\le\mathbb E[E^+\mid\mathcal F],\quad p=\Pr(E_{K,L}=1\mid\mathcal F).\tag{28}$$

这些是允许补全集合上的sharp界：存在相容的补全达到下端点，也存在相容补全达到上端点。

**证明。** 对任意相容补全$z$逐坐标有$z^-\le z\le z^+$。连续全1事件对每个坐标单调，故$\phi(z^-)\le\phi(z)\le\phi(z^+)$。条件期望保序得到概率界。全部未知补0或全部补1分别相容并达到两个端点；逐观测模式采用这两种补全可构造达到条件期望端点的相容世界。若附加动力学/容量耦合约束排除这些补全，sharp性需重新证明。证毕。

等价的计算形式：$E^-=1$当且仅当有长度$L$的窗口全部为已确认1；$E^+=1$当且仅当有长度$L$的窗口中没有已确认0。二者都可线性扫描计算。

### 9.1 从不完整端口覆盖得到阈值区间

若总容量$C_z$已知，当前记录中有$b_k$个确认不可用端口、$m_k$个状态未知端口，则实际不可用率位于

$$\frac{b_k}{C_z}\le\frac{O_k}{C_z}\le\frac{b_k+m_k}{C_z}.\tag{29}$$

若左端已达到$\tau$，则$Z_k=1$；若右端严格小于$\tau$，则$Z_k=0$；否则为未知。如果连总容量或“未知端口数量”都未知，式(29)也不能直接使用。不能把缺失站点当作零不可用率。

这里的未来缺失模式只用于**未来结果发生后的目标区间标签构造**，不作为预测原点输入。训练时可以利用过去完整发生的$(E^-,E^+)$，但能否从这些区间标签学习真实$p$，取决于缺失机制与额外假设。只在完整样本上训练通常得到$\Pr(E=1\mid\mathcal F,\text{完整观测})$，不自动等于总体$p$。

### 9.2 可以学习识别端点，而不伪造精确事件标签

对已经完整发生、但观测有缺口的历史窗口，$E^-,E^+$本身仍可计算。以两个Brier目标学习

$$\min_{q_-,q_+}\;\mathbb E[(E^--q_-(\mathcal F))^2+(E^+-q_+(\mathcal F))^2],\quad 0\le q_-\le q_+\le1.\tag{28a}$$

在不限制函数类的总体问题中，由T1分别应用于两个标签，唯一最优解为$q_-^*=\mathbb E[E^-\mid\mathcal F]$、$q_+^*=\mathbb E[E^+\mid\mathcal F]$；由于条件期望保序，它们满足约束，所以约束不排除Bayes解。再由T13，$q_-^*\le p\le q_+^*$。

这是一个完整的总体识别目标，但有限样本的$\widehat q_-,\widehat q_+$不自动夹住真实$p$。可用共享小模型并通过$q_-=\sigma(a)$、$q_+=q_-+(1-q_-)\sigma(b)$保证输出顺序，其中$\sigma$是logistic函数；该参数化只是实现建议，有限实数参数不能精确表示0或1端点，模型逼近误差仍需评价。

必须区分三个对象：**识别区间**描述无限样本下仍可能存在的机制歧义；**端点估计的置信区间**描述有限样本误差；**单次未来状态的预测区间**描述随机结果的不确定性。它们不是同一个“90%区间”，不能混用覆盖率解释。只预测两个端点也不等同解决了未知缺失机制。

### 命题T14：概率区间对应的稳健决策

令告警行动$a\in\{0,1\}$，误报成本$c_{\rm FP}>0$，漏报成本$c_{\rm FN}>0$。已知$p$时，行动0和1的期望成本分别为$c_{\rm FN}p$与$c_{\rm FP}(1-p)$，最优告警阈值为$\tau_c=c_{\rm FP}/(c_{\rm FP}+c_{\rm FN})$。

若仅知$p\in[\underline p,\overline p]$，则最坏成本为

$$\overline R(0)=c_{\rm FN}\overline p,\qquad
\overline R(1)=c_{\rm FP}(1-\underline p).\tag{30}$$

故minimax规则在$c_{\rm FP}(1-\underline p)\le c_{\rm FN}\overline p$时告警，反之不告警；相等时两者都最优。

**证明。** 两个成本关于$p$分别单调增加与单调减少，故区间上极大值分别在上端与下端取得。比较两个极大值即得规则。证毕。

若允许一个明确付费的人工复核/弃权行动，且其成本恒为$c_A\ge0$，则在三者$c_{\rm FN}\overline p,c_{\rm FP}(1-\underline p),c_A$中选择最小值。这只是指定成本下的决策，不是现实运营收益证明；成本需在新任务中事先约定或实测。

### 推论T14a：事件Brier改进如何联系决策价值

令$p=\mathbb E[E\mid\mathcal F]$，并要求$\widehat p: \Omega\to[0,1]$是**预测原点$\mathcal F$可测、在相应结果发生前固定**的概率预测。成本、plug-in阈值和平局约定也须在评价前固定，不能使用该评价标签选择。Brier风险满足

$$\mathbb E(E-\widehat p)^2=\mathbb E[p(1-p)]+\mathbb E(p-\widehat p)^2.\tag{31}$$

证明是T1在Bernoulli目标上的直接应用：$E$有界，$\widehat p$有界且$\mathcal F$可测，取标量$W=1$，所需二阶矩和可测性均满足，条件方差为$p(1-p)$。取正成本$c_{\rm FP},c_{\rm FN}$与其诱导阈值$\tau_c=c_{\rm FP}/(c_{\rm FP}+c_{\rm FN})$，明确规定$a(\widehat p)=\mathbf1\{\widehat p\ge\tau_c\}$、$a^*=\mathbf1\{p\ge\tau_c\}$，二者使用同一平局约定。此时

$$\mathbb E[\ell(E,a(\widehat p))-\ell(E,a^*)]\le(c_{\rm FP}+c_{\rm FN})\sqrt{\mathbb E(p-\widehat p)^2}.\tag{32}$$

**证明。** 当两个行动相同时，条件遗憾为0。否则$p$与$\widehat p$位于阈值两侧，条件遗憾为$(c_{\rm FP}+c_{\rm FN})|p-\tau_c|$，而$|p-\tau_c|\le|p-\widehat p|$。取期望并用Cauchy–Schwarz得到式(32)。阈值相等的平局约定不影响上界。证毕。

因此Brier超额风险有明确决策含义，但仅降低边际占用RMSE不能代替式(31)中的事件概率误差。反过来，某次Brier降低也不保证实际告警成本一定按同一比例下降；式(32)只是上界。

### 推论T14b：未知事件概率下的minimax Brier点预测

如果必须用一个概率$q\in[0,1]$代替识别区间$[\underline p,\overline p]$，以最坏Brier风险为准则，则

$$q^{\rm rob}=\operatorname{proj}_{[\underline p,\overline p]}(1/2).\tag{32a}$$

**证明。** Bernoulli$(p)$下的Brier风险为$q^2+p(1-2q)$。当$q\le1/2$时，最坏$p$为$\overline p$，对应二次函数在$q=\overline p$达到无约束极小；当$q\ge1/2$时，最坏$p$为$\underline p$，无约束极小为$q=\underline p$。分别考虑$\overline p\le1/2$、$\underline p\ge1/2$及区间跨越$1/2$，全局极小点为$\overline p$、$\underline p$、$1/2$，即投影公式。边界相等时连续衔接。证毕。

所以稳健概率不一定是区间中点。它刻画指定识别集合上的最坏风险折中，不保证频率校准；若业务允许，保留整个区间比压成一个数更能表达信息不足。


## 10. 新数据和新传感信息的价值：先补什么才值得

### 命题T15：Gaussian条件下的信息价值闭式

设给定$\mathcal F$后，未来目标$Y$与**原点时已实际获得的新增信息**$Z$联合Gaussian，条件均值为$m_Y,m_Z$，条件协方差块为$\Sigma_{YY},\Sigma_{YZ},\Sigma_{ZZ}$，且$\Sigma_{ZZ}\succ0$。所有二阶矩有限。令$\mathcal G=\sigma(\mathcal F,Z)$，则

$$\begin{aligned}
\mathbb E[Y\mid\mathcal G]&=m_Y+\Sigma_{YZ}\Sigma_{ZZ}^{-1}(Z-m_Z),\\
\Sigma_{Y\mid\mathcal G}&=\Sigma_{YY}-\Sigma_{YZ}\Sigma_{ZZ}^{-1}\Sigma_{ZY},\\
\mathcal R_W^*(\mathcal F)-\mathcal R_W^*(\mathcal G)&=\mathbb E\operatorname{tr}\big(W\Sigma_{YZ}\Sigma_{ZZ}^{-1}\Sigma_{ZY}\big).
\end{aligned}\tag{33}$$

**证明。** 令$K_0=\Sigma_{YZ}\Sigma_{ZZ}^{-1}$、$\xi=Y-m_Y-K_0(Z-m_Z)$。条件协方差$\operatorname{Cov}(\xi,Z\mid\mathcal F)=0$。在联合Gaussian条件下，联合特征函数的二次型交叉项为零，因此分解为边际特征函数乘积，得到条件独立。于是$\mathbb E[\xi\mid\mathcal F,Z]=0$，并由展开二阶矩得到协方差公式。再用T3：均值增量为$K_0(Z-m_Z)$，其条件二次矩为$\operatorname{tr}(WK_0\Sigma_{ZZ}K_0^\top)$，对$\mathcal F$外层取期望即可。证毕。

这是经典Gaussian conditioning的应用[R9]，不是新传感算法。若$Z$是未来才知道的真实能量、占用或缺失掩码，它不满足新增可用信息的前提。若$Z$只是旧输入的确定变换，T3仍要求信息价值为零；有限拟合得到的非零估计不能推翻该结论。若$\Sigma_{ZZ}$奇异，需去掉确定性冗余方向或另行定义广义逆版本，本命题没有覆盖它。

### 10.1 从公式反推数据优先级

式(33)说明，值得补充的信息须同时具有：对当前信息无法解释的部分仍有剩余方差；与目标残差具有条件关联；噪声和冗余没有完全吞掉该关联。记录版本生效时间的价值不一定是直接提分，它还可能让模型从“未知合同”转入可识别条件，从而使实验解释成立。

建议按问题分别补数据：

- 对观测合同路线，优先补发布版本、逐条可见时间、缺失与修正日志。没有这些字段，实时漂移识别和严格在线评估缺少依据。
- 对持续不可用路线，优先补连续、带mask的端口状态与固定容量；同时记录被动占用/故障的语义，避免把不可用误当排队。
- 对混合能量路线，优先补同站同时间的主动状态、会话时长、电量与功率/采样语义。只补另一个不同口径的能量城市数据不能识别原有微观分解。

这是信息价值导向的数据设计，不是已证明的采集收益。实际新增观测若有成本，必须把风险和成本转换为同一决策单位后，才能比较价值减成本；不能直接把RMSE下降百分比等同货币收益。

## 11. 理论可获益，不等于有限样本能证明

### 命题T16：固定候选的有界验证下界

设候选$j=1,\ldots,J$已经在验证块出现前冻结。给定训练信息，$n$个验证块独立同分布。块$b$中基线损失与候选损失之差为$D_{b,j}\in[-M_\ell,M_\ell]$，其中$M_\ell>0$为事先确定的常数；不同候选之间可以相关。记$\mu_j=\mathbb E[D_{b,j}\mid\text{训练信息}]$、$\overline D_j=n^{-1}\sum_bD_{b,j}$。对任意$\delta\in(0,1)$，以至少$1-\delta$的条件概率同时有

$$\mu_j\ge\overline D_j-M_\ell\sqrt{\frac{2\log(J/\delta)}n},\quad j=1,\ldots,J.\tag{34}$$

**证明。** 每个$D_{b,j}$区间长度为$2M_\ell$。独立有界变量的Hoeffding不等式[R11]给出$\Pr(\overline D_j-\mu_j\ge r\mid\text{训练信息})\le\exp\{-nr^2/(2M_\ell^2)\}$。取$r=M_\ell\sqrt{2\log(J/\delta)/n}$，每个事件概率至多$\delta/J$。对$J$个候选使用并集界得到式(34)，不需要候选之间独立。证毕。

这项命题适合解释样本需求，不能直接对现有连续30天数据宣称有限样本保证：时间相关未消除；扩展训练还使后续模型使用前期标签，不满足“所有候选在全部验证块前冻结”的简单条件。预先写好block bootstrap也不自动产生独立性。

对于Brier损失，预测限制在$[0,1]$可使损失在$[0,1]$，因此$M_\ell=1$。原始MSE和NLL一般无界，不能不加尾部条件就套用式(34)。若将损失截断到有界范围，必须承认验证对象变成了截断风险。

式(34)还提示：如果希望识别的**绝对**平均损失改善为$\gamma>0$，使误差项小于$\gamma$至少要求$n>2M_\ell^2\log(J/\delta)/\gamma^2$。这是保守最坏情形数量级，不是本项目实际所需天数；也不是把“2%相对RMSE”直接当$\gamma=0.02$。两者的单位与函数都不同。

### 11.1 对已有试点的正确解释

T2给出总体信号与估计误差竞争；T16说明样本较少时难以验证微小改善。当前0.001%左右的占用增益或略负的能量增益，不应被扩大成普遍存在/不存在信息的结论。与此同时，既有分诊门已失败，也不能用“样本不足”作为无限加模型的理由。

后续确认应至少固定目标、信息合同、损失、模型集合、最小效应、选择规则与相互独立程度可辩护的时间/来源边界。bootstrap的统计量必须与主指标一致，或者明确另报；现有试点的pooled-day区间不等于逐折平均增益的区间。

## 12. 从推导得到的研究路线，而非先选模型再找公式

### 12.1 三条路线的条件分岔

| 条件 | 理论判断 | 推荐行动 |
|---|---|---|
| 已知线性合同、相同信息、同线性设计 | T4/T5：解析法和直接回归可等价 | 作为强控制；暂停重新包装线性适配 |
| 发布差异可由过去配对观测定义，且T2剩余信号足够 | 修正存在风险下降空间，但估计成本必须可控 | 先比强直接预测和已有软校正，再谈适配方法 |
| 未知覆盖/分母/发布机制造成观测等价 | T8/T9：某些微观对象不可唯一恢复 | 补元数据/观测；或报告识别集合，不能强造latent真值 |
| 目标是能量乘积、阈值或持续事件 | T10/T11：均值/边际不足以确定目标 | 研究联合依赖，但必须有相应真值和强直接目标基线 |
| 未来事件标签因缺失只能部分观察 | T13/T14：概率与决策可有可达界 | 优先研究缺失条件下的持续事件风险与可靠决策 |

### 12.2 建议的主研究问题

**候选主线：不完全观测下，持续不可用事件的可识别概率预测与决策。**

它比“再加一个合同MLP”有更清楚的理论必要性：T11给出边际不足的反例；T13给出缺失条件下可识别的信息；T12提供小而具体的联合模型；T14将概率误差联系到明确告警成本。这里的单项算法都不是首创，潜在贡献须落在**真实EV观测条件下，这四者是否形成一个不能被简单直接事件分类器替代的闭合问题与方法**。

这一建议是基于理论与现有负结果的路线判断，不是已完成的创新证明。现阶段不能声称已掌握事件频率、缺失模式、Markov充分性或运营成本；这些尚待数据资格与独立实验。

### 12.3 最小方法规格：先声明可观察什么

输入：原点可用的历史端口状态、容量、mask、日历和发布时间；输出：指定$(K,L,\tau)$下的事件概率，或在缺失不可忽略时输出识别区间。目标只计未来窗口内部持续段，初始化$R_0=0$。方法顺序：

1. 先审计目标可识别性：若大量历史目标只能得到宽区间，不先用任意填补训练“精确标签”。
2. 训练直接事件logistic/树模型作为强小模型；其本身可能已经足够。
3. 在同信息、近似容量和相同选择预算下比较有限run-length模型。状态充分性是需要反证的模型假设。
4. 缺失机制未获支持时，以T13界和T14稳健决策报告信息边界；不得将conformal标签附加为自动覆盖保证。

不把联合混合能量、图传播、动态定价、基础模型和蒸馏一起并入主线。若未来获得同站状态/电量数据，T10与式(23)可形成独立扩展，但当前应保持一篇工作一个核心问题。

### 12.4 必须能否定该路线的三个实验

**实验A：联合依赖必要性。** 固定边际校准与同一历史信息，比较直接事件分类、独立边际构造、一阶状态模型、run-length模型。主指标为Brier和固定告警预算下漏报；同时报告事件基率与可靠性。若run-length不优于强直接分类器，则不把路径模型当贡献。

**实验B：缺失与识别边界。** 首先评价真实记录中的缺失；人工遮蔽只作明确标注的压力测试。验证真实完整标签落入区间的逻辑正确性，并检查区间宽度和稳健决策代价。若真实数据几乎完全观察，部分识别只是边界说明，不应成为主标题；若缺失端口数也未知，则必须补数据或收紧任务。

**实验C：新证据与行动价值。** 在方法和阈值选择前锁定新的时间/来源；在明确成本比或固定告警预算下评价，并报告相同目标上的强简单对照。若只有人为扰动情形改善、实际事件过少或效果仅来自换指标/增加输入，则否定原方法主张。

这些是后续设计而非本次执行结果。旧UrbanEV开发区可用于探索目标是否存在，不能重新命名为确认集；Paris formal/protected和原隔离库仍不因本报告而开放。

### 12.5 理论成熟度与研究成熟度分开

| 项目 | 当前状态 |
|---|---|
| 条件风险、线性等价、投影代价、识别下界 | 在声明条件下有完整推导 |
| 非线性协方差与Taylor界 | 精确恒等式或带显式矩/光滑条件的近似 |
| 持续事件递推、缺失sharp界、稳健行动 | 在声明Markov/相容补全/成本条件下成立 |
| EV数据满足上述额外条件 | 部分未验证 |
| 新方法优于强基线、跨城市稳定、部署有效 | 未证明 |
| 世界首创/顶会录用资格 | 未建立，不能由理论文稿或自动审查分数推出 |

## 13. 参考文献与知识来源边界

以下文献用于定位既有理论和最近邻，不把其报告分数转录为本项目成绩。原始数学推导在正文给出，文献引用不能替代证明条件。

[R1] Li, H. et al. UrbanEV: An Open Benchmark Dataset for Urban Electric Vehicle Charging Demand Prediction. Scientific Data, 2025. [数据论文](https://www.nature.com/articles/s41597-025-04874-4)。支持数据背景，不替代本地发布版本审计。

[R2] Girolimetto, D.; Di Fonzo, T. Point and probabilistic forecast reconciliation for general linearly constrained multiple time series. Statistical Methods & Applications. [出版社页面](https://link.springer.com/article/10.1007/s10260-023-00738-6)。用于一般线性约束边界。

[R3] Wickramasuriya, S. L.; Athanasopoulos, G.; Hyndman, R. J. Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization. JASA, 114(526), 804–819, 2019. [作者页面与DOI](https://robjhyndman.com/publications/mint/)。MinT及协方差协调已有先例。

[R4] Girolimetto, D. et al. Forecast reconciliation with non-linear constraints. arXiv:2510.21249, 2025. [预印本](https://arxiv.org/abs/2510.21249)。

[R5] Biswas, A.; Zambon, L.; Nespoli, L.; Corani, G. Nonlinear Probabilistic Forecast Reconciliation. arXiv:2604.26668, 2026. [全文](https://arxiv.org/html/2604.26668v1)。方法及§3.3限制用于非线性概率近邻定位，不声称一般概率协调空白。

[R6] Zambon, L. et al. Probabilistic reconciliation of mixed-type hierarchical time series. UAI 2024, PMLR 244, 4078–4095. [会议页面](https://proceedings.mlr.press/v244/zambon24a.html)。

[R7] Athanasopoulos, G.; Hyndman, R. J.; Kourentzes, N.; Panagiotelis, A. Forecast reconciliation: A review. International Journal of Forecasting, 40(2), 430–456, 2024. [论文](https://doi.org/10.1016/j.ijforecast.2023.10.010)。统计discrepancy与软约束并非本报告首创。

[R8] Gneiting, T.; Raftery, A. E. Strictly Proper Scoring Rules, Prediction, and Estimation. JASA, 102(477), 359–378, 2007. [作者保存全文](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)。用于proper score与分布评价。

[R9] Rasmussen, C. E.; Williams, C. K. I. Gaussian Processes for Machine Learning. MIT Press, 2006, Chapter 2. [作者官方开放章节](https://gaussianprocess.org/gpml/chapters/RW2.pdf)。用于Gaussian条件分布的标准背景。

[R10] Fu, J. C.; Koutras, M. V. Distribution Theory of Runs: A Markov Chain Approach. JASA, 89(427), 1050–1058, 1994. [DOI](https://doi.org/10.1080/01621459.1994.10476841)。有限Markov嵌入是经典方法，正文对本任务的递推独立证明。

[R11] Hoeffding, W. Probability Inequalities for Sums of Bounded Random Variables. JASA, 58(301), 13–30, 1963. [DOI](https://doi.org/10.1080/01621459.1963.10500830)。式(34)使用独立、有界前提，未对当前时间序列自动套用。

[R12] Molinari, F. Microeconometrics with Partial Identification. Handbook of Econometrics 7A, 355–486, 2020. [作者预印本](https://arxiv.org/abs/2004.11751)。用于观测等价与部分识别的概念边界。

[R13] Zheng, K. et al. Coherent Hierarchical Probabilistic Forecasting of Electric Vehicle Charging Demand. arXiv:2411.00337. [预印本](https://arxiv.org/abs/2411.00337)。不将arXiv上传年份等同首次发表年份。

[R14] Li, C. et al. A Behavior-Guided Online Probabilistic Forecasting Method for Electric vehicle Charging Loads. arXiv:2608.24441, 2026. [预印本](https://arxiv.org/abs/2608.24441)。EV行为变化与延迟反馈已有近邻，不把普通在线校准作为创新。

[R15] Forecasting Electric Vehicle Charging Station Occupancy: Smarter Mobility Data Challenge. arXiv:2306.06142, 2023. [原始任务](https://arxiv.org/abs/2306.06142)。Paris数据可用性与其许可/分析边界需单独遵守。

## 附录A. 推导依赖与假设账本

| 结果 | 直接依赖 | 额外条件 | 在EV应用中是否已验证 |
|---|---|---|---|
| T1–T3 | 条件期望、Cauchy–Schwarz、平方展开 | 二阶矩、原点信息、固定正定W | 数学条件可声明；真实发布时间未全知 |
| T4 | T1、线性条件期望 | 已知有界H；协方差交叉项保留 | 部分发布合同已核验 |
| T5 | 固定设计凸二次优化 | 同$Z_{\rm des}/D_{\rm w}/P_{\rm reg}/\lambda$，正规矩阵可逆 | CPU试点满足并数值核验 |
| T6–T7 | 正交投影、二次优化 | 同度量；C满行秩；有限二阶矩 | 真实等式误差必须单独测量 |
| T8–T9 | 行空间/零空间、两世界构造 | 状态集合或观测等价世界明确 | 对微观恢复尚不足以识别 |
| T10 | 条件中心矩、Fubini/Taylor | A0正规条件核、可积乘积；近似另需C3和三阶矩 | 状态/功率同构数据未齐 |
| T11–T12 | 反例、全概率、归纳 | T12需run-length充分状态 | 未验证 |
| T13 | 单调布尔函数、条件期望保序 | 任意未知位补全相容才sharp | mask及容量资格待核验 |
| T14及推论 | 条件成本比较、T1、Cauchy–Schwarz | 固定正成本，概率/区间定义正确 | 运营成本尚未实测 |
| T15 | T3、Gaussian条件分布 | 新观测原点可用；协方差非奇异 | 未有完整新增观测试验 |
| T16 | Hoeffding、并集界 | 固定候选、条件独立验证块、有界损失 | 当前30天试点不满足直接套用条件 |

依赖图无循环：T1为风险基础，T2/T3/T4从其或条件期望出发；T5、T6、T8、T9、T11、T13是独立代数/构造节点；T7使用T6符号；T12是有限归纳；T14只使用明确成本和T1；T15使用T3；T16是独立验证层。所有近似和研究候选均不作为已证明命题的前提。

## 附录B. 证据、复现与未开放数据

本报告读取的是既有汇总、协议与论文来源；没有新增真实数据训练或打开新的目标时间段。关键本地证据包括CPU试点协议、宏指标表、线性与发布差异诊断、完整性核验和最新整合报告；随附工件索引保存精确路径与哈希。PDF随附可编辑Markdown、LaTeX与证明审查记录。

数值反例检验只用于核对公式和边界，不构成新的EV实验。任何自动证明审查都不是外部人工同行评议，也不是机器形式化验证。原有失败协议、Paris formal/protected与隔离库边界不因理论报告而变更。


## 附录C. 证明审查、反例核查与交付边界

本报告经过三轮独立自动数学审查，并对关键修复节另做一次独立盲审及修订核查。最终主审判定为PASS，修复节的最终独立核查也为PASS。这表示在文内明确假设下未发现剩余实质证明缺口，不等同外部人工同行评议、形式化证明验证或新方法有效性证明。

### C.1 本轮确实修复了什么

| 问题 | 初始缺口 | 修复方式 | 最终状态 |
|---|---|---|---|
| Brier推导的可测性 | 若允许概率预测偷看评价标签，分解不成立 | 明确原点可测、结果前冻结及成本诱导阈值 | 关闭；另做独立节级核查 |
| 细粒度基线条件 | 均值估计未在推论中显式写入信息集 | 补充二阶可积与原点可测，训练信息并入信息集 | 关闭 |
| 条件期望与时间积分 | 绝对可积说明未完全交代条件版本 | 添加标准Borel设定、正规条件核和逐条件Fubini | 关闭 |
| 线性代数符号 | 正则、投影和目标矩阵使用相同字母 | 用分节专名与类型表区分 | 关闭 |
| 事件的退化边界 | 原始定义域未包含零视野/长度不足 | 统一非负视野、正连续长度、空索引事件为0 | 关闭 |
| 线性融合的适用域 | 未在该段重申权重正定与矩阵自由度 | 固定正定损失矩阵，并排除结构化/奇异变体 | 关闭 |

初始Brier缺口的反例可以手算：信息集平凡，事件以一半概率发生，若非法取预测等于评价结果，经验平方误差为0，而按未补条件的分解右侧为1/2。修复后该预测不满足命题前提。这个反例的价值是暴露信息条件，而不是证明真实数据模型存在泄漏。

### C.2 代数与有限枚举检验

另外完成19组代数/边界核查，共18,432个有限案例，结果通过。其中包括风险正交分解、同设计ridge等价、错误约束投影、run-length递推与完整路径枚举、缺失位所有补全、稳健Brier解、Gaussian协方差半正定性和独立Bernoulli尾界核对。

这些检验帮助发现符号和实现错误，但不替代正文证明；人工构造并非新的EV训练或测试数据。没有把枚举案例数解释成真实样本量，也没有用数值测试推断未证明的普遍命题。

### C.3 输出与仍待研究的事项

随附Markdown保存完整理论文字；LaTeX可重新编译PDF；证明账本、逐轮原始评审和代数检查结果保留在理论工作目录。引用核验区分原期刊卷年、后来的在线数字化日期和arXiv上传日期，未把它们混作研究发生时间。

尚未建立的内容包括：真实EV过程满足run-length充分状态假设、未知缺失机制下端点可稳定估计、新数据的信息价值、非线性机制收益、独立跨期/跨城确认和运营收益。报告提出的成熟推导框架不能代替这些实证条件。

原有受保护数据、隔离库与历史失败协议的状态保持不变。
