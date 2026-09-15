# 基线与来源登记

本页按“已有开发结果、正式注册、相关文献”区分状态。它不是排行榜，也不是重新开展的文献检索。来源与查阅时点沿用截至 2026-09-14 的公开研究记录。

## 已有开发比较

| 系统 | 信息与身份 | 公开证据 |
|---|---|---|
| RIDGE_O / RIDGE_OD | 本区域 168 小时 O，或 O＋滞后 D；均含上下文 | 两次确定性拟合，四 H 核心比较 |
| OD_PRODUCT / OD_SEPARABLE / OD_CONCAT_MLP | 固定 RIDGE_OD 底座上的残差模型 | 每配置三种子，同开发训练合同 |
| TimeXer LOCAL-OD / GLOBAL-O | 固定作者核心的本地适配，信息轨道不同 | 每配置三种子；不是官方六折逐字复现 |
| Chronos-2 native | 已缓存预训练输出 | H3/H12 共同支持，不能进入四 H 完整宏表 |
| Last / day / week | 确定性历史参考 | 按具体桥接或实验范围报告 |

精确成绩来自[结果总览](../../results/README.md)，方法与参数细节来自[开发比较报告](../reports/benchmark/COMPREHENSIVE_COMPARISON_REPORT_20260913.md)。

## 正式六折注册

固定清单包含 Last/Day/Week，三种 ridge，四种残差配置，TimeXer 两个信息轨道、DLinear、PatchTST、iTransformer，以及本地 Chronos-2 和 TimesFM3。注册不等于运行完成。完整输入、作者提交、权重身份与训练预算见[六折配置](../../configs/research/URBANEV_MATCHED_SIX_FOLD_V1_20260914.json)。

DLinear 的早期合成验证及“尚未真实运行”是准备阶段状态；其后已进入正式任务清单。各方法的完成状态应通过该轮回执确认，不能从安装环境或可用源码推断。

## 文献与历史来源

[完整来源表](../../artifacts/summaries/comprehensive_development_comparison_v1/baseline_sources.csv)登记官方 UrbanEV、交通基础模型工作、DyConfuse-Net、TriCast、Urban-CSTPNet、MDFANet、MAGE 等来源及协议差异。旧查阅记录见[文献与基线快照](../reviews/BASELINE_REVIEW_20260914.md)。

原论文表格、摘要分数、ST-EVCDP247 区 5 分钟任务、能量/负荷目标以及本项目局部占用率开发值不构成同一比较总体。未复现的已发表强方法应保留为证据缺口；不能直接按其小数值计算本项目距离全球前沿的差距。

模型核心、完整本地系统和作者原实验分别命名；CAPER 的项目实现身份不能改称作者源码复现。第三方快照与许可见[模型身份](../MODEL_IDENTITY.md)及[第三方声明](../../THIRD_PARTY_NOTICES.md)。
