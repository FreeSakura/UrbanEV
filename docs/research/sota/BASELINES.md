# 强基线清单与结果状态

更新日期：2026-09-09。该清单用于规划公平对比，尚不是SOTA排行榜。统一新协议结果为空，不以文献数字补空位。

| 系统 | 来源与角色 | 新协议实现状态 | 新协议结果 |
|---|---|---|---|
| TimeXer | 已有上游快照；训练模型强基线 | 新训练入口已接通，合成前向/反向和训练通过 | 未运行 |
| Seasonal Linear / MLP | 季节锚与容量控制 | 已实现 | 未运行 |
| Innovation / Level Attention | 等参数输入表示探针 | 已实现，非成熟创新 | 未运行 |
| Chronos-2 native median | 原生多变量基础模型 | 历史版本已筛查；新版本适配待做 | 未运行 |
| TimesFM-3 native median | 2026年8月发布的原生多变量基础模型 | 源码、输出语义和权重许可已核查；推理适配待做 | 未运行 |
| Bounded quantile midpoint | 基础模型有限分位的有界均值近似 | NumPy核心已实现 | 未运行 |
| Bias / affine calibration | 排除普通校准即可解释增益 | 设计已明确，基础模型预测缓存待做 | 未运行 |
| Full-quantile ridge | 与约束头同分位信息的公平控制 | 设计已明确，实现与真实比较待做 | 未运行 |
| Simplex MSE quantile head | 一个受约束共享头的候选路线 | NumPy优化及合成测试入口已实现 | 未运行 |

## 历史结果仅作研究起点

旧内部协议的全局固定融合RMSE约0.071367，TimeXer约0.073709，Chronos-2中位数约0.073542。它们来自既有研究，不是本轮新训练成绩。必须核对原点、区域顺序、目标构造、H步范围、上下文、划分、归一化、裁剪和聚合，才能复用到新表。新入口配置与旧紧凑TimeXer的所有超参数尚未逐项建立相同身份。

旧router、蒸馏和Paris教师的失败门均保留。新的占用率RMSE研究不会把事件Brier收益或缺失评价区间缩窄计作预测成绩。

## 一手来源与适用范围

- [UrbanEV数据论文](https://www.nature.com/articles/s41597-025-04874-4)与[官方代码](https://github.com/IntelligentSystemsLab/UrbanEV)：小时区域占用率、逐月扩展折及3/6/9/12小时任务的主要对齐对象。论文表格与本项目168小时上下文不能未经配置桥接直接作优越性比较。
- [TimeXer](https://arxiv.org/abs/2402.19072)：外生信息感知时间序列预测。现有快照复用许可证和来源清单；新入口使用统一配置，尚未复现其最佳UrbanEV配置。
- [Chronos-2](https://arxiv.org/abs/2510.15821)与[官方仓库](https://github.com/amazon-science/chronos-forecasting)：支持多变量和协变量；新比较固定历史168小时及同原点275区域，不默认额外未来信息。
- [TimesFM-3官方发布](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)与[官方仓库](https://github.com/google-research/timesfm)：330M参数，原生多变量。当前3.0权重采用单独的非商业、非生产许可；代码许可与权重许可不同。本项目仅规划本地研究使用，不重新发布权重。
- [Urban-CSTPNet](https://www.mdpi.com/2079-9292/15/15/3297)：2026年EV时空概率预测相关工作；其概率目标、量纲、视野及划分与本主任务不能默认一致。公开数据声明为处理数据与补充结果按请求获取，尚未建立本项目可复现的匹配实现，不能用其数字直接排名。
- [LTSF-Linear](https://arxiv.org/abs/2205.13504)：提醒季节/线性基线不可缺席；日差分和静态线性混合不是新颖性证据。

## 点预测语义

Chronos-2的`mean`和TimesFM-3默认`forecast`均不能按名称当作均值。固定源码核查及比较规则见[POINT_FORECAST_CONTRACT.md](POINT_FORECAST_CONTRACT.md)。以中位数计算RMSE是合法基线，但训练期MSE适配需独立标为adapted系统，不能称零样本。

在声称SOTA之前，仍需补齐最新可比方法检索、源码/权重版本锁定、完整多种子结果、信息与训练预算对齐以及独立确认范围。当前只报告研究目标和已实现能力。
