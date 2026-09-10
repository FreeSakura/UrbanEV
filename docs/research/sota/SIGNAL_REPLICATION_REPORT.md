# 第二折信息线索复核报告

2026-09-10。实验配置与代码在准备第二折输入前以821dd96冻结，之后未调参。

## 1. 结论

- 区域相关分组：**NO_GO**。
- 时长信息：ridge为**GO_DEVELOPMENT_ONLY**；冻结Chronos-2为**NO_GO**。
- 两类预测器（ridge与Chronos-2）联合复现：**未成立**。首折已失败机制的状态不改变。

## 2. 相同窗口中的结果

全部结果为275区域、H3/H12等权宏平均，RMSE和MAE在[0,1]占用率尺度上计算。原始及裁剪指标、逐H结果与预测文件哈希均保留在汇总JSON。

### Chronos-2

| 系统 | RMSE | MAE |
|---|---:|---:|
| full275 | 0.08805563 | 0.05603717 |
| random32_s42 | 0.08825081 | 0.05616995 |
| random32_s43 | 0.08817718 | 0.05618483 |
| random32_s44 | 0.08830954 | 0.05623888 |
| raw_correlation32 | 0.08842643 | 0.05606706 |
| duplicate_occupancy | 0.08842643 | 0.05606706 |
| lagged_duration | 0.08835392 | 0.05595254 |
| permuted_duration | 0.08850191 | 0.05598642 |

| 候选 / 参考 | RMSE改善% | 95%块区间 | MAE差值 | 点门 |
|---|---:|---|---:|---|
| raw_correlation32 / full275 | -0.4211 | [-0.8248, 0.1343] | 0.00002989 | NO_GO |
| lagged_duration / duplicate_occupancy | 0.0820 | [-0.4126, 0.4191] | -0.00011452 | NO_GO |
| lagged_duration / permuted_duration | 0.1672 | [-0.0532, 0.3669] | -0.00003388 | NO_GO |

### 固定ridge

| 系统 | RMSE | MAE |
|---|---:|---:|
| occupancy | 0.09011314 | 0.05928088 |
| duplicate_occupancy | 0.09015135 | 0.05931081 |
| lagged_duration | 0.08843656 | 0.05860989 |
| permuted_duration | 0.09017394 | 0.05936931 |

| 候选 / 参考 | RMSE改善% | 95%块区间 | MAE差值 | 点门 |
|---|---:|---|---:|---|
| lagged_duration / occupancy | 1.8605 | [0.1006, 3.0322] | -0.00067099 | GO_DEVELOPMENT_ONLY |
| lagged_duration / permuted_duration | 1.9267 | [0.2267, 3.0714] | -0.00075942 | GO_DEVELOPMENT_ONLY |

错配对照行的JSON `point_gate`也计算了通用1%诊断阈值；冻结协议的联合判断实际只要求正确时长胜过错配时长、MAE不劣，且相对最佳同信息控制通过1%主门。机器可读summary按这一协议判断。

## 3. 范围与解释

第二折训练[0,1171)，验证[1171,1318)，测试[1318,1464)；只准备前1318行。前折末段进入后折训练符合扩展协议，不能称所有历史测试时间均未读取。本窗口对本候选首次使用，但历史审计接触过全数据集，不能称全局盲测或独立确认。

区域分组取最后288训练小时的绝对相关，固定K32；三个随机分组均报告，并以最强对照定门。时长输入截止origin−2；每个占用组与辅助通道组成独立任务，增广从275变为550通道。复制占用与错配时长是有限控制，不能当作因果检验。

ridge与Chronos使用各自固定的同信息控制，不做新验证集选超参数。确定性ridge没有伪造多种子重复；三个随机分组种子反映分区变化，不能当作神经训练种子。

块区间在预测原点上重采样，保留全部区域和H步，共2000次、块长24。区间只描述本窗口，最佳参考身份固定；没有纳入所有模型选择、城市和季节不确定性。MAE区间跨零时只可说点估计改善，不可说稳定非劣。

时长的区间累计小时单位已由官方文档支持，但具体端点、发布延迟和原始预处理的因果性仍未确认；现阶段是离线预测研究。数据版本限定与出处见[source semantics](../../../artifacts/summaries/signal_replication_v1/source_semantics.json)。

## 4. 失败分层与下一步

| 项目 | 已观察到的问题 | 可以保留的结论 | 不能声称的解释 |
|---|---|---|---|
| 简单相关分组 | 第二折宏RMSE比全区域差0.4211%，H3/H12均更差 | 首折正收益没有在本窗口复现；停止把它作为稳定改进主线 | 尚未证明由时间漂移、冗余上下文或特定注意力机制造成 |
| 直接给Chronos添加时长 | 相对最佳同信息控制仅改善0.0820%，95%区间[-0.4126%,0.4191%] | 没有建立达到1%门槛的实质优势 | 不能证明D完全无信息，也不能断言只是基础模型利用方式有问题 |
| ridge添加时长 | RMSE改善1.8605%，但MAE差值区间跨零 | 保留线性模型中的开发信息线索 | 不能扩大为所有模型均有效或MAE稳定非劣 |
| 绝对强度与成本 | 时长Chronos的RMSE仍比全区域Chronos差约0.3388%；增广从275增至550通道 | 复制和错配输入提供有限对照 | 小幅相对收益不等于SOTA或更高效率 |
| 数据语义 | 单位已找到文档支持，端点和上线发布时间仍未知 | 可以进行限定版本的离线预测比较 | 不能认定主动/非主动充电潜状态或线上可部署性 |

原始相关分组在第二折的增益区间也跨零；这里的NO_GO表示未满足预定推进门，不是已证明所有相关分组都显著有害。两个路线均不再针对这段开发窗调整组大小、时间窗、正则或输入延迟。

下一候选应先检验训练内部滚动留出中，时长是否能预测**基础模型残差**。仅对Y可预测未必说明还有超越基础模型的空间，也可能模型已从占用历史提取了相同信息。[理论第6节](SIGNAL_INFORMATION_THEORY.md)给出修正风险差ΔR(w)=2wᵀc−wᵀΣw及其总体ridge形式。该候选本轮未运行；必须另立协议，采用仅占用修正和错配时长的等容量控制，再评价新开发窗口。旧单纯形头、残差引导交换、观测算子的失败状态均不回写。

本轮实际运行16个Chronos系统/视野单元和8个ridge单元。Chronos推理总用时约17.6分钟；CPU ridge与部分运行重叠，且增广输入量不同，因此不构成严格速度基准。真实前向核查确认增广上下文为550×168，9组大小为8×64+38，见[结构回执](../../../artifacts/summaries/signal_replication_v1/augmented_structure_receipt.json)。

## 5. 工件

独立核查结论为 **WARN，无完整性失败**。全部ridge与Chronos预测指标、哈希、区间和推进门经本地重算一致，模型身份与此前固定快照匹配，公开叙述通过核查。WARN涉及开发范围、时长代理和MAE区间边界，见[核查回执](../../../artifacts/summaries/signal_replication_v1/integrity_review.json)。本地59项测试通过。

[固定方案](SIGNAL_REPLICATION_PLAN.md) · [理论反馈](SIGNAL_INFORMATION_THEORY.md) · [机器可读结论](../../../artifacts/summaries/signal_replication_v1/summary.json) · [Chronos结果](../../../artifacts/summaries/signal_replication_v1/foundation.json) · [ridge结果](../../../artifacts/summaries/signal_replication_v1/ridge.json)。

原始输入、分组成员、预测数组与权重只保存在本地。公开导出保留原执行哈希与规范化JSON哈希。首次基础模型启动因解释器缺chronos而失败，随后使用此前验证的环境、相同代码和配置成功重跑；失败不是科学结果，也未被删除。

复现使用此前固定的Chronos专用环境、冻结模型权重和经身份核验的第二折训练/验证CSV；普通CPU环境可以单独运行ridge：

```bash
python scripts/research/replicate_signals.py --route ridge --csv local-data/prepared-f2.csv --duration local-data/UrbanEV/duration.csv --info local-data/UrbanEV/inf.csv --output local-data/replication-ridge
python scripts/research/replicate_signals.py --route foundation --csv local-data/prepared-f2.csv --duration local-data/UrbanEV/duration.csv --info local-data/UrbanEV/inf.csv --model-dir local-data/chronos2 --output local-data/replication-chronos
```

两种运行都要求新输出目录，元数据中记录输入、脚本、协议、模型及依赖身份；不要用另一版本的百分比occupancy文件替换本项目计数快照。
