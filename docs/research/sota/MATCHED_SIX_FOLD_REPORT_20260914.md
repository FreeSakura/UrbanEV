# UrbanEV 完整六折匹配比较

当前状态：**正式配置、代码和数据身份已冻结，等待按注册清单执行。此时没有新的测试成绩。** 完整是指本合同24格和必做清单完成；不是穷尽全球模型或复现官方论文每个运行细节。

## 任务与样本

使用现有已核验count版本转275区域小时占用率；数量先float32除以float32区域容量，评分使用float64。容量由站级charge_count按TAZID求和。完整日期与合法性检查通过；不换数据、不填补、不裁剪目标。

每月扩展六折，累计小时720/1464/2184/2928/3672/4344。每折80%训练、末10%测试，其余验证。历史168小时，H=3/6/9/12独立训练，所有方法共同训练原点169起、步长1。预测原点o的目标为[o,o+H)，主评分是最后一步。24格分别含44,780训练、5,984验证及5,960测试原点任务。完整端点每实例有1,639,000个原点×区域条目，不是独立重复样本。

历史测试类数据已经曝光。结果将表述为固定协议后的回顾性比较，不能因重新冻结就称新盲测。较早折test进入较晚折train是扩展设计的一部分；任何test分数不得反馈模型或选择。

## 固定比较清单与预算

| 组 | 系统 | 范围 |
|---|---|---|
| 零拟合 | Last、Day、Week | 相同原点的简单历史参考 |
| 普通线性 | RIDGE_O、RIDGE_O_CTX、RIDGE_OD_CTX | 分别O、O＋上下文、OD＋上下文；72次拟合，λ0.01 |
| 自有方法 | PRODUCT_MSE、SEPARABLE_MSE、CONCAT_MSE、PRODUCT_POSREG | 同格RIDGE_OD_CTX底座，H独立输出 |
| 作者核心 | TimeXer LOCAL-OD/GLOBAL-O、DLinear、PatchTST、iTransformer | 固定作者提交，明确本地资源配置 |
| 本地预训练 | Chronos-2、TimesFM3 | 各5,960原点×H调用；原生point及Q0.5分列 |

九个神经配置×24格×两个种子20260921/20260922，共432次训练。每次固定20epoch、AdamW学习率0.001余弦降至0.0001，矩阵衰减1e-4、bias/一维参数不衰减，梯度截断1。有效batch16原点×275区域、microbatch8，实际尾批按原点数加权，合计1,011,960次注册更新。保持float32、关闭AMP/TF32/compile，不做额外学习率、正则或结构网格搜索。

micro8是正式冻结前的资源修订，不是micro2逐位等价替换；BatchNorm更新及dropout随机数消耗按micro8冻结。PatchTST正式patch16/stride8已验证，不沿用早期patch12的资源估计。36个正式形状均通过合成前后向，两个基础模型8个合成API请求通过。

每次训练完成20epoch后，才比较0/5/10/15/20检查点的验证集raw端点RMSE，完全相同取较早者。不选最好种子、不集成，不合并验证数据重训。选中epoch0的残差模型是底座别名，作者模型epoch0是未训练初始化，两者分开标注。

POSREG沿用训练残差e32转float64计算RMS的规则，τ=float32(0.5×max(RMS,1e-6))，目标为MSE＋τ·mean([|e−f|−|e|]₊)，不按测试结果改权重。PRODUCT/SEPARABLE参数为8232＋73H，CONCAT为8808＋25H，仅H12时均为9108，不将其他H也称等参数。

## 完整性与评分

所有监督训练和选择完成后，冻结全部参数，再生成监督及基础模型预测；全部必做预测保存并核验哈希之后，才统一评分。训练阶段无test构造器。单次claim支持从模型、优化器、CPU/CUDA随机状态及保存步数恢复；完成任务不重新训练择优。合成GPU测试已验证一次受控epoch边界中断与连续训练的参数和验证分数完全相同。

主表按每种子24格RMSE等权平均，再平均种子指标；MAE、各H六折均值、初始化范围及raw/clip、terminal/path独立报告。禁止缺格平均、原点错配或合同混用。基础模型原生point和Q0.5分别保存；两者即使相同也标注别名，不根据成绩二选一，不将API point自动称条件均值。

每H补充相对RIDGE_OD_CTX及Chronos原生point的配对循环移动块bootstrap：块长24原点、1000次、固定种子20260914，各方法及对应种子共用同一抽样。只描述给定训练结果下的时间抽样敏感性，不生成四H总体区间或自动显著性门。

## 当前执行证据

- [完整配置与源身份](../../../configs/research/URBANEV_MATCHED_SIX_FOLD_V1_20260914.json)
- [注册预算与工程核验](../../../artifacts/summaries/urbanev_matched_six_fold_v1/registration.json)
- [72项划分原点记录](../../../artifacts/summaries/urbanev_matched_six_fold_v1/origin_plan.csv)
- [432个注册神经任务](../../../artifacts/summaries/urbanev_matched_six_fold_v1/registered_tasks.json)

本地303项测试通过；真实结果与累计成本将在执行完成后补入。其他历史方向和未对齐近期专用论文仍保留在[已有成果比较](COMPREHENSIVE_COMPARISON_REPORT_20260913.md)与[基线来源](BASELINES.md)，不冒充本次已复现。

当前不作SOTA声明。20epoch、两个种子不代表充分调优或普遍收敛；输入信息、归一化、ridge底座与直接预测、预训练成本等差异始终披露。无新基础权重下载，无新增付费算力，无定时下一轮，无额度重置。
