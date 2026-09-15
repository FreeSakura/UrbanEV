# 一次H3/H12训练内部校准：限定授权与运行清单

> 阶段记录：下文保留原研究时点的结论和执行范围；当前公开状态见[项目状态](../PROJECT_STATUS.md)。目录迁移不改变原结果。

研究验收已接受求解器提交e7deb7754efea619e540dccdc810b5c3ebdd7e38，固定数值配置SHA256为6c905dc4c4f7a98376d13277f6dba4ca13da027ffe9de31e7ff8f774345cf9ca。本次新建[授权记录](../../configs/research/CONTINUOUS_V2_CALIBRATION_2H_AUTHORIZATION.json)，不覆盖旧配置中的pending或历史失败记录。

授权只包含一次`calibrate_h3_h12`。新增基础推理为0，自动推进为false；无论本阶段信息门结果如何，都停在研究验收点。H6/H9、[1560,1747)尾部、第三折验证[1747,1966)、测试[1966,2184)及Paris保护数据保持关闭。

[冻结运行清单](../../configs/research/CONTINUOUS_V2_CALIBRATION_2H_RUN.json)保留三个窗口，拟合原点分别44/72/100个，校准各14个，标签全部在相应较早拟合前缀或校准段内。监督拟合、特征和标签的语义读取最大到1559行，D最后使用origin−2。

准入在真实标签读取前检查：固定阶段、注册和数值配置身份、已验收代码身份、执行入口身份、完整七系统与全部时间原点。默认未提供明确授权时仍拒绝。代码身份检查保持求解器、精确边界、科学评分/过门及V1拟合源码与已接受版本一致，仅允许授权函数和新接入模块变化。

只复用V1的H3/H12基础预测缓存。完整缓存不透明字节哈希与数值读取分别记账；映射缓存后只复制所需114个原点，末目标不超过1559，既有尾部预测不读取其数值。V1残差方向用原CPU函数、原float32目标减法顺序和2线程限制重建，并要求α=1重建预测与V1相应缓存精确数值相等；这是血缘核验，不是补选步长。

native固定0；其余六系统分别调用同一个已接受求解器，在三个窗口、两个视野之间共享一个α。合法零步长外层状态使用其内层score参与过门，不能误判成软件故障。任一必要系统INPUT_BLOCKED或NUMERICAL_BLOCKED就阻塞整个阶段，剩余系统和科学门明确NOT_RUN。

执行前完成纯合成的错误阶段、配置/代码哈希、时间标签越界、缓存失配和必要对照阻塞检查。入口和运行清单须提交且保持干净，执行时以一次性claim防止自动重复。准备阶段只为[0,1560)前缀计算不透明校验和，不解析标签；实际语义读取账本和模型调用监测随结果回传。

```bash
python scripts/research/run_continuous_calibration_2h.py --csv local-data/v1-prepared.csv --duration local-data/duration.csv --info local-data/inf.csv --cache local-data/v1-screen --output local-data/calibration-2h
```

结果只可能为CALIBRATION_2H_BLOCKED、CALIBRATION_2H_INFORMATION_NO_GO或CALIBRATION_2H_PASS_REVIEW_REQUIRED。详细数组留在本地，公开授权/读取回执、七系统与六单元指标、完整门判定及求解诊断。这是训练内部开发校准，不是盲测、第三折验证成绩或SOTA声明。
