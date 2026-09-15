# 连续步长V2：H3/H12训练内部校准结果

> 阶段记录：下文保留原研究时点的结论和执行范围；当前公开状态见[项目状态](../../PROJECT_STATUS.md)。目录迁移不改变原结果。

**结论：CALIBRATION_2H_INFORMATION_NO_GO。** 所有计算有效，信息门和结构门均未通过。本次合法连续校准取得非零步长与小幅RMSE改善，但幅度不足1%；阶段结束后停止。

## 授权与执行

研究验收求解器提交为e7deb7754efea619e540dccdc810b5c3ebdd7e38，新增阶段入口和运行清单在9a965f3d9c87a630f71f55ffe35df3b15a02ddb5冻结后执行一次。错误阶段、配置/代码身份、越界原点、缓存失配和必要对照阻塞均经过合成检查；127项本地测试通过。旧配置pending及旧失败记录未覆盖。

只复用既有全275区域H3/H12基础预测缓存，新增基础推理为0。原V1 CPU规则重建的42个系统/单元α=1预测与旧缓存精确数值相等。所有监督拟合、特征和标签的语义读取限于[0,1560)，D最后使用origin−2；完整缓存的字节哈希单独记账，尾部预测未作数值读取。

三个窗口拟合原点44/72/100，校准各14原点，间隔12小时。它们是已使用过的训练内部校准窗口，不能当作盲测、第三折验证或SOTA结果。

该扩展训练前缀包含先前折的测试时刻；不声称所有历史测试都未接触。本次保持关闭的是所列尾部和当前第三折验证/测试范围。

## 七系统结果

| 系统 | 外层状态 | α | 宏RMSE | 宏MAE |
|---|---|---:|---:|---:|
| native | FIXED_NATIVE | 0.000000000000 | 0.081440181345 | 0.043121046533 |
| bias | ZERO_ONLY_FEASIBLE | 0.000000000000 | 0.081440181345 | 0.043121046533 |
| occupancy | ZERO_ONLY_FEASIBLE | 0.000000000000 | 0.081440181345 | 0.043121046533 |
| duplicate | ZERO_ONLY_FEASIBLE | 0.000000000000 | 0.081440181345 | 0.043121046533 |
| raw_duration | OK | 0.081825092173 | 0.081174590180 | 0.043121046533 |
| orthogonal_duration | OK | 0.097963741177 | 0.081122896363 | 0.043121046533 |
| permuted_orthogonal | ZERO_ONLY_FEASIBLE | 0.000000000000 | 0.081440181345 | 0.043121046533 |

C*为native。orthogonal_duration相对它的宏RMSE改善0.389593%，MAE不高于native；三个时间块均改善，全部单元损害约束、错配对照条件也通过。信息门唯一失败项为‘相对C*改善至少1%’。

bias、occupancy、duplicate和permuted_orthogonal返回ZERO_ONLY_FEASIBLE；这是已完成连续可行性判定后仅0可行的合法状态，不是数值故障，也不是仅因为旧离散网格缺少小步长。raw_duration和orthogonal_duration返回OK。

结构门中，相对raw_duration的RMSE改善仅0.063682%，仍低于1%。其MAE数值差约3.073e-14，按零恶化预算照实为未通过；这个极小差值不单独构成有实际意义的MAE差异论证。结构RMSE条件本身也未达到。

## 完整性与边界

全部七系统、六单元的raw/clipped指标与保存的预测/标签重新核算一致，详见核验回执。求解器未决边界、未决分段、事件连续性失败均为0；合法零步长状态使用内层score参与既有门函数，未删除任何对照，也未为过门补选α。

执行计时约79.43秒，覆盖读取/核验、CPU重建与求解等运行段；不包含提交、预检和最终报告写出。完整进程峰值内存未捕获：外部采样时进程已退出，本次没有为了补资源日志重复校准。持久化数组大小另列，只表示磁盘占用，不冒充峰值内存。

本次计算并非数值阻塞，不能把未通过1%门的结果改写成软件异常。反过来，虽有小幅改善，也不能放宽原门或宣称旧V1已通过。结果已回传研究对话；截至阶段收尾尚未收到独立结果验收。按用户最新要求结束本阶段，后续路线不自动执行，见[阶段总结](../../history/proposals/PHASE_CLOSEOUT_20260910.md)。

H6/H9推理、[1560,1747)尾部评价、第三折验证[1747,1966)、测试[1966,2184)和Paris保护数据均未执行；未使用额度重置卡。

## 工件

[运行方案](../../protocols/CONTINUOUS_V2_CALIBRATION_2H_PLAN.md) · [授权记录](../../../configs/research/CONTINUOUS_V2_CALIBRATION_2H_AUTHORIZATION.json) · [冻结清单](../../../configs/research/CONTINUOUS_V2_CALIBRATION_2H_RUN.json) · [阶段结论](../../../artifacts/summaries/continuous_v2_calibration_2h/summary.json) · [全部门判定](../../../artifacts/summaries/continuous_v2_calibration_2h/gate_decision.json) · [逐单元CSV](../../../artifacts/summaries/continuous_v2_calibration_2h/calibration_cells.csv) · [读取回执](../../../artifacts/summaries/continuous_v2_calibration_2h/execution_receipt.json)。

详细预测、标签、残差和逐事件轨迹留本地，公开仅汇总、全局求解诊断与哈希。现有数据与过去开发过程的曝光边界保持原说明，不将该阶段重新包装为独立确认。
