# 配置与协议

根目录配置主要对应历史审计与模型执行；`research/` 保存后续预测研究的版本化合同。配置存在表示规则已写入，不等于实验已经完成。

当前六折配置是 [URBANEV_MATCHED_SIX_FOLD_V1_20260914.json](research/URBANEV_MATCHED_SIX_FOLD_V1_20260914.json)。其阶段状态由[注册与公开回执](../artifacts/summaries/urbanev_matched_six_fold_v1/)说明，报告见[完整六折合同](../docs/reports/benchmark/MATCHED_SIX_FOLD_REPORT_20260914.md)。

其他配置按[研究登记表](../results/studies.json)关联报告、入口与结果；文字协议见[阶段协议目录](../docs/protocols/README.md)。

冻结配置包含数据、模型或执行源码身份。维护仓库时保留原配置，新的科学实验另建显式版本；不能为消除哈希错误而覆盖原身份字段。历史 GPU/模型下载清单只适用于注明的运行，不自动代表新基础模型的完整安装环境。
