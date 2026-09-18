# 代码架构与运行入口

仓库采用 `src` 布局。当前持续事件几何与评价在 `urbanev_audit` 中；它也保留历史审计模块。`urbanev_forecast` 保留原充电预测工程。包名与已有导入保持兼容，实验协议和冻结证据不因导航调整而迁移。

## 当前论文的实现链

| 环节 | 入口 |
|---|---|
| 最小游程区间和私人见证 | `src/urbanev_audit/event_cover_geometry.py` |
| 全部一元蕴含LP | `src/urbanev_audit/event_relaxations.py` |
| 轨迹精确端点 | `src/urbanev_audit/persistent_events.py` |
| 补全见证与重放 | `src/urbanev_audit/event_witnesses.py` |
| 完整cover LP/ILP与DP对照 | `scripts/research/validate_persistent_event_covers.py` |
| 固定预测生成 | `scripts/research/run_shared_missing_ap1.py`、`run_shared_missing_ap2.py` |
| 论文图表及Word | `scripts/build_persistent_event_figures.py`、`build_persistent_event_manuscript.py` |

完整参数见[论文复现说明](../../paper/persistent_events/README.md)。这些入口不依赖旧六折执行。

## 历史预测与共用维护模块

| 层 | 位置 | 职责 |
|---|---|---|
| 数据与通用模型 | `src/urbanev_forecast/data.py`、`prepare.py`、`models.py` | 率数据准备、滚动窗口、基线与通用训练输入 |
| 比较合同 | `benchmark_contract.py`、`benchmark_metrics.py`、`comparison_core.py` | 数据身份、原点、末点/path 评分及汇总约束 |
| 方法模块 | `short_state.py`、`conditional_od.py`、`continuous_*.py` 等 | 特定研究的预测器、校准和机制实现 |
| 完整比较 | `full_benchmark_*.py` 与相应 scripts | 按冻结合同组织训练、模型、数据及完整性守卫 |
| 审计工具 | `src/urbanev_audit/` | 工件模式、目标重建、指标、来源与公开边界检查 |
| 实验编排 | `scripts/research/` | 版本化研究入口，消费 `configs/research/` |
| 结果与说明 | `artifacts/summaries/`、`results/`、`docs/reports/` | 原始公开证据、派生总览与研究解释 |
| 项目维护 | `scripts/repository.py`、`scripts/build_manifest.py` | 资料导航、结果汇总、链接检查与发布清单 |

## 常用入口

| 任务 | 命令 |
|---|---|
| 通用预测训练 | `python -m urbanev_forecast --help` |
| 数量转率 | `python -m urbanev_forecast.prepare --help` |
| 固定六折训练/预测 | `python scripts/research/run_full_benchmark.py --help` |
| 固定六折基础模型预测 | `python scripts/research/run_full_foundation.py --help` |
| 全部载荷完成后评分 | `python scripts/research/score_full_benchmark.py --help` |
| 历史工件校验 | `python -m urbanev_audit verify --help` |
| 历史目标与指标重算 | `python -m urbanev_audit recompute --help` |
| 重建资料总览 | `python scripts/repository.py build` |
| 校验资料总览 | `python scripts/repository.py check` |
| 校验当前树哈希 | `python scripts/repository.py verify-manifest` |

实验脚本的对应研究、配置和证据由[研究登记表](../../results/studies.json)连接。所有脚本的逐项用途见[脚本目录](../../scripts/README.md)。

## 为什么保留部分执行路径

训练、评分、数据和模型源码参与冻结合同的哈希计算；旧 checkpoint 也绑定代码指纹。因此历史六折执行依赖及原配置保留身份。通用开发 CLI 已去除代码指纹硬阻断，只将代码身份写入评价报告。路径布局本身不是重训理由，历史执行回执也不应更新成新源码哈希。

新研究逻辑应在对应模块中实现并用独立配置声明。通用工具保持无数据副作用；导入包或重建索引不能触发下载、训练或读取真实目标。
