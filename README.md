# UrbanEV Forecast

[![CI](https://github.com/FreeSakura/UrbanEV/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FreeSakura/UrbanEV/actions/workflows/ci.yml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

**面向 UrbanEV 标准多变量预测的研究项目，目标是突破可复现强基线与 SOTA。** 主任务是275个区域的小时占用率预测，视野为3、6、9、12小时，以RMSE为主指标、MAE共同报告。当前尚未取得SOTA结果。

这不是UrbanEV数据集官方仓库。原始数据与官方方法见 [IntelligentSystemsLab/UrbanEV](https://github.com/IntelligentSystemsLab/UrbanEV)。本项目从证据审计转向预测方法研究：新模型、强基线、实验和结果是主线，既有审计工具承担验证工作。

## 最新执行结果

[双方向试验报告](docs/research/sota/DUAL_DIRECTION_PILOT_REPORT.md)：区域残差引导交换未通过，RMSE比最佳相关分组对照差1.73%；观测算子的额外优势也未成立。追加优化后，直接加入滞后时长相对占用输入的RMSE/MAE下降2.86%/3.54%，仅保留为开发线索。原始门、后验诊断和失败原因分别保留，尚无独立确认或SOTA结果。

[基础模型校准开发报告](docs/research/sota/FM_CALIBRATION_REPORT.md)：Chronos-2与TimesFM-3均完成275区域联合推理及H3/H12训练/验证实验。单纯形头在两个骨干上均为**NO_GO**；相对原始Q0.5的RMSE改善约2.18%/2.37%，但没有建立相对同信息ridge的实质优势，MAE也上升。停止扩建该候选，保留校准与ridge为强对照。完整测试与SOTA仍未建立。

## 最新选题判断（2026-09-10）

[文献与研究方向报告](docs/research/sota/RESEARCH_DIRECTIONS_20260910.md)提出的两个方向均已完成首折开发试验。当前保留简单相关分组和滞后时长的信息线索，暂停扩展未通过的机制；已有单纯形头NO_GO保持不变。

## 当前研究主线

- [研究目标与路线](docs/research/sota/RESEARCH_ROADMAP.md)：标准预测任务、现代强基线和可否定的机制研究。
- [当前完整方案与评审](docs/research/sota/FINAL_PROPOSAL.md)：五轮自动评审后转入实验，尚未达到成熟SOTA方法阈值。
- [实验协议](configs/research/URBANEV_SOTA_V1.json)：任务与划分口径、种子和推进门槛。
- [基线清单与结果状态](docs/research/sota/BASELINES.md)：区分文献结果、历史工件、尚未运行的候选，避免跨协议混排。
- [理论积累](docs/research/PAIRED_AUDIT_V3_REPORT.md)：V1–V3关于风险、事件与缺失评价的研究；这部分不是标准预测SOTA成绩。

[有界分位点预测适配](docs/research/sota/QUANTILE_MEAN_THEORY.md)的数值核心与开发实验已完成；[理论反馈](docs/research/sota/FM_CALIBRATION_THEORY_FEEDBACK.md)解释稳健中点和RMSE–MAE权衡的边界。本候选没有通过原定推进门。

首轮实现提供季节线性、季节MLP、创新注意力和原始水平注意力探针，以及现有TimeXer源代码适配。探针用于验证机制，尚未确立方法优势或新颖性。Chronos-2和TimesFM-3已接通本地冻结推理与分位缓存；首折开发对照完成，六折完整基线与测试比较尚未完成。

## 开始研究

在仓库根目录，使用Python 3.10或更高版本：

```bash
python -m pip install -e .[test,research]
python -m urbanev_forecast smoke --model innovation_attention --epochs 2 --output local-data/smoke
pytest
```

`smoke`生成合成周期序列并执行训练/验证，用于检查代码，不产生真实UrbanEV成绩。无需下载数据或大模型。

取得数据授权后，先准备第一折训练/验证前缀：

```bash
python -m urbanev_forecast.prepare --data-root local-data/UrbanEV --hours 648 --output local-data/urbanev-rates.csv
python -m urbanev_forecast train --csv local-data/urbanev-rates.csv --model seasonal_linear --fold 1 --horizon 3 --epochs 30 --output local-data/seasonal-f1-h3-s42
python -m urbanev_forecast train --csv local-data/urbanev-rates.csv --model timexer --fold 1 --horizon 3 --epochs 30 --output local-data/timexer-f1-h3-s42
```

`prepare`将占用数量除以对应区域容量；输入应为上游`occupancy.csv`和`inf.csv`。`train`只读取选定折的训练/验证前缀，保存最佳验证checkpoint、配置、数据与代码哈希。输出目录必须为空，目标与checkpoint仅留在本地。可用`--device cuda`启用现有GPU。

在方法和协议冻结后，单独使用`test`评价完整测试前缀：

```bash
python -m urbanev_forecast test --csv local-data/urbanev-rates-complete.csv --checkpoint local-data/seasonal-f1-h3-s42/checkpoint.pt --output local-data/seasonal-f1-h3-s42-test
```

该步骤要求CSV覆盖所选折的完整时间范围，并核对训练/验证前缀及当前代码与checkpoint一致。测试输出不用于选模型。详见[路线与运行顺序](docs/research/sota/RESEARCH_ROADMAP.md)。

## 代码与研究资产

| 位置 | 用途 |
|---|---|
| `src/urbanev_forecast/` | 预测模型、数据窗口、训练/测试入口 |
| `configs/research/` | 新研究协议 |
| `docs/research/sota/` | 研究路线、基线清单、实验与评审 |
| `models/timexer/` | 保留原许可证和溯源的TimeXer实现 |
| `artifacts/summaries/` | 公开汇总结果及历史证据 |
| `src/urbanev_audit/` | 数据身份、指标、隐私和工件验证工具 |
| `paper/` | 理论报告及历史审计论文 |

## 历史成果与公开边界

[原审计项目入口](docs/history/AUDIT_PROJECT_README.md)、[历史论文](paper/main/UrbanEV_Evidence_Audit_Main.pdf)及其[引用信息](docs/history/AUDIT_CITATION.cff)完整保留。历史负结果不会因项目转型被改判。旧`python -m urbanev_audit`命令继续可用。

仓库公开代码、协议、汇总与理论，不上传原始目标、缺失mask、私有路径或模型checkpoint。当前署名使用FreeSakura；旧提交和标签仍可能包含历史署名，详见[公开署名与隐私范围](docs/PUBLIC_ATTRIBUTION.md)。Paris formal/protected材料不属于本轮标准UrbanEV预测研究的输入。

原创代码采用MIT许可证；论文和原创文档采用CC BY 4.0；第三方源码与数据遵循各自许可证。运行环境依赖不会自动下载基础模型权重。引用本研究软件请使用[CITATION.cff](CITATION.cff)。
