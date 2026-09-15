# 复现指南

仓库级检查、已存预测重算和重新训练回答不同问题。选择所需层级，记录使用的 Git 提交、配置与输出目录。

| 层级 | 输入 | 可验证的内容 |
|---|---|---|
| 公开资料复核 | 当前仓库 | 文档链接、研究覆盖、公开数值汇总、哈希与历史审计一致性 |
| 合成工程验证 | 仓库和 CPU PyTorch | 模型、窗口、评分和训练流程 |
| 已存预测重算 | 相应 Release 预测包和合法取得的数据 | 目标身份一致后重算历史审计指标 |
| 原模型重执行 | 固定源码、环境、数据及必要权重 | 指定协议中的完整模型执行 |

## 公开资料复核

```bash
python -m pip install -e ".[test]"
python scripts/repository.py check
python -m pytest
python scripts/repository.py verify-manifest
python scripts/privacy_audit.py --root . --git-history
```

`repository.py check` 会从原 CSV 重算开发核心宏指标，核对冻结汇总，检查登记表是否覆盖每个证据目录，检查 Markdown 本地链接及生成文件是否过期。它不读取原始目标，也不重新训练。Git 历史审计需要保留 `.git`；使用源码归档时，可去掉 `--git-history` 只检查当前文件。

维护者在修改公开文件后更新派生资料：

```bash
python scripts/repository.py build
python scripts/build_model_source_manifest.py
python scripts/build_manifest.py
python scripts/repository.py check
```

`build_manifest.py` 更新本次工作树的完整性清单；原实验的执行源码哈希仍表示当时版本，不应被批量改成当前源码哈希。

`verify-manifest` 只验证当前仓库的文件哈希。历史 `urbanev_audit verify` 还要求 `release-assets/` 中的 ZIP 与校验文件齐全，不能作为全新克隆的无下载检查；外部工件审计使用下节的单独流程。

## 合成工程验证

参照[快速开始](quickstart.md)安装 CPU PyTorch 和 research 扩展，然后运行完整测试及 `smoke`。CI 分别执行轻量资料检查、预测测试和历史论文构建。合成数据通过测试不意味着真实预测优势。

## 历史审计结果重算

适用于 v0.9.1 审计论文及其目标不随包公开的 Release 工件：

```bash
python scripts/download_release_assets.py --tag v0.9.1-preprint --output local-data/release-assets
python scripts/audit_release.py --asset-root local-data/release-assets
python -m urbanev_audit recompute --scope headline --data-root local-data/targets --asset-root local-data/release-assets --development-shard local-data/paris/development_state_shard.csv --output local-data/recomputed-headline.json
```

先按[数据指南](data.md)组织和登记数据。上例将 UrbanEV 的 `occupancy.csv` 和 `inf.csv` 直接放在 `local-data/targets/`，Paris 分片通过 `--development-shard` 单独指定。工具只在目标哈希精确匹配后评分；具体输入重建见[目标实现](../../src/urbanev_audit/targets.py)。

Paris 默认只接受开发分片 `development_state_shard.csv`，列为 `date, Station, Available, Charging, Passive, Other`。开发日期上界为 2020-11-30。完整 `train.csv` 必须额外显式使用 `--allow-full-source`；正式及受保护部分不在公开复现范围内。

## 完整六折比较

冻结协议为 [URBANEV_MATCHED_SIX_FOLD_V1_20260914.json](../../configs/research/URBANEV_MATCHED_SIX_FOLD_V1_20260914.json)，任务和输入要求见[阶段报告](../reports/benchmark/MATCHED_SIX_FOLD_REPORT_20260914.md)。该轮实际评分结果尚未完整公开。

执行顺序是：全部监督训练与验证选择完成 → 冻结模型 → 保存监督及基础模型预测 → 校验全部载荷 → 统一评分。对应入口：

```bash
python scripts/research/run_full_benchmark.py --help
python scripts/research/run_full_foundation.py --help
python scripts/research/score_full_benchmark.py --help
```

训练入口需要 `--config`、`--data`、`--private`、`--author-roots` 和 `--phase train|predict`；基础模型入口另需固定 `--model-dir` 和 `--backend`。作者源码与预训练权重由本地路径提供，身份按合同核验。通用 research 扩展不包含所有基础模型后端；后端版本以该轮配置和下载清单为准，不能把[历史 GPU 环境](../../environment-gpu-cu121.yml)当作所有新实验的环境锁。

这一高成本任务不属于安装测试，也不由资料重建自动触发。新的执行应明确其范围，并使用匹配的源提交；不得绕过代码身份、已完成任务或预测完整性检查。

## 论文重建

LaTeX 和字体依赖见[论文说明](../../paper/README.md)。公开 PDF 已提供，阅读无需安装 TeX。重建后的 PDF 需要通过论文清单和字体检查；文档导航重构不会自动刷新历史论文数值。
