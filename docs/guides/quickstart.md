# 快速开始

当前任务是持续事件结构与评价。安装检查、小构造验证和稿件构建都不需要 GPU，也不需要恢复旧充电预测六折。

## 建立环境

要求 Python 3.10 或更高版本和 Git。所有命令从仓库根目录执行。

```bash
git clone https://github.com/FreeSakura/UrbanEV.git
cd UrbanEV
python -m venv .venv
```

Linux / macOS：

```bash
source .venv/bin/activate
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

安装当前科学与测试依赖：

```bash
python -m pip install -e ".[test,evidence]"
python scripts/repository.py check
python scripts/repository.py verify-manifest
```

`evidence` 提供 SciPy、scikit-learn 和 Numba 等持续事件依赖；仅安装 `test` 不足以运行所有当前科学测试。

## 不下载数据的小检查

```bash
python scripts/research/validate_persistent_event_covers.py --selfcheck --output local-data/cover-check
python -m pytest tests/test_event_cover_geometry.py tests/test_persistent_events.py tests/test_event_witnesses.py tests/test_ap1_baselines.py
```

验证器要求新输出目录，避免覆盖已有证据。这里检查构造实例、约束与轨迹实现，不运行真实模型拟合。完整的短序列穷举入口为 `scripts/research/verify_persistent_event_structure.py`。

## 选择需要的复现层级

| 需要完成的工作 | 下一步 |
|---|---|
| 阅读和修改论文 | [Word 全稿](../../paper/persistent_events/WORD_MANUSCRIPT.docx)、[稿源](../../paper/persistent_events/manuscript.md) |
| 核对冻结数值及证据 | [结果总览](../../results/README.md)、[证据映射](../../paper/persistent_events/EVIDENCE_MAP.md) |
| 使用已有预测缓存 | [缓存验证参数](../../paper/persistent_events/README.md#rebuild-the-two-fixed-evaluations-from-public-data)；只有原始缓存使用严格冻结哈希检查 |
| 从原始数据重建预测 | 同一复现说明中的 UCI 下载、AP1拟合和AP2复用流程；这是单独的科学执行 |
| 重建图表与 Word | 安装 `.[manuscript]`、Pandoc 和 Poppler，见[稿件构建](../../paper/persistent_events/README.md#regenerate-manuscript-figures-and-word) |
| 使用旧充电预测工程 | [历史预测指南](forecasting.md)，该路径另需 PyTorch |

## 常见问题

| 现象 | 处理 |
|---|---|
| 提示缺少 SciPy、Numba 或 sklearn | 在当前虚拟环境安装 `.[evidence]` |
| 找不到当前包或导入旧副本 | 确认当前 Python/虚拟环境，并在本仓库重新 `pip install -e .` |
| 验证器输出目录非空 | 指定新目录，不覆盖冻结结果 |
| 新拟合缓存不匹配历史哈希 | 配套传入新 AP1/AP2 results，不加 `--strict-frozen-hashes` |
| 找不到 Pandoc 或 pdftoppm | 安装相应本地程序；阅读已交付 Word 无需这些工具 |

[完整复现分级](reproducibility.md) · [当前研究状态](../PROJECT_STATUS.md) · [贡献说明](../../CONTRIBUTING.md)
