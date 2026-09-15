# 论文、补充材料与理论 PDF

这里保存历史审计论文及后续理论报告。当前预测研究的结果和状态见[项目状态](../docs/PROJECT_STATUS.md)；本目录的审计主文不是新的预测方法 SOTA 论文。

| 文件 | 角色 |
|---|---|
| [Main v0.9.1](main/UrbanEV_Evidence_Audit_Main.pdf) | 历史审计预印本主文 |
| [Supplement v0.9.1](supplement/UrbanEV_Evidence_Audit_Supplement.pdf) | 详细协议、运行链与补充证据 |
| [历史封面归档](archive/UrbanEV_Evidence_Audit_Historical_Archive_v0.9.0.pdf) | 显式标明历史身份的 v0.9.0 归档 |
| [原始归档](archive/UrbanEV_Evidence_Audit_Archive.pdf) | 保留的原 42 页 PDF |
| [V3 理论反馈](research/UrbanEV_Theory_Feedback_V3.pdf) | 缺失观测下的配对事件审计；[中文报告](../docs/reports/audit/PAIRED_AUDIT_V3_REPORT.md)和[推导](../docs/theory/DERIVATION_V3.md) |

全部 PDF 和 Markdown 可从[完整资料目录](../docs/catalog.md)检索。`main/`、`supplement/`、`archive/` 保存 LaTeX 入口，`shared/` 保存共享正文、结果、图表和参考文献。`main_archive_original.tex` 保留原始入口，`main_archive.tex` 为明确标记的历史重建入口。

## 重建

需要 Python 基础/测试依赖，以及 latexmk、LaTeX 扩展宏包和字体。CI 在 Ubuntu 安装 `latexmk texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra poppler-utils`。从仓库根目录运行：

```bash
python scripts/build_paper.py --variant main
python scripts/build_paper.py --variant supplement
python scripts/build_paper.py --variant archive
python scripts/build_historical_archive.py
python scripts/verify_paper_manifest.py
```

已有 PDF 可直接阅读；编辑论文时需重新构建并检查页数、字体和文本身份。导航整理不重新计算或改写论文结果。完整性信息见[论文构建清单](../artifacts/manifests/PAPER_BUILD_MANIFEST.json)。

原创论文和文档采用 [CC BY 4.0](../licenses/CC-BY-4.0.txt)。引用历史审计论文使用 [AUDIT_CITATION.cff](../docs/history/AUDIT_CITATION.cff)，软件引用使用根目录 [CITATION.cff](../CITATION.cff)。
