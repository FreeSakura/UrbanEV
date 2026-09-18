# 当前论文与历史归档

## 当前持续事件论文

**Persistent Event Evaluation under Partial Observations — A Sharp Structural Regime for Overlapping Labels**

- [可编辑 Word 全稿](persistent_events/WORD_MANUSCRIPT.docx)：英文正文、完整证明附录与中文摘要。
- [Markdown 稿源](persistent_events/manuscript.md) · [图表与数据](persistent_events/figures) · [主张证据映射](persistent_events/EVIDENCE_MAP.md)。
- [端到端复现与 Word 构建](persistent_events/README.md)：当前稿件使用 Pandoc 和原生 Word 公式，读取和编辑不需要 LaTeX。

贡献已冻结为核心定理 A、具有经典来源的几何推论 B、冻结实证。当前稿件是匿名完整工作稿，尚无期刊发表信息。[当前状态](../docs/PROJECT_STATUS.md)

## 历史审计论文与理论报告

以下为原 UrbanEV 审计论文与阶段理论文件，沿用各自的结果和协议；不是当前持续事件论文的投稿主文。

| 文件 | 角色 |
|---|---|
| [Main v0.9.1](main/UrbanEV_Evidence_Audit_Main.pdf) | 历史审计预印本主文 |
| [Supplement v0.9.1](supplement/UrbanEV_Evidence_Audit_Supplement.pdf) | 详细协议、运行链与补充证据 |
| [历史封面归档](archive/UrbanEV_Evidence_Audit_Historical_Archive_v0.9.0.pdf) | 显式标明历史身份的 v0.9.0 归档 |
| [原始归档](archive/UrbanEV_Evidence_Audit_Archive.pdf) | 保留的原 42 页 PDF |
| [V3 理论反馈](research/UrbanEV_Theory_Feedback_V3.pdf) | 缺失观测下的配对事件审计；[中文报告](../docs/reports/audit/PAIRED_AUDIT_V3_REPORT.md)和[推导](../docs/theory/DERIVATION_V3.md) |

全部 PDF 和 Markdown 可从[完整资料目录](../docs/catalog.md)检索。`main/`、`supplement/`、`archive/` 保存 LaTeX 入口，`shared/` 保存共享正文、结果、图表和参考文献。`main_archive_original.tex` 保留原始入口，`main_archive.tex` 为明确标记的历史重建入口。

## 历史 LaTeX 稿重建

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
