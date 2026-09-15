# 仓库维护与资料迁移

## 目录责任

入口说明集中在根 README、文档中心和项目状态；`docs/research/` 描述当前问题与规则。日期报告归入 `docs/reports/`，理论归入 `docs/theory/`，阶段设计归入 `docs/protocols/`，早期方案归入 `docs/history/`。

`artifacts/summaries/` 保存原公开证据，`results/studies.json` 登记每个目录的研究身份和对应说明，`results/README.md` 为派生结果总览。不要在多个导航页持续追加逐轮日志。

## 2026-09-15 迁移

基于主分支 `5f38f7d151b58fd4f1dd8ab17981b5ee59659678`，将原集中在 `docs/research/sota/` 的资料按内容迁移，重写 README、状态、综合报告、研究路线及使用指南，保留日期报告的科学结论。完整旧路径、原文 SHA256 和新路径记录在[path-map.json](path-map.json)。旧链接可在该文件的 source_commit 对应版本下访问。

原研究登记、训练和评分代码、模型源码、冻结配置、公开数值及执行回执保持原身份。论文 PDF 与其来源未因资料整理而改写。绘图/报告渲染脚本更新了文档输出位置；历史 source manifest 仍表示原运行版本，不应被改成新路径下的当前哈希。

V3 PDF 已嵌入旧推导页面的主分支链接，因此该旧路径保留一个简短兼容入口，正文仅位于新的理论目录。冻结 CSV/JSON 内的历史文档路径保持原值，当前对应关系由路径映射和研究登记提供。

旧“当前方案”页中有过期范围和阶段状态。它们的有用科学内容保留在日期记录或专题文件；新的当前入口以主分支公开证据为依据，不推断本地训练任务的实时完成情况。

## 生成与校验

```bash
python scripts/repository.py build
python scripts/build_model_source_manifest.py
python scripts/build_manifest.py
python scripts/repository.py check
```

资料构建器使用标准库，从登记表和已有 CSV/JSON 构建结果页、证据清单、分类索引及 Markdown/PDF 全目录。生成输出不含当前时间或本机路径，因此相同输入产生相同内容。检查模式只读，不自动修复文件。

开发核心表只接受 `NEW_MATCHED_CORE`、`terminal_H`、`raw`、全部四个 H 及正确实例集合；先按实例平均四 H，再平均实例，不混入 H3/H12 缓存参考。它同时对照原 `model_summary.json`，确保派生解释没有改变原结果。

链接检查覆盖本地 Markdown/图片和引用式链接，验证 Markdown 标题锚点及路径大小写；外部网页可达性不在该离线检查的结论范围。新增资料应先登记再生成，避免把缺失文件从检查范围中排除。
