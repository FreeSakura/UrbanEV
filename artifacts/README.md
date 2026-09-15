# 公开工件

`summaries/` 保存各阶段公开的数值、登记、选择和执行回执；`manifests/` 保存当前树、历史来源、论文和外部 Release 的完整性信息。

| 要做的事 | 入口 |
|---|---|
| 阅读综合数值和适用范围 | [结果总览](../results/README.md) |
| 查某个证据目录属于哪项研究 | [研究登记表](../results/studies.json) |
| 查文件大小和当前公开内容哈希 | [逐文件清单](../results/inventory.csv) |
| 核对完整公开仓库 | [FULL_RELEASE_MANIFEST.json](manifests/FULL_RELEASE_MANIFEST.json) |
| 核对外部预测包 | [RELEASE_ASSET_MANIFEST.json](manifests/RELEASE_ASSET_MANIFEST.json) |
| 重算或重新执行 | [复现指南](../docs/guides/reproducibility.md) |

原实验目录是证据来源。登记、校准、合成验证和真实开发成绩的含义各不相同；目录存在不代表完整训练已经结束。修改说明不应批量改写原执行源码哈希或原决定。

部分历史 CSV/JSON 存有迁移前的文档路径。这些属于原时点引用，保留原字节；通过[路径映射](../docs/maintenance/path-map.json)查新位置，或在映射中的 source_commit 下查看旧文件。当前可导航关联使用研究登记表。
