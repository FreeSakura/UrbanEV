# 贡献指南

欢迎改进预测工程、结果复现、研究说明和资料完整性。提交应说明具体问题、改动后的行为及验证范围。

## 开发环境

按[快速开始](docs/guides/quickstart.md)创建虚拟环境。安装 `.[test]` 可运行公开资料和审计检查；修改预测模型时同时安装 CPU PyTorch 与 `.[test,research]`，确保相关测试没有因缺少依赖而跳过。

## 修改类型

| 类型 | 维护要求 |
|---|---|
| 说明与导航 | 使用相对链接，更新对应入口，运行资料检查 |
| 新研究 | 独立版本化配置、报告和公开证据目录；登记到 `results/studies.json` |
| 科学执行逻辑 | 测试真实行为变化，明确数据/代码身份变化；保留原执行回执 |
| 既有数值纠错 | 给出来源、受影响主张及版本记录，不静默改写旧实验 |
| 论文 | 重新构建、核对论文清单，并检查版面与字体 |

## 提交前检查

```bash
python -m pytest
python scripts/repository.py build
python scripts/build_model_source_manifest.py
python scripts/build_manifest.py
python scripts/repository.py check
python scripts/repository.py verify-manifest
python scripts/privacy_audit.py --root . --git-history
git diff --check
```

派生的结果页、CSV、资料目录和完整性清单应随源文件一起提交。CI 会拒绝失效本地链接、遗漏证据目录、缺失视野、混入其他 cohort 或过期派生文件。`scripts/repository.py` 只处理公开资料，不触发实验。

## 结果与公开边界

开发、校准、合成、历史审计和完整匹配评价应明确区分。缺失结果写明状态，不能以计划预算代替完成数量。改变科学结论需要对应证据；重写说明不构成重新选择模型的依据。

原始数据、目标数组、模型权重和运行缓存留在被忽略的本地目录。第三方材料保留许可与来源。问题反馈可使用工件错误或主张不一致模板；敏感暴露请遵循 [SECURITY.md](SECURITY.md)。
