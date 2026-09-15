# Python 包

| 包 | 用途 | 入口 |
|---|---|---|
| `urbanev_forecast` | 数据窗口、预测模型、校准方法和版本化评价合同 | `python -m urbanev_forecast --help` |
| `urbanev_audit` | 历史工件校验、目标重建、指标及公开边界检查 | `python -m urbanev_audit --help` |

从仓库根目录安装 `python -m pip install -e ".[test,research]"`；完整 CPU 环境步骤见[快速开始](../docs/guides/quickstart.md)。模型执行依赖 PyTorch，通用资料索引工具只使用标准库。

[架构指南](../docs/guides/architecture.md)介绍模块关系，[脚本目录](../scripts/README.md)列出实验编排入口。受冻结合同保护的科学源码保留原执行身份；仅移动文件也可能改变 checkpoint 指纹，因此需区分资料维护与新实验版本。
