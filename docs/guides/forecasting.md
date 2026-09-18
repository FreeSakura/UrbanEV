# 历史 UrbanEV 预测工程

本页保留充电占用预测的开发与 checkpoint 评价入口。当前持续事件论文从[快速开始](quickstart.md)和[论文复现说明](../../paper/persistent_events/README.md)进入；不会通过阅读本页启动训练。

要求 Python 3.10 或更高版本和 Git。所有命令从仓库根目录执行；CPU 示例不需要真实数据、GPU 或基础模型权重。

## 安装

```bash
git clone https://github.com/FreeSakura/UrbanEV.git
cd UrbanEV
python -m venv .venv
```

Linux / macOS 激活：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活：

```powershell
.venv\Scripts\Activate.ps1
```

安装基础依赖并核验公开资料：

```bash
python -m pip install -e ".[test,evidence]"
python scripts/repository.py check
python -m pytest
```

没有安装 PyTorch 时，测试套件会跳过依赖它的测试。要检查预测训练与模型，请继续安装研究依赖并重新执行测试。

## 合成预测示例

```bash
python -m pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test,research]"
python -m urbanev_forecast smoke --model innovation_attention --epochs 2 --output local-data/smoke
python -m pytest
```

示例生成 8 通道周期序列，执行训练和验证；结果、配置与 checkpoint 保存在指定输出目录。它只能证明接口可以运行，不产生 UrbanEV 测试成绩。可以使用空目录或包含笔记的目录；同名结果文件已存在时更换输出目录。

## 开发训练

取得正确数据版本后，将 `occupancy.csv` 和 `inf.csv` 放入 `local-data/UrbanEV/`。详见[单位和版本要求](data.md)。

```bash
python -m urbanev_forecast.prepare --data-root local-data/UrbanEV --hours 648 --output local-data/urbanev-rates.csv
python -m urbanev_forecast train --csv local-data/urbanev-rates.csv --model seasonal_linear --fold 1 --horizon 3 --epochs 30 --output local-data/seasonal-f1-h3-s42
```

`prepare` 按区域容量把占用数量转换为率。`train` 读取所选折的训练/验证前缀，记录数据与代码身份，并保存最佳验证 checkpoint；使用现有 GPU 时可显式添加 `--device cuda`。

这个通用开发入口采用 `path_1_to_H` 口径。完整六折的独立 H 训练与 `terminal_H` 比较使用另一套冻结入口，见[复现指南](reproducibility.md#完整六折比较)。两套入口的分数不能直接混排。

## 固定 checkpoint 评价

方法和协议冻结后，准备覆盖对应测试前缀的率 CSV：

```bash
python -m urbanev_forecast test --csv local-data/urbanev-rates-complete.csv --checkpoint local-data/seasonal-f1-h3-s42/checkpoint.pt --output local-data/seasonal-f1-h3-s42-test
```

评价会核对训练/验证数据前缀并加载匹配的模型参数。代码修改不会直接阻止评价；`result.json` 同时记录训练代码和当前代码哈希，以及 `evaluation_code_matches_training`。模型结构变化导致参数不兼容时，由模型加载报告具体错误。

## 常见问题

| 现象 | 处理 |
|---|---|
| 同名结果文件已存在 | 更换输出目录，保留已有结果 |
| 提示缺少 torch | 安装上面的 CPU PyTorch 和 research 扩展 |
| 测试数据长度不足 | 检查所选折需要的完整时间前缀 |
| checkpoint 数据前缀不一致 | 使用与训练匹配的数据；代码差异只记录，不阻塞 |
| 只有公开结果，没有原始数据 | 使用仓库级复核；真实指标重算需要合法取得目标数据 |

下一步：[阅读结果](../../results/README.md) · [复现分级](reproducibility.md) · [代码架构](architecture.md)。
