# 复现首折基础模型与校准实验

从仓库根目录运行。为Chronos-2和TimesFM-3分别建立环境；模型不与项目安装自动下载。原始数据、权重、预测缓存和学习参数保留在`local-data/`或其他不公开的位置。

## 环境与权重

共同参考环境：Python3.10.8、PyTorch2.5.1+cu121、NumPy2.2.6、Pandas2.3.3、huggingface_hub0.36.2、safetensors0.8.0。完整版本记录见[环境锁](../../../artifacts/summaries/fm_calibration_v1/environment_locks.json)。复现GPUprofile需要匹配设备和配置；其他环境的时间不能直接与报告相减。

Chronos-2使用`chronos-forecasting==2.2.2`和`transformers==4.57.6`。TimesFM使用下列固定源码安装，避免把可变分支当版本锁：

```bash
python -m pip install --no-deps git+https://github.com/google-research/timesfm.git@8cb7eda91c2b416b37e99a979afc50f2a18e1791
```

在已安装项目及相应依赖的环境中，把官方权重下载到本地，revision分别固定为：

| 骨干 | 官方仓库 | 权重revision |
|---|---|---|
| Chronos-2 | `amazon/chronos-2` | `29ec3766d36d6f73f0696f85560a422f50e8498c` |
| TimesFM-3 | `google/timesfm-3.0-pytorch` | `43046b85ec22d584a13f8098c2ed39c889e129c2` |

本地模型目录至少包含`config.json`和`model.safetensors`。TimesFM-3权重用于非商业研究，禁止将权重和适配参数作为本仓库发布物；代码授权与权重授权分开。文件SHA见公开profile与[TimesFM权重锁](../../../artifacts/summaries/fm_calibration_v1/timesfm_weight_lock.json)。

## 数据与执行顺序

使用自行取得的上游`occupancy.csv`和`inf.csv`，只准备第一折训练/验证648小时：

```bash
python -m urbanev_forecast.prepare --data-root local-data/UrbanEV --hours 648 --output local-data/fold1-rates.csv
```

以下示例在Chronos-2环境运行：

```bash
python scripts/research/profile_foundation.py --backend chronos2 --model-dir local-data/models/chronos2 --csv local-data/fold1-rates.csv --output local-data/profile-chronos2
python scripts/research/cache_foundation.py --backend chronos2 --model-dir local-data/models/chronos2 --csv local-data/fold1-rates.csv --profile local-data/profile-chronos2/profile.json --output local-data/cache-chronos2
python scripts/research/evaluate_point_heads.py --cache local-data/cache-chronos2 --csv local-data/fold1-rates.csv --output local-data/evaluation-chronos2
```

在TimesFM环境中将骨干名和模型/输出目录替换为`timesfm3`，执行同样三个阶段。profile必须验证一个原点的完整275变量，缓存阶段才会放行。默认GPU；只有显式指定`--device cpu`才执行CPU版本，并作为不同profile记录。

输出目录默认必须不存在。新版缓存若中断，可用相同命令追加`--resume`；它核对数据、代码、模型、原点、shape、dtype与已经完成部分的内容哈希。没有内容封存字段的旧progress不能恢复。完整缓存评价前仍要验证全文件SHA。

## 输出与公开范围

- `profile.json`：源前缀、实际前向shape、模型身份、时间和显存。
- `receipt.json`与`cache_summary.json`：配置与缓存身份，不含观测真值。
- `quantiles.npy`、`native.npy`、`origins.npy`：逐条本地预测，保留在本地。
- `evaluation.json`：所有预注册系统的原始/裁剪分数、正则候选、收敛和门结果，可审查后公开。
- `private_head_parameters_h*.npz`：学习参数，保留在本地，不上传。

本次缓存与评分代码对应`ffeb4a9`；后续提交增加恢复校验和profile放行检查，未改变已经完成的原始缓存或评价。跨硬件重跑不要求逐比特预测相同，但每次运行必须保存自己的版本、哈希与数值结果。
