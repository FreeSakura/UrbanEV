# 数据、单位与本地输入

本项目主要预测 UrbanEV 的区域小时占用率。上游数据文件名相同不代表版本或单位相同；先确认数据版本、文件哈希、空间粒度和目标单位，再使用转换工具。

## UrbanEV

| 项目 | 约定 |
|---|---|
| 空间粒度 | 275 区域；区域顺序必须与容量和各输入一致 |
| 主要文件 | `occupancy.csv`、`inf.csv`；特定研究另用 duration 等观测 |
| 当前六折合同输入 | 已固定的占用数量版本，按区域容量转换为占用率 |
| 容量 | 站级 `charge_count` 按 `TAZID` 聚合；已经是区域容量时不重复求和 |
| 滞后输入 | 只使用预测原点可得的观测；六折合同另将 D 滞后一小时 |
| 输出位置 | 数据、目标数组、预测缓存与权重放入被忽略的 `local-data/` 或 `private/` |

已是 occupancy rate 的文件不能再次除容量。价格的单位、发布时间与天气的预测可得性也需要明确；仅有事后观测不等于预测时已知。详细字段证据见[信息地图](../reports/information/INFORMATION_EVIDENCE_MAP_20260913.md)，当前输入身份见[六折配置](../../configs/research/URBANEV_MATCHED_SIX_FOLD_V1_20260914.json)。

通用开发准备示例见[快速开始](quickstart.md)。历史审计数据登记命令只登记用户已取得的数据，不下载或重新授权数据：

```bash
python -m urbanev_audit register-data --dataset urbanev --data-root local-data/UrbanEV
```

## Paris 与历史证据

Paris 仅作为历史审计及 V3 配对事件研究的开发证据，不属于当前标准 UrbanEV 六折预测输入。开发重算需要独立分片和精确目标哈希；正式及受保护角色不在公开复现范围内。见[历史复现流程](reproducibility.md#历史审计结果重算)。

## 公开工件

仓库内的 CSV/JSON 是可公开汇总、登记和回执，不包含完整原始目标。较大的目标不随包公开的历史预测包位于 [GitHub Releases](https://github.com/FreeSakura/UrbanEV/releases)，应下载到本地目录并核验其 SHA256SUMS 和内容模式。

许可与获取来源以[数据许可](../DATA_LICENSES.md)为准，模型源码身份见[模型说明](../MODEL_IDENTITY.md)。下载数据或权重的许可与软件 MIT 许可相互独立。
