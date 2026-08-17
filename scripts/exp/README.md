# `scripts/exp/` —— 在用的训练器、判定口径与诊断

**规矩一：放在这里的脚本必须进 git。** 2026-08-14 发现 `exp_head_features.py`、
`exp_horizon_7d_vs_15d.py`、`exp_xgb_feature_group_ablation.py` 三个文件从未被提交
且已从磁盘消失，直接导致所有训练轴跑不动（`exp_nam_gate.py` 依赖前两个里的函数）。
`_metrics.py` 是从台账数值反推重建出来的替代品，代价是半天时间。

**规矩二：已关轴的一次性脚本迁到 `../archive/`，不留在这里。** 判断标准是「下一个
实验还会不会用它」——训练器、判定口径、可复用诊断留下；某一轮的专用判定脚本和
驱动 `.sh` 迁走。2026-08-17 按此清理，本目录从 40 个文件降到 22 个。

## 目录内容

### 训练器
| 文件 | 用途 |
|---|---|
| `exp_nam_gate.py` | **所有 NAM 轴的入口**；`--disable-gate` 下是严格加法 NAM，即当前生产载体 |
| `exp_tree_vs_nam.py` | XGBoost / LightGBM 对照臂，与 NAM 同折同种子 |

### 判定口径（改打分层必须过这一套）
| 文件 | 用途 |
|---|---|
| `analyze_multifold.py` | **晋级判定（G4/G5）**：折均 IC + 跨种子配对 + 行情分层门槛 |
| `analyze_holdout.py` | 读 `rank_ic_holdout`（选型不可见段）横比 HPO 臂；两栏差值即选型偏差实测值 |
| `analyze_ic.py` | 单次结果的 IC 明细 |
| `diag_ic_by_regime.py` | **强制门槛**：涨跌日分层 IC。合并 IC 会把「涨日赢跌日输」平均成净正 |
| `diag_random_null.py` / `null_metrics.py` | 零假设与功效，定 MDE |
| `diag_paired_bootstrap.py` | 同种子配对 bootstrap 显著性 |

### 可复用诊断
| 文件 | 用途 |
|---|---|
| `_metrics.py` | `_daily_metrics` / `_head_percentile` / `_compare`；驱动 checkpoint 选型 |
| `diag_forecast_ic.py` | 业绩预告单因子残差 IC |
| `diag_score_blend.py` | 线性混合打分（免重训的快速上限探测） |
| `analyze_top5_failures.py` | 头部选股失败案例归因 |
| `compare_training_candidates.py` | 候选模型横比 |
| `diagnose_label_return_calibration.py` | 标签与真实收益的标定（含 `FUTURE_DAYS` 耦合） |
| `diagnose_xgb_head_regime.py` | XGBoost 头部的 regime 依赖 |

### 工具与生产基线配方
| 文件 | 用途 |
|---|---|
| `run_t098_full.sh` | **生产基线复现**：全量池纯加性 NAM，4 种子存档 |
| `run_t099_backtest.sh` | 上述存档在熊/牛两个 OOS 窗口的回测 |
| `bench_store_dtype.sh` | fp16/fp32 驻留的速度基准（T109 用它定位显存超配） |
| `stamp.py` | 给任意命令的每行输出打上累计/增量耗时戳。**是命令包装器**：`python stamp.py python -u xxx.py`，不是管道过滤器（当过滤器用会 `WinError 87` 秒退） |

## 四个坑

1. **`_metrics.py::_head_percentile` 没能复现旧定义**（差 2~3 倍），已降级为「仅报告、
   不参与任何门槛」。绝对值**不能**与 T083 及之前的台账数字比较，同批次内臂间比较仍有效。
2. **写 `nohup python` 时务必加 `-u`**。重定向到文件时 stdout 是块缓冲的，训练每 5 个
   epoch 才 print，日志会长时间冻住——那不是进程死了。判活用 `ps -W | grep workbuddy`，
   或看 `nvidia-smi` 的显存占用。
3. **重定向 stdout 前必设 `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`**。否则 GBK 编码错误会
   在 5 分钟数据预加载**之后**才崩，白等一遍。
4. **性能读数必须标注股票池规模，且不得跨规模外推。** `--chunk-days 4` 在 800 只池上
   微基准快 3.8x，在全量池上是**负收益**（75.5 vs 52.8 µs/样本）——补齐块加重显存压力。
   同类错误犯过两次，判据见下。

## 全量池的显存陷阱（T109，2026-08-17）

全量池 `Xtr`(5.93 GiB) + `Xva`(1.46 GiB) = **7.39 GiB 放不进 6.00 GiB 显存**。
Windows 的 NVIDIA 驱动开着 sysmem fallback：它**不报 OOM**，而是静默把溢出部分页到
主机内存，于是每个 day-step 都走 PCIe 取特征。

症状是 `utilization.gpu 100%` + `utilization.memory 0%` + 28W/90W + sm 时钟满频
2565 MHz——看着像「算力打满」，实际是计算核心在等 PCIe。

**判据：每样本耗时随池规模的方向。** 池大 4.3 倍而单样本成本反而上升，就是分页；
若瓶颈是 kernel 启动开销，大池必须**更便宜**。

`--store-dtype fp16`（特征 fp16 驻留、计算仍 fp32）降到 3.70 GiB，
383.4 → 30.7 s/epoch = **12.5x**；`--seeds 11,23,37` 多种子复用同一份数据集，
省掉每种子 11.5 分钟的构建。一次 n=4 全量池判决从 18.3h 降到 2.3h。

等价性已验（T110）：s11 fp32 holdout 0.09818 vs fp16 0.09733（Δ −0.00085 < 阈值
0.002），且**早停落在同一 epoch 42**——轨迹没被改道，比小 ΔIC 更强的证据。

## 外部依赖与不可搬迁项

- `exp_nam_gate.py` 从 `../diagnose_xgb_oof_head.py` 导入 `_fold_dataset`，
  那个文件**留在 `scripts/` 根目录**。
- `../migrate_downside_risk_cache.py` 被 `tests/test_business_logic.py` 导入 `_migrate_one`，
  也不能挪。
- `diag_ic_by_regime.py`、`diag_paired_bootstrap.py`、`exp_tree_vs_nam.py` 用
  `from scripts.exp.exp_nam_gate import ...` 这种**包路径**导入，所以本目录的
  `__init__.py` 和 `exp_nam_gate.py` 的位置都不能动。

跑实验一律带 `--cache-dir database/system_data/factors_cache_2026-08-14-fwdadjust`，
否则会静默用回旧的、复权不全的缓存。
