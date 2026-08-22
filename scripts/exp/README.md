# `scripts/exp/` —— 常设判定口径与可复用诊断

**训练器已搬走**（2026-08-22）：NAM 与树的训练入口现在都在 `scripts/` 顶层，
按模型族分工、共用同一份数据基座：

| 入口 | 模型族 | 面板 |
|---|---|---|
| `../train_nam_model.py` | NAM（`--disable-gate` = 严格加性），**当前生产载体** | 224 列 |
| `../train_tree_model.py` | XGBoost / LightGBM | 236 列 |

本目录只留**下一个实验还会用到**的东西：判定口径、常设门槛、同折对照臂。

**规矩一：放在这里的脚本必须进 git。** 2026-08-14 发现 `exp_head_features.py`、
`exp_horizon_7d_vs_15d.py`、`exp_xgb_feature_group_ablation.py` 三个文件从未被提交
且已从磁盘消失，直接导致所有训练轴跑不动。`eval_metrics.py`（原 `_metrics.py`）
是从台账数值反推重建出来的替代品，代价是半天时间。

**规矩二：已关轴的一次性脚本迁到 `../archive/`，不留在这里。** 判断标准是「下一个
实验还会不会用它」。2026-08-17 清理过一次（40 → 22 个），2026-08-22 再清一次：
20 个 `run_t*.sh` 驱动脚本迁到 `../archive/t_series/`，22 个已终审关闭轴的诊断迁到
`../archive/diag_closed_axes/`（门控 / 择时 / 下行风险 / 预告混合 / 聚类瘦身 /
种子集成 / 分数混合 / 头部 regime）。本目录降到 10 个 `.py`。

## 目录内容

### 同折对照
| 文件 | 用途 |
|---|---|
| `exp_tree_vs_nam.py` | XGBoost / LightGBM 对照臂，与 NAM 同折同种子同归一化。复用 `train_nam_model._prepare_fold`，逐字节相同的折边界是结构保证 |

### 判定口径（改打分层必须过这一套）
| 文件 | 用途 |
|---|---|
| `analyze_multifold.py` | **晋级判定（G4/G5）**：折均 IC + 跨种子配对 + 行情分层门槛 |
| `analyze_ic.py` | **打分层轴的唯一筛选口径**：任意 `<PREFIX>_s{seed}.json` 与基线同种子配对比 IC |
| `diag_ic_by_regime.py` | **强制门槛**：涨跌日分层 IC。合并 IC 会把「涨日赢跌日输」平均成净正 |
| `diag_random_null.py` / `null_metrics.py` | 零假设与功效，定 MDE |
| `diag_paired_bootstrap.py` | 同种子配对 bootstrap 显著性 |
| `judge_arm.py` | 串行协议的单臂判定器：候选 vs 基线，一次打全三道门 |

### 可复用诊断
| 文件 | 用途 |
|---|---|
| `eval_models_on_window.py` | **跨窗横比的唯一正确做法**：把多个存档拉到同一窗口同一考卷上算 IC。holdout IC 换了 years/end 就换了考卷，不能直接横比 |
| `diag_backtest_daily_paired.py` | 日频配对回测。回测终值的 MDE 约 80pp，同种子浮点抖动就差 27~60pp —— 先做日频配对再看终值 |
| `compare_training_candidates.py` | 两个已归档模型在同一验证截面上直接比 |

评估指标库 `_metrics.py` 已提升为 `core/factors/eval_metrics.py`（训练入口依赖它，
不该住在实验目录里）；`_fold_dataset` 也从原 `scripts/diagnose_xgb_oof_head.py`
一并并入该模块。

## 四个坑

1. **`eval_metrics._head_percentile` 没能复现旧定义**（差 2~3 倍），已降级为「仅报告、
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

- `diag_ic_by_regime.py`、`diag_paired_bootstrap.py`、`exp_tree_vs_nam.py` 用
  `from scripts.train_nam_model import ...` 这种**包路径**导入，所以 `scripts/` 与
  本目录的 `__init__.py` 都不能删。
- `tests/test_business_logic.py` 导入 `scripts.archive.oneoff.migrate_downside_risk_cache`
  的 `_migrate_one`，归档目录因此也带了 `__init__.py`。

缓存目录不必再手传：`--cache-dir` 默认已是 `TrainingConfig.CURRENT_CACHE_DIR`
（当前 08-18 idxrel 基座），两个训练入口与本目录的对照臂共用同一份。
