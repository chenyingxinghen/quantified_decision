# 已归档的实验产物（2026-08-14 整理）

`diagnose_output/` 根目录只保留 **`*.json` 结果文件** 和**当前在跑**的日志。
原因：`scripts/exp/analyze_multifold.py` 用
`os.path.join(ROOT, 'diagnose_output', f'{prefix}_s{seed}.json')` 直接拼路径找结果，
**JSON 一旦挪走，历史臂就没法再复算 Δ 表**。所以 JSON 一律留在上一级不动（198 个，1.9M）。

搬进来的是不影响任何工具的东西：

| 目录 | 内容 | 说明 |
|---|---|---|
| `logs/*.log.gz` | 312 份训练/回测 stdout | 判定已写进 `TRAINING_ITERATIONS.md`，日志只作证据留存，gzip 压缩 |
| `misc/` | png / csv / 几个漏了扩展名的 JSON（`T058_hidden8_s42` 等） | 一次性图表与手滑产物 |
| `nam_gate_pre2026-08-14/` | T042~T056 的 16 个早期 NAM 门控产物子目录（11M），外加搬走那一刻的一份扁平快照 | 该轴已关（见 [[e6-scalar-gate-regime-tradeoff]]） |
| `rankdbg/` | 排名调试的 txt/log | 一次性排查 |

⚠ **`diagnose_output/nam_gate/`（上一级、无后缀）是活的**：`exp_nam_gate.py` 每次训练
都往那里写 `post_train_factor_dashboard.png` / `factor_effectiveness.csv` /
`gate_weights.csv` 等扁平文件，**下一次训练会直接覆盖**。要留证据得自己另存。
本目录里那份扁平快照就是搬迁瞬间（T090_base seed=11/23 之间）的状态。

`diagnose_output/` 根目录另有 4 个被 `.gitignore` 白名单放行、**已进 git** 的报告产物：
`factor_ic_report.csv`、`factor_quality.png`、`ic_timeseries.png`、`label_distribution.png`
—— 它们必须留在根目录，别归档（我一度搬走过，导致 git 报 4 个 deleted，已还原）。

`misc/T058_hidden8_s42` 这类无扩展名文件是当年 `--output` 忘了写 `.json`，
内容是正常 JSON；它们**不会**被 `analyze_multifold` 认出来，所以搬走无损失。
