# `scripts/archive/` —— 已关闭实验轴的代码证据

这里的东西**不要再拿来跑新实验**。它们是 `TRAINING_ITERATIONS.md` 里已落地判定的
证据来源，留着只为可复现——判定写在台账，代码留在这里，两边对得上。
在用的工具见 `../exp/README.md`。

**2026-08-17：这些文件此前从未进过 git。** `.gitignore` 里有一条 `scripts/archive/*`，
把整条证据链对 git 屏蔽了 63 个文件，"两边对得上" 这句话在那之前是假的——与
2026-08-14 丢掉 `exp_head_features.py` 等三个文件是同一根因（文件只存在于磁盘）。
规则已删除并加了注释。**归档 ≠ 忽略。**

同批修掉的三类断链（历次搬迁只挪文件、没改路径）：

| 类别 | 症状 | 修法 |
|---|---|---|
| Python 根锚点 | `dirname(dirname(__file__))` 在 `scripts/` 下对，搬进 `archive/` 后解析成 `scripts/` | 按文件实际深度显式拼 `..` |
| Bash 根锚点 | `cd "$(dirname "$0")/../.."` 深度不对；`ROOT="G:/ai_proj/..."` 与 `cd /g/ai_proj/...` 写死盘符 | 统一成相对脚本自身的 `cd` |
| 引用路径 | 还指着搬迁前的扁平 `scripts/exp_nam_gate.py`、`from null_metrics import` | 按 basename 重定向到当前真实位置 |

修完逐个实跑：**17/17 归档判定脚本能从新位置跑起来并复现台账数字**。校验脚本检查
编译、根锚点实算、`cd` 深度、引用存在性四项。

## 目录分类

深度固定为 `scripts/archive/<bucket>/`（3 层）。加新桶时注意根锚点要跟着改。

### `e_series/` —— E3~E16 与 K 轴
| 脚本 | 轴 | 结论 |
|---|---|---|
| `run_e3_*.sh` `analyze_e3.py` | E3 标签残差化 | 决定性劣化 0/4 种子，标签里的 vol/size 暴露是可交易收益 |
| `run_e6_*.sh` `analyze_e6.py` | E6 标量门控 | 牛市 4/4 +37pp、熊市 0/4 −12pp，是 regime 交易不是改进。**见下方失效说明** |
| `run_e7_indrel.sh` `analyze_e7.py` | E7 行业内分位 | 落在噪声带内 |
| `run_e8_prune.sh` | E8 特征剪枝 | 删列单调劣化 |
| `run_e9_multifold.sh` | E9 多折评估器 | 已晋级为常规判定口径，评估器留在 `../exp/analyze_multifold.py` |
| `run_e11_*.sh` `analyze_e11_bt.py` | E11 多持有期标签 | 验证 IC 4/4 涨但 OOS 熊市回测 4/4 掉 9pp |
| `run_e12_ictransfer.sh` | E12 IC 可迁移性 | 产出行情分层判定，工具留在 `../exp/diag_ic_by_regime.py` |
| `run_e13_centered_label.sh` | E13 居中 5~9 标签 | 增益与倾斜同时消失，多期标签族整体关闭 |
| `run_e14_universe.sh` | E14 训练股票池外推 | holdout IC 只掉 1σ 内（注：此结论后被 T098 在**全量池**上推翻，见台账） |
| `run_e15_head.sh` | E15 极端头部 | 头部拿的是尾部幅度不是命中率 |
| `run_e16_timedecay.sh` | E16 时间衰减 | τ=3/7 都 2/4 中位为负，斜率无指向 |
| `run_k*.sh` `analyze_k_sweep.py` | K 轴 | K∈{10,20,40,60} 单峰峰在 20；K=40 只当低噪尺子 |
| `run_ens_k40.sh` | 跨种子集成 | 等权 0.5/0.5，按 IR 加权已证伪 |
| `run_regime_filter.sh` | 行情过滤规则 | 用户明令禁止用风控规则修回测 |
| `run_seed_null_sweep.sh` `analyze_seed_null.py` `analyze_power.py` | 零假设与功效 | 已固化为协议，工具留在 `../exp/diag_random_null.py` / `null_metrics.py` |
| `diag_effective_sample.py` `diag_holding_period.py` `diag_label_vs_return.py` `diag_regime_conditionality.py` | 配套诊断 | 各自服务于上面某一轴 |
| `diagnose_fundamental_feature_stability.py` | 基本面特征稳定性 | **已跑不动**：`from scripts.exp_xgb_feature_group_ablation import classify_feature`，那个文件从未进 git 且已丢失 |

### `t_series/` —— T090~T111
| 脚本 | 轴 | 结论 |
|---|---|---|
| `run_t090_forecast_and_magnitude.sh` | T090 预告族 / 幅度标签 | 预告单因子残差 IC +0.03 为真，但混进打分 ΔIC 仅 +0.0005；幅度被**反向**预测 |
| `run_t091_ss.sh` | T091 样本选择 | 未晋级 |
| `run_t092_trees.sh` `run_t092_backtest.sh` | T092 树 vs NAM（800 只池） | 树赢 0.022 |
| `run_t093_fullpanel.sh` | T093 全面板 | 加行业内分位无效、删交叉项单调劣化，219 列在平台顶 |
| `run_t095_final.sh` `analyze_t095.py` | T095 800 只池终审 | 定下 NAM 冠军配置 y2/h16/lr2e-3 |
| `run_t096_hpo.sh` | T096 NAM 调参三轴 | 四臂 0 晋级；选型偏差稳定占 12~13% |
| `run_t100_tree_hpo.sh` `analyze_tree_hpo.py` `chain_t100.sh` | T100 树调参 | 未晋级 |
| `analyze_blend.py` | T094/T105 xgb+lgb 混合 | 风格互补为真，但「何时用谁」折外 R²=−0.19；n=3 终审关闭。**输入 npz 已删**（140MB，见 `diagnose_output/archive/DELETED_ARTIFACTS.txt`），要跑得先重跑 T094 |
| `run_t102_nam_interaction.sh` `analyze_t102.py` `run_t104_gate_interaction.sh` | T102/T104 门控 + 显式交互列 | 单种子超加性是噪声，配对后关闭 |
| `run_t103_tree_backtest.sh` `analyze_t103.py` | T103 树头部选股 | 全量池单模型树两窗皆负且胜率更低 |
| `run_t105_blend_final.sh` `analyze_t105.py` | T105 等权混合终审 | 优势是单种子运气，赢在 β>1.3 不是选股 |
| `run_t106_gate_lambda.sh` `analyze_t106.py` | T106 门控 λ 下界 | λ=0 塌陷成静态重加权、λ=0.01 几乎均匀，无稳定第三态 |
| `run_t107_gate_s42.sh` `run_t108_gate_n4.sh` `run_t111_gate_multiseed.sh` `analyze_t108.py` | T107/T108/T111 门控终审 | **门控轴永久关闭**：IC 0/4、跌日 ΔIC 4/4 为负、熊市回测 1/4 |
| `run_t110_fp16_equivalence.sh` | T110 fp16 等价性 | PASS（Δ −0.00085、早停同轮）；技术已固化为 `--store-dtype` |
| `cmp_t099_vs_t095_bt.py` | 全量池 vs 800 只池回测横比 | 全量池胜，是台账唯一验证有效的方向（加数据） |
| `watch_t108.sh` | 回测完成哨兵 | 用 `.done` 标记而非 `pgrep`——Git-Bash 下 `pgrep` 匹配不到 `nohup` 起的脚本 |

### `oneoff/` —— 跨轴一次性排查
`_diag_model_identity.py`（模型混淆排除）、`_bench_preload.py`（预加载耗时实测）、
`_smoke_diag.py`（冒烟）、`scale_gate_sensitivity.py`（门控尺度敏感性）。

## ⚠ E6 那批判定已失去前提

`analyze_e6.py` 现在跑出来会打印「门禁**通过**」。**不要采信**：标量门控的双重中心化
bug（复合成 `4r-3`）后来已修，牛市 4/4 / 熊市 0/4 那批读数建立在有 bug 的实现上。
门控轴的**有效**终审是 T108/T111（见上表），结论是关闭。

留着 E6 的代码只为对照台账里的历史记录，不作为门控轴的证据。

## 2026-08-14 更正（保留）

- 上表原先把 `migrate_downside_risk_cache.py` 记在这里，**实际它在 `../` 根目录**，
  而且不能挪——`tests/test_business_logic.py` 导入了它的 `_migrate_one`。
- `ingest_performance_forecast.py` 属数据落库，留在 `../` 根目录。
