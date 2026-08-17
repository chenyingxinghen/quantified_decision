"""评估指标：从 `scripts/exp_horizon_7d_vs_15d.py`（已丢失）重建。

**为什么要重建而不是随便写一个**：`_daily_metrics` 不只用于报告——
`exp_nam_gate.py:579` 用它的 `rank_ic` 做检查点选型，写错会选出不同的 epoch、
产出不同的模型。所以定义必须与旧实现逐位一致，验收标准是
`T083_base_s42` 三折的 `rank_ic / rank_ic_std / positive_ic_ratio /
top5_excess / days / head_pct_top*` 全部复现 `diagnose_output/T083_base_s42.json`
（同配置同种子复跑在本机是确定性的）。

定义反推自该 JSON 的键位（`folds[*].nam_gate`）：
    rank_ic, rank_ic_std, rank_ic_ir, positive_ic_ratio, top5_excess, days,
    head.head_pct_top{0.01,0.05,0.2}

校验结果（2026-08-14，`diagnose_output/T090_repro_s42.json` vs
`diagnose_output/T083_base_s42.json`，同配置同种子，3258 个数值键）
---------------------------------------------------------------------
- **逐位复现**：`rank_ic`、`positive_ic_ratio`、`top5_excess`、`days`，
  以及 regime 分层（`up_days` / `down_days`）全部三折完全一致 ⇒
  **决定检查点选型与 G4 判定的那条路径已确认无误**。
- `history[*].val_top5_excess` 有 ~1e-9 相对差：float32 累加顺序噪声，无意义。
- `rank_ic_std` / `rank_ic_ir`：已修（ddof=1 → ddof=0），见下方注释。
- ``head_pct_top*`` **未能复现**（如折 60-80% f=0.01：台账 0.5876 vs 本实现
  0.5426；超额幅度差 2~3 倍）。旧定义没能反推出来 —— 排除了「标签均值口径」
  （`ret_val` 才是入参）与 ddof 类差异，剩下的候选（池化排名 vs 逐日排名等）
  需要落盘预测才能判别，成本一次 28 分钟训练，不值得。
  ⇒ ``head_pct_top*`` 一律视为**仅供报告**的重建指标：
  它不参与检查点选型（选型走 `_daily_metrics(pred_va, y_val)`，见
  `exp_nam_gate.py:662`），也不参与任何门槛。
  **绝对值不得与 T083 及之前的台账数字比较**；同一批新代码内部的臂间比较仍有效。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import rankdata

HEAD_FRACTIONS = (0.01, 0.05, 0.2)
MIN_DAY_SAMPLES = 10


def _day_slices(dates: np.ndarray) -> List[Tuple[int, int]]:
    """每个交易日的 [start, end)。要求 dates 已按日期升序（切折保证了这一点）。"""
    _, starts, counts = np.unique(np.asarray(dates), return_index=True, return_counts=True)
    return [(int(s), int(s + c)) for s, c in zip(starts, counts)]


def _daily_metrics(pred: np.ndarray, target: np.ndarray, dates: np.ndarray,
                   topk: int = 5) -> Dict[str, object]:
    """逐日截面 Rank IC 与 Top-K 超额的汇总。

    target 传 `ret_val`（原始前向收益）时得到的是可比的绝对口径；传 `y_val`
    （标签）时得到的是「对标签的拟合度」。两者的 Rank IC 在标签为逐日单调变换
    时相等——这也是选型能用标签口径的原因。
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    ics: List[float] = []
    excess: List[float] = []
    for s, e in _day_slices(dates):
        n = e - s
        if n < MIN_DAY_SAMPLES:
            continue
        p, r = pred[s:e], target[s:e]
        if not (np.isfinite(p).all() and np.isfinite(r).all()):
            continue
        if np.std(p) < 1e-12 or np.std(r) < 1e-12:
            continue
        ic = np.corrcoef(rankdata(p), rankdata(r))[0, 1]
        if not np.isfinite(ic):
            continue
        ics.append(float(ic))
        kk = min(topk, n)
        top = np.argpartition(p, -kk)[-kk:]
        excess.append(float(r[top].mean() - r.mean()))
    if not ics:
        return {'rank_ic': float('nan'), 'rank_ic_std': float('nan'),
                'rank_ic_ir': None, 'positive_ic_ratio': float('nan'),
                f'top{topk}_excess': float('nan'), 'days': 0}
    a = np.asarray(ics)
    # ddof=0：由 T083_base_s42 反推确定。三折上 std(ddof=1) 恰好等于台账值
    # × sqrt(n/(n-1))（如折 60-80%：0.10879600 × sqrt(593/594) = 0.10870438
    # = 台账值，逐位吻合），所以旧实现用的是**总体**标准差。
    # 这条直接影响 rank_ic_ir，必须与旧口径一致才能跨台账比较。
    sd = float(a.std(ddof=0)) if len(a) > 1 else 0.0
    return {
        'rank_ic': float(a.mean()),
        'rank_ic_std': sd,
        'rank_ic_ir': float(a.mean() / sd) if sd > 0 else None,
        'positive_ic_ratio': float((a > 0).mean()),
        f'top{topk}_excess': float(np.mean(excess)),
        'days': int(len(a)),
    }


def _head_percentile(pred: np.ndarray, ret: np.ndarray, dates: np.ndarray,
                     fractions=HEAD_FRACTIONS) -> Dict[str, float]:
    """预测头部若干比例的股票，其**实际收益截面分位**的日均值。

    ⚠️ 重建未通过校验（见模块 docstring）：绝对值与 T083 及之前的台账口径不同，
    只能在同一批代码内部做臂间比较，且不参与任何门槛判定。

    0.5 = 与随机挑无异；>0.5 表示头部确实排在收益前面。用分位而不是收益本身，
    是为了不被个别涨停/暴涨样本带偏（头部收益的日度方差极大，见 T056）。
    """
    pred = np.asarray(pred, dtype=np.float64)
    ret = np.asarray(ret, dtype=np.float64)
    acc: Dict[str, List[float]] = {f'head_pct_top{f}': [] for f in fractions}
    for s, e in _day_slices(dates):
        n = e - s
        if n < MIN_DAY_SAMPLES:
            continue
        p, r = pred[s:e], ret[s:e]
        if not (np.isfinite(p).all() and np.isfinite(r).all()):
            continue
        r_pct = (rankdata(r) - 0.5) / n
        for f in fractions:
            kk = max(1, int(round(n * f)))
            top = np.argpartition(p, -kk)[-kk:]
            acc[f'head_pct_top{f}'].append(float(r_pct[top].mean()))
    return {k: (float(np.mean(v)) if v else float('nan')) for k, v in acc.items()}


def _compare(nam_row: Dict[str, object], base_row: Dict[str, object]) -> Tuple[Dict[str, float], bool]:
    """NAM vs XGBoost-LambdaRank 基线的逐指标差值 + 是否通过。

    只在未加 `--skip-baseline` 时用到。判定沿用台账口径：Rank IC 与 top5 超额
    都不劣化即算通过（基线臂只是尺子，不是门槛，真正的门槛在 analyze_multifold）。
    """
    deltas: Dict[str, float] = {}
    for k in ('rank_ic', 'rank_ic_ir', 'positive_ic_ratio', 'top5_excess'):
        a, b = nam_row.get(k), base_row.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            deltas[k] = float(a) - float(b)
    passed = deltas.get('rank_ic', 0.0) >= 0.0 and deltas.get('top5_excess', 0.0) >= 0.0
    return deltas, passed
