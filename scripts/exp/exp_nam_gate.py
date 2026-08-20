"""T039：NAM 单因子专家 + 市场状态门控（NAMGateModel）受控 A/B。

背景
----
T027–T038 证明生产 XGBoost LambdaRank 的头部瓶颈是"弱绝对信号 + regime 依赖"，
而中长期市场状态在日频截面上零方差、被迫剔除，只能靠手工 ``{mkt}_regime_{stock}``
乘积注入。本实验换一条路：

  score = β0 + Σ_k  w_t[k] · Σ_{i∈group k} f_i(x_i)

- ``f_i``：每个因子一条独立形状函数（NAM）→ 可解释性到"单因子曲线"级别
- ``w_t``：多时间尺度市场状态 ``m_t`` 经门控网络输出的因子族权重
           → 直接回答"什么行情下哪类因子被放大"

受控条件（唯一变量 = 模型族 + 市场信息注入方式）
- 相同股票池 / 相同 7 日标签 / 相同折切分（历史折 70–80% 选型，最终折 80–100% 确认）
- 相同截面归一化（复用 trainer 的 rank + robust skip-col 逻辑，保证与回测端一致）
- NAM 线**剔除** ``*_regime_*`` 手工交互列（改由门控端到端学习）
- 基线 = 同折 XGBoost LambdaRank（复用 exp_head_features._train_and_predict）

防专家坍塌（DeepSeek 研讨指出的核心风险）
- 负载均衡损失 CV² + 门控熵正则 + 温度 warmup
- 逐 epoch 记录门控权重分布，输出监控曲线；坍塌时可切 ``--gate-mode multiplicative``
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from config.baostock_config import DATABASE_PATH
from config.factor_config import ModelConfig, TrainingConfig
from config.factor_groups import build_group_index
from core.data.baostock_main import BaostockDataManager
from core.factors.nam_gate_model import (
    GateUsageTracker, NAMGateModel, gate_entropy, listnet_loss, load_balance_loss,
    save_sidecar_metadata,
)
from core.factors.regime_features import align_regime_to_samples, build_regime_matrix
from core.factors.train_ml_model import MLModelTrainer
from scripts.diagnose_xgb_oof_head import _fold_dataset
# _daily_metrics / _head_percentile / _compare 原在 scripts/exp_horizon_7d_vs_15d.py，
# 该文件与 exp_head_features.py 一并丢失（从未进 git）。前三个已在 _metrics.py 重建
# 并用 T083_base_s42 复现校验；_train_and_predict（XGB 基线臂）无法重建，
# 因此 --skip-baseline 现在是硬要求，缺它时延迟到真正用到才报错。
from scripts.exp._metrics import _compare, _daily_metrics, _head_percentile


def _train_and_predict(*_a, **_kw):
    raise RuntimeError(
        'XGBoost 基线臂依赖已丢失的 scripts/exp_head_features.py::_train_and_predict，'
        '无法重建。请加 --skip-baseline；跨臂比较用 scripts/exp/analyze_multifold.py 做。')


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------

def _rank_labels_by_day(scores: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """复刻 train_models 的标签处理：逐日截面 rank → (0,1) → ** exponent。

    必须在 split 之后对训练/验证分别做，否则跨集合排名会引入标签泄漏。
    """
    from core.factors.train_ml_model import _fast_rankdata_1d

    exponent = getattr(TrainingConfig, 'LABEL_WEIGHT_EXPONENT', 1.0)
    weighted = getattr(TrainingConfig, 'LABEL_WEIGHTED_FOR_XGB', False)
    out = np.empty(len(dates), dtype=np.float32)
    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    for s, c in zip(starts, counts):
        e = s + c
        if c > 1:
            r = _fast_rankdata_1d(scores[s:e]) / (c + 1)
            out[s:e] = np.power(r.astype(np.float32), exponent) if weighted else r
        else:
            out[s:e] = 0.5
    return out


def _magnitude_labels_by_day(src: np.ndarray, dates: np.ndarray, mode: str,
                             clip: float = 3.0) -> np.ndarray:
    """E19：**不排名**的标签，把幅度信息留在损失里。

    背景（为什么这是个真问题）：基线管线把幅度销毁了两次。
    ① `_rank_labels_by_day` 把当日收益压成 (0,1) 均匀分位——涨 30% 和涨 3%
       只要都在第一名附近就完全等价；
    ② `listnet_loss` 再做 `t = softmax(y * y_scale)`，y_scale=2 且 y∈(0,1)，
       头尾权重比只有 e²≈7.4，进一步抹平。
    结果是模型只学「谁在前面」，学不到「前面的人领先多少」。

    **控制变量是这个实验的关键。** z-score 的展幅（±3σ，宽度 6）远大于 rank 的
    宽度 1，直接塞进 `softmax(y*2)` 会让头部集中度暴涨——那测到的就不是「幅度
    信息有没有用」，而是「y_scale 调大有没有用」（后者 T036 已扫过）。所以这里
    把每个交易日的 z 线性缩放到**与当日 rank 标签相同的均值(0.5)与标准差**，
    使得与基线唯一的差别是截面内的**间距形状**，softmax 的温度效应保持不变。

    mode:
      winsor_z    逐日 robust z-score（median / 1.4826·MAD），clip 到 ±clip。
                  保留线性幅度：领先两倍就是两倍。
      signed_sqrt sign(z)·sqrt(|z|)，介于 rank 与线性之间。A 股收益厚尾，
                  纯线性会让极端日的少数样本吃掉整个 softmax 质量，这一档
                  用来分辨「幅度有用」和「极值有用」——两个 mode 斜率同向
                  才算幅度轴真的成立。
    """
    if mode not in ('winsor_z', 'signed_sqrt'):
        raise ValueError(f'未知 --label-transform: {mode}')
    src = np.asarray(src, dtype=np.float64)
    out = np.empty(len(dates), dtype=np.float32)
    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    for s, c in zip(starts, counts):
        e = s + c
        if c <= 1:
            out[s:e] = 0.5
            continue
        x = src[s:e]
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale < 1e-12:
            # 当日收益几乎全同（极端停牌日）：退回 rank，避免除零放大噪声
            scale = np.std(x)
            if not np.isfinite(scale) or scale < 1e-12:
                out[s:e] = 0.5
                continue
        z = np.clip((x - med) / scale, -clip, clip)
        if mode == 'signed_sqrt':
            z = np.sign(z) * np.sqrt(np.abs(z))
        # 与当日 rank 标签对齐一阶/二阶矩：只改间距形状，不改 softmax 温度。
        # 直接复用 `_rank_labels_by_day` 的当日输出，而不是自己再算一次 rank——
        # 后者的分位口径（_fast_rankdata_1d、LABEL_WEIGHT_EXPONENT）不保证一致，
        # 差一点就会把「幅度」和「温度」混在一起测。
        ref = _rank_labels_by_day(x, dates[s:e]).astype(np.float64)
        z_sd = np.std(z)
        if z_sd > 1e-12:
            z = (z - z.mean()) / z_sd * np.std(ref) + np.mean(ref)
        else:
            z = ref
        out[s:e] = z.astype(np.float32)
    return out


INDREL_CORE = [
    # 动量/反转：A 股这两族的截面差异很大一部分是行业共同驱动
    'return_5d', 'return_20d', 'return_60d', 'momentum_10d', 'momentum_20d',
    'rsi_12', 'bias_18', 'price_percentile_60d',
    # 估值/质量/规模：可比性只在行业内成立（银行 PB 和芯片 PB 不同量纲）
    'inv_pe', 'inv_pb', 'dynamic_pe', 'peg', 'roeAvg', 'npMargin', 'market_cap',
    # 波动/量能
    'price_volatility_20', 'amount_change_rate', 'vr_30',
]


def _add_industry_relative(dataset, meta, feature_list, min_group=10):
    """E7：为 ``feature_list`` 追加「当日行业内百分位」列 ``<base>__ind``。

    现有 219 列全部是**全市场**截面量（训练前统一做当日全市场 rank 归一化），
    模型因此无法区分「这只票强」和「它所在行业强」。行业内分位把行业共同驱动
    剥掉，是一条不需要新数据、也不需要重建因子缓存的新信息轴。

    只用当日同一横截面内的信息，无前视。行业成员数 < ``min_group`` 的当日分组
    退化为 0.5（信息不足时保持中性，而不是给一个 1 只票 = 1.0 的假极值）。
    """
    import pandas as pd

    X, names = dataset[0], list(dataset[3])
    dates = dataset[4]
    if len(meta) != len(X):
        raise ValueError(f'sample_metadata 行数 {len(meta)} 与 X {len(X)} 不一致')

    from core.data.baostock_fetcher import BaostockFetcher
    fetcher = BaostockFetcher()
    ind = fetcher._get_stock_industry_from_db()
    fetcher.close()
    if ind is None or ind.empty:
        raise RuntimeError('行业表为空，拒绝产出 __ind 列（否则会训练在全 0.5 的死列上）')
    ind = ind[['code', 'industry']].drop_duplicates('code')
    code_to_ind = dict(zip(ind['code'], ind['industry'].fillna('Unknown')))

    codes = np.asarray(meta['code'].values)
    ind_arr = np.array([code_to_ind.get(c, 'Unknown') for c in codes], dtype=object)
    cover = float(np.mean(ind_arr != 'Unknown'))
    print(f'  行业覆盖率 {cover:.2%}（Unknown 视为一个独立行业参与排名）')
    if cover < 0.8:
        raise RuntimeError(f'行业覆盖率仅 {cover:.2%}，拒绝训练')

    use = [f for f in feature_list if f in names]
    missing = [f for f in feature_list if f not in names]
    if missing:
        print(f'  [警告] 以下列不在特征集中，已跳过: {missing}')
    if not use:
        raise RuntimeError('没有任何可用的 __ind 基础列')

    idx = [names.index(f) for f in use]
    df = pd.DataFrame(np.asarray(X[:, idx], dtype=np.float64), columns=use)
    df['_d'] = dates
    df['_i'] = ind_arr
    g = df.groupby(['_d', '_i'], sort=False)
    ranked = g[use].rank(pct=True, na_option='keep')
    small = g[use[0]].transform('size').values < min_group
    new = np.array(ranked.to_numpy(dtype=np.float32), dtype=np.float32, copy=True)
    new[small, :] = 0.5
    new = np.nan_to_num(new, nan=0.5)

    new_names = [f'{f}__ind' for f in use]
    out = list(dataset)
    out[0] = np.hstack([X, new]).astype(X.dtype, copy=False)
    out[3] = names + new_names
    print(f'  已追加行业内相对列 {len(new_names)} 条（小行业置中比例 {small.mean():.2%}）')
    return tuple(out[:10])


def _forward_returns_by_horizon(stocks_data, meta, horizons):
    """按 ``(code, date)`` 对齐各持有期的前向收益。

    口径与 ``prepare_dataset`` 的标签完全一致（``close.shift(-h)/close − 1``，
    同一份 ``stocks_data``），所以多期标签和 7 日基线标签是可比的、都无前视。
    """
    import pandas as pd

    frames = []
    for code, df in stocks_data.items():
        if df is None or len(df) == 0 or 'close' not in df.columns:
            continue
        d = df[['date', 'close']].copy()
        d['date'] = d['date'].astype(str).str[:10]
        d = d.sort_values('date')
        c = d['close'].astype(float).to_numpy()
        out = pd.DataFrame({'code': str(code), 'date': d['date'].to_numpy()})
        for h in horizons:
            fc = np.concatenate([c[h:], np.full(min(h, len(c)), np.nan)])[:len(c)]
            with np.errstate(invalid='ignore', divide='ignore'):
                out[f'fwd{h}'] = fc / np.where(c == 0, np.nan, c) - 1.0
        frames.append(out)
    if not frames:
        raise RuntimeError('stocks_data 里没有可用的 close 序列')
    allf = pd.concat(frames, ignore_index=True)

    key = pd.DataFrame({
        'code': meta['code'].astype(str).to_numpy(),
        'date': pd.Series(meta['date'].to_numpy()).astype(str).str[:10].to_numpy(),
    })
    merged = key.merge(allf, on=['code', 'date'], how='left')
    out = {}
    for h in horizons:
        v = merged[f'fwd{h}'].to_numpy(dtype=np.float64)
        cov = float(np.mean(np.isfinite(v)))
        print(f'  前向收益 {h}d 对齐覆盖率 {cov:.2%}')
        if cov < 0.5:
            raise RuntimeError(f'{h}d 前向收益覆盖率仅 {cov:.2%}，(code,date) 对齐可能坏了')
        out[h] = v
    return out


def _multi_horizon_label(fwd, dates, horizons):
    """多持有期复合标签：逐日逐持有期做截面 rank，再对持有期取平均。

    **先 rank 再平均**：不同持有期收益的量纲和波动差好几倍，直接平均会被最长的
    那一期主导；rank 之后每期等权，才是真正的「降标签噪声」。
    缺失（序列末尾取不到未来价）用当日截面中位数补，等价于「该期无观点」。
    """
    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    acc = np.zeros(len(dates), dtype=np.float64)
    for h in horizons:
        x = np.asarray(fwd[h], dtype=np.float64).copy()
        for s, c in zip(starts, counts):
            e = s + c
            seg = x[s:e]
            bad = ~np.isfinite(seg)
            if bad.any():
                med = np.nanmedian(seg) if np.isfinite(seg).any() else 0.0
                seg[bad] = med
                x[s:e] = seg
        acc += _rank_labels_by_day(x, dates).astype(np.float64)
    return acc / len(horizons)


def _residualize_returns(label_src: np.ndarray, dates: np.ndarray,
                         X_raw: np.ndarray, all_names, mode: str) -> np.ndarray:
    """E3：在截面 rank **之前**把原始前向收益里的风险成分剥掉。

    动机
    ----
    ``rank(原始 7 日收益)`` 同时编码三样东西：(1) beta 暴露 —— 截面 rank 只减掉
    市场**水平**，减不掉 beta **离散度**；(2) 波动率量纲 —— 高波动股天然占据
    标签两端；(3) 真 alpha。前两者可由 atr / 振幅 / 换手等特征高精度预测，
    模型会优先把容量花在它们身上，真正的超额收益反而被淹没。

    重要限定
    --------
    - 严格逐日截面运算，不含任何跨日信息，无前视。
    - 必须使用**未做截面归一化**的原始特征列（归一化后已丢掉量纲）。
    - 残差化后验证 Rank IC 必然下降（删掉的正是最好预测的部分），
      因此该轴只能用双窗口随机零假设分位判定，**不能用 IC 判定**。

    mode:
      none      不改（当前生产口径）
      vol       r / price_volatility_20，按波动率归一（最小改动）
      beta_size 对 [price_volatility_20, log(market_cap)] 做逐日截面 OLS 取残差
    """
    if mode == 'none':
        return label_src

    index = {n: i for i, n in enumerate(all_names)}
    need = {'vol': ['price_volatility_20'],
            'beta_size': ['price_volatility_20', 'market_cap']}[mode]
    missing = [n for n in need if n not in index]
    if missing:
        raise ValueError(
            f'--label-residualize {mode} 需要原始特征列 {missing}，数据集中不存在')

    eps = 1e-8
    out = np.asarray(label_src, dtype=np.float64).copy()
    vol = np.asarray(X_raw[:, index['price_volatility_20']], dtype=np.float64)
    mcap = (np.asarray(X_raw[:, index['market_cap']], dtype=np.float64)
            if mode == 'beta_size' else None)

    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    for s, c in zip(starts, counts):
        e = s + c
        if c < 10:
            continue
        r = out[s:e]
        v = vol[s:e]
        good_v = np.isfinite(v) & (v > 0)
        if good_v.sum() < max(10, c // 5):
            continue                       # 当日波动率列不可用 -> 保持原值
        v = np.where(good_v, v, np.median(v[good_v]))

        if mode == 'vol':
            out[s:e] = r / (v + eps)
            continue

        # beta_size: 逐日截面 OLS  r ~ 1 + vol + log(mcap)，取残差
        m = mcap[s:e]
        good_m = np.isfinite(m) & (m > 0)
        if good_m.sum() < max(10, c // 5):
            out[s:e] = r / (v + eps)       # 市值列不可用则退化为 vol 归一
            continue
        m = np.where(good_m, m, np.median(m[good_m]))
        A = np.column_stack([np.ones(c), v, np.log(m + eps)])
        rr = np.where(np.isfinite(r), r, 0.0)
        try:
            coef, *_ = np.linalg.lstsq(A, rr, rcond=None)
            out[s:e] = rr - A @ coef
        except np.linalg.LinAlgError:
            out[s:e] = r / (v + eps)
    return out


def _prepare_fold(trainer, dataset, feature_names, train_fraction, val_end, regime_matrix,
                 target='scores', label_residualize='none',
                 multi_horizon=None, fwd_returns=None,
                 label_transform='rank', label_clip=3.0):
    """切折 + 选列 + 截面归一化 + 标签处理 + 市场状态对齐。"""
    full_dates = dataset[4]
    train_date = full_dates[int(len(full_dates) * train_fraction)]
    split_idx = int(np.searchsorted(full_dates, train_date, side='left'))

    fold, _, val_start, _, split_date, end_date = _fold_dataset(dataset, train_fraction, val_end)
    X, y, returns, all_names, dates, unbuyable, limits, scores, is_st, w_sig = fold

    index = {n: i for i, n in enumerate(all_names)}
    Xs = X[:, [index[n] for n in feature_names]].copy()

    # 截面归一化：训练集算统计量，验证集复用（与生产/回测端完全一致）
    skip_stats = trainer._apply_cross_sectional_normalization_inplace(
        Xs[:split_idx], dates[:split_idx], list(feature_names)
    )
    trainer._apply_cross_sectional_normalization_inplace(
        Xs[val_start:], dates[val_start:], list(feature_names), skip_col_stats=skip_stats
    )
    Xs = np.nan_to_num(Xs, nan=0.5, posinf=1.0, neginf=0.0)

    label_src = scores if scores is not None else y
    if target == 'returns':
        # 用原始前向收益排名当训练目标：去掉 vol_booster 污染，门控学真正的 regime 调节
        label_src = returns
    label_src = np.nan_to_num(label_src, nan=0.5)

    # E3 标签残差化：必须在 rank 之前，且用**未归一化**的原始列 X（不是 Xs）
    if label_residualize != 'none':
        if target != 'returns':
            raise ValueError('--label-residualize 仅在 --target returns 下有意义')
        label_src = _residualize_returns(label_src, dates, X, all_names, label_residualize)

    y_train = _rank_labels_by_day(label_src[:split_idx], dates[:split_idx])
    y_val = _rank_labels_by_day(label_src[val_start:], dates[val_start:])

    # E19 幅度标签：**只换训练标签**。y_val 必须保持 rank 口径——它在
    # exp_nam_gate:579 参与检查点选型，换掉就等于同时换了尺子和被测物；
    # 而且 winsor 的截断会在两端造出并列，连 Rank IC 都不再与基线严格可比。
    # 与 E11 多期标签同一条纪律（见下方注释）。
    if label_transform != 'rank':
        if target != 'returns':
            raise ValueError('--label-transform 只在 --target returns 下有意义'
                             '（scores 已被 vol_booster 加过权，幅度不是收益幅度）')
        y_train = _magnitude_labels_by_day(
            label_src[:split_idx], dates[:split_idx], label_transform, label_clip)

    # E11 多持有期复合标签：**只换训练标签**，验证标签仍是 7 日基线口径，
    # 这样 val Rank IC 与基线严格可比（要回答的是「训练目标噪声更小，能不能更准地
    # 预测同一个 7 日结果」，而不是换一把尺子）。
    keep_tr = None
    if multi_horizon:
        if fwd_returns is None:
            raise ValueError('multi_horizon 需要 fwd_returns')
        # fwd_returns 是按全量样本建的；折数据是它的前缀（切折只从尾部截断），
        # 所以直接取前 split_idx 个即对齐。
        fwd_tr = {h: np.asarray(v)[:split_idx] for h, v in fwd_returns.items()}
        y_train = _rank_labels_by_day(
            _multi_horizon_label(fwd_tr, dates[:split_idx], multi_horizon),
            dates[:split_idx])
        # 禁运期：训练标签最长看 max_h 天，而切折只留了 FUTURE_DAYS(7) 天间隔，
        # 差额必须从训练集尾部砍掉，否则 20 日标签会穿进验证窗口。
        extra = int(max(multi_horizon)) - int(getattr(TrainingConfig, 'FUTURE_DAYS', 7))
        if extra > 0:
            uniq_tr = np.unique(dates[:split_idx])
            if extra >= len(uniq_tr):
                raise RuntimeError('禁运期长于训练集')
            cutoff = uniq_tr[-extra]
            keep_tr = dates[:split_idx] < cutoff
            print(f'  多期标签禁运：训练集尾部砍掉 {extra} 个交易日（< {cutoff}），'
                  f'样本 {split_idx} → {int(keep_tr.sum())}')

    X_train, d_train, ret_train = Xs[:split_idx], dates[:split_idx], returns[:split_idx]
    M = align_regime_to_samples(dates, regime_matrix)
    M_train = M[:split_idx]
    if keep_tr is not None:
        X_train = X_train[keep_tr]
        y_train = y_train[keep_tr]
        d_train = d_train[keep_tr]
        ret_train = ret_train[keep_tr]
        M_train = M_train[keep_tr]

    return {
        'X_train': X_train, 'X_val': Xs[val_start:],
        'y_train': y_train, 'y_val': y_val,
        'd_train': d_train, 'd_val': dates[val_start:],
        'M_train': M_train, 'M_val': M[val_start:],
        'ret_train': ret_train, 'ret_val': returns[val_start:],
        'split_date': str(split_date), 'end_date': None if end_date is None else str(end_date),
        'skip_stats': skip_stats,
    }


def _day_slices(dates: np.ndarray):
    """返回每个交易日的 (start, end) 切片（dates 已按日期升序）。"""
    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    return [(int(s), int(s + c)) for s, c in zip(starts, counts)]


def _topk_excess(pred: np.ndarray, ret: np.ndarray, day_slices, k: int = 20) -> float:
    """预测 Top-K 组合相对当日全截面均值的超额收益，按交易日平均。

    该指标仅用于训练诊断与最终评估，不用于检查点选型。T056 已证明其日度估计
    方差过高，用于早停会在验证期头部噪声上过拟合；默认选型必须保持 rank_ic。
    """
    vals = []
    for s, e in day_slices:
        n = e - s
        if n < 10:
            continue
        kk = min(k, n)
        p = pred[s:e]
        r = ret[s:e]
        top = np.argpartition(p, -kk)[-kk:]
        vals.append(float(np.mean(r[top]) - np.mean(r)))
    return float(np.mean(vals)) if vals else float('-inf')


def _regime_stratified_metrics(pred: np.ndarray, ret: np.ndarray, dates: np.ndarray,
                               k: int = 40) -> dict:
    """T082/E12：按「这 7 天市场涨没涨」把验证日分层，分别算 Rank IC 与 Top-K 超额。

    为什么必须分层：E11 的多期标签在**合并**口径上折均 IC 4/4 上升、过了门槛，
    结果 OOS 熊市回测超额 4/4 掉 ~9pp。分层后原因一目了然——增益全长在上涨日，
    下跌日 12 个（窗口×种子）里只有 2 个为正、中位 −0.0134。合并统计量把一次
    **符号翻转**平均成了净正，于是一个 regime 押注混过 IC 门槛、白烧一轮回测确认。

    市场状态用当日全截面平均前向收益的符号定义。它不是交易时可知的信号，
    只作研究期诊断：问的是「行情下跌时这个改动还灵不灵」，不是拿它去择时。
    """
    day_ic, day_ex, day_mkt = [], [], []
    for s, e in _day_slices(np.asarray(dates)):
        n = e - s
        if n < 10:
            continue
        p, r = np.asarray(pred[s:e], np.float64), np.asarray(ret[s:e], np.float64)
        ic = np.corrcoef(rankdata(p), rankdata(r))[0, 1]
        if not np.isfinite(ic):
            continue
        kk = min(k, n)
        top = np.argpartition(p, -kk)[-kk:]
        day_ic.append(float(ic))
        day_ex.append(float(r[top].mean() - r.mean()))
        day_mkt.append(float(r.mean()))
    day_ic = np.asarray(day_ic)
    day_ex = np.asarray(day_ex)
    up = np.asarray(day_mkt) > 0
    out = {'topk': int(k)}
    for name, mask in (('all', np.ones_like(up)), ('up_days', up), ('down_days', ~up)):
        if mask.sum() < 20:      # 日数太少不给数，免得拿噪声当结论
            out[name] = {'n_days': int(mask.sum()), 'rank_ic': None, 'topk_excess': None}
            continue
        out[name] = {'n_days': int(mask.sum()),
                     'rank_ic': float(day_ic[mask].mean()),
                     'topk_excess': float(day_ex[mask].mean())}
    return out


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

class _DayBatchLoader:
    """DataLoader 形式的「逐交易日 / 逐块」取数器（2026-08-19）。

    为什么需要它
    ------------
    T109 把特征整表搬进显存换来 12.5x 提速，但代价是**窗口长度被显存钉死**：
    本机 6.00 GiB 显存下 fp16 224 列只能吃约 9.6M 样本（9 年窗），13 年窗
    (12.1M / 5.05 GiB) 加上 1.7 GiB 的权重+激活+optimizer 就会超配 ——
    而 Windows 驱动**不报 OOM**，它静默页到主机内存，于是回到 12.5x 慢的老路。
    本类把「整表驻留」换成「批取数」：特征可以留在 pinned 主机内存里，
    每个 step 只把当前这一块搬上去，显存占用与窗口长度**解耦**。

    为什么不用 torch.utils.data.DataLoader + num_workers>0
    ---------------------------------------------------------
    特征是一整块 10+ GiB 的常驻张量。Windows 用 spawn 起 worker，会把它 pickle
    一遍再传给子进程 —— 代价远大于收益。所以这里是 num_workers=0 的语义，
    但保留 DataLoader 的两个关键特性：**批组装**与**预取**（用独立 CUDA stream
    把 H2D 传输藏在当前块的计算后面）。

    速度预期（别期待加速）
    ----------------------
    每个交易日约 4700 行 × 224 列 × 2B ≈ 2.1 MB，pinned H2D 约 12 GB/s ⇒ 0.18 ms；
    而一个 chunk 的前向+反向是 ~15 ms。所以纯传输开销约 1%，开预取后接近 0。
    **本类的价值是拿掉容量天花板，不是提速**。
    ``device='cuda'`` 时退化为零拷贝切片，与 T109~T122 的行为逐位一致。
    """

    def __init__(self, store, y, M, day_slices, dev, chunk_days=1,
                 prefetch=True, min_rows=5):
        self.store = store              # [N, F] 张量，cuda 或 pinned cpu
        self.y = y                      # [N] cuda
        self.M = M                      # [D, d_m] cuda（逐日 regime，已按日取首行）
        self.days = day_slices
        self.dev = dev
        self.chunk_days = max(1, int(chunk_days))
        self.min_rows = min_rows
        self.on_gpu = store.device.type == dev.type and store.device.type == 'cuda'
        self.prefetch = bool(prefetch) and not self.on_gpu and dev.type == 'cuda'
        self.stream = torch.cuda.Stream() if self.prefetch else None

    def _rows(self, s, e):
        """取 [s,e) 行并放到计算设备上（GPU 驻留时零拷贝）。"""
        blk = self.store[s:e]
        return blk if self.on_gpu else blk.to(self.dev, non_blocking=True)

    # ── 逐日路径 ─────────────────────────────────────────────────────────
    def iter_days(self, order):
        """yield (di, x_fp32, y_slice, is_last)。x 已升 fp32（fp16 驻留时逐日升精度）。"""
        idx = [int(d) for d in order
               if (self.days[int(d)][1] - self.days[int(d)][0]) >= self.min_rows]
        yield from self._pipelined(idx, self._make_day)

    def _make_day(self, di):
        s, e = self.days[di]
        return (di, self._rows(s, e).float(), self.y[s:e])

    # ── 分块路径（chunk_days>1）────────────────────────────────────────────
    def iter_chunks(self, order):
        """yield (blk, Xb, yb, mask, is_last)，语义与原地组装一致（补位被 mask 掉）。"""
        valid = [int(d) for d in order
                 if (self.days[int(d)][1] - self.days[int(d)][0]) >= self.min_rows]
        groups = [valid[c0:c0 + self.chunk_days]
                  for c0 in range(0, len(valid), self.chunk_days)]
        yield from self._pipelined(groups, self._make_chunk)

    def _make_chunk(self, blk):
        C = len(blk)
        L = max(self.days[d][1] - self.days[d][0] for d in blk)
        F = self.store.shape[1]
        Xb = torch.zeros((C, L, F), dtype=torch.float32, device=self.dev)
        yb = self.y.new_zeros((C, L))
        mask = torch.zeros((C, L), dtype=torch.bool, device=self.dev)
        for i, d in enumerate(blk):
            s, e = self.days[d]
            Xb[i, :e - s] = self._rows(s, e)
            yb[i, :e - s] = self.y[s:e]
            mask[i, :e - s] = True
        return (blk, Xb, yb, mask)

    # ── 深度 1 的预取流水线 ───────────────────────────────────────────────
    def _pipelined(self, items, make):
        """惰性产出，并在独立 stream 上预建**下一**批；消费前 wait_stream 保序。

        产出元组末位追加 is_last，供调用方做末尾梯度 flush —— 调用方不该自己
        len() 整个序列（那会把所有批一次性建出来，显存瞬间爆掉）。
        GPU 驻留或非 cuda 时退化为直接构造：无 stream、无额外同步、零拷贝。
        """
        n = len(items)
        if not self.prefetch:
            for i, it in enumerate(items):
                yield make(it) + (i == n - 1,)
            return
        nxt = None
        for i, it in enumerate(items):
            if nxt is None:
                with torch.cuda.stream(self.stream):
                    nxt = make(it)
            cur, nxt = nxt, None
            if i + 1 < n:
                with torch.cuda.stream(self.stream):
                    nxt = make(items[i + 1])
            cs = torch.cuda.current_stream()
            cs.wait_stream(self.stream)
            # 必须 record_stream：cur 里的张量是在 self.stream 上分配的，caching
            # allocator 只跟踪「分配流」的生命周期。不登记消费流的话，本批张量一旦
            # 在 self.stream 看来空闲，就可能被下一批的分配复用 —— 而默认流上的
            # 前向可能还没读完，表现为**偶发的脏数据**（无报错、只是结果不对）。
            for t in cur:
                if torch.is_tensor(t) and t.is_cuda:
                    t.record_stream(cs)
            yield cur + (i == n - 1,)


def train_nam_gate(fold, feature_names, group_names, group_ids, regime_cols,
                   epochs=60, lr=2e-3, weight_decay=1e-5,
                   lambda_lb=0.01, lambda_ent=0.0, lambda_div=0.0, ema_momentum=0.02,
                   y_scale=10.0,
                   expert_hidden=16, gate_mode='softmax', gate_scalar_col='vol_expand',
                   accum_days=4,
                   warmup_epochs=5, temp_start=2.0, patience=12, min_epochs=20, seed=42,
                   device='auto', verbose=True, disable_gate=False,
                   select_metric='rank_ic', select_topk=20, time_decay_years=0.0,
                   select_holdout=0.0, chunk_days=1, store_dtype='auto', group_norm=False,
                   store_device='auto', prefetch=True):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = NAMGateModel(
        feature_names=list(feature_names), group_names=list(group_names),
        group_ids=group_ids, regime_cols=list(regime_cols),
        expert_hidden=expert_hidden, gate_mode=gate_mode,
        gate_scalar_col=gate_scalar_col, device=device,
    )
    model.disable_gate = disable_gate
    dev = torch.device(model.device)

    Xtr_np, Xva_np = fold['X_train'], fold['X_val']
    mean = Xtr_np.mean(axis=0)
    std = Xtr_np.std(axis=0)
    std[std < 1e-6] = 1.0
    model.input_mean, model.input_std = mean.astype(np.float32), std.astype(np.float32)

    # ── 显存驻留策略（T109，2026-08-17）────────────────────────────────────
    # 全量池 Xtr(5.93 GiB) + Xva(1.46 GiB) = 7.39 GiB **放不进 6.00 GiB 显存**。
    # Windows 的 NVIDIA 驱动开着 sysmem fallback：它不报 OOM，而是**静默**把溢出
    # 部分页到主机内存，于是每个 day-step 都要走 PCIe 取特征。这就是
    # 「utilization.gpu 100% / utilization.memory 0% / 28W / 2565MHz」的真实成因
    # —— 不是算力不足，也不是 DataLoader 喂不上，是计算核心在等 PCIe。
    # 实测每样本耗时：800 只池 29.6us → 全量池 52.8us；而 --chunk-days 4 反而
    # 恶化到 75.5us（补齐块 + 更大的中间激活把显存压得更紧），这正是页面抖动
    # 而非 kernel 启动开销的签名 —— 后者只会随批量变大而摊薄。
    # 对策：特征以 fp16 驻留，逐日切片时升回 fp32；模型权重、前向、反向、优化器
    # 全程 fp32 不变。7.39 → 3.70 GiB，装得下。z-score 后输入落在 ±10 内，
    # fp16 相对精度约 1e-3，远低于因子自身的截面噪声。
    _bytes_fp32 = (len(Xtr_np) + len(Xva_np)) * Xtr_np.shape[1] * 4
    _vram = (torch.cuda.get_device_properties(dev).total_memory
             if dev.type == 'cuda' else 0)
    if store_dtype == 'auto':
        # 留 15% 给权重、激活、优化器状态与 cuda context
        _fp16 = dev.type == 'cuda' and _bytes_fp32 > 0.85 * _vram
    else:
        _fp16 = store_dtype == 'fp16'
    store_dt = torch.float16 if _fp16 else torch.float32
    _bytes_store = _bytes_fp32 // (2 if _fp16 else 1)

    # ── 驻留设备（2026-08-19）：显存装不下时改走「主机驻留 + 批取数」──────────
    # auto 的判据是 0.68×显存：实测 T122（9 年窗）特征 3.95 GiB + 非特征开销
    # 1.70 GiB = 5.65 / 6.00 GiB，即非特征部分约占 28%，所以留 32% 余量。
    # 旧配方全部落在 cuda 分支（9y 66%、13y/2022 62%），行为逐位不变。
    # 关键：**auto 以前没有逃生口** —— fp16 仍超配时只能任驱动静默分页（12.5x 慢），
    # 现在会自动降为主机驻留，代价只有约 1% 的 PCIe 传输（还能被预取藏掉）。
    _store_dev = dev
    if dev.type == 'cuda':
        if store_device == 'host':
            _store_dev = torch.device('cpu')
        elif store_device == 'auto' and _bytes_store > 0.68 * _vram:
            _store_dev = torch.device('cpu')
    if verbose and dev.type == 'cuda':
        print(f"  特征驻留: {'fp16' if _fp16 else 'fp32'} "
              f"{_bytes_store / 2 ** 30:.2f} GiB "
              f"/ 显存 {_vram / 2 ** 30:.2f} GiB"
              + ("（fp32 需 %.2f GiB > 85%% 显存，自动降为 fp16 以避免 sysmem 分页）"
                 % (_bytes_fp32 / 2 ** 30) if _fp16 and store_dtype == 'auto' else ''))
        if _store_dev.type == 'cpu':
            _why = ('由 --store-device host 显式指定' if store_device == 'host'
                    else f'{_bytes_store / 2 ** 30:.2f} GiB 超过 0.68×显存 '
                         f'{0.68 * _vram / 2 ** 30:.2f} GiB，auto 转主机驻留以免驱动静默分页')
            print(f"  → **主机驻留 + DataLoader 批取数**（{_why}），"
                  f"预取={'开' if prefetch else '关'}")

    _mean32, _std32 = mean.astype(np.float32), std.astype(np.float32)

    def _to_gpu(a):
        """分块归一化并落到驻留设备 —— 整表 (a-mean)/std 会在主机侧再吃一份 5.93 GiB。

        主机驻留时用 pinned 内存：非 pinned 的 H2D 要先在驱动里过一次中转缓冲，
        且无法与计算重叠（`non_blocking` 失效）。pin 失败（内存不足/系统限制）
        则退回普通内存，只是慢一点，不影响正确性。
        """
        pin = _store_dev.type == 'cpu'
        try:
            out = torch.empty((len(a), a.shape[1]), dtype=store_dt,
                              device=_store_dev, pin_memory=pin)
        except RuntimeError as _e:
            if not pin:
                raise
            print(f"  [警告] pinned 内存分配失败（{_e}），退回普通主机内存："
                  f"H2D 无法与计算重叠，会比预期慢")
            out = torch.empty((len(a), a.shape[1]), dtype=store_dt, device=_store_dev)
        step = 1 << 19
        for s in range(0, len(a), step):
            blk = (np.asarray(a[s:s + step], dtype=np.float32) - _mean32) / _std32
            out[s:s + step] = torch.from_numpy(blk).to(_store_dev, dtype=store_dt)
        return out

    Xtr = _to_gpu(Xtr_np)
    Xva = _to_gpu(Xva_np)
    ytr = torch.as_tensor(fold['y_train'], dtype=torch.float32, device=dev)

    tr_days = _day_slices(fold['d_train'])
    va_days = _day_slices(fold['d_val'])
    # T096/2026-08-15：**选型集与报告集分离**。
    # 此前选 checkpoint 用的验证集和报告 rank_ic 用的是同一个，报告值就是
    # `max(history[*].val_rank_ic)` —— 一个 40 次抽样的顺序统计量。实测虚高
    # 25~35%（0.096 报告 vs 0.075 诚实），且虚高幅度随种子在 0.0189~0.0260 间
    # 摆动，是 σ_seed 的主要来源（见 [[checkpoint-selection-inflates-ic]]）。
    # 拿这种指标做 HPO 等于按噪声选超参。
    # 切法：验证折**按时间前段**做选型（inner），**后段**只用于报告（outer）。
    # 按时间切而非随机切，因为 7 日标签重叠会让随机切的两半互相泄漏。
    # select_holdout=0 时退化为旧行为（两者同一个集合），保证历史结果可复现。
    n_sel = len(va_days) if select_holdout <= 0 else \
        max(1, int(round(len(va_days) * (1.0 - select_holdout))))
    sel_slice, rep_slice = slice(0, n_sel), slice(n_sel, len(va_days))
    if verbose and select_holdout > 0:
        print(f"  选型/报告分离: 选型 {n_sel} 日 → 报告 {len(va_days) - n_sel} 日"
              f"（holdout={select_holdout:g}）")

    def _slice_days(day_slice):
        """把「第 i..j 个交易日」翻译成样本行索引（va_days 是 [(start,end)] 列表）。"""
        chunk = va_days[day_slice]
        if not chunk:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in chunk])
    # T086/E16：训练日的时间衰减权重 —— 越久远的交易日对损失贡献越小。
    # 归一化成均值 1，这样有效学习率和不加权时可比（否则等于偷偷调小了 lr）。
    if time_decay_years and time_decay_years > 0:
        _td = pd.to_datetime(pd.Series([fold['d_train'][s] for s, _ in tr_days]))
        _age = (_td.max() - _td).dt.days.to_numpy() / 365.25
        _w = np.exp(-_age / float(time_decay_years))
        _w = _w / _w.mean()
        if verbose:
            print(f"  时间衰减 τ={time_decay_years}y：最老日权重 {_w.min():.4f}、"
                  f"最新 {_w.max():.4f}（均值归一），跨度 {_age.max():.1f}y")
        day_w = torch.as_tensor(_w, dtype=torch.float32, device=dev)
    else:
        day_w = None
    # 每日的市场状态向量：同日全市场同值，只取该日首行
    Mtr = torch.as_tensor(np.stack([fold['M_train'][s] for s, _ in tr_days]),
                          dtype=torch.float32, device=dev)
    Mva = torch.as_tensor(np.stack([fold['M_val'][s] for s, _ in va_days]),
                          dtype=torch.float32, device=dev)

    # DataLoader 形式的取数器。GPU 驻留时是零拷贝切片（与 T109~T122 逐位一致）；
    # 主机驻留时按块 H2D 并用独立 stream 预取。验证侧不预取：它每 epoch 只跑一遍
    # 且在 no_grad 下，藏不出什么，多一条 stream 反而增加同步点。
    n_feat = Xtr.shape[1]
    train_loader = _DayBatchLoader(Xtr, ytr, Mtr, tr_days, dev,
                                   chunk_days=chunk_days, prefetch=prefetch)
    val_loader = _DayBatchLoader(Xva, ytr.new_zeros(0), Mva, va_days, dev,
                                 chunk_days=1, prefetch=False, min_rows=0)

    net = model.build(d_regime=Mtr.shape[1], disable_gate=disable_gate,
                      group_norm=group_norm)
    if chunk_days > 1 and (lambda_lb > 0 or lambda_div > 0 or lambda_ent > 0):
        # 门控正则项作用在 GateUsageTracker 的跨日 EMA 上，逐日更新的节奏是它的
        # 语义的一部分；分块会把 chunk_days 天并成一次更新，等于偷偷改了 momentum。
        # 与其给出一个「快但不等价」的结果，不如直接拒绝。
        raise ValueError('--chunk-days > 1 目前只支持 lambda_lb/div/ent 全为 0 '
                         '（门控正则依赖逐日 EMA 更新节奏）；生产配置本就是 --disable-gate --lambda-lb 0')
    n_params = sum(p.numel() for p in net.parameters())
    if verbose:
        print(f"  NAMGate: {len(feature_names)} 因子 / {len(group_names)} 族 / "
              f"regime {Mtr.shape[1]} 维 / 参数 {n_params:,} / device {model.device}")
        print(f"  训练 {len(tr_days)} 日 {len(Xtr_np)} 样本，验证 {len(va_days)} 日 {len(Xva_np)} 样本")

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    def _predict_val(temp=1.0):
        """验证期前向。全部结果留在 GPU 上拼接，最后只做一次 device→host 同步。"""
        net.eval()
        score_parts, gate_parts = [], []
        with torch.no_grad():
            for j, (s, e) in enumerate(va_days):
                sc, w, _ = net.forward_day(val_loader._rows(s, e).float(), Mva[j], temp)
                score_parts.append(sc.detach())
                gate_parts.append(w.detach())
            out = torch.cat(score_parts).to(torch.float32).cpu().numpy()
            gates = torch.stack(gate_parts).to(torch.float32).cpu().numpy()
        return out, gates

    best_ic, best_state, since_best, best_epoch_idx = -np.inf, None, 0, 0
    last_state = None
    history = []
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        temp = temp_start - (temp_start - 1.0) * min(1.0, epoch / max(1, warmup_epochs)) \
            if warmup_epochs > 0 else 1.0
        net.train()
        order = rng.permutation(len(tr_days))
        steps = 0
        opt.zero_grad(set_to_none=True)
        pending = 0
        # 全部标量/向量统计留在 GPU 上累加，epoch 末一次性取回。
        # 逐日 .cpu()/float() 会强制 device 同步，2500 训练日 × 60 epoch ≈ 15 万次阻塞。
        tot_rank_t = torch.zeros((), dtype=torch.float64, device=dev)
        w_accum_t = torch.zeros(len(group_names), dtype=torch.float64, device=dev)
        ent_day_sum_t = torch.zeros((), dtype=torch.float64, device=dev)
        w_days = 0
        # 长期均衡跟踪器：每个 epoch 重置，惩罚作用在跨日 EMA 上而非单日 batch
        tracker = GateUsageTracker(len(group_names), momentum=ema_momentum, device=dev)

        for pos, di in enumerate(order):
            if chunk_days > 1:
                break        # 走下方的分块路径
            s, e = tr_days[di]
            if e - s < 5:
                continue
            # .float() 在 fp32 驻留时返回自身（无拷贝），fp16 驻留时逐日升精度。
            # 逐日路径**刻意不走预取**：`pos == len(order) - 1` 这个 flush 条件依赖
            # 未过滤的 order（末日若样本 <5 会被 continue 掉、于是不 flush），
            # 换成预取器的已过滤序列会改变末尾梯度的处置 —— 不是错，但破坏逐位复现。
            score, w, _ = net.forward_day(train_loader._rows(s, e).float(), Mtr[di], temp)
            l_rank = listnet_loss(score, ytr[s:e], y_scale=y_scale)
            if day_w is not None:
                l_rank = l_rank * day_w[di]
            loss = l_rank
            w_batch = w.unsqueeze(0)
            balance, diversity = tracker.update(w_batch)
            if lambda_lb > 0:
                # 长期使用率均衡（允许单日偏科）
                loss = loss + lambda_lb * balance * (e - s)
            if lambda_div > 0:
                # 奖励当日权重偏离长期均值 —— 直接对抗"死门控"
                loss = loss - lambda_div * diversity * (e - s)
            if lambda_ent > 0:
                loss = loss - lambda_ent * gate_entropy(w_batch) * (e - s)
            (loss / accum_days).backward()
            pending += 1
            tot_rank_t += l_rank.detach().to(torch.float64)
            steps += 1
            _wd = w.detach().to(torch.float64)
            w_accum_t += _wd
            _p = _wd / _wd.sum().clamp_min(1e-8)
            ent_day_sum_t -= (_p * torch.log(_p + 1e-8)).sum()
            w_days += 1
            if pending >= accum_days or pos == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                pending = 0

        if chunk_days > 1:
            # ── 分块路径（T097）：一次前向算 chunk_days 天 ──────────────────
            # 瓶颈实测不在算力而在 kernel 启动开销：单日 800×219 的前向只要
            # 微秒级算力，却要付一次完整的 launch + 同步，2500 日 × 60 epoch
            # ≈ 15 万次。实测 chunk=4 相对逐日 **3.8x**，且提速几乎全部来自
            # 批前向而非「少走几步」—— 所以把 step 频率仍锁在 accum_days，
            # 与逐日路径语义等价（只有浮点求和顺序不同）。
            # 各交易日股票数不等，按块内最长补齐 + mask：补位 logit 填 -inf
            # 使其在 softmax 里权重恒为 0，目标分布同样置 0。
            # 2026-08-19：批组装搬进 _DayBatchLoader（DataLoader 形式）。分块序列
            # 本来就是先过滤 <5 行的日再切块，与预取器完全同构，所以这条路径可以
            # 安全地开预取；末尾 flush 用 `_last_chunk` 判定，与原 `c0 + chunk_days
            # >= len(valid)` 等价。
            _chunks = train_loader.iter_chunks(order)
            for blk, Xb, yb, mask, _last_chunk in _chunks:
                C = len(blk)
                L = Xb.shape[1]
                contrib = net.experts(Xb.reshape(C * L, -1))
                gsum = (contrib @ net.group_onehot).reshape(C, L, -1)
                if net.disable_gate:
                    score = gsum.sum(-1) + net.bias
                    w_blk = torch.ones(C, len(group_names), device=dev)
                else:
                    w_blk = net.gate(Mtr[blk], temp)              # [C, K]
                    score = (gsum * w_blk.unsqueeze(1)).sum(-1) + net.bias
                # 掩码 softmax：补位不参与归一化，也不贡献损失
                neg = torch.finfo(score.dtype).min
                logp = torch.log_softmax(score.masked_fill(~mask, neg), dim=1)
                tgt = torch.softmax((yb * y_scale).masked_fill(~mask, neg), dim=1)
                l_days = -(tgt * logp.masked_fill(~mask, 0.0)).sum(1)   # [C]
                if day_w is not None:
                    l_days = l_days * day_w[torch.as_tensor(blk, device=dev)]
                loss = l_days.sum()
                (loss / accum_days).backward()
                pending += C
                tot_rank_t += l_days.detach().sum().to(torch.float64)
                steps += C
                _wd = w_blk.detach().to(torch.float64)
                w_accum_t += _wd.sum(0)
                _p = _wd / _wd.sum(1, keepdim=True).clamp_min(1e-8)
                ent_day_sum_t -= (_p * torch.log(_p + 1e-8)).sum()
                w_days += C
                if pending >= accum_days or _last_chunk:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    pending = 0

        pred_va, gates_va = _predict_val(1.0)
        m = _daily_metrics(pred_va, fold['y_val'], np.asarray(fold['d_val']))
        # 与 Top-K 策略对齐的验证量：用真实前向收益而非标签排名
        m['top20_excess'] = _topk_excess(pred_va, fold['ret_val'], va_days, k=select_topk)
        if select_holdout > 0:
            # 选型只看 inner 段，报告段（outer）对早停完全不可见 —— 这是本次修复的核心。
            _sd = _slice_days(sel_slice)
            m_sel = _daily_metrics(pred_va[_sd], fold['y_val'][_sd],
                                   np.asarray(fold['d_val'])[_sd])
            m_sel['top20_excess'] = _topk_excess(pred_va[_sd], fold['ret_val'][_sd],
                                                 _day_slices(np.asarray(fold['d_val'])[_sd]),
                                                 k=select_topk)
            _rd = _slice_days(rep_slice)
            m_rep = _daily_metrics(pred_va[_rd], fold['y_val'][_rd],
                                   np.asarray(fold['d_val'])[_rd])
            m['rank_ic_sel'] = m_sel['rank_ic']
            m['rank_ic_holdout'] = m_rep['rank_ic']
            sel_score = m_sel['rank_ic'] if select_metric == 'rank_ic' else m_sel['top20_excess']
        else:
            sel_score = m['rank_ic'] if select_metric == 'rank_ic' else m['top20_excess']
        sched.step(-sel_score)

        tot_rank = float(tot_rank_t)
        w_accum = w_accum_t.cpu().numpy()
        ent_day_sum = float(ent_day_sum_t)
        w_mean = w_accum / max(w_days, 1)
        share = w_mean / max(w_mean.sum(), 1e-8)
        ent = float(-(share * np.log(share + 1e-8)).sum())
        rec = {
            'epoch': epoch, 'temp': round(temp, 3),
            'train_listnet': tot_rank / max(steps, 1),
            'val_rank_ic': m['rank_ic'], 'val_top5_excess': m['top5_excess'],
            'val_top20_excess': m['top20_excess'], 'select_score': sel_score,
            # 分离模式下这两栏才是可信读数：holdout 段对早停不可见，是无偏的
            'val_rank_ic_sel': m.get('rank_ic_sel'),
            'val_rank_ic_holdout': m.get('rank_ic_holdout'),
            'gate_entropy': ent, 'gate_max_share': float(share.max()),
            # 单日熵：越低说明门控当天越"有主张"；与上面的长期熵配合看
            'gate_entropy_day': ent_day_sum / max(w_days, 1),
            'gate_mean': [round(float(v), 4) for v in w_mean],
        }
        history.append(rec)
        last_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        # 仅在训练足够轮数后才允许记录最佳检查点：避免噪声验证 IC 选中欠训练
        # 的检查点（如 epoch 2 看似峰值、真实输入下形状函数近零、得分退化成常数）。
        if epoch >= min_epochs and sel_score > best_ic:
            best_ic = sel_score
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best_epoch_idx = len(history) - 1
            since_best = 0
        else:
            if epoch >= min_epochs:
                since_best += 1

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1 or since_best >= patience):
            print(f"    [ep {epoch:>3}] listnet={rec['train_listnet']:.4f} "
                  f"val_ic={m['rank_ic']:.4f} top5={m['top5_excess']:.4f} "
                  f"top20={m['top20_excess']:.5f}{'*' if select_metric != 'rank_ic' else ''} "
                  f"gate_H={ent:.3f}/{np.log(len(group_names)):.3f} "
                  f"H_day={rec['gate_entropy_day']:.3f} "
                  f"max_share={share.max():.3f} T={temp:.2f}")
        if share.max() > 0.60:
            print(f"    [警告] 门控最大族占比 {share.max():.3f} > 0.60，存在专家坍塌风险："
                  f"建议上调 --lambda-lb 或切 --gate-mode multiplicative")
        if epoch >= min_epochs and since_best >= patience:
            print(f"    [早停] epoch {epoch}（验证 {select_metric} 连续 {patience} 轮无改善，且已过 min_epochs）")
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    elif last_state is not None:
        net.load_state_dict(last_state)
        best_epoch_idx = len(history) - 1
    net.to(dev).eval()
    model.is_trained = True

    pred_va, gates_va = _predict_val(1.0)
    model.compute_feature_importance(Xtr_np)
    return model, pred_va, gates_va, history, best_ic, best_epoch_idx


# ---------------------------------------------------------------------------
# 可解释性出图
# ---------------------------------------------------------------------------

def export_interpretability(model, gates_va, va_dates, history, out_dir, top_n=12):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    produced = []

    # 1. 因子形状函数
    tops = [n for n, _ in model.get_top_factors(top_n)]
    curves = model.shape_curves(tops)
    if curves:
        ncol = 4
        nrow = int(np.ceil(len(curves) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow))
        for ax, (name, (g, v)) in zip(np.ravel(axes), curves.items()):
            # 去掉每条曲线的常数偏置再画：偏置 ~10 而形状幅度 ~0.1，纵轴从 0 画起
            # 时所有曲线都是水平线，图完全不可读（T098 一批的历史图即如此）。
            v = v - v.mean()
            ax.plot(g, v, color='#c0392b', lw=1.8)
            ax.axhline(0, color='#999', lw=0.6, ls='--')
            ax.set_title(name, fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in np.ravel(axes)[len(curves):]:
            ax.axis('off')
        fig.suptitle('因子形状函数 f_i(x_i)−均值：横轴=当日截面分位，纵轴=对得分的贡献（已去常数偏置）', fontsize=11)
        fig.tight_layout()
        p = os.path.join(out_dir, 'shape_functions.png')
        fig.savefig(p, dpi=130)
        plt.close(fig)
        produced.append(p)

    # 2. Regime → 因子族 门控热力图
    if gates_va is not None and len(gates_va):
        uniq = pd.DatetimeIndex(pd.to_datetime(pd.Series(va_dates).unique())).sort_values()
        gdf = pd.DataFrame(gates_va, index=uniq[:len(gates_va)], columns=model.group_names)
        fig, ax = plt.subplots(figsize=(13, 4.5))
        im = ax.imshow(gdf.T.to_numpy(), aspect='auto', cmap='RdYlBu_r', origin='lower')
        ax.set_yticks(range(len(model.group_names)))
        ax.set_yticklabels(model.group_names, fontsize=8)
        step = max(1, len(gdf) // 12)
        ax.set_xticks(range(0, len(gdf), step))
        ax.set_xticklabels([d.strftime('%y-%m') for d in gdf.index[::step]], fontsize=8, rotation=45)
        ax.set_title('市场状态 → 因子族门控权重（红=放大，蓝=压制；1.0 为中性）', fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        p = os.path.join(out_dir, 'gate_heatmap.png')
        fig.savefig(p, dpi=130)
        plt.close(fig)
        produced.append(p)
        gdf.to_csv(os.path.join(out_dir, 'gate_weights.csv'), encoding='utf-8-sig')

    # 3. 训练稳定性：门控熵与验证 IC
    if history:
        h = pd.DataFrame(history)
        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax1.plot(h['epoch'], h['gate_entropy'], color='#2980b9', label='门控熵')
        ax1.axhline(np.log(len(model.group_names)), color='#2980b9', ls=':', lw=1,
                    label='均匀分布上界 lnK')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('门控熵', color='#2980b9')
        ax2 = ax1.twinx()
        ax2.plot(h['epoch'], h['val_rank_ic'], color='#c0392b', label='验证 Rank IC')
        ax2.set_ylabel('验证 Rank IC', color='#c0392b')
        ax1.set_title('训练稳定性：门控熵未塌陷则说明各因子族都在被使用')
        fig.tight_layout()
        p = os.path.join(out_dir, 'gate_stability.png')
        fig.savefig(p, dpi=130)
        plt.close(fig)
        produced.append(p)
        h.to_csv(os.path.join(out_dir, 'train_history.csv'), index=False, encoding='utf-8-sig')

    return produced


# ---------------------------------------------------------------------------

def _strip(payload):
    return {k: v for k, v in payload.items() if k not in {'predictions', 'returns', 'dates'}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stocks', type=int, default=800)
    ap.add_argument('--years', type=int, default=8)
    ap.add_argument('--end', required=True)
    ap.add_argument('--estimators', type=int, default=500)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--min-epochs', type=int, default=20,
                    help='早停与最佳检查点捕获的最少训练轮数（防止噪声验证 IC 选中欠训练检查点）')
    ap.add_argument('--select-metric', default='rank_ic', choices=['rank_ic', 'top20_excess'],
                    help='早停与最佳检查点选型指标。rank_ic=低方差整截面统计量（生产默认）；'
                         'top20_excess=仅保留用于复现实验T056，已证实会过拟合验证期头部噪声，禁止用于新实验')
    ap.add_argument('--select-topk', type=int, default=20,
                    help='select-metric=top20_excess 时的头部档位 K')
    ap.add_argument('--chunk-days', type=int, default=1,
                    help='T097 训练加速：一次前向计算多少个交易日。1=逐日（历史行为）。'
                         '训练段孤立微基准 3.8x，但端到端只有 1.07~1.23x（800 只池上'
                         '大头是数据加载）。**全量池上反而更慢**（每样本 75.5us vs '
                         '逐日 52.8us）：补齐块与更大的中间激活加重显存压力，触发'
                         'sysmem 分页。所以全量池请配 --store-dtype fp16 后再评估是否要开。'
                         'optimizer 步频仍锁在 --accum-days，与逐日路径语义等价 —— '
                         '同种子同折三折早停 epoch 完全一致，holdout IC 差 +0.0004~0.0012。'
                         '门控正则（lambda_lb/div/ent）非 0 时不可用')
    ap.add_argument('--contrib-train-days', type=int, default=400,
                    help='T110：训练侧贡献诊断的抽样交易日数（0=全量）。全量池训练侧 2727 日，'
                         '逐日算三组 219 列 float64 IC 约 9 分钟，占单次运行收尾的大头，'
                         '而这些数只进 factor_effectiveness.csv 的 train_* 列与稳定性散点，'
                         '**不参与任何晋级判定**。等间隔抽样保留全时段覆盖；'
                         '跨日均值标准误放大 √(2727/400)≈2.6 倍。验证侧一律精确不抽样。')
    ap.add_argument('--store-dtype', choices=('auto', 'fp32', 'fp16'), default='auto',
                    help='T109：特征矩阵在显存里的驻留精度（计算全程仍是 fp32）。'
                         'auto=fp32 占用超过 85%% 显存时自动降 fp16。'
                         '全量池 Xtr+Xva 是 7.39 GiB > 6.00 GiB 显存，驱动会静默页到'
                         '主机内存，每个 day-step 走 PCIe —— 这是全量池慢的主因，'
                         '不是算力不足。fp16 驻留后 3.70 GiB 装得下。'
                         'fp32=强制旧行为（复现 T098/T106 及更早结果时用）')
    ap.add_argument('--store-device', choices=('auto', 'cuda', 'host'), default='auto',
                    help='2026-08-19：特征矩阵驻留在哪。cuda=整表进显存（T109~T122 行为，'
                         '最快但**窗口长度被显存钉死**）；host=驻留 pinned 主机内存，'
                         '按 DataLoader 形式逐块 H2D（显存占用与窗口长度解耦）。'
                         'auto=fp16 后仍超过 68%%显存就自动转 host —— 这是 auto 以前缺的'
                         '逃生口：超配时 Windows 驱动不报 OOM 而是静默分页，实测慢 12.5x'
                         '（[[vram-oversubscription-was-the-bottleneck]]），比走 PCIe 批取数差得多。'
                         '本机 6.00 GiB 显存下：9 年窗 3.95 GiB 走 cuda，13 年窗 5.05 GiB 转 host。'
                         '开销参考：每日约 2.1 MB，pinned H2D 约 0.18 ms 对一个 chunk 约 15 ms '
                         '的前反向 ⇒ 约 1%%，且被预取藏掉。**它买的是容量不是速度。**')
    ap.add_argument('--prefetch', action='store_true',
                    help='主机驻留时用独立 CUDA stream 预取下一批。**默认关，因为实测无收益**：'
                         '用真实 NAMGateNet 测得 66.01(预取) vs 66.08(不预取) vs 64.83(cuda驻留) '
                         'ms/chunk —— 每日 2.1MB 的 H2D 只要 0.18ms，对 65ms 的前反向本就完全'
                         '被掩盖，藏不出东西；而多一条 stream 要付跨流分配与 record_stream 的开销。'
                         '留着这个开关是为了将来「窗口大到内存也装不下、需从 parquet 惰性读」时'
                         '真正需要预取的场景。')
    ap.set_defaults(prefetch=False)
    ap.add_argument('--select-holdout', type=float, default=0.0,
                    help='T096：把验证折按时间切成「前段选型 / 后段报告」，比例=后段占比。'
                         '0=旧行为（选型与报告同一集合，报告值是 40 次抽样的最大值，'
                         '实测虚高 25~35%%，见 [[checkpoint-selection-inflates-ic]]）。'
                         '做 HPO / 跨配置横比时必须设为 0.3~0.5，否则是在按噪声选超参')
    ap.add_argument('--regime-topk', type=int, default=40,
                    help='行情分层诊断（nam_gate.regime）的头部档位 K。默认 40 而非生产的 20：'
                         'K=40 噪声更低，只当尺子用（见 T069/T072 的 K 轴结论）')
    ap.add_argument('--time-decay-years', type=float, default=0.0,
                    help='T086/E16：训练日损失的时间衰减常数 τ（年），0=不衰减。'
                         '权重 exp(−样本距今年数/τ) 后归一化成均值 1（保持有效 lr 可比）。'
                         '立论：13 年训练跨度里 A 股结构变过多次，老样本可能是负资产；'
                         '但下调老样本也会减少有效样本量，方向要靠实验定')
    ap.add_argument('--seed', type=int, default=42,
                    help='随机种子（torch/numpy/采样）。回测结论对种子敏感，多种子复跑取中位数')
    ap.add_argument('--seeds', default=None,
                    help='T110 效率：在**同一进程内**依次跑多个种子，逗号分隔（如 11,23,37），'
                         '复用同一份已构建的数据集与因子审计。全量池单次数据构建约 11.5 分钟，'
                         'n=4 判决分进程跑等于白扔 34 分钟。给定时覆盖 --seed。'
                         '输出路径自动插入 `_s{seed}`（--output/--save-model-dir/--plot-dir），'
                         '也可在路径里显式写 `{seed}` 占位符自定义位置。')
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-5,
                    help='AdamW 权重衰减；生产基线为 1e-5，0 表示关闭正则')
    ap.add_argument('--y-scale', type=float, default=2.0,
                    help='ListNet 目标 softmax 温度（越大越聚焦头部；2 为经验收敛值，10 会不收敛）')
    ap.add_argument('--lambda-lb', type=float, default=0.0,
                    help='门控长期使用率均衡（CV²）正则；>0 会把门控压向均匀，抑制 regime 路由，默认关闭')
    ap.add_argument('--lambda-ent', type=float, default=0.0)
    ap.add_argument('--expert-hidden', type=int, default=16)
    ap.add_argument('--gate-mode', default='softmax',
                    choices=['softmax', 'multiplicative', 'scalar'],
                    help="scalar = E6 标量条件化：只有 n_groups 个门控参数，无 MLP")
    ap.add_argument('--gate-scalar-col', default='vol_expand',
                    help='gate-mode=scalar 时驱动门控的 regime 列名（缺列硬失败）')
    ap.add_argument('--disable-gate', action='store_true',
                    help='纯加性 NAM（门控权重恒为 1，隔离门控贡献）')
    ap.add_argument('--group-norm', action='store_true',
                    help='T119：门控前对族求和做**当日截面标准化**。'
                         '乘积 w_k·S_k 在 (w_k→c·w_k, S_k→S_k/c) 下不变 ⇒ 损失曲面有平坦方向 ⇒ '
                         '门控权重随噪声漂移（T116 实测 status 族 gate_std 4.45 > gate_mean 3.97）。'
                         '标准化锁死 f 的幅度，门控只能调相对重要性、无法被专家吸收。'
                         '仅作用于门控通路，--disable-gate 时不生效（纯加性基线逐位不变）。')
    ap.add_argument('--target', default='returns', choices=['scores', 'returns'],
                    help='训练目标：returns=原始收益排名（与回测收益对齐，默认）；'
                         'scores=生产 vol_boosted 标签（已证实与收益解耦，会拟合反号噪声）')
    ap.add_argument('--label-residualize', default='none',
                    choices=['none', 'vol', 'beta_size'],
                    help='E3 标签去噪：在截面 rank 之前剥掉收益里的风险成分。'
                         'vol=按 price_volatility_20 归一；'
                         'beta_size=对 [vol, log(market_cap)] 逐日截面 OLS 取残差。'
                         '注意：该轴会使验证 Rank IC 下降（删掉的正是最好预测的部分），'
                         '判定必须用双窗口随机零假设分位，不能用 IC。')
    ap.add_argument('--label-transform', default='rank',
                    choices=['rank', 'winsor_z', 'signed_sqrt'],
                    help='E19 训练标签的截面变换。rank=基线（逐日分位，销毁幅度）；'
                         'winsor_z=robust z-score 截断到 ±--label-clip，保留线性幅度；'
                         'signed_sqrt=sign(z)·sqrt(|z|)，压缩厚尾后的幅度。'
                         '非 rank 时已把每日矩对齐到 rank 标签，softmax 温度效应不变；'
                         '只换训练标签，验证标签恒为 rank 口径。')
    ap.add_argument('--label-clip', type=float, default=3.0,
                    help='--label-transform 的 winsor 截断（单位：robust σ）')
    ap.add_argument('--lambda-div', type=float, default=0.0,
                    help='跨日多样性奖励系数（对抗死门控）')
    ap.add_argument('--ema-momentum', type=float, default=0.02,
                    help='门控长期使用率 EMA 动量')
    ap.add_argument('--accum-days', type=int, default=4)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--keep-manual-interaction', action='store_true',
                    help='保留手工 *_regime_* 交互列（默认剔除，改由门控学习）')
    ap.add_argument('--drop-groups', default='',
                    help='逗号分隔的因子族名，训练前整族剔除（依据 group_effectiveness.csv 的 effective IC）')
    ap.add_argument('--drop-features', default='',
                    help='逗号分隔的因子名，训练前逐列剔除（T114 瘦身轴）。未知名硬失败，'
                         '防止拼错列名后静默训练在错误面板上')
    ap.add_argument('--drop-features-file', default=None,
                    help='文本文件，每行一个因子名（# 开头为注释行）；与 --drop-features 取并集。'
                         '用于把杀名单固化成可审计的文件而不是命令行长串')
    ap.add_argument('--group-map-file', default=None,
                    help='T118：用 {feature: cluster_id} 的 JSON 覆盖 config/factor_groups 的'
                         '手工分族。立论见 T117 —— 手工 12 族的内聚度与随机划分不可区分'
                         '（z≈+0.1），而数据驱动簇内聚 0.35~0.50（z≈+73）。'
                         '映射必须**只由训练段导出**（`diag_factor_clustering.py '
                         '--emit-group-map`），否则等于把验证信息带进架构选择。'
                         '缺列/多列一律硬失败。')
    ap.add_argument('--folds', default='0.7:0.8,0.8:1.0',
                    help='折列表，形如 "0.7:0.8,0.8:1.0"')
    ap.add_argument('--skip-baseline', action='store_true',
                    help='跳过 XGBoost 基线重跑（用于快速迭代 NAM 超参）')
    ap.add_argument('--allow-degenerate-downside-risk', action='store_true',
                    help='仅用于复现旧实验：允许 downside_risk 缓存退化；新训练禁止使用')
    ap.add_argument('--cache-dir', default=None,
                    help='独立版本化因子缓存目录；新公式训练必须显式指定，禁止覆盖历史共享缓存')
    ap.add_argument('--industry-relative', default='none',
                    help="E7：追加当日行业内百分位列 <base>__ind。'none' 关闭；"
                         "'core' 用内置核心列表；也可传逗号分隔的因子名")
    ap.add_argument('--multi-horizon', default='none',
                    help="E11：用多持有期复合排名当**训练**标签（验证标签仍是 7 日基线，"
                         "保持可比）。'none' 关闭；例 '5,10,20'")
    ap.add_argument('--save-model-dir', default=None)
    ap.add_argument('--plot-dir', default='diagnose_output/nam_gate')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    end_dt = datetime.strptime(args.end, '%Y-%m-%d')
    end = end_dt.strftime('%Y-%m-%d')
    start = (end_dt - timedelta(days=365 * args.years)).strftime('%Y-%m-%d')
    started = time.time()

    TrainingConfig.FUTURE_DAYS = 7  # 与 7 日生产基线对齐（注意：不能改 SHORT_PREDICTION）
    trainer = MLModelTrainer(db_path=DATABASE_PATH, cache_dir=args.cache_dir)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()['code'].tolist()[:args.stocks]
    manager.close()
    stocks_data = trainer.load_label_data(codes, start, end)
    _indrel = args.industry_relative.strip().lower() != 'none'
    _horizons = None
    if args.multi_horizon.strip().lower() != 'none':
        _horizons = sorted({int(s) for s in args.multi_horizon.split(',') if s.strip()})
        if not _horizons:
            raise SystemExit('--multi-horizon 解析为空')
    if _horizons and args.label_transform != 'rank':
        raise SystemExit('--multi-horizon 与 --label-transform 互斥：多期标签是把多个'
                         '窗口的分位相加，本身已经是 rank 复合，再谈幅度没有定义')
    dataset = trainer.prepare_dataset(
        stocks_data, train_start_date=start, train_end_date=end,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
        n_jobs=args.workers, target_features=None, use_factor_cache_only=True,
        return_sample_metadata=_indrel or bool(_horizons),
    )
    _fwd = None
    if _horizons:
        print(f'  多持有期标签: {_horizons}（训练标签换成各期截面 rank 的均值）')
        _fwd = _forward_returns_by_horizon(stocks_data, dataset[10], _horizons)
    if _indrel:
        _meta = dataset[10]
        _flist = INDREL_CORE if args.industry_relative.strip().lower() == 'core' else \
            [s.strip() for s in args.industry_relative.split(',') if s.strip()]
        dataset = _add_industry_relative(dataset, _meta, _flist)
        del _meta
    all_features = list(dataset[3])
    print(f"  全部特征 {len(all_features)}，样本 {len(dataset[0])}")

    # 释放标签行情原表：`prepare_dataset` 已把需要的东西全部物化进 dataset，
    # `_forward_returns_by_horizon` 也已在上面算完。4777 只股票的 DataFrame 字典
    # 是纯冗余占用。实测该进程提交量 29.01 GB / 峰值工作集 21.54 GB，压在 31.7 GB
    # 的机器上 —— 内存压力会让 `_to_gpu` 上传时把已换出的页从磁盘错回来，是
    # 「某个种子随机慢 5~50 倍」的候选成因之一（另一半是显存 95% 贴顶）。
    del stocks_data
    gc.collect()

    # 已知缓存契约审计：旧 downside_risk 公式会使该列几乎全为 0。
    # exp_nam_gate 使用 cache-only，代码公式修复不会自动重建同名缓存；若不硬失败，
    # 下一轮仍会悄悄训练在坏特征上。
    if 'downside_risk' in all_features:
        _ds_idx = all_features.index('downside_risk')
        _ds = np.asarray(dataset[0][:, _ds_idx], dtype=float)
        _finite = _ds[np.isfinite(_ds)]
        _nonzero_ratio = float(np.mean(np.abs(_finite) > 1e-12)) if len(_finite) else 0.0
        _std = float(np.std(_finite)) if len(_finite) else 0.0
        print(f"  downside_risk 缓存审计: 非零率={_nonzero_ratio:.3%}, std={_std:.6g}")
        if (_nonzero_ratio < 0.01 or _std < 1e-8) and not args.allow_degenerate_downside_risk:
            raise RuntimeError(
                'downside_risk 缓存已退化；拒绝训练。请先用修复后的因子公式全量重建缓存，'
                '再训练并绑定新的因子版本。仅复现旧实验可传 --allow-degenerate-downside-risk。'
            )

    nam_features = all_features if args.keep_manual_interaction else \
        [f for f in all_features if '_regime_' not in f]
    dropped = len(all_features) - len(nam_features)
    print(f"  NAM 输入特征 {len(nam_features)}（剔除手工交互 {dropped} 列）")

    group_names, group_ids = build_group_index(nam_features)

    # 按因子族裁剪：训练后诊断（group_effectiveness.csv）显示部分族的 effective IC
    # 长期为零或为负，却占据可观的贡献量级，属于纯噪声源。此开关允许直接剔除。
    if args.drop_groups:
        _drop = {g.strip() for g in args.drop_groups.split(',') if g.strip()}
        _unknown = _drop - set(group_names)
        if _unknown:
            raise ValueError(f"--drop-groups 含未知因子族: {sorted(_unknown)}；可选: {group_names}")
        _keep = [f for f, gid in zip(nam_features, group_ids)
                 if group_names[int(gid)] not in _drop]
        print(f"  按族裁剪 {sorted(_drop)}: 特征 {len(nam_features)} → {len(_keep)}")
        nam_features = _keep
        group_names, group_ids = build_group_index(nam_features)

    # 逐列裁剪（T114 瘦身轴）：死亡交集 / 精确重复 / 低覆盖列
    _dropf = {s.strip() for s in args.drop_features.split(',') if s.strip()}
    if args.drop_features_file:
        with open(args.drop_features_file, encoding='utf-8') as _fh:
            for _ln in _fh:
                # 行尾注释要剥掉：杀名单用 `列名  # [dead]` 记录每列的杀因来源，
                # 只跳整行注释会把整条 "名字 # 标签" 当成列名，撞上下面的硬失败。
                _name = _ln.split('#', 1)[0].strip()
                if _name:
                    _dropf.add(_name)
    if _dropf:
        _unknown = _dropf - set(nam_features)
        if _unknown:
            raise ValueError(f"--drop-features 含未知/已被剔除的因子名: {sorted(_unknown)[:10]}"
                             f"（共 {len(_unknown)} 个）")
        nam_features = [f for f in nam_features if f not in _dropf]
        group_names, group_ids = build_group_index(nam_features)
        print(f"  逐列裁剪 {len(_dropf)} 列: 特征 → {len(nam_features)}")

    # T118：用数据驱动簇覆盖手工分族。放在所有裁剪之后，因为映射是对**最终面板**
    # 逐列给出的；缺列或多列都硬失败（静默 fallback 会让"数据驱动"实际跑成手工族）。
    if args.group_map_file:
        with open(args.group_map_file, encoding='utf-8') as _fh:
            _gmap = json.load(_fh)
        _miss = [f for f in nam_features if f not in _gmap]
        _extra = [f for f in _gmap if f not in set(nam_features)]
        if _miss or _extra:
            raise ValueError(
                f"--group-map-file 与最终面板不匹配：缺 {len(_miss)} 列 "
                f"{_miss[:5]}，多 {len(_extra)} 列 {_extra[:5]}。"
                f"映射必须对当前面板（{len(nam_features)} 列）逐列给出。")
        _cids = sorted({int(v) for v in _gmap.values()})
        _remap = {c: i for i, c in enumerate(_cids)}
        group_names = [f'c{c}' for c in _cids]
        group_ids = np.asarray([_remap[int(_gmap[f])] for f in nam_features], dtype=np.int64)
        _sz = pd.Series(group_ids).value_counts().sort_index().to_dict()
        print(f"  分族来源: 数据驱动簇 {args.group_map_file} "
              f"（{len(group_names)} 簇，规模 {_sz}）")

    print(f"  因子族 K={len(group_names)}: {group_names}")

    regime = build_regime_matrix(DATABASE_PATH)
    print(f"  市场状态矩阵: {regime.shape[0]} 日 × {regime.shape[1]} 维")

    folds = []
    for chunk in args.folds.split(','):
        a, b = chunk.split(':')
        folds.append((float(a), float(b)))

    # T110 效率：多种子在同一进程内复用同一份 fold。全量池数据集构建要 11.5 分钟，
    # 而每个判决都是 n=3/4 —— 分进程跑等于把它重复 3~4 次。
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else [args.seed]
    multi_seed = len(seeds) > 1

    def _seed_path(p, sd):
        """多种子时给输出路径插入 `_s{seed}`；也支持显式 `{seed}` 占位符。"""
        if p is None:
            return None
        if '{seed}' in p:
            return p.format(seed=sd)
        if not multi_seed:
            return p
        root, ext = os.path.splitext(p)   # 目录名 ext='' → 直接追加后缀
        return f'{root}_s{sd}{ext}'

    orig_save, orig_split = TrainingConfig.SAVE_DIR, TrainingConfig.TRAIN_TEST_SPLIT
    orig_params = ModelConfig.XGBOOST_PARAMS.copy()
    cache_dir = os.path.join('models', 'diagnostics', 'nam_gate_cache')
    os.makedirs(cache_dir, exist_ok=True)

    results_by_seed = {sd: {} for sd in seeds}
    last_model_by_seed = {}
    try:
        TrainingConfig.SAVE_DIR = cache_dir
        ModelConfig.XGBOOST_PARAMS['n_estimators'] = args.estimators

        for tf, ve in folds:
            tag = f"{int(tf*100)}-{int(ve*100)}%"
            print(f"\n{'='*68}\n=== 折 {tag} ===\n{'='*68}")

            base_row = None
            if not args.skip_baseline:
                print('\n-- 基线 XGBoost LambdaRank --')
                b = _train_and_predict(trainer, dataset, all_features, tf, ve,
                                      target=args.target)
                base_row = _daily_metrics(b['predictions'], b['returns'], b['dates'])
                base_row['best_iteration'] = b['best_iteration']
                base_row['head'] = _head_percentile(b['predictions'], b['returns'], b['dates'])
                base_row['features_used'] = len(b['trained_feature_names'])
                print(f"   {base_row}")

            print('\n-- 候选 NAMGateModel --')
            fold = _prepare_fold(trainer, dataset, nam_features, tf, ve, regime,
                                 target=args.target,
                                 label_residualize=args.label_residualize,
                                 multi_horizon=_horizons, fwd_returns=_fwd,
                                 label_transform=args.label_transform,
                                 label_clip=args.label_clip)
            # 最后一折的 fold 备好之后，原始 dataset 就再无用处（fold 里已是切好、
            # 归一化好的副本）。单折运行（生产配方全是 --folds 0.8:1.0）能就此砍掉
            # 约 9 GB：dataset[0] 是 9.5M × 231 float32。多折时保留给后续折用。
            if (tf, ve) == folds[-1]:
                dataset = None
                gc.collect()
                print('  已释放原始 dataset（最后一折，fold 副本已就绪）')
            for sd in seeds:
                if multi_seed:
                    print(f"\n--- 种子 {sd}（折 {tag}，数据集复用）---")
                model, pred, gates, history, best_ic, best_epoch_idx = train_nam_gate(
                    fold, nam_features, group_names, group_ids, list(regime.columns),
                    epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                    lambda_lb=args.lambda_lb, lambda_ent=args.lambda_ent,
                    lambda_div=args.lambda_div,
                    ema_momentum=args.ema_momentum, expert_hidden=args.expert_hidden,
                    gate_mode=args.gate_mode, gate_scalar_col=args.gate_scalar_col,
                    accum_days=args.accum_days, device=args.device,
                    disable_gate=args.disable_gate, y_scale=args.y_scale,
                    min_epochs=args.min_epochs, seed=sd,
                    select_metric=args.select_metric, select_topk=args.select_topk,
                    time_decay_years=args.time_decay_years,
                    select_holdout=args.select_holdout,
                    chunk_days=args.chunk_days, store_dtype=args.store_dtype,
                    store_device=args.store_device, prefetch=args.prefetch,
                    group_norm=args.group_norm,
                )
                nam_row = _daily_metrics(pred, fold['ret_val'], np.asarray(fold['d_val']))
                if args.select_holdout > 0:
                    # 报告段的无偏读数：这一段对早停完全不可见。跨配置横比只该看它。
                    _vd = np.asarray(fold['d_val'])
                    _days = _day_slices(_vd)
                    _n_sel = max(1, int(round(len(_days) * (1.0 - args.select_holdout))))
                    _rows = np.concatenate([np.arange(s, e) for s, e in _days[_n_sel:]]) \
                        if len(_days) > _n_sel else np.zeros(0, dtype=np.int64)
                    if len(_rows):
                        _h = _daily_metrics(pred[_rows], fold['ret_val'][_rows], _vd[_rows])
                        nam_row['rank_ic_holdout'] = _h['rank_ic']
                        nam_row['holdout_days'] = int(len(_days) - _n_sel)
                nam_row['head'] = _head_percentile(pred, fold['ret_val'],
                                                   np.asarray(fold['d_val']))
                # T082/E12：行情分层 —— 合并 IC 会把「上涨日赢、下跌日输」平均成净正
                nam_row['regime'] = _regime_stratified_metrics(
                    pred, fold['ret_val'], np.asarray(fold['d_val']), k=args.regime_topk)
                nam_row['features_used'] = len(nam_features)
                nam_row['groups'] = len(group_names)
                nam_row['best_val_select_score'] = float(best_ic)
                nam_row['select_metric'] = args.select_metric
                if args.select_metric == 'rank_ic':
                    nam_row['best_val_rank_ic_on_label'] = float(best_ic)
                # 用「最佳检查点」的熵，而不是最后一轮的（早停后两者不同）
                best_rec = history[best_epoch_idx] if history else {}
                nam_row['gate_entropy'] = best_rec.get('gate_entropy')
                nam_row['gate_max_share'] = best_rec.get('gate_max_share')
                nam_row['best_epoch_idx'] = best_epoch_idx
                nam_row['epochs_run'] = len(history)
                # 调试：同一份预测分别对"训练标签"和"真实收益"算 IC，定位两者背离
                _lab = _daily_metrics(pred, fold['y_val'], np.asarray(fold['d_val']))
                nam_row['rank_ic_vs_label'] = _lab['rank_ic']
                _lab2 = _daily_metrics(fold['y_val'], fold['ret_val'],
                                       np.asarray(fold['d_val']))
                nam_row['label_vs_return_ic'] = _lab2['rank_ic']
                print(f"   {  {k: v for k, v in nam_row.items() if k != 'head'} }")
                print(f"   head: {nam_row['head']}")

                entry = {'nam_gate': nam_row, 'split_date': fold['split_date'],
                         'validation_end_date_exclusive': fold['end_date'],
                         'history': history}
                if base_row is not None:
                    deltas, passed = _compare(nam_row, base_row)
                    entry['baseline_xgboost'] = base_row
                    entry['deltas'] = deltas
                    entry['passes_four_metric_gate'] = bool(passed)
                    print(f"   四指标门: {'通过' if passed else '未通过'}  Δ={deltas}")
                results_by_seed[sd][tag] = entry

                model.attach_regime(regime)
                last_model_by_seed[sd] = (model, gates, fold['d_val'], history, tag, fold)

        # 产出模型与可解释性图（用最后一折，即最终确认折），每个种子各一份
        for sd in seeds:
            results = results_by_seed[sd]
            last_model = last_model_by_seed.get(sd)
            if last_model is not None:
                model, gates, dval, history, tag, fold = last_model
                save_dir = _seed_path(args.save_model_dir, sd) or \
                    os.path.join('models', 'nam_gate')
                os.makedirs(save_dir, exist_ok=True)
                mp = os.path.join(save_dir, 'nam_gate_factor_model.pkl')
                model.save_model(mp)
                save_sidecar_metadata(model, save_dir)
                if args.cache_dir:
                    from core.factors.cache_manifest import bind_model_to_cache
                    bind_model_to_cache(save_dir, trainer.factors_cache_dir)
                    print(f"模型已绑定因子缓存: {trainer.factors_cache_dir}")

                # 关键：必须与模型同目录落盘 norm_stats.pkl。
                # 训练端对 skip-rank 连续列做了 robust-sigmoid 归一化，回测端若找不到该文件
                # 会跳过这一步、以原始量纲喂入模型，造成训练/推理特征尺度错配
                # （表现为打分被市值类原始大数主导、横截面排名近乎静态、不同模型回测结果雷同）。
                import pickle as _pickle
                _norm_stats = {
                    'skip_col_stats': fold.get('skip_stats'),
                    'factor_names': list(model.feature_names),
                }
                _np_path = os.path.join(save_dir, 'norm_stats.pkl')
                with open(_np_path, 'wb') as _nf:
                    _pickle.dump(_norm_stats, _nf)
                print(f"归一化统计量已保存: {_np_path}")
                print(f"\n模型已保存: {mp}")

                plots = export_interpretability(model, gates, dval, history,
                                                _seed_path(args.plot_dir, sd))
                print(f"可解释性产出 ({len(plots)}): " + ', '.join(plots))
                results['artifacts'] = {'model': mp, 'plots': plots, 'fold': tag}
                # 强制每轮训练后输出因子诊断（训练/验证 IC、加权贡献、分布漂移、季度稳定性）
                diag = export_post_train_analysis(model, fold, _seed_path(args.plot_dir, sd),
                                                  train_contrib_days=args.contrib_train_days)
                print(f"训练后因子诊断 ({len(diag)}): "
                      + ', '.join(os.path.basename(v) for v in diag.values()))
                results['artifacts']['diagnostics'] = diag

            payload = {
                'metadata': {
                    'experiment': 'T039_nam_gate',
                    'stocks': args.stocks, 'years': args.years, 'end': end,
                    'future_days': 7,
                    'nam_features': len(nam_features), 'dropped_manual_interaction': dropped,
                    'dropped_features': sorted(_dropf) if _dropf else [],
                    'group_map_file': args.group_map_file,
                    'groups': group_names, 'regime_dims': int(regime.shape[1]),
                    'gate_mode': args.gate_mode, 'gate_scalar_col': args.gate_scalar_col,
                    'disable_gate': args.disable_gate,
                    'group_norm': args.group_norm,
                    'target': args.target, 'label_residualize': args.label_residualize,
                    'label_transform': args.label_transform, 'label_clip': args.label_clip,
                    'lambda_lb': args.lambda_lb,
                    'store_dtype': args.store_dtype,
                    'store_device': args.store_device, 'prefetch': args.prefetch,
                    'select_metric': args.select_metric, 'select_topk': args.select_topk,
                    'seed': sd, 'seeds_in_process': seeds,
                    'y_scale': args.y_scale,
                    'lambda_ent': args.lambda_ent, 'lambda_div': args.lambda_div,
                    'expert_hidden': args.expert_hidden,
                    'epochs': args.epochs, 'lr': args.lr,
                    'device': str(torch.cuda.get_device_name(0))
                    if torch.cuda.is_available() else 'cpu',
                    'elapsed_sec': round(time.time() - started, 1),
                },
                'folds': results,
            }
            out_path = _seed_path(args.output, sd)
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"\n已保存结果: {out_path}  (累计耗时 {payload['metadata']['elapsed_sec']}s)")
    finally:
        TrainingConfig.SAVE_DIR = orig_save
        TrainingConfig.TRAIN_TEST_SPLIT = orig_split
        ModelConfig.XGBOOST_PARAMS.clear()
        ModelConfig.XGBOOST_PARAMS.update(orig_params)


# ---------------------------------------------------------------------------
# 训练后因子诊断（强制产出，用于发现解释与问题）
# ---------------------------------------------------------------------------

def _rank_ic_columns(values: np.ndarray, target: np.ndarray):
    """每日截面 Spearman 相关，逐列返回。values: [B, N], target: [B]。"""
    if len(values) < 5 or np.ptp(target) <= 1e-12:
        return np.full(values.shape[1], np.nan, dtype=np.float64)
    xr = rankdata(values, axis=0).astype(np.float64)
    yr = rankdata(target).astype(np.float64)
    xr -= xr.mean(axis=0, keepdims=True)
    yr -= yr.mean()
    denom = np.sqrt((xr * xr).sum(axis=0) * float(np.dot(yr, yr)))
    out = (xr * yr[:, None]).sum(axis=0) / np.maximum(denom, 1e-12)
    out[np.asarray(denom <= 1e-12)] = np.nan
    return out


def _ic_summary(ic_by_day: np.ndarray, prefix: str):
    valid = np.isfinite(ic_by_day)
    n = valid.sum(axis=0).astype(np.float64)
    safe = np.where(valid, ic_by_day, 0.0)
    mean = safe.sum(axis=0) / np.maximum(n, 1)
    centered = np.where(valid, ic_by_day - mean, 0.0)
    std = np.sqrt((centered * centered).sum(axis=0) / np.maximum(n, 1))
    return {
        f'{prefix}_ic': mean,
        f'{prefix}_ic_std': std,
        f'{prefix}_ic_ir': mean / np.maximum(std, 1e-12),
        f'{prefix}_positive_ratio': (np.where(valid, ic_by_day > 0, False).sum(axis=0)
                                     / np.maximum(n, 1)),
        f'{prefix}_days': n,
    }


def _evaluate_contributions(model, X, returns, dates, M, max_days=0):
    """按日测量原始因子 IC 与 NAM 加权贡献 IC / 占比。

    区分三件事：因子「形状函数自身有效性」(raw IC)、被门控加权后「实际驱动得分的
    贡献有效性」(effective IC)、以及「贡献幅度占比」(mean_abs/rms)。后者高但 IC 低
    的因子，是形状函数噪声被门控放大的典型病态，必须能被诊断出来。

    ``max_days``>0 时按**等间隔**抽样到最多这么多个交易日（0=全量）。
    逐日 `_rank_ic_columns` 在 float64 上算 3 组 219 列，是全量池收尾耗时的大头
    （2727 训练日约 9 分钟）。这里的产物**只进诊断表**，不参与任何晋级判定，
    所以抽样是安全的：跨日均值的标准误只放大 √(2727/max_days)，
    而等间隔抽样保留了全时段覆盖且确定性可复现。
    """
    net = model.net
    dev = torch.device(model.device)
    mean = model.input_mean
    std = model.input_std
    # 归一化按日切片做，不在整表上做 —— 全量池 X_train 是 (7.3M, 219)，
    # 整表 astype+归一化要再吃 2×5.9GiB，2026-08-16 s23 实测在这里 OOM
    #（训练本体早已成功，只是诊断导出把 JSON 一起带崩了）。
    Xn = np.asarray(X, dtype=np.float32)
    _mean = np.asarray(mean, dtype=np.float32) if mean is not None else None
    _std = np.asarray(std, dtype=np.float32) if std is not None else None
    slices = _day_slices(dates)
    if max_days and len(slices) > max_days:
        keep = np.linspace(0, len(slices) - 1, max_days).round().astype(int)
        slices = [slices[i] for i in np.unique(keep)]
    n_features = X.shape[1]
    n_groups = len(model.group_names)
    raw_ics = np.full((len(slices), n_features), np.nan, dtype=np.float32)
    eff_ics = np.full_like(raw_ics, np.nan)
    group_ics = np.full((len(slices), n_groups), np.nan, dtype=np.float32)
    abs_sum = np.zeros(n_features, dtype=np.float64)
    sq_sum = np.zeros(n_features, dtype=np.float64)
    group_abs_sum = np.zeros(n_groups, dtype=np.float64)
    gate_rows = np.zeros((len(slices), n_groups), dtype=np.float32)
    rows = 0

    net.eval()
    with torch.no_grad():
        for di, (s, e) in enumerate(slices):
            if e - s < 5:
                continue
            xt = torch.as_tensor(Xn[s:e], dtype=torch.float32, device=dev)
            if _mean is not None and _std is not None:
                xt = (xt - torch.as_tensor(_mean, device=dev)) / \
                     torch.as_tensor(_std, device=dev)
            mt = torch.as_tensor(np.asarray(M[s], dtype=np.float32),
                                 dtype=torch.float32, device=dev)
            _, w, group_sums = net.forward_day(xt, mt)
            contrib = net.experts(xt).cpu().numpy().astype(np.float64)
            w_np = w.cpu().numpy().astype(np.float64)
            group_ids = np.asarray(model.group_ids)
            gs_np = group_sums.cpu().numpy().astype(np.float64)
            col_scale = 1.0
            if getattr(net, 'group_norm', False) and not net.disable_gate:
                # forward_day 返回的是**归一化前**的 group_sums，而打分实际用的是
                # _std_groups(group_sums)（逐日截面标准化）。这里必须补上同一口径，
                # 否则 --group-norm 跑出来的 group_effectiveness.csv 量的是归一化前的
                # 专家幅度 —— 与真实打分无关。2026-08-19 实测：T119 未修时门前量级
                # 极差 243x（与不带归一化的 T116 的 208x 几乎一样），一眼看去像
                # 「归一化没生效」，实则是测量口径错。
                sd = gs_np.std(axis=0, ddof=0, keepdims=True) + 1e-6
                gs_np = (gs_np - gs_np.mean(axis=0, keepdims=True)) / sd
                col_scale = 1.0 / sd[0][group_ids]      # 同族各列共用该缩放
            effective = contrib * w_np[group_ids] * col_scale   # [B, N]
            group_effective = gs_np * w_np                      # [B, K]
            y = np.asarray(returns[s:e], dtype=np.float64)
            raw_ics[di] = _rank_ic_columns(np.asarray(X[s:e], dtype=np.float64), y)
            eff_ics[di] = _rank_ic_columns(effective, y)
            group_ics[di] = _rank_ic_columns(group_effective, y)
            # 幅度统计必须先做当日截面去均值：ListNet 平移不变，专家输出可以带
            # 任意大的常数偏置（T098 实测中位 ~10.6、atr_14 到 16），不去均值时
            # mean_abs/rms 是偏置排行榜（cross_fund 的"25.1% 贡献"恰好=列数占比
            # 55/219），high_contribution_weak_ic 红旗随之失真。IC 列不受影响
            #（Spearman 平移不变），无需回算历史 CSV，但横比只能用去均值后的口径。
            eff_c = effective - effective.mean(axis=0, keepdims=True)
            grp_c = group_effective - group_effective.mean(axis=0, keepdims=True)
            abs_sum += np.abs(eff_c).sum(axis=0)
            sq_sum += (eff_c * eff_c).sum(axis=0)
            group_abs_sum += np.abs(grp_c).sum(axis=0)
            gate_rows[di] = w_np
            rows += e - s

    factor_stats = {}
    factor_stats.update(_ic_summary(raw_ics, 'raw'))
    factor_stats.update(_ic_summary(eff_ics, 'effective'))
    factor_stats['effective_mean_abs'] = abs_sum / max(rows, 1)
    factor_stats['effective_rms'] = np.sqrt(sq_sum / max(rows, 1))
    group_stats = _ic_summary(group_ics, 'effective')
    group_stats['effective_mean_abs'] = group_abs_sum / max(rows, 1)
    return factor_stats, group_stats, gate_rows


def _regime_ood_table(M_train: np.ndarray, M_val: np.ndarray, columns):
    """逐 regime 特征检查验证期分布是否落在训练期支撑集之外。"""
    rows = []
    for i, name in enumerate(columns):
        tr = np.asarray(M_train[:, i], dtype=np.float64)
        va = np.asarray(M_val[:, i], dtype=np.float64)
        lo, hi = np.nanmin(tr), np.nanmax(tr)
        rows.append({
            'regime_feature': name,
            'train_min': float(lo), 'train_max': float(hi),
            'val_min': float(np.nanmin(va)), 'val_max': float(np.nanmax(va)),
            'val_outside_train_ratio': float(np.mean((va < lo) | (va > hi))),
            'train_mean': float(np.nanmean(tr)), 'val_mean': float(np.nanmean(va)),
            'mean_shift': float(np.nanmean(va) - np.nanmean(tr)),
        })
    return pd.DataFrame(rows)


def export_post_train_analysis(model, fold, out_dir, train_contrib_days=400):
    """训练后强制诊断：因子有效性 / 加权贡献 / regime OOD / 季度稳定性。

    返回各产物路径；同时落盘 CSV、JSON、仪表盘 PNG，供下一轮迭代直接读问题。

    ``train_contrib_days``：训练侧贡献诊断抽样到多少个交易日（0=全量）。
    全量池训练侧有 2727 日，逐日算三组 219 列 IC 约 9 分钟，占单次运行收尾的大头；
    这些数只进诊断表，不参与晋级判定。**验证侧一律精确不抽样** ——
    `gates_val` 要与 `d_val` 逐日对齐供可解释性图使用，且验证 IC 是要看的量。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    f_train, _, _ = _evaluate_contributions(
        model, fold['X_train'], fold['ret_train'], fold['d_train'], fold['M_train'],
        max_days=train_contrib_days)
    f_val, g_val, gates_val = _evaluate_contributions(
        model, fold['X_val'], fold['ret_val'], fold['d_val'], fold['M_val'])

    factor_rows = []
    for i, name in enumerate(model.feature_names):
        row = {'feature': name, 'group': model.group_names[int(model.group_ids[i])]}
        row.update({f'train_{k}': float(v[i]) for k, v in f_train.items()})
        row.update({f'val_{k}': float(v[i]) for k, v in f_val.items()})
        row['effective_ic_decay'] = row['val_effective_ic'] - row['train_effective_ic']
        row['importance_ic_alignment'] = abs(row['val_effective_ic']) * row['val_effective_mean_abs']
        factor_rows.append(row)
    factors = pd.DataFrame(factor_rows).sort_values('importance_ic_alignment', ascending=False)
    factors.to_csv(os.path.join(out_dir, 'factor_effectiveness.csv'), index=False,
                   encoding='utf-8-sig')

    groups = pd.DataFrame({
        'group': model.group_names,
        **{k: np.asarray(v) for k, v in g_val.items()},
        'gate_mean': np.nanmean(gates_val, axis=0),
        'gate_std': np.nanstd(gates_val, axis=0),
    }).sort_values('effective_ic', ascending=False)
    groups.to_csv(os.path.join(out_dir, 'group_effectiveness.csv'), index=False,
                  encoding='utf-8-sig')

    ood = _regime_ood_table(fold['M_train'], fold['M_val'], model.regime_cols)
    ood.to_csv(os.path.join(out_dir, 'regime_ood.csv'), index=False, encoding='utf-8-sig')

    # 季度模型稳定性（基于验证期预测）
    dts = pd.to_datetime(fold['d_val'])
    pred = np.empty(len(fold['X_val']), dtype=np.float32)
    mean = model.input_mean
    std = model.input_std
    Xn = fold['X_val'].astype(np.float32)
    if mean is not None and std is not None:
        Xn = ((Xn - np.asarray(mean, dtype=np.float32))
              / np.asarray(std, dtype=np.float32)).astype(np.float32)
    model.net.eval()
    with torch.no_grad():
        for s, e in _day_slices(fold['d_val']):
            xt = torch.as_tensor(Xn[s:e], dtype=torch.float32, device=model.device)
            mt = torch.as_tensor(np.asarray(fold['M_val'][s], dtype=np.float32),
                                 dtype=torch.float32, device=model.device)
            sc, _, _ = model.net.forward_day(xt, mt)
            pred[s:e] = sc.cpu().numpy()
    quarterly = []
    for period in pd.PeriodIndex(dts, freq='Q').unique().sort_values():
        mask = np.asarray(pd.PeriodIndex(dts, freq='Q') == period)
        metrics = _daily_metrics(pred[mask], fold['ret_val'][mask], np.asarray(fold['d_val'])[mask])
        metrics['period'] = str(period)
        quarterly.append(metrics)
    quarters = pd.DataFrame(quarterly)
    quarters.to_csv(os.path.join(out_dir, 'quarterly_model_metrics.csv'), index=False,
                    encoding='utf-8-sig')

    high_contrib_weak = factors[
        (factors['val_effective_mean_abs'] >= factors['val_effective_mean_abs'].quantile(0.8))
        & (factors['val_effective_ic'].abs() < 0.01)]
    sign_flip = factors[
        (np.sign(factors['train_effective_ic']) != np.sign(factors['val_effective_ic']))
        & (factors['train_effective_ic'].abs() >= 0.02)]
    issues = {
        'high_contribution_weak_ic': high_contrib_weak['feature'].tolist(),
        'train_to_val_ic_sign_flip': sign_flip['feature'].tolist(),
        'regime_features_with_ood': ood.loc[
            ood['val_outside_train_ratio'] > 0.01, 'regime_feature'].tolist(),
        'negative_quarters': quarters.loc[quarters['rank_ic'] < 0, 'period'].tolist(),
    }
    with open(os.path.join(out_dir, 'diagnostic_flags.json'), 'w', encoding='utf-8') as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)

    top = factors.head(15).sort_values('val_effective_ic')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].barh(top['feature'], top['val_effective_ic'], color='#2f6f9f')
    axes[0, 0].axvline(0, color='#888', lw=0.7)
    axes[0, 0].set_title('Top factors: realized contribution Rank IC')
    axes[0, 1].scatter(factors['train_effective_ic'], factors['val_effective_ic'], s=12, alpha=0.6)
    lim = max(0.05, float(np.nanmax(np.abs(factors[['train_effective_ic', 'val_effective_ic']]))))
    axes[0, 1].plot([-lim, lim], [-lim, lim], '--', color='#888', lw=0.8)
    axes[0, 1].set(xlabel='Train IC', ylabel='Validation IC', title='Factor IC stability')
    axes[1, 0].bar(groups['group'], groups['effective_ic'], color='#8b5a2b')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].axhline(0, color='#888', lw=0.7)
    axes[1, 0].set_title('Factor-group realized contribution IC')
    axes[1, 1].plot(quarters['period'], quarters['rank_ic'], marker='o', color='#8f3b3b')
    axes[1, 1].axhline(0, color='#888', lw=0.7)
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].set_title('Quarterly model Rank IC')
    fig.tight_layout()
    dashboard = os.path.join(out_dir, 'post_train_factor_dashboard.png')
    fig.savefig(dashboard, dpi=140)
    plt.close(fig)
    return {
        'factor_table': os.path.join(out_dir, 'factor_effectiveness.csv'),
        'group_table': os.path.join(out_dir, 'group_effectiveness.csv'),
        'regime_ood': os.path.join(out_dir, 'regime_ood.csv'),
        'quarterly_metrics': os.path.join(out_dir, 'quarterly_model_metrics.csv'),
        'flags': os.path.join(out_dir, 'diagnostic_flags.json'),
        'dashboard': dashboard,
    }


if __name__ == '__main__':
    main()
