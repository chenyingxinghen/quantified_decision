"""E1a：配对 block bootstrap —— 判定两个配置的验证 IC 差异是否真实。

动机
----
标签是 7 日前向收益、逐日采样，相邻日标签重叠 6/7，自相关 ≈ 0.86。
末折验证期约 630 交易日，独立时间块只有约 90 个。以 ICIR≈0.95、
rank_ic≈0.097 计，SE(IC̄) ≈ 0.102/√90 ≈ 0.011 —— 而 T045–T067 全部实验的
验证 IC 跨度也恰好约 0.011。也就是说，整个超参搜索空间的信号差异
约等于一个标准误。

但配置之间是**同数据同折的配对比较**，模型高度相关，差分的标准误会
显著小于单配置的 SE。因此不能用单配置 SE 直接判定，必须直接对
「逐日 IC 差分序列」做 bootstrap。

方法
----
1. 用完全相同的折切分与截面归一化，重建验证集（复用 train_nam_model._prepare_fold，
   保证与训练/回测三端一致）。
2. 对每个模型逐日算 Spearman IC，得到长度 D 的序列 ic_a[d]、ic_b[d]。
3. 差分序列 diff[d] = ic_a[d] - ic_b[d]。
4. 对 diff 做 **moving-block bootstrap**（默认块长 7 = FUTURE_DAYS，
   与标签重叠长度一致），重采样 B 次，得到 mean(diff) 的分布 → 均值、CI、双侧 p。

判据
----
  CI 跨越 0  -> 该轴的 IC 差异不可测量，不足以支撑晋级/否决
  CI 不含 0  -> 差异真实（但仍需回测 + 随机零假设确认是否有经济价值）

用法
----
  python scripts/exp/diag_paired_bootstrap.py \
      --model-a models/nam_gate/T045 --model-b models/nam_gate/T058_hidden8_s42 \
      --label-a T045 --label-b T058_hidden8 \
      --out diagnose_output/paired_bootstrap_T045_vs_T058.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.baostock_config import DATABASE_PATH
from config.factor_config import TrainingConfig
from core.data.baostock_main import BaostockDataManager
from core.factors.train_ml_model import MLModelTrainer, _fast_rankdata_1d
from core.factors.nam_gate_model import NAMGateModel
from scripts.train_nam_model import _prepare_fold, build_group_index, build_regime_matrix


# ---------------------------------------------------------------------------
# 逐日 IC 序列
# ---------------------------------------------------------------------------

def daily_ic_series(pred: np.ndarray, ret: np.ndarray, dates: np.ndarray,
                    min_count: int = 10):
    """逐日截面 Spearman IC。返回 (ic[D], day_index[D])，day_index 为 unique 日序号。"""
    uniq, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    ics, days = [], []
    for i, (s, c) in enumerate(zip(starts, counts)):
        if c < min_count:
            continue
        e = s + c
        pr = _fast_rankdata_1d(pred[s:e])
        rr = _fast_rankdata_1d(ret[s:e])
        if np.ptp(pr) <= 0 or np.ptp(rr) <= 0:
            continue
        ic = np.corrcoef(pr, rr)[0, 1]
        if np.isfinite(ic):
            ics.append(float(ic))
            days.append(i)
    return np.asarray(ics, dtype=np.float64), np.asarray(days, dtype=np.int64)


# ---------------------------------------------------------------------------
# moving-block bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_mean(x: np.ndarray, block: int = 7, n_boot: int = 10000,
                         seed: int = 42):
    """对序列 x 的均值做 moving-block bootstrap。

    块长 block 应 >= 标签重叠长度（7 日标签 → 7），使块内保留自相关、
    块间近似独立。返回 (boot_means[n_boot],)。
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < block * 2:
        raise ValueError(f'序列长度 {n} 过短，无法用块长 {block} 做 bootstrap')
    n_blocks = int(np.ceil(n / block))
    max_start = n - block  # moving block: 起点可取 0..n-block
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start + 1, size=(n_boot, n_blocks))
    # 构造索引矩阵 [n_boot, n_blocks*block] 再截断到 n
    offs = np.arange(block)
    idx = (starts[:, :, None] + offs[None, None, :]).reshape(n_boot, -1)[:, :n]
    return x[idx].mean(axis=1)


def summarize(diff: np.ndarray, block: int, n_boot: int, seed: int, alpha: float = 0.05):
    boots = block_bootstrap_mean(diff, block=block, n_boot=n_boot, seed=seed)
    obs = float(diff.mean())
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # 双侧 p：以 0 为原点，看 bootstrap 分布落在对侧的比例（centered）
    centered = boots - boots.mean()
    p_two = float(np.mean(np.abs(centered) >= abs(obs)))
    return {
        'mean_diff': obs,
        'ci_low': float(lo),
        'ci_high': float(hi),
        'ci_level': 1 - alpha,
        'p_two_sided': p_two,
        'se_boot': float(boots.std(ddof=1)),
        'significant': bool(lo > 0 or hi < 0),
        'n_days': int(len(diff)),
        'n_eff_blocks': int(np.ceil(len(diff) / block)),
        'block': block,
        'n_boot': n_boot,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _load_model(path: str) -> NAMGateModel:
    pkl = path
    if os.path.isdir(path):
        pkl = os.path.join(path, 'nam_gate_factor_model.pkl')
    return NAMGateModel().load_model(pkl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-a', required=True, help='模型A目录或pkl')
    ap.add_argument('--model-b', required=True, help='模型B目录或pkl')
    ap.add_argument('--label-a', default='A')
    ap.add_argument('--label-b', default='B')
    ap.add_argument('--stocks', type=int, default=800)
    ap.add_argument('--years', type=int, default=13)
    ap.add_argument('--end', default='2022-09-05')
    ap.add_argument('--train-fraction', type=float, default=0.8,
                    help='与训练 --folds "0.8:1.0" 对齐')
    ap.add_argument('--target', default='returns', choices=['scores', 'returns'])
    ap.add_argument('--block', type=int, default=7,
                    help='bootstrap 块长，应 >= FUTURE_DAYS')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--out', default='diagnose_output/paired_bootstrap.json')
    args = ap.parse_args()

    end_dt = datetime.strptime(args.end, '%Y-%m-%d')
    end = end_dt.strftime('%Y-%m-%d')
    start = (end_dt - timedelta(days=365 * args.years)).strftime('%Y-%m-%d')

    TrainingConfig.FUTURE_DAYS = 7

    print(f'加载模型 A: {args.model_a}')
    ma = _load_model(args.model_a)
    print(f'加载模型 B: {args.model_b}')
    mb = _load_model(args.model_b)
    if list(ma.feature_names) != list(mb.feature_names):
        raise ValueError('两模型特征列表不一致，无法配对比较')
    feature_names = list(ma.feature_names)

    print(f'重建数据集: {args.stocks} 股 / {start} ~ {end}')
    trainer = MLModelTrainer(db_path=DATABASE_PATH, cache_dir=args.cache_dir)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()['code'].tolist()[:args.stocks]
    manager.close()
    stocks_data = trainer.load_label_data(codes, start, end)
    dataset = trainer.prepare_dataset(
        stocks_data, train_start_date=start, train_end_date=end,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
        n_jobs=args.workers, target_features=None, use_factor_cache_only=True,
        return_sample_metadata=False,
    )
    regime = build_regime_matrix(DATABASE_PATH)

    fold = _prepare_fold(trainer, dataset, feature_names, args.train_fraction,
                         1.0, regime, target=args.target)
    Xv, retv, dv = fold['X_val'], fold['ret_val'], np.asarray(fold['d_val'])
    print(f'验证集: {len(Xv)} 样本 / {len(np.unique(dv))} 交易日')

    # 预测（模型内部会用自身 input_mean/std 做全局标准化）
    ma.attach_regime(regime)
    mb.attach_regime(regime)
    pa = ma.predict(Xv)
    pb = mb.predict(Xv)

    ic_a, days_a = daily_ic_series(pa, retv, dv)
    ic_b, days_b = daily_ic_series(pb, retv, dv)
    if not np.array_equal(days_a, days_b):
        common = np.intersect1d(days_a, days_b)
        ic_a = ic_a[np.isin(days_a, common)]
        ic_b = ic_b[np.isin(days_b, common)]
    diff = ic_a - ic_b

    res_a = summarize(ic_a, args.block, args.n_boot, args.seed)
    res_b = summarize(ic_b, args.block, args.n_boot, args.seed)
    res_d = summarize(diff, args.block, args.n_boot, args.seed)

    print('\n' + '=' * 72)
    print(f'配对 block bootstrap（块长 {args.block}，{args.n_boot} 次）')
    print('=' * 72)
    print(f'  {args.label_a:<22} IC = {res_a["mean_diff"]:+.5f}  '
          f'CI[{res_a["ci_low"]:+.5f}, {res_a["ci_high"]:+.5f}]')
    print(f'  {args.label_b:<22} IC = {res_b["mean_diff"]:+.5f}  '
          f'CI[{res_b["ci_low"]:+.5f}, {res_b["ci_high"]:+.5f}]')
    print('-' * 72)
    print(f'  差分 A-B              = {res_d["mean_diff"]:+.5f}  '
          f'CI[{res_d["ci_low"]:+.5f}, {res_d["ci_high"]:+.5f}]')
    print(f'  配对 SE = {res_d["se_boot"]:.5f}   单配置 SE = {res_a["se_boot"]:.5f}'
          f'   （收窄 {res_a["se_boot"]/max(res_d["se_boot"],1e-12):.1f}×）')
    print(f'  双侧 p = {res_d["p_two_sided"]:.4f}   '
          f'有效独立块 ≈ {res_d["n_eff_blocks"]}')
    print(f'  判定: {"差异真实（CI 不含 0）" if res_d["significant"] else "★ 不可测量（CI 跨越 0）"}')
    print('=' * 72)

    payload = {
        'model_a': args.model_a, 'model_b': args.model_b,
        'label_a': args.label_a, 'label_b': args.label_b,
        'ic_a': res_a, 'ic_b': res_b, 'diff': res_d,
        'ic_series_a': ic_a.tolist(), 'ic_series_b': ic_b.tolist(),
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'已保存: {args.out}')


if __name__ == '__main__':
    main()
