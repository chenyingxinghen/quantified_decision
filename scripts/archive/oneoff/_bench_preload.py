"""R2 微基准：因子面板预加载的耗时与常驻内存（全量 vs 按窗口裁剪）。

只跑 `_preload_factor_cache`，不跑回测日循环。用于确认 R2 之后多种子并行是否安全。
用法: python scripts/archive/oneoff/_bench_preload.py
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import psutil  # noqa: E402

from core.backtest.strategies.ml_factor_strategy import MLFactorBacktestStrategy  # noqa: E402
from config.factor_config import TrainingConfig  # noqa: E402

MODEL = os.path.join(ROOT, 'models', 'nam_gate', 'T068_seed37')
WIN = ('2022-09-05', '2024-08-05')


def bench(label, preload_start, preload_end):
    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 1e9
    s = MLFactorBacktestStrategy(
        model_path=MODEL, min_confidence=0, use_cache=True,
        cache_dir=TrainingConfig.CACHE_DIR, max_positions=40,
        preload_start=preload_start, preload_end=preload_end,
        name=f'bench_{label}')
    s._get_model_feature_names = lambda: _FEATS
    t = time.time()
    s._preload_factor_cache()
    dt = time.time() - t
    rss1 = proc.memory_info().rss / 1e9
    rows = sum(len(v) for v in s._factor_dates_cache.values())
    print(f'[{label}] preload {dt:6.1f}s  RSS +{rss1 - rss0:5.2f} GB  '
          f'stocks={len(s._factor_matrix_cache)} rows={rows:,}')
    del s
    return dt, rss1 - rss0


if __name__ == '__main__':
    # 用真实模型的特征名，保证列数与生产一致
    import pickle
    pkls = [f for f in os.listdir(MODEL) if f.endswith('_factor_model.pkl')]
    with open(os.path.join(MODEL, sorted(pkls)[-1]), 'rb') as f:
        m = pickle.load(f)
    _FEATS = list(getattr(m, 'feature_names', None) or m['feature_names'])
    print(f'特征列数: {len(_FEATS)}')
    del m
    bench('window', *WIN)
    bench('full', None, None)
