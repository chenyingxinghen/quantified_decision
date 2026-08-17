"""轻量冒烟测试：用合成模型验证训练后因子诊断管线端到端可用。"""
import os, sys, tempfile
import numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from core.factors.nam_gate_model import NAMGateModel
import scripts.exp.exp_nam_gate as E

rng = np.random.default_rng(0)
n_feat, n_groups, d_reg = 12, 3, 4
group_ids = np.array([0] * 4 + [1] * 4 + [2] * 4)
group_names = ['mom', 'value', 'vol']
regime_cols = [f'rm{i}' for i in range(d_reg)]


def make_fold(n_days, per_day, seed):
    rng = np.random.default_rng(seed)
    dates = np.repeat(np.arange(n_days), per_day)
    X = rng.standard_normal((n_days * per_day, n_feat)).astype(np.float32)
    M = rng.standard_normal((n_days * per_day, d_reg)).astype(np.float32)
    # 让因子 0 与收益正相关（构造一个弱信号），其余噪声
    ret = (0.6 * X[:, 0] + rng.standard_normal(n_days * per_day) * 0.8).astype(np.float64)
    return {'X_train': X, 'X_val': X, 'ret_train': ret, 'ret_val': ret,
            'd_train': dates, 'd_val': dates, 'M_train': M, 'M_val': M}


model = NAMGateModel(feature_names=[f'f{i}' for i in range(n_feat)],
                     group_names=group_names, group_ids=group_ids,
                     regime_cols=regime_cols, expert_hidden=8, device='cpu')
model.input_mean = np.zeros(n_feat, dtype=np.float32)
model.input_std = np.ones(n_feat, dtype=np.float32)
model.build(d_regime=d_reg)
model.is_trained = True

fold = make_fold(40, 30, 1)
out = tempfile.mkdtemp()
diag = E.export_post_train_analysis(model, fold, out)
print('产物:', list(diag.keys()))
for k, v in diag.items():
    print(' ', k, os.path.basename(v), os.path.exists(v))

import pandas as pd
fac = pd.read_csv(diag['factor_table'])
print('因子表行数:', len(fac), '列:', [c for c in fac.columns][:8])
grp = pd.read_csv(diag['group_table'])
print('分组表 effective_ic:', grp['effective_ic'].round(3).tolist())
q = pd.read_csv(diag['quarterly_metrics'])
print('季度指标 periods:', q['period'].tolist()[:3], 'rank_ic sample:', q['rank_ic'].round(3).head(2).tolist())
print('SMOKE_OK')
