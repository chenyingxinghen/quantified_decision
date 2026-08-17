"""诊断：三个 NAM 存档在同一真实特征矩阵上的打分是否真的不同。"""
import os, sys, pickle, hashlib
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from core.factors.nam_gate_model import NAMGateModel

ROOT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')), 'models', 'nam_gate')
PATHS = {
    'root':       os.path.join(ROOT, 'nam_gate_factor_model.pkl'),
    'T042d':      os.path.join(ROOT, 'T042d', 'nam_gate_factor_model.pkl'),
    'T042e':      os.path.join(ROOT, 'T042e', 'nam_gate_factor_model.pkl'),
    'T042e_fixed':os.path.join(ROOT, 'T042e_fixed', 'nam_gate_factor_model.pkl'),
}

for k, p in PATHS.items():
    h = hashlib.md5(open(p, 'rb').read()).hexdigest()
    print(f'{k:12s} md5={h} size={os.path.getsize(p)}')

models = {}
for k, p in PATHS.items():
    m = NAMGateModel()
    m.load_model(p)
    models[k] = m
    print(f'{k:12s} feat={len(m.feature_names)} groups={len(m.group_names)} gate_mode={m.gate_mode} '
          f'disable_gate={m.disable_gate} trained={m.is_trained}')

# state_dict 指纹
print('\n--- state_dict fingerprint ---')
for k, m in models.items():
    sd = m.net.state_dict()
    flat = np.concatenate([v.detach().cpu().numpy().ravel() for v in sd.values()])
    print(f'{k:12s} n_param={flat.size} sum={flat.sum():.6f} std={flat.std():.6f} '
          f'sha={hashlib.md5(flat.tobytes()).hexdigest()[:12]}')

# 同一随机输入下的打分排序对比
rng = np.random.default_rng(0)
n, d = 3000, len(models['T042d'].feature_names)
X = rng.random((n, d)).astype(np.float32)
import pandas as pd
Xdf = pd.DataFrame(X, columns=models['T042d'].feature_names)

scores = {}
for k, m in models.items():
    if m.regime_matrix is not None:
        try:
            m.set_context_date(str(m.regime_matrix.index[-1])[:10])
        except Exception as e:
            print('set_context_date fail', k, e)
    scores[k] = np.asarray(m.predict(Xdf)).ravel()

print('\n--- score stats ---')
for k, s in scores.items():
    print(f'{k:12s} mean={s.mean():.4f} std={s.std():.4f} top1_idx={int(np.argmax(s))}')

keys = list(scores)
print('\n--- pairwise top-20 overlap / spearman ---')
from scipy.stats import spearmanr
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = scores[keys[i]], scores[keys[j]]
        ta = set(np.argsort(-a)[:20]); tb = set(np.argsort(-b)[:20])
        rho = spearmanr(a, b).statistic
        print(f'{keys[i]:12s} vs {keys[j]:12s} top20_overlap={len(ta & tb)}/20 spearman={rho:.6f}')
