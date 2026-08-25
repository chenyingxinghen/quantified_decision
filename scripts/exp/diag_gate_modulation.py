"""诊断 scheme B 的 gate 是否真正调制了族权重（而非近均匀退化）。"""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, '.')
from core.factors.nam_gate_model import NAMGateModel
from config import DATABASE_PATH
from core.factors.regime_features import build_regime_matrix

MP = 'models/nam_gate/schemeb_s42/nam_gate_factor_model.pkl'
m = NAMGateModel(); m.load_model(MP)

# gate 参数幅度
net = m.net
a = net.gate.a.detach().cpu().numpy()
b = net.gate.b.detach().cpu().numpy() if net.gate.b is not None else None
print(f"gate_mode={m.gate_mode} | n_groups={net.gate.n_groups}")
print(f"a 向量 (per-family 系数 on s1): norm={np.linalg.norm(a):.4f}  range=[{a.min():.4f},{a.max():.4f}]")
if b is not None:
    print(f"b 向量 (per-family 系数 on s2): norm={np.linalg.norm(b):.4f}  range=[{b.min():.4f},{b.max():.4f}]")
print(f"a/b 量级 ≈ 0 ?  {'YES(退化)' if (np.abs(a).max()<0.05 and (b is None or np.abs(b).max()<0.05)) else 'NO(有调制)'}")

# 全样本 gate 权重
W = m.gate_weights_over_time()   # index=date, cols=groups
print(f"\n全样本 gate 权重矩阵: {W.shape[0]} 日 × {W.shape[1]} 族")
print(f"理论均匀权重 = 1.0 (K={net.gate.n_groups})")

# 每个族的跨日统计
per = pd.DataFrame({
    'mean': W.mean(),
    'std': W.std(),
    'min': W.min(),
    'max': W.max(),
})
per['spread(max-min)'] = per['max'] - per['min']
print("\n每族权重跨日统计 (mean/std/min/max/spread):")
print(per.round(4).to_string())

# 跨日整体调制强度: 所有 (日,族) 权重偏离 1.0 的 RMS
dev = np.sqrt(((W.to_numpy() - 1.0)**2).mean())
print(f"\n全样本权重偏离均匀(1.0)的 RMS = {dev:.4f}  (越小=越退化)")

# 熊市 vs 牛市
bear = W.loc['2022-09-05':'2024-08-05']
bull = W.loc['2024-08-05':'2026-08-01']
mb = bear.mean(); mu = bull.mean()
print(f"\n熊市窗 {bear.shape[0]} 日 族权重均值:")
print(mb.round(3).to_string())
print(f"\n牛市窗 {bull.shape[0]} 日 族权重均值:")
print(mu.round(3).to_string())
diff = (mb - mu).sort_values()
print("\n熊市-牛市 族权重差 (最大的 3 个族 vs 最小的 3 个族):")
print("  熊市更重:", list(diff.tail(3).index), "差值", diff.tail(3).round(3).to_dict())
print("  牛市更重:", list(diff.head(3).index), "差值", diff.head(3).round(3).to_dict())
print(f"\n熊市/牛市 族权重差异的 RMS = {np.sqrt(((mb-mu)**2).mean()):.4f}")

# 每日熵
ent = -(W.div(W.sum(axis=1), axis=0) * np.log(W.div(W.sum(axis=1), axis=0) + 1e-8)).sum(axis=1)
print(f"\n每日 gate 熵: mean={ent.mean():.4f} / 均匀上限 ln{net.gate.n_groups}={np.log(net.gate.n_groups):.4f}"
      f" (占比 {ent.mean()/np.log(net.gate.n_groups)*100:.1f}%)")
