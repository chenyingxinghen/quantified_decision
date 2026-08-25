"""把 scheme B 的门控权重按时间区域拆开，判定过拟合。

gate 输出 = softmax(logits) * 13  → 均匀态每族=1.0、总和=13。
故：概率权重 w_prob = raw/13；离散度用 raw 偏离 1.0 度量；熵用 w_prob。
"""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, '.')
from core.factors.nam_gate_model import NAMGateModel

mp = 'models/nam_gate/schemeb_s42/nam_gate_factor_model.pkl'
m = NAMGateModel(); m.load_model(mp)
Wraw = m.gate_weights_over_time()              # raw, 每族均匀=1.0
K = Wraw.shape[1]
Wp = Wraw / K                                  # 概率分布
dates = Wraw.index
TRAIN_N = 2540
CUT = pd.Timestamp('2022-09-05')
insample = dates <= CUT
in_dates = dates[insample]
train_dates = in_dates[:TRAIN_N]
val_dates   = in_dates[TRAIN_N:]
oos_dates   = dates[~insample]
print(f"总天数={len(dates)} | in-sample={insample.sum()} (TRAIN={len(train_dates)} VAL={len(val_dates)}) | OOS={len(oos_dates)}")
Huni = np.log(K)
print(f"均匀熵上限 H_uni = {Huni:.3f}  |  均匀 raw 每族=1.0\n")

def region_stats(name, dts):
    w = Wraw.loc[dts].to_numpy(dtype=np.float64)
    wp = w / K
    if len(w) == 0:
        print(f"  [{name}] 空"); return
    H = -(wp * np.log(wp + 1e-12)).sum(axis=1)
    rms = np.sqrt(((w - 1.0) ** 2).mean())
    maxw = w.max(axis=1)
    # 哪些族被系统性 boost / suppress（均值偏离 1.0）
    fam_mean = w.mean(axis=0)
    fam_names = np.array(Wraw.columns)
    boosted = fam_names[np.argsort(-(fam_mean - 1.0))[:4]]
    bval = (fam_mean - 1.0)[np.argsort(-(fam_mean - 1.0))[:4]]
    supp = fam_names[np.argsort(fam_mean - 1.0)[:4]]
    sval = (fam_mean - 1.0)[np.argsort(fam_mean - 1.0)[:4]]
    print(f"  [{name}] 天数={len(w)}")
    print(f"    逐日熵 H: 均值={H.mean():.3f} (占均匀 {H.mean()/Huni:.1%})  min={H.min():.3f} max={H.max():.3f}")
    print(f"    raw 偏离均匀 RMS={rms:.3f} | 每日最大 raw 权重 均值={maxw.mean():.3f} 峰={maxw.max():.3f}")
    print(f"    系统性 boost: " + ", ".join(f"{n}({v:+.2f})" for n, v in zip(boosted, bval)))
    print(f"    系统性 suppress: " + ", ".join(f"{n}({v:+.2f})" for n, v in zip(supp, sval)))

print("=== 门控调制强度（按区域，正确归一化）===")
region_stats("TRAIN", train_dates)
region_stats("VAL  ", val_dates)
region_stats("OOS  ", oos_dates)

print("\n=== leverage 族 raw 权重 × 市场方向（mkt_pc1 符号，均匀=1.0）===")
rm = m.regime_matrix
pc1 = rm.loc[Wraw.index, 'mkt_pc1'].to_numpy()
lev = Wraw['leverage'].to_numpy()
for nm, dts in [("TRAIN", train_dates), ("VAL", val_dates), ("OOS", oos_dates)]:
    sub = pc1[np.isin(dates, dts)]
    lsub = lev[np.isin(dates, dts)]
    u = sub > 0; d = sub <= 0
    if u.sum() and d.sum():
        print(f"  [{nm}] 牛杠杆={lsub[u].mean():.3f} 熊杠杆={lsub[d].mean():.3f} 差={lsub[d].mean()-lsub[u].mean():+.3f} "
              f"(牛{dates[np.isin(dates,dts)][u].shape[0]}日/熊{dates[np.isin(dates,dts)][d].shape[0]}日)")
    else:
        print(f"  [{nm}] 单边缺失 (涨{u.sum()} 跌{d.sum()})")
