"""判别实验：regime 特征到底有没有「因子族条件化」的信息？

背景：T043 训练后诊断显示门控完全冻结（乘性模式全族饱和在下界 0.2，std≈1e-6；
softmax 模式熵恒等于 ln(K)）。有两种互斥解释：

  A. 市场状态特征本身无效 —— 数据里就不存在「不同 regime 下该切换因子族」的结构；
  B. 门控没学起来 —— 目标函数对门控常数分量不可识别 + sigmoid 饱和梯度消失。

本脚本用三层证据把两者分开：

  1. 因子族日度 IC 是否真的随时间变动（超出抽样噪声）→ 条件化结构是否存在；
  2. Oracle 上界：若当天就知道各族 IC，按 IC 加权能把组合 IC 提到多高（相对等权）；
  3. 可实现性：只用 regime 特征（严格样本外）预测未来 N 日各族 IC，
     用预测权重组合，看 IC 能否显著超过等权。

第 3 步显著为正 → 解释 B（门控该修）；第 3 步为零 → 解释 A（换市场特征）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from config.baostock_config import DATABASE_PATH  # noqa: E402
from config.factor_config import TrainingConfig  # noqa: E402
from core.data.baostock_main import BaostockDataManager  # noqa: E402
from core.factors.nam_gate_model import NAMGateModel  # noqa: E402
from core.factors.regime_features import build_regime_matrix  # noqa: E402
from core.factors.train_ml_model import MLModelTrainer  # noqa: E402
from scripts.exp.exp_nam_gate import _prepare_fold, _day_slices  # noqa: E402


def _rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    """单日 Spearman IC。"""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    xv, yv = x[m], y[m]
    if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return np.nan
    rx, ry = rankdata(xv), rankdata(yv)
    return float(np.corrcoef(rx, ry)[0, 1])


def collect_group_panel(model: NAMGateModel, X, returns, dates, M):
    """逐日抽取各因子族的未加权贡献、族 IC、门控权重。"""
    net = model.net
    dev = torch.device(model.device)
    Xn = X.astype(np.float32)
    if model.input_mean is not None and model.input_std is not None:
        Xn = ((Xn - np.asarray(model.input_mean, dtype=np.float32))
              / np.asarray(model.input_std, dtype=np.float32)).astype(np.float32)

    slices = _day_slices(dates)
    K = len(model.group_names)
    day_list, gic, gate_w = [], [], []
    ic_uniform, ic_oracle, ic_oracle_sign = [], [], []
    group_sums_by_day = []

    net.eval()
    with torch.no_grad():
        for (s, e) in slices:
            if e - s < 20:
                continue
            xt = torch.as_tensor(Xn[s:e], dtype=torch.float32, device=dev)
            mt = torch.as_tensor(np.asarray(M[s], dtype=np.float32),
                                 dtype=torch.float32, device=dev)
            _, w, gsum = net.forward_day(xt, mt)
            gs = gsum.cpu().numpy().astype(np.float64)          # [B, K] 未加权族贡献
            y = np.asarray(returns[s:e], dtype=np.float64)
            ics = np.array([_rank_ic(gs[:, k], y) for k in range(K)])
            if not np.isfinite(ics).all():
                continue
            day_list.append(dates[s])
            gic.append(ics)
            gate_w.append(w.cpu().numpy().astype(np.float64))
            group_sums_by_day.append((gs, y))
            # 等权（= 去掉门控的纯 NAM）
            ic_uniform.append(_rank_ic(gs.sum(axis=1), y))
            # oracle：当天真实 IC 当权重（含未来信息，仅作上界）
            ic_oracle.append(_rank_ic(gs @ ics, y))
            ic_oracle_sign.append(_rank_ic(gs @ np.sign(ics), y))

    return {
        'dates': np.array(day_list),
        'group_ic': np.array(gic),                     # [D, K]
        'gate': np.array(gate_w),                      # [D, K]
        'ic_uniform': np.array(ic_uniform),
        'ic_oracle': np.array(ic_oracle),
        'ic_oracle_sign': np.array(ic_oracle_sign),
        'panel': group_sums_by_day,
    }


def ridge_fit(Xtr, Ytr, alpha=10.0):
    """闭式 ridge（含截距），Xtr [n,p] / Ytr [n,K]。"""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = (Xtr - mu) / sd
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    p = Z.shape[1]
    R = np.eye(p) * alpha
    R[-1, -1] = 0.0
    W = np.linalg.solve(Z.T @ Z + R, Z.T @ Ytr)
    return (mu, sd, W)


def ridge_pred(model, Xte):
    mu, sd, W = model
    Z = (Xte - mu) / sd
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    return Z @ W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/nam_gate/T043/nam_gate_factor_model.pkl')
    ap.add_argument('--stocks', type=int, default=800)
    ap.add_argument('--years', type=int, default=13)
    ap.add_argument('--end', default='2022-09-05')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--horizon', type=int, default=20,
                    help='预测目标：未来 N 日平均族 IC（N=1 即日度）')
    ap.add_argument('--out', default='diagnose_output/nam_gate/regime_conditionality.json')
    args = ap.parse_args()

    model = NAMGateModel()
    model.load_model(args.model)
    feature_names = list(model.feature_names)
    print(f"模型 {args.model}: {len(feature_names)} 特征 / K={len(model.group_names)} 族")

    end_dt = datetime.strptime(args.end, '%Y-%m-%d')
    end = end_dt.strftime('%Y-%m-%d')
    start = (end_dt - timedelta(days=365 * args.years)).strftime('%Y-%m-%d')

    TrainingConfig.FUTURE_DAYS = 7
    trainer = MLModelTrainer(db_path=DATABASE_PATH)
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
    fold = _prepare_fold(trainer, dataset, feature_names, 0.8, 1.0, regime, target='returns')

    out = {}
    for part in ('train', 'val'):
        X = fold[f'X_{part}']
        r = fold[f'ret_{part}']
        d = fold[f'd_{part}']
        M = fold[f'M_{part}']
        print(f"\n=== {part}: {len(X)} 样本 / {d[0]} → {d[-1]} ===")
        res = collect_group_panel(model, X, r, d, M)
        out[part] = res
        D, K = res['group_ic'].shape
        print(f"  有效交易日 {D}")

        # --- 层 1：族 IC 时变性 ---
        gic = res['group_ic']
        print("\n  [层1] 因子族日度 IC 的时变性")
        print(f"  {'族':<12}{'均值IC':>9}{'日IC标准差':>12}{'20日平滑std':>13}{'|均值|/std':>11}")
        sm = pd.DataFrame(gic).rolling(20, min_periods=10).mean().to_numpy()
        layer1 = []
        for k, g in enumerate(model.group_names):
            mu, sd = np.nanmean(gic[:, k]), np.nanstd(gic[:, k])
            smsd = np.nanstd(sm[:, k])
            layer1.append({'group': g, 'ic_mean': mu, 'ic_std': sd, 'ic_smooth_std': smsd})
            print(f"  {g:<12}{mu:>9.4f}{sd:>12.4f}{smsd:>13.4f}{abs(mu)/(sd+1e-9):>11.3f}")

        # --- 层 2：oracle 上界 ---
        u = np.nanmean(res['ic_uniform'])
        o = np.nanmean(res['ic_oracle'])
        osg = np.nanmean(res['ic_oracle_sign'])
        print(f"\n  [层2] 组合 IC：等权 {u:.4f} | oracle(IC加权) {o:.4f} | oracle(符号) {osg:.4f}")
        print(f"        oracle 相对等权提升 {(o - u):.4f} ({(o/u - 1)*100 if u else float('nan'):.1f}%)")

        out[part] = {
            'n_days': int(D),
            'date_range': [str(res['dates'][0]), str(res['dates'][-1])],
            'layer1_group_ic': layer1,
            'layer2': {'ic_uniform': float(u), 'ic_oracle': float(o),
                       'ic_oracle_sign': float(osg)},
            'gate_mean': res['gate'].mean(0).tolist(),
            'gate_std': res['gate'].std(0).tolist(),
            '_res': res,
        }

    # --- 层 3：regime → 未来族 IC 的样本外可预测性 ---
    print(f"\n{'='*70}\n[层3] regime 特征预测未来 {args.horizon} 日族 IC（严格样本外）\n{'='*70}")
    tr, va = out['train']['_res'], out['val']['_res']
    H = args.horizon

    def fwd_ic(gic, h):
        """未来 h 日平均 IC（严格不含当日之前信息，用于当日决策的目标）。"""
        df = pd.DataFrame(gic)
        f = df.shift(-1).rolling(h, min_periods=max(3, h // 3)).mean().shift(-(h - 1))
        return f.to_numpy()

    Ytr = fwd_ic(tr['group_ic'], H)
    Mtr_full = fold['M_train']
    Mva_full = fold['M_val']
    # 把样本级 M 折到交易日级
    def day_M(M, dates_all, day_dates):
        idx = {d: i for i, d in enumerate(dates_all)}
        return np.array([M[idx[d]] for d in day_dates], dtype=np.float64)

    Mtr = day_M(Mtr_full, fold['d_train'], tr['dates'])
    Mva = day_M(Mva_full, fold['d_val'], va['dates'])
    Mtr = np.nan_to_num(Mtr, nan=0.0, posinf=0.0, neginf=0.0)
    Mva = np.nan_to_num(Mva, nan=0.0, posinf=0.0, neginf=0.0)

    ok = np.isfinite(Ytr).all(axis=1)
    rg = ridge_fit(Mtr[ok], Ytr[ok], alpha=50.0)
    Pva = ridge_pred(rg, Mva)                      # [Dv, K] 预测的族 IC
    Yva = fwd_ic(va['group_ic'], H)

    print(f"\n  {'族':<12}{'OOS corr(预测,实现)':>20}{'OOS R2':>10}")
    layer3 = []
    for k, g in enumerate(model.group_names):
        m = np.isfinite(Yva[:, k])
        if m.sum() < 30:
            continue
        c = float(np.corrcoef(Pva[m, k], Yva[m, k])[0, 1])
        ss_res = float(((Yva[m, k] - Pva[m, k]) ** 2).sum())
        ss_tot = float(((Yva[m, k] - Yva[m, k].mean()) ** 2).sum())
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        layer3.append({'group': g, 'oos_corr': c, 'oos_r2': r2})
        print(f"  {g:<12}{c:>20.4f}{r2:>10.4f}")

    # 用预测权重实际组合，比较 IC
    ic_pred_w, ic_pred_sign, ic_uni = [], [], []
    for di, (gs, y) in enumerate(va['panel']):
        if di >= len(Pva):
            break
        w = Pva[di]
        ic_uni.append(_rank_ic(gs.sum(axis=1), y))
        ic_pred_w.append(_rank_ic(gs @ w, y))
        ic_pred_sign.append(_rank_ic(gs @ np.sign(w), y))
    ic_uni = np.array(ic_uni); ic_pred_w = np.array(ic_pred_w)
    ic_pred_sign = np.array(ic_pred_sign)
    diff = ic_pred_w - ic_uni
    t = float(np.nanmean(diff) / (np.nanstd(diff) / np.sqrt(np.isfinite(diff).sum()) + 1e-12))

    print(f"\n  [层3 结论] 验证期组合 IC")
    print(f"    等权（纯 NAM）      : {np.nanmean(ic_uni):.4f}")
    print(f"    regime 预测权重     : {np.nanmean(ic_pred_w):.4f}")
    print(f"    regime 预测符号     : {np.nanmean(ic_pred_sign):.4f}")
    print(f"    oracle 上界         : {np.nanmean(va['ic_oracle']):.4f}")
    print(f"    差值 t 统计量       : {t:.2f}")

    # ------------------------------------------------------------------
    # 层 4：收缩化门控。层 3 证明 regime→族 IC 的「方向」有信息（corr 最高 0.50）
    # 但「幅度」完全不可信（OOS R² 普遍为负），直接当权重必然放大噪声。
    # 这里用三段式无泄漏流程检验「弱信号 + 强收缩」是否可用：
    #   A = 训练期前 80%  → 拟合 ridge
    #   B = 训练期后 20%  → 选族（corr>0）+ 选收缩系数 γ
    #   val               → 纯样本外评估
    # ------------------------------------------------------------------
    print(f"\n{'='*70}\n[层4] 收缩化门控（无泄漏三段式：拟合A / 选参B / 评估val）\n{'='*70}")
    nA = int(len(Mtr) * 0.8)
    okA = np.isfinite(Ytr[:nA]).all(axis=1)
    rgA = ridge_fit(Mtr[:nA][okA], Ytr[:nA][okA], alpha=50.0)
    Pb = ridge_pred(rgA, Mtr[nA:])
    Yb = Ytr[nA:]
    mb = np.isfinite(Yb).all(axis=1)
    keep, corrB = [], []
    for k, g in enumerate(model.group_names):
        c = float(np.corrcoef(Pb[mb, k], Yb[mb, k])[0, 1]) if mb.sum() > 30 else 0.0
        corrB.append(c)
        if c > 0.05:
            keep.append(k)
    print(f"  B 段入选族({len(keep)}/{len(model.group_names)}): "
          f"{[model.group_names[k] for k in keep]}")

    pmu, psd = Pb.mean(0), Pb.std(0) + 1e-12

    def build_w(P):
        z = np.clip((P - pmu) / psd, -2.0, 2.0)
        mask = np.zeros(len(model.group_names))
        mask[keep] = 1.0
        return z * mask

    # B 段选 γ
    tr_panel_B = tr['panel'][nA:]
    Zb = build_w(Pb)
    best_g, best_ic = 0.0, -np.inf
    for gam in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]:
        ics = []
        for di, (gs, y) in enumerate(tr_panel_B):
            if di >= len(Zb):
                break
            ics.append(_rank_ic(gs @ (1.0 + gam * Zb[di]), y))
        v = float(np.nanmean(ics))
        print(f"    γ={gam:<4} B段组合IC {v:.4f}")
        if v > best_ic:
            best_ic, best_g = v, gam
    print(f"  B 段最优 γ = {best_g}")

    Zv = build_w(ridge_pred(rgA, Mva))
    ic_shrink, ic_u2 = [], []
    for di, (gs, y) in enumerate(va['panel']):
        if di >= len(Zv):
            break
        ic_u2.append(_rank_ic(gs.sum(axis=1), y))
        ic_shrink.append(_rank_ic(gs @ (1.0 + best_g * Zv[di]), y))
    ic_u2 = np.array(ic_u2); ic_shrink = np.array(ic_shrink)
    dd = ic_shrink - ic_u2
    t4 = float(np.nanmean(dd) / (np.nanstd(dd) / np.sqrt(np.isfinite(dd).sum()) + 1e-12))
    print(f"\n  [层4 结论] 验证期（严格样本外）")
    print(f"    等权         : {np.nanmean(ic_u2):.4f}")
    print(f"    收缩门控 γ={best_g}: {np.nanmean(ic_shrink):.4f}")
    print(f"    提升 {np.nanmean(dd):+.4f}   t = {t4:.2f}")

    # γ 全曲线（验证期）：与层5 低参查表版做「参数量 → 估计噪声」对照。
    # 这是事后诊断曲线，只用于观察信噪比形态，不参与选参（选参已在 B 段完成）。
    sweep_val4 = {}
    for gam in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]:
        ics = []
        for di, (gs, y) in enumerate(va['panel']):
            if di >= len(Zv):
                break
            ics.append(_rank_ic(gs @ (1.0 + gam * Zv[di]), y))
        sweep_val4[str(gam)] = float(np.nanmean(ics))
    print("    [γ 扫描·验证期] " + "  ".join(f"γ={g}:{v:.4f}" for g, v in sweep_val4.items()))

    # ------------------------------------------------------------------
    # 层 5：极简离散 regime 桶。层 3/4 用 29 维 ridge → 11 输出（330 参数）在
    # 低信噪比目标上必然过拟合。这里退到 4 桶 × 11 族 = 44 个查表参数，
    # 用样本量换稳定性 —— 若连这个都无效，才能说 regime 特征这条路走不通。
    # ------------------------------------------------------------------
    print(f"\n{'='*70}\n[层5] 离散 regime 桶（trend × vol 四象限查表）\n{'='*70}")
    rc = list(model.regime_cols)
    i_tr, i_vol = rc.index('trend_ma20_dev'), rc.index('vol_20')
    vol_med = float(np.nanmedian(Mtr[:nA, i_vol]))

    def bucket(M):
        return ((M[:, i_tr] >= 0).astype(int) * 2 + (M[:, i_vol] >= vol_med).astype(int))

    bA, bB, bV = bucket(Mtr[:nA]), bucket(Mtr[nA:]), bucket(Mva)
    gicA = tr['group_ic'][:nA]
    table = np.zeros((4, len(model.group_names)))
    names4 = ['跌势-低波', '跌势-高波', '涨势-低波', '涨势-高波']
    print(f"  {'桶':<12}{'A段日数':>8}  各族平均IC (z化后用于加权)")
    for b in range(4):
        m = bA == b
        if m.sum() < 50:
            continue
        v = np.nanmean(gicA[m], axis=0)
        table[b] = (v - v.mean()) / (v.std() + 1e-12)
        print(f"  {names4[b]:<12}{int(m.sum()):>8}  "
              + ' '.join(f'{x:+.2f}' for x in table[b]))

    def sweep(panel, buckets, offset=0):
        res = {}
        for gam in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]:
            ics = []
            for di, (gs, y) in enumerate(panel):
                if di >= len(buckets):
                    break
                ics.append(_rank_ic(gs @ (1.0 + gam * table[buckets[di]]), y))
            res[gam] = float(np.nanmean(ics))
        return res

    rB = sweep(tr['panel'][nA:], bB)
    print('\n  B 段 γ 扫描: ' + '  '.join(f'{g}:{v:.4f}' for g, v in rB.items()))
    g5 = max(rB, key=rB.get)
    rV = sweep(va['panel'], bV)
    print('  val  γ 扫描: ' + '  '.join(f'{g}:{v:.4f}' for g, v in rV.items()))
    print(f"\n  [层5 结论] B 段最优 γ={g5} → val IC {rV[g5]:.4f} "
          f"(等权 {rV[0.0]:.4f}, 提升 {rV[g5]-rV[0.0]:+.4f})")

    payload = {
        'model': args.model,
        'horizon': H,
        'groups': list(model.group_names),
        'train': {k: v for k, v in out['train'].items() if k != '_res'},
        'val': {k: v for k, v in out['val'].items() if k != '_res'},
        'layer3_predictability': layer3,
        'layer3_portfolio': {
            'ic_uniform': float(np.nanmean(ic_uni)),
            'ic_regime_weight': float(np.nanmean(ic_pred_w)),
            'ic_regime_sign': float(np.nanmean(ic_pred_sign)),
            'ic_oracle': float(np.nanmean(va['ic_oracle'])),
            't_stat': t,
        },
        'layer4_shrinkage': {
            'kept_groups': [model.group_names[k] for k in keep],
            'corr_B': {g: c for g, c in zip(model.group_names, corrB)},
            'gamma': best_g,
            'ic_uniform': float(np.nanmean(ic_u2)),
            'ic_shrink': float(np.nanmean(ic_shrink)),
            'delta': float(np.nanmean(dd)),
            't_stat': t4,
            'sweep_val': sweep_val4,
        },
        'layer5_bucket': {
            'buckets': names4,
            'table': table.tolist(),
            'gamma_B': g5,
            'sweep_B': rB,
            'sweep_val': rV,
            'delta_val': rV[g5] - rV[0.0],
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n已写出: {args.out}")


if __name__ == '__main__':
    main()
