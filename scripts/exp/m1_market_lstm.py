# -*- coding: utf-8 -*-
"""
M1 判别实验：LSTM 市场时序表示 vs 静态 regime，预测未来市场收益。

判据：OOS R² / IC。LSTM 显著强于静态 regime → 时序表示有价值，
双轴 NAM 架构（LSTM 调制单因子权重）的必要条件成立。
若持平或更弱 → 市场状态与未来收益关系本身弱（T046 ρ²≈0.054），
换表示救不了，双轴架构纵截面轴不成立，此轴关闭。

切分与项目协议一致：train ≤2022-09-05，OOS-熊 2022-09-05~2024-08-05（主判），
OOS-牛 2024-08-05~2026-08-05（参考）。val = train 末尾 10%（时间序）。
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from core.factors.regime_features import build_regime_matrix, load_market_sentiment

OUT = os.path.join(ROOT, 'diagnose_output', 'm1')
os.makedirs(OUT, exist_ok=True)

WINDOW = 20          # LSTM 时序窗口（交易日）
HORIZONS = (5, 20)   # 未来收益 horizon
HIDDEN = 32
EPOCHS = 200
PATIENCE = 25
LR = 1e-3
WD = 1e-4
SEEDS = (42, 11, 23)
TRAIN_END = pd.Timestamp('2022-09-05')
BEAR_END = pd.Timestamp('2024-08-05')

DEVICE = torch.device('cpu')


def spearmanr(a, b):
    a = pd.Series(a).rank().to_numpy()
    b = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


class LSTMMarket(nn.Module):
    def __init__(self, d_in, hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def evaluate(pred, y):
    pred = np.asarray(pred, np.float64)
    y = np.asarray(y, np.float64)
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    ic = spearmanr(pred, y) if len(pred) > 2 else float('nan')
    acc = float(np.mean((pred > 0) == (y > 0))) if len(pred) else float('nan')
    return {'n': int(len(y)), 'r2': r2, 'ic': ic, 'acc': acc}


def train_lstm(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LSTMMarket(Xtr.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.MSELoss()

    mu, sd = float(ytr.mean()), float(ytr.std()) or 1.0
    ytr_n = (ytr - mu) / sd
    yva_n = (yva - mu) / sd

    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr_n)
    Xva_t = torch.from_numpy(Xva); yva_t = torch.from_numpy(yva_n)
    n = len(Xtr_t)
    best_va, best_state, patience = float('inf'), None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(lossf(model(Xva_t), yva_t))
        if vloss < best_va - 1e-5:
            best_va, patience = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_tr = model(Xtr_t).numpy() * sd + mu
        p_va = model(Xva_t).numpy() * sd + mu
    return model, p_tr, p_va


def main():
    # ---- 数据 ----
    mat = build_regime_matrix().astype(np.float32)
    sent = load_market_sentiment()
    mret = sent['mean_return'].reindex(mat.index).fillna(0.0)
    idx = (1.0 + mret).cumprod()

    dates = pd.DatetimeIndex(mat.index)
    X_all = mat.to_numpy(dtype=np.float32)  # (T, 30)

    last_need = max(HORIZONS)
    valid = np.arange(WINDOW - 1, len(dates) - last_need)

    y_by_h = {}
    for h in HORIZONS:
        y_by_h[h] = (idx.shift(-h) / idx - 1.0).to_numpy(dtype=np.float32)[valid]

    X_windows = np.stack([X_all[t - WINDOW + 1:t + 1] for t in valid])  # (N, W, 30)
    X_flat = X_all[valid]  # (N, 30) 静态对照输入
    D = dates[valid]

    train_m = D <= TRAIN_END
    bear_m = (D > TRAIN_END) & (D <= BEAR_END)
    bull_m = D > BEAR_END

    report = {'meta': {'window': WINDOW, 'horizons': list(HORIZONS), 'hidden': HIDDEN,
                       'train_end': str(TRAIN_END.date()), 'bear_end': str(BEAR_END.date()),
                       'n_train': int(train_m.sum()), 'n_bear': int(bear_m.sum()),
                       'n_bull': int(bull_m.sum())},
              'results': {}}

    # ---- 静态对照：Ridge（当日 regime 30 维）----
    from sklearn.linear_model import Ridge
    # val = train 末尾 10%（时间序）
    tr_idx = np.where(train_m)[0]
    n_tr = len(tr_idx)
    va_idx = tr_idx[int(n_tr * 0.9):]
    tr_idx = tr_idx[:int(n_tr * 0.9)]
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    for h in HORIZONS:
        yh = y_by_h[h]
        best_a, best_va_r2 = None, -np.inf
        for a in alphas:
            m = Ridge(alpha=a).fit(X_flat[tr_idx], yh[tr_idx])
            p = m.predict(X_flat[va_idx])
            ss_res = np.sum((p - yh[va_idx]) ** 2)
            ss_tot = np.sum((yh[va_idx] - yh[va_idx].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
            if r2 > best_va_r2:
                best_va_r2, best_a = r2, a
        m = Ridge(alpha=best_a).fit(X_flat[tr_idx], yh[tr_idx])
        p_bear = m.predict(X_flat[bear_m])
        p_bull = m.predict(X_flat[bull_m])
        report['results'][f'h{h}_static'] = {
            'alpha': best_a,
            'bear': evaluate(p_bear, yh[bear_m]),
            'bull': evaluate(p_bull, yh[bull_m]),
        }
        print(f"[static h={h}] alpha={best_a} "
              f"bear R²={report['results'][f'h{h}_static']['bear']['r2']:.4f} "
              f"IC={report['results'][f'h{h}_static']['bear']['ic']:.4f}")

    # ---- LSTM ----
    for h in HORIZONS:
        yh = y_by_h[h]
        seeds_out = []
        for seed in SEEDS:
            model, _, _ = train_lstm(
                X_windows[tr_idx], yh[tr_idx],
                X_windows[va_idx], yh[va_idx], seed)
            with torch.no_grad():
                p_bear = model(torch.from_numpy(X_windows[bear_m])).numpy()
                p_bull = model(torch.from_numpy(X_windows[bull_m])).numpy()
            seeds_out.append({
                'seed': seed,
                'bear': evaluate(p_bear, yh[bear_m]),
                'bull': evaluate(p_bull, yh[bull_m]),
            })
            print(f"[lstm h={h} seed={seed}] "
                  f"bear R²={seeds_out[-1]['bear']['r2']:.4f} "
                  f"IC={seeds_out[-1]['bear']['ic']:.4f} | "
                  f"bull R²={seeds_out[-1]['bull']['r2']:.4f}")

        r2s = [s['bear']['r2'] for s in seeds_out]
        ics = [s['bear']['ic'] for s in seeds_out]
        report['results'][f'h{h}_lstm'] = {
            'seeds': seeds_out,
            'bear_r2_mean': float(np.mean(r2s)), 'bear_r2_std': float(np.std(r2s)),
            'bear_ic_mean': float(np.mean(ics)), 'bear_ic_std': float(np.std(ics)),
        }

    # ---- 判定 ----
    j = {}
    for h in HORIZONS:
        st = report['results'][f'h{h}_static']['bear']
        ls = report['results'][f'h{h}_lstm']
        j[f'h{h}'] = {
            'static_bear_r2': st['r2'], 'static_bear_ic': st['ic'],
            'lstm_bear_r2_mean': ls['bear_r2_mean'], 'lstm_bear_ic_mean': ls['bear_ic_mean'],
            'r2_delta': ls['bear_r2_mean'] - st['r2'],
            'verdict': 'LSTM 强于静态' if ls['bear_r2_mean'] > st['r2'] + 0.01
                       else ('持平' if abs(ls['bear_r2_mean'] - st['r2']) <= 0.01
                             else 'LSTM 弱于静态'),
        }
    report['judgement'] = j
    with open(os.path.join(OUT, 'm1_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(json.dumps(j, ensure_ascii=False, indent=2))
    print(f"报告: {os.path.join(OUT, 'm1_report.json')}")


if __name__ == '__main__':
    main()
