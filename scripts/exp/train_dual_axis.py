# -*- coding: utf-8 -*-
"""
T139 双轴 NAM 训练：纵截面 LSTM 市场时序调制 + 横截面 NAM 专家。

用法（冒烟 800×3y，与 T045 同折/同标签/同 ListNet 配置）：
    python scripts/exp/train_dual_axis.py --stocks 800 --years 3 --end 2022-09-05 \
        --folds 0.8:1.0 --seed 42 \
        --save-model-dir models/nam_gate/dual_axis_smoke \
        --output diagnose_output/dual_axis_smoke

生产配方在冒烟验证后扩展（--stocks 5480 --years 8 全量）。
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
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from core.factors.dual_axis_model import DualAxisNAM, DualAxisModel
from core.factors.nam_gate_model import listnet_loss
from core.factors.regime_features import build_regime_matrix
from core.factors.train_ml_model import MLModelTrainer
from config.baostock_config import DATABASE_PATH
from config.factor_config import TrainingConfig
from core.data.baostock_main import BaostockDataManager
from train_nam_model import _prepare_fold, _day_slices, _rank_labels_by_day

WINDOW = 20
HIDDEN = 32


def day_windows(regime: pd.DataFrame, days: np.ndarray, window: int) -> np.ndarray:
    """对 days 中每个唯一日期，取 regime 的 [window, d] 窗口（首段前填）。"""
    uniq = pd.DatetimeIndex(pd.to_datetime(pd.Series(np.unique(days)))).sort_values()
    lookup = {}
    for d in uniq:
        pos = regime.index.get_indexer([d], method='pad')[0]
        if pos < 0:
            raise ValueError(f'regime 矩阵无 {d} 当日或更早数据')
        start = max(0, pos - window + 1)
        arr = regime.iloc[start:pos + 1].to_numpy(dtype=np.float32)
        if arr.shape[0] < window:
            arr = np.concatenate([np.repeat(arr[:1], window - arr.shape[0], axis=0), arr])
        lookup[d] = arr
    return lookup


def daily_rank_ic(pred: np.ndarray, y: np.ndarray, days: np.ndarray) -> float:
    """按日 rank IC（Spearman 均值）——与生产 select_metric=rank_ic 同口径。"""
    ics = []
    for s, e in _day_slices(days):
        if e - s < 3:
            continue
        p = pd.Series(pred[s:e]).rank().to_numpy()
        t = pd.Series(y[s:e]).rank().to_numpy()
        if np.std(p) == 0 or np.std(t) == 0:
            continue
        ics.append(float(np.corrcoef(p, t)[0, 1]))
    return float(np.mean(ics)) if ics else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stocks', type=int, default=800)
    ap.add_argument('--years', type=int, default=3)
    ap.add_argument('--end', required=True)
    ap.add_argument('--folds', default='0.8:1.0')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--min-epochs', type=int, default=20)
    ap.add_argument('--warmup-epochs', type=int, default=10,
                    help='冻结 LSTM+调制层的轮数（安全网：先让专家收敛再开调制）')
    ap.add_argument('--init-experts-from', default=None,
                    help='从纯 NAM 存档（nam_gate_factor_model.pkl）加载 experts 权重。'
                         '配合 --freeze-experts 实现"先练纵截面"：横截面专家保持基线不变，'
                         '只训 LSTM+调制层，判别调制是否独立学得市场条件化。')
    ap.add_argument('--freeze-experts', action='store_true',
                    help='冻结 experts 权重（requires_grad=False），只训练 LSTM+调制+bias。'
                         '与 --warmup-epochs 区别：warmup 是暂时冻结（之后解冻），'
                         'freeze 是全程冻结。')
    ap.add_argument('--unfreeze-epoch', type=int, default=None,
                    help='纵截面 stage 结束的 epoch：在此解冻 experts 进入联合微调。'
                         'None=全程冻结（纯纵截面 stage 判别）；给 N 则第 N 轮起联合训练。')
    ap.add_argument('--mod-wd', type=float, default=0.0,
                    help='m_t 收缩正则系数：loss += mod_wd · mean((m_t − 1)²)。'
                         '强制调制权重靠近安全网 1，直接对抗"重写 experts"捷径——'
                         '在调制只能小幅波动的约束下若 val_ic 仍提升，才是真实市场条件化。')
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-5)
    ap.add_argument('--y-scale', type=float, default=2.0)
    ap.add_argument('--accum-days', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--include-mkt', action='store_true',
                    help='regime 矩阵追加 10 个 mkt_* 市场级列（纵截面输入更宽）')
    ap.add_argument('--save-model-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    end_dt = datetime.strptime(args.end, '%Y-%m-%d')
    end = end_dt.strftime('%Y-%m-%d')
    start = (end_dt - timedelta(days=365 * args.years)).strftime('%Y-%m-%d')
    dev = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
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
    del stocks_data
    gc.collect()
    all_features = list(dataset[3])
    nam_features = [f for f in all_features if '_regime_' not in f]
    print(f'  全部特征 {len(all_features)}，NAM 特征 {len(nam_features)}，样本 {len(dataset[0])}')

    regime = build_regime_matrix(DATABASE_PATH, include_mkt=args.include_mkt)
    print(f'  市场状态矩阵: {regime.shape[0]} 日 × {regime.shape[1]} 维'
          + ('（含 mkt_*）' if args.include_mkt else ''))

    tf, ve = 0.8, 1.0
    fold = _prepare_fold(trainer, dataset, nam_features, tf, ve, regime,
                         target='returns')
    dataset = None
    gc.collect()
    print(f'  fold: train {fold["X_train"].shape}, val {fold["X_val"].shape}')

    net = DualAxisNAM(n_factors=len(nam_features), d_regime=regime.shape[1],
                      expert_hidden=16, hidden=HIDDEN, window=WINDOW).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    print(f'  DualAxisNAM: 参数 {n_params:,}（专家 {sum(p.numel() for p in net.experts.parameters()):,}'
          f' + LSTM/调制 {n_params - sum(p.numel() for p in net.experts.parameters()):,}）')

    # ── 纵截面优先：从纯 NAM 加载 experts，可选冻结只训调制层 ──────────────
    if args.init_experts_from:
        from core.factors.nam_gate_model import NAMGateModel
        src = NAMGateModel()
        src.load_model(args.init_experts_from)
        if list(src.feature_names) != list(nam_features):
            raise ValueError(
                f'--init-experts-from 特征不匹配: 源 {len(src.feature_names)} '
                f'vs 当前 {len(nam_features)}')
        own = net.state_dict()
        _ek = [k for k in own if k.startswith('experts.')]
        for k in _ek:
            own[k] = src.net.state_dict()[k].to(own[k].device)
        net.load_state_dict(own)
        print(f'  已从 {args.init_experts_from} 加载 experts 权重（{len(_ek)} 组特征一致）')
    if args.freeze_experts:
        n_frz = sum(p.numel() for n, p in net.named_parameters()
                    if n.startswith('experts.'))
        for n, p in net.named_parameters():
            if n.startswith('experts.'):
                p.requires_grad_(False)
        print(f'  已冻结 experts（{n_frz:,} 参数）—— 纵截面 stage：只训 LSTM+调制+bias')

    # 市场窗口：训练/验证每日一个 [window, d]
    wtr = day_windows(regime, fold['d_train'], WINDOW)
    wva = day_windows(regime, fold['d_val'], WINDOW)

    # 数据转张量（800×3y 冒烟规模全量驻留）
    Xtr = torch.from_numpy(fold['X_train'].astype(np.float32)).to(dev)
    ytr = torch.from_numpy(fold['y_train'].astype(np.float32)).to(dev)
    dtr = fold['d_train']
    Xva = torch.from_numpy(fold['X_val'].astype(np.float32)).to(dev)
    yva = torch.from_numpy(fold['y_val'].astype(np.float32)).to(dev)
    dva = fold['d_val']
    tr_days = _day_slices(dtr)
    va_days = _day_slices(dva)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mod_params = set(net.lstm.parameters()) | set(net.mod.parameters())
    exp_params = set(net.experts.parameters())

    best_ic, best_state, patience = -np.inf, None, 0
    history = []
    for ep in range(args.epochs):
        # warmup：冻结纵截面轴（安全网，先让专家收敛再开调制）
        frozen = ep < args.warmup_epochs
        for p in mod_params:
            p.requires_grad_(not frozen)
        # 纵截面优先 stage：--freeze-experts 全程冻结 experts，或到 unfreeze_epoch 解冻
        if args.freeze_experts:
            unfreeze = (args.unfreeze_epoch is not None) and (ep >= args.unfreeze_epoch)
            for p in exp_params:
                p.requires_grad_(unfreeze)
        stage = 'mod-frozen' if frozen else \
            ('exp-frozen' if (args.freeze_experts and not
                              ((args.unfreeze_epoch is not None) and ep >= args.unfreeze_epoch))
             else 'joint')
        net.train()
        opt.zero_grad()
        ep_loss, n_steps = 0.0, 0
        for s, e in tr_days:
            xb, yb = Xtr[s:e], ytr[s:e]
            w = torch.from_numpy(wtr[pd.Timestamp(dtr[s])]).to(dev)
            score, m, _ = net(xb, w)
            loss = listnet_loss(score, yb, temp=1.0, y_scale=args.y_scale)
            if args.mod_wd > 0:
                # m_t 收缩到安全网 1（逐日一个调制向量，mean over factors）
                loss = loss + args.mod_wd * torch.mean((m - 1.0) ** 2)
            loss.backward()
            if (n_steps + 1) % args.accum_days == 0 or e == len(dtr):
                opt.step()
                opt.zero_grad()
            ep_loss += float(loss.detach())
            n_steps += 1
        opt.step()
        opt.zero_grad()

        # 验证：整截面 rank IC（低方差选型统计量，T056 铁律）
        net.eval()
        pred_va = np.empty(len(yva), dtype=np.float64)
        with torch.no_grad():
            for s, e in va_days:
                w = torch.from_numpy(wva[pd.Timestamp(dva[s])]).to(dev)
                score, _, _ = net(Xva[s:e], w)
                pred_va[s:e] = score.cpu().numpy()
        ic = daily_rank_ic(pred_va, fold['y_val'], dva)
        hist = {'epoch': ep, 'loss': ep_loss / max(n_steps, 1), 'val_rank_ic': ic,
                'frozen': frozen, 'stage': stage}
        # m_t 变异度（warmup 结束后才有意义）
        if not frozen:
            ms = []
            with torch.no_grad():
                for s, e in va_days[:200]:
                    w = torch.from_numpy(wva[pd.Timestamp(dva[s])]).to(dev)
                    _, m, _ = net(Xva[s:e], w)
                    ms.append(m.cpu().numpy())
            ms = np.stack(ms)
            hist['m_std'] = float(ms.std(axis=0).mean())
            hist['m_mean_dev'] = float(np.abs(ms.mean(axis=0) - 1.0).mean())
        history.append(hist)
        print(f"  [ep {ep:3d} {stage:9s}] loss={hist['loss']:.4f} val_ic={ic:.4f}"
              + (f" m_std={hist.get('m_std', float('nan')):.4f}"
                 f" m_mean_dev={hist.get('m_mean_dev', float('nan')):.4f}" if not frozen else ' (m≡1)'))
        if ic > best_ic + 1e-5 and ep >= args.min_epochs:
            best_ic, patience = ic, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience and ep >= args.min_epochs:
                print(f'  [早停] epoch {ep}（验证 rank_ic 连续 {args.patience} 轮无改善）')
                break

    if best_state is None:
        best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    print(f'  最佳检查点已加载，best_val_ic={best_ic:.4f}')

    # 最终验证 IC + m_t 变异度诊断
    net.eval()
    pred_va = np.empty(len(yva), dtype=np.float64)
    ms_all = []
    with torch.no_grad():
        for s, e in va_days:
            w = torch.from_numpy(wva[pd.Timestamp(dva[s])]).to(dev)
            score, m, _ = net(Xva[s:e], w)
            pred_va[s:e] = score.cpu().numpy()
            ms_all.append(m.cpu().numpy())
    final_ic = daily_rank_ic(pred_va, fold['y_val'], dva)
    ms_all = np.stack(ms_all)  # [n_days, 228]
    m_std_mean = float(ms_all.std(axis=0).mean())
    m_std_max = float(ms_all.std(axis=0).max())
    m_dev_mean = float(np.abs(ms_all.mean(axis=0) - 1.0).mean())
    print(f'  最终 val rank_ic = {final_ic:.4f}')
    print(f'  m_t 变异度: 因子均值 std={m_std_mean:.4f} max={m_std_max:.4f} '
          f'| 均值偏离1的幅度 {m_dev_mean:.4f}')
    print('  [判定] m_t 时间变异接近 0 → 未利用市场信息，判负；变异大且 IC 有增益 → 方向成立')

    # 保存
    model = DualAxisModel(net, nam_features, list(regime.columns),
                          norm_stats=fold.get('skip_stats'))
    pkl = model.save(args.save_model_dir)
    os.makedirs(args.output, exist_ok=True)
    out = {
        'val_rank_ic': final_ic, 'best_val_rank_ic': best_ic,
        'm_std_mean': m_std_mean, 'm_std_max': m_std_max, 'm_mean_dev': m_dev_mean,
        'n_params': n_params, 'seed': args.seed, 'stocks': args.stocks,
        'years': args.years, 'epochs_run': len(history),
        'init_experts_from': args.init_experts_from, 'freeze_experts': args.freeze_experts,
        'unfreeze_epoch': args.unfreeze_epoch, 'mod_wd': args.mod_wd,
        'stage_breakdown': _stage_summary(history),
        'elapsed_sec': time.time() - t0,
    }
    with open(os.path.join(args.output, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'  已保存模型: {pkl}  (累计耗时 {time.time()-t0:.1f}s)')


def _stage_summary(history):
    """各 stage 的 val_ic 均值/最大，用于判别调制层独立增益。"""
    out = {}
    for h in history:
        st = h.get('stage', 'joint')
        d = out.setdefault(st, {'n': 0, 'ic_mean': 0.0, 'ic_max': -9.0, 'm_std_mean': 0.0})
        d['n'] += 1
        d['ic_mean'] += h['val_rank_ic']
        d['ic_max'] = max(d['ic_max'], h['val_rank_ic'])
        d['m_std_mean'] += h.get('m_std', 0.0)
    for st, d in out.items():
        d['ic_mean'] /= max(d['n'], 1)
        d['m_std_mean'] /= max(d['n'], 1)
    return out


if __name__ == '__main__':
    main()
