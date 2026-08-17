"""T092：同折对照 NAM vs XGBoost vs LightGBM。

**为什么要有这个脚本**：T039 以来约 50 轮实验全部在 NAM 上做，而实盘 automation 加载的是
`models/mark/automation/lightgbm_factor_model.pkl`（单模型 LightGBM）。翻遍 78 个历史结果
文件，**没有任何一个含第二个模型臂** —— `exp_nam_gate.py` 的同折 XGBoost 基线依赖已丢失的
`exp_head_features._train_and_predict`，所有实验一律 `--skip-baseline`。
也就是说「NAM 比树好」这件事**从未被验证过**，却已经决定了 50 轮实验的载体。

**可比性靠什么保证**：直接复用 `exp_nam_gate._prepare_fold`，所以三个模型吃到的是
逐字节相同的 `X_train/y_train/d_train` 与 `X_val/ret_val/d_val`（同一折边界、同一截面
归一化统计量、同一 rank 标签、同一族裁剪）。评估用同一个 `_metrics._daily_metrics`，
口径与 `T090_base` 完全一致，可以直接和台账里的 0.09650 比。

**两套读数，因为选型偏差**（见 [[checkpoint-selection-inflates-ic]]）：
- `rank_ic`：早停选出的最优迭代 —— 与 NAM 的「最优 epoch」同样带选择偏差，**这一栏才和
  T090_base 可比**。
- `rank_ic_fixed`：固定 `--fixed-iter` 棵树、不做任何验证集选型 —— 无偏，三模型之间横比更公平。

输出 JSON 用 `folds[tag]['nam_gate']` 这个键名纯粹是为了让 `analyze_multifold.py` 能直接读
（它硬编码了该键），**不代表跑的是 NAM**。文件名区分模型。

用法（种子只影响树的 subsample/colsample 抽样）：
  python -u scripts/exp/exp_tree_vs_nam.py --model xgboost --seed 42 \
      --stocks 800 --years 13 --end 2022-09-05 --drop-groups forecast \
      --cache-dir database/system_data/factors_cache_2026-08-14-fwdadjust \
      --folds 0.6:0.8,0.7:0.9,0.8:1.0 --allow-degenerate-downside-risk \
      --output diagnose_output/T092_xgb_s42.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.baostock_config import DATABASE_PATH
from config.factor_config import ModelConfig, TrainingConfig
from config.factor_groups import build_group_index
from core.data.baostock_main import BaostockDataManager
from core.factors.regime_features import build_regime_matrix
from core.factors.train_ml_model import MLModelTrainer
from scripts.exp._metrics import _daily_metrics, _day_slices
from scripts.exp.exp_nam_gate import _prepare_fold, _regime_stratified_metrics


def _day_groups(dates):
    """lambdarank 的 query 分组 = 每个交易日一组。dates 已按日期升序。"""
    _, counts = np.unique(dates, return_counts=True)
    return counts.astype(np.int64)


def _discretize(y_cont, dates, n_bins):
    """连续 rank → 固定边界档位，与 train_ml_model.train_models 的实现同口径。

    固定边界（而非按日 qcut）是关键：涨跌停日大量并列会让 qcut 的分位边界重合，
    不同日期切出不同档位数，同一「档位 k」跨 query 语义就不一致，直接毁掉 lambdarank 信号。
    """
    import pandas as pd
    edges = np.concatenate([[0.0], np.linspace(1.0 / n_bins, 1.0, n_bins)])
    out = np.empty(len(y_cont), dtype=np.int32)
    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    mid = n_bins // 2
    for s, c in zip(starts, counts):
        e = s + c
        if c <= 1:
            out[s:e] = mid
            continue
        r = np.clip(y_cont[s:e], 1e-6, 1.0 - 1e-9)
        b = pd.cut(r, bins=edges, labels=False, include_lowest=True)
        out[s:e] = np.clip(np.nan_to_num(b, nan=float(mid)).astype(np.int32), 0, n_bins - 1)
    return out


def _sel_rep(d_val, select_holdout):
    """把验证折按**时间**切成 (选型行索引, 报告行索引)。

    按时间而非随机切：7 日重叠标签会跨随机切分泄露，随机切出来的「holdout」
    仍然被早停看见过。select_holdout=0 时报告段=整折（旧行为，带选型偏差）。
    """
    slices = _day_slices(np.asarray(d_val))
    if select_holdout <= 0:
        idx = np.arange(len(d_val), dtype=np.int64)
        return idx, idx
    n_sel = max(1, int(round(len(slices) * (1.0 - select_holdout))))
    cat = lambda ch: (np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in ch])
                      if ch else np.zeros(0, dtype=np.int64))
    return cat(slices[:n_sel]), cat(slices[n_sel:])


def _fit_xgb(fold, seed, fixed_iter, sel, overrides):
    import xgboost as xgb
    params = ModelConfig.get_model_params('xgboost', task='ranking')
    params.update({'random_state': seed, 'n_estimators': 1000})
    params.update(overrides)
    early = params.pop('early_stopping_rounds', 200)
    model = xgb.XGBRanker(**params, early_stopping_rounds=early)
    # 早停只看 sel 段；rep 段对选型完全不可见 —— 这是无偏读数的前提。
    Xs, ys, ds = fold['X_val'][sel], fold['y_val'][sel], np.asarray(fold['d_val'])[sel]
    model.fit(
        fold['X_train'], fold['y_train'],
        group=_day_groups(fold['d_train']),
        eval_set=[(Xs, ys)],
        eval_group=[_day_groups(ds)],
        verbose=False,
    )
    best = int(getattr(model, 'best_iteration', fixed_iter) or fixed_iter) + 1
    p_best = model.predict(fold['X_val'], iteration_range=(0, best))
    p_fix = model.predict(fold['X_val'], iteration_range=(0, min(fixed_iter, best if best > fixed_iter else fixed_iter)))
    return p_best, p_fix, best


def _fit_lgb(fold, seed, fixed_iter, n_bins, sel, overrides):
    import lightgbm as lgb
    y_tr = _discretize(fold['y_train'], fold['d_train'], n_bins)
    y_va = _discretize(fold['y_val'], fold['d_val'], n_bins)
    g_tr = _day_groups(fold['d_train'])
    params = ModelConfig.get_model_params('lightgbm', task='ranking')
    params.update({'random_state': seed, 'n_estimators': 1000})
    params.update(overrides)
    early = params.pop('early_stopping_rounds', 200)

    ds = np.asarray(fold['d_val'])[sel]
    model = lgb.LGBMRanker(**params)
    model.fit(fold['X_train'], y_tr, group=g_tr,
              eval_set=[(fold['X_val'][sel], y_va[sel])], eval_group=[_day_groups(ds)],
              callbacks=[lgb.early_stopping(early, first_metric_only=True, verbose=False)])
    best = int(model.best_iteration_ or fixed_iter)
    p_best = model.booster_.predict(fold['X_val'], num_iteration=best)

    # LightGBM 早停会**截断 booster**（实测 num_trees() == best_iteration），
    # 所以 num_iteration=300 取不到第 300 棵树 —— 无选型读数只能重训一棵固定轮数的。
    # XGBoost 不同，它保留全部树，直接 iteration_range 即可。
    fixed = lgb.LGBMRanker(**{**params, 'n_estimators': fixed_iter})
    fixed.fit(fold['X_train'], y_tr, group=g_tr)
    p_fix = fixed.booster_.predict(fold['X_val'])
    return p_best, p_fix, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['xgboost', 'lightgbm'])
    ap.add_argument('--stocks', type=int, default=800)
    ap.add_argument('--years', type=int, default=13)
    ap.add_argument('--end', default='2022-09-05')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--folds', default='0.6:0.8,0.7:0.9,0.8:1.0')
    ap.add_argument('--drop-groups', default='')
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--fixed-iter', type=int, default=300,
                    help='不做验证集选型的固定迭代数，用于给出无偏读数')
    ap.add_argument('--full-panel', action='store_true',
                    help='保留手工 *_regime_* 交互列（默认剔除以与 NAM 对齐）；'
                         '配合不传 --drop-groups 即为满配 231 列')
    ap.add_argument('--save-preds', default=None,
                    help='把每折验证集的 (dates, ret, p_best, p_fix) 存成 npz 前缀，'
                         '供 T094 条件混合分析用（逐日 IC 序列 / oracle 上界）')
    ap.add_argument('--allow-degenerate-downside-risk', action='store_true')
    ap.add_argument('--select-holdout', type=float, default=0.0,
                    help='T100：把验证折按**时间**切成「前段早停 / 后段报告」，比例=后段占比。'
                         '0=旧行为（早停与报告同一集合）。做树的 HPO 横比时必须设 0.3~0.5，'
                         '否则测的是「哪组超参更容易撞到验证高点」。'
                         '注意树的选型偏差方向与 NAM 不同：lgb 的 ndcg@20 早停在 109~161 棵树，'
                         '远没到 IC 最优点，实测偏差是**负**的 −0.003')
    ap.add_argument('--topk', type=int, default=20,
                    help='头部能力指标的档位 K，默认 20 = 生产持仓数。'
                         'ndcg 的头部截断与特有优化让整截面 IC 低估 lambdarank，'
                         '所以 HPO 主判据用逐日 top-K 超额，IC 只作副证')
    ap.add_argument('--params', default='',
                    help='超参覆盖，JSON 字典，如 \'{"learning_rate":0.03,"max_depth":8}\'。'
                         'lgb 的 ndcg 截断用 {"eval_at":[40]}')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    end_dt = datetime.strptime(args.end, '%Y-%m-%d')
    end = end_dt.strftime('%Y-%m-%d')
    start = (end_dt - timedelta(days=365 * args.years)).strftime('%Y-%m-%d')
    started = time.time()

    TrainingConfig.FUTURE_DAYS = 7
    np.random.seed(args.seed)
    trainer = MLModelTrainer(db_path=DATABASE_PATH, cache_dir=args.cache_dir)
    manager = BaostockDataManager()
    codes = manager.get_stock_list_from_db()['code'].tolist()[:args.stocks]
    manager.close()
    stocks_data = trainer.load_label_data(codes, start, end)
    dataset = trainer.prepare_dataset(
        stocks_data, train_start_date=start, train_end_date=end,
        include_fundamentals=TrainingConfig.INCLUDE_FUNDAMENTALS,
        n_jobs=args.workers, target_features=None, use_factor_cache_only=True,
    )
    all_features = list(dataset[3])

    if 'downside_risk' in all_features and not args.allow_degenerate_downside_risk:
        col = np.asarray(dataset[0][:, all_features.index('downside_risk')], dtype=float)
        fin = col[np.isfinite(col)]
        if len(fin) and (np.mean(np.abs(fin) > 1e-12) < 0.01 or np.std(fin) < 1e-8):
            raise RuntimeError('downside_risk 缓存已退化；拒绝训练')

    # 特征集：默认与 NAM 完全一致（剔手工交互列 + 同样的族裁剪），保证单变量可比；
    # --full-panel 则让树吃满全部 231 列 —— 手工 *_regime_* 交互本来是**为 NAM 造的**
    # （它严格可加、学不了交互），砍掉它们对能自学交互的树是双重惩罚。
    # 这一档回答的是「各自满配时哪个模型更强」，是选型该看的读数。
    if args.full_panel:
        feats = list(all_features)
    else:
        feats = [f for f in all_features if '_regime_' not in f]
    group_names, group_ids = build_group_index(feats)
    if args.drop_groups:
        drop = {g.strip() for g in args.drop_groups.split(',') if g.strip()}
        unknown = drop - set(group_names)
        if unknown:
            raise ValueError(f'未知因子族 {sorted(unknown)}；可选 {group_names}')
        feats = [f for f, g in zip(feats, group_ids) if group_names[int(g)] not in drop]
        group_names, group_ids = build_group_index(feats)
    print(f'  特征 {len(feats)} / 族 {len(group_names)}  样本 {len(dataset[0])}')

    regime = build_regime_matrix(DATABASE_PATH)
    n_bins = ModelConfig.get_n_bins()
    overrides = json.loads(args.params) if args.params.strip() else {}
    if overrides:
        print(f'  超参覆盖: {overrides}')
    results = {}
    for chunk in args.folds.split(','):
        tf, ve = (float(x) for x in chunk.split(':'))
        tag = f'{int(tf*100)}-{int(ve*100)}%'
        print(f"\n{'='*60}\n=== 折 {tag} · {args.model} ===\n{'='*60}", flush=True)
        fold = _prepare_fold(trainer, dataset, feats, tf, ve, regime, target='returns')
        t0 = time.time()
        sel, rep = _sel_rep(fold['d_val'], args.select_holdout)
        if args.select_holdout > 0:
            print(f'  选型/报告分离: 早停 {len(sel)} 行 → 报告 {len(rep)} 行'
                  f'（holdout={args.select_holdout}）', flush=True)
        if args.model == 'xgboost':
            p_best, p_fix, best = _fit_xgb(fold, args.seed, args.fixed_iter, sel, overrides)
        else:
            p_best, p_fix, best = _fit_lgb(fold, args.seed, args.fixed_iter, n_bins, sel, overrides)
        d_val = np.asarray(fold['d_val'])
        row = _daily_metrics(p_best, fold['ret_val'], d_val, topk=args.topk)
        row_fix = _daily_metrics(p_fix, fold['ret_val'], d_val, topk=args.topk)
        row['rank_ic_fixed'] = row_fix['rank_ic']
        row['best_iteration'] = best
        row['fixed_iteration'] = args.fixed_iter
        row['features_used'] = len(feats)
        row['groups'] = len(group_names)
        row['regime'] = _regime_stratified_metrics(p_best, fold['ret_val'], d_val)
        if args.select_holdout > 0:
            # 无偏读数：报告段对早停完全不可见。HPO 横比只能看这两栏。
            m_rep = _daily_metrics(p_best[rep], np.asarray(fold['ret_val'])[rep],
                                   d_val[rep], topk=args.topk)
            row['rank_ic_holdout'] = m_rep['rank_ic']
            row[f'top{args.topk}_excess_holdout'] = m_rep[f'top{args.topk}_excess']
            row['holdout_days'] = m_rep['days']
            row['regime_holdout'] = _regime_stratified_metrics(
                p_best[rep], np.asarray(fold['ret_val'])[rep], d_val[rep])
        if args.save_preds:
            np.savez_compressed(
                f"{args.save_preds}_{tag.replace('%','')}.npz",
                dates=d_val.astype('U10'), ret=np.asarray(fold['ret_val'], dtype=np.float64),
                p_best=np.asarray(p_best, dtype=np.float64),
                p_fix=np.asarray(p_fix, dtype=np.float64))
        results[tag] = {'nam_gate': row, 'split_date': fold['split_date'],
                        'validation_end_date_exclusive': fold['end_date']}
        print(f"  {args.model}: rank_ic={row['rank_ic']:.5f} (best_iter={best})  "
              f"fixed@{args.fixed_iter}={row['rank_ic_fixed']:.5f}  "
              + (f"holdout_ic={row['rank_ic_holdout']:.5f} "
                 f"holdout_top{args.topk}={row[f'top{args.topk}_excess_holdout']:.5f}  "
                 if args.select_holdout > 0 else '')
              + f"{time.time()-t0:.0f}s", flush=True)

    payload = {'metadata': {'model': args.model, 'seed': args.seed, 'stocks': args.stocks,
                            'years': args.years, 'end': end, 'cache_dir': args.cache_dir,
                            'drop_groups': args.drop_groups, 'fixed_iter': args.fixed_iter,
                            'select_holdout': args.select_holdout, 'topk': args.topk,
                            'params_override': overrides,
                            'elapsed_sec': round(time.time() - started, 1)},
               'folds': results}
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
    ic = np.mean([r['nam_gate']['rank_ic'] for r in results.values()])
    icf = np.mean([r['nam_gate']['rank_ic_fixed'] for r in results.values()])
    print(f"\n已保存 {args.output}  折均 rank_ic={ic:.5f}  折均 fixed={icf:.5f}  "
          f"(耗时 {time.time()-started:.1f}s)")


if __name__ == '__main__':
    main()
