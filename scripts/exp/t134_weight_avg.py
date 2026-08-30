# -*- coding: utf-8 -*-
"""
T134 多种子权重平均（weight averaging，phase1 训练策略层）
==========================================================
4 个 base 种子（s42/s11/s23/s37）的 NAM 模型 state_dict 逐参数平均 → 1 个模型。

动机：T131 投票（选股层离散共识）已证伪；权重平均是**参数层连续平均**——
不同种子在等价解空间里采样，平均 = 取解空间中心，理论上保留全部信息、
消除种子抖动，且 NAM 的加性结构（每因子独立 expert）没有 MLP 的隐层排列
问题，平均比通用网络更安全。

不碰特征/标签/架构，零训练成本。输出 models/nam_gate/T134_wavg4/。
用法：python scripts/exp/t134_weight_avg.py [--seeds 42,11,23,37] [--out models/nam_gate/T134_wavg4]
"""
import argparse
import os
import pickle
import shutil
import sys
import numpy as np

ROOT = r"G:/ai_proj/quantified_decision"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from core.factors.nam_gate_model import NAMGateModel  # noqa: E402

BASE_DIR = os.path.join(ROOT, "models", "nam_gate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,11,23,37")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "T134_wavg4"))
    args = ap.parse_args()
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]

    models = []
    for s in seeds:
        p = os.path.join(BASE_DIR, f"sweepA_base_s{s}", "nam_gate_factor_model.pkl")
        m = NAMGateModel()
        m.load_model(p)
        if m.pca_W is not None:
            raise SystemExit(f"T134 不适用于 PCA 模型: {p}")
        models.append(m)
        print(f"  [load] seed={s}: feature_names={len(m.feature_names)}")

    # 校验一致性
    fns = [tuple(m.feature_names) for m in models]
    if len(set(fns)) != 1:
        raise SystemExit("种子间 feature_names 不一致，拒绝平均")
    sds = [m.net.state_dict() for m in models if m.net is not None]
    if len(sds) != len(models):
        raise SystemExit("存在未训练模型")

    # state_dict 逐参数平均
    import torch
    avg_sd = {}
    keys = sds[0].keys()
    for k in keys:
        stacked = np.stack([sd[k].detach().cpu().numpy().astype(np.float64)
                            for sd in sds])
        avg_sd[k] = torch.from_numpy(stacked.mean(axis=0).astype(np.float32))

    # 非网络参数取平均（input_mean/std 各种子统计应几乎一致；feature_importance 平均）
    base = models[0]
    base.input_mean = np.stack([m.input_mean for m in models]).mean(axis=0).astype(np.float32)
    base.input_std = np.stack([m.input_std for m in models]).mean(axis=0).astype(np.float32)
    base.feature_importance = {
        f: float(np.mean([m.feature_importance.get(f, 0.0) for m in models]))
        for f in base.feature_names
    }
    base.net.load_state_dict(avg_sd)
    base.is_trained = True
    base.net.eval()

    os.makedirs(args.out, exist_ok=True)
    mp = os.path.join(args.out, "nam_gate_factor_model.pkl")
    base.save_model(mp)
    # norm_stats 随种子相同（共享面板），复制第一个的
    src_ns = os.path.join(BASE_DIR, f"sweepA_base_s{seeds[0]}", "norm_stats.pkl")
    if os.path.exists(src_ns):
        shutil.copy2(src_ns, os.path.join(args.out, "norm_stats.pkl"))
    # sidecar
    import json
    with open(os.path.join(args.out, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(base.feature_names, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "factor_groups.json"), "w", encoding="utf-8") as f:
        json.dump({g: [] for g in base.group_names}, f, ensure_ascii=False, indent=2)

    # 验证：加载回来 predict 冒烟
    m2 = NAMGateModel()
    m2.load_model(mp)
    rng = np.random.default_rng(0)
    X = rng.random((20, len(m2.feature_names))).astype(np.float32)
    import pandas as pd
    df = pd.DataFrame(X, columns=m2.feature_names)
    pred = m2.predict(df)
    print(f"\n[OK] 权重平均模型已保存: {mp}")
    print(f"  predict 冒烟: shape={pred.shape} range=[{pred.min():.4f}, {pred.max():.4f}]")
    print(f"  参与种子: {seeds}")


if __name__ == "__main__":
    main()
