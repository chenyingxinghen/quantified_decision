"""把 NAM 标量门控的敏感度向量整体缩放 ``a -> λ·a``，写出一个新模型目录。

λ=0 时 logits 全零 → ``softmax`` 给均匀权重，严格等价于 ``--disable-gate`` 基线；
λ=1 保持原模型。因此这是一条从基线到 T073 的连续插值，且**不需要重新训练**。

同时复制 `norm_stats.pkl` 等 sidecar（T043 铁律：缺它回测会以原始量纲喂特征），
并复制因子缓存绑定清单（T052 铁律）。

用法:
  python scripts/archive/oneoff/scale_gate_sensitivity.py --src models/nam_gate/T073_e6_scalar_s42 \
      --dst models/nam_gate/T074_e6_lam0p5_s42 --lam 0.5
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PKL = 'nam_gate_factor_model.pkl'
SIDECARS = ['norm_stats.pkl', 'feature_names.json', 'factor_groups.json',
            'regime_cols.json', 'cache_manifest.json', 'model_metadata.json']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--lam', type=float, required=True)
    args = ap.parse_args()

    import torch  # noqa: F401  (加载 state_dict 需要)
    from core.factors.nam_gate_model import NAMGateModel

    src_pkl = os.path.join(args.src, PKL)
    if not os.path.isfile(src_pkl):
        raise SystemExit(f'源模型不存在: {src_pkl}')

    m = NAMGateModel(device='cpu').load_model(src_pkl)
    if m.gate_mode != 'scalar':
        raise SystemExit(f"只支持 gate_mode='scalar'，实得 {m.gate_mode}")

    a = m.net.gate.a.detach().clone()
    with torch.no_grad():
        m.net.gate.a.copy_(a * float(args.lam))
    print(f'  a: |max| {float(a.abs().max()):.3f} -> '
          f'{float(m.net.gate.a.abs().max()):.3f}  (λ={args.lam})')

    os.makedirs(args.dst, exist_ok=True)
    m.save_model(os.path.join(args.dst, PKL))
    for name in SIDECARS:
        p = os.path.join(args.src, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(args.dst, name))
    if not os.path.isfile(os.path.join(args.dst, 'norm_stats.pkl')):
        raise SystemExit('缺 norm_stats.pkl，拒绝产出（T043 铁律）')
    print(f'已写出: {args.dst}')


if __name__ == '__main__':
    main()
