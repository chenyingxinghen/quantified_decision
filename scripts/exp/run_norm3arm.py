# -*- coding: utf-8 -*-
"""三臂归一化对照 driver（2026-08-25，T139 后续）：
A 全 rank（现状基线）/ B 全 robust-sigmoid（验证 rank 是否多余）/
C 双通道（rank + __z 幅值旁路）。

800×3y s42，T045 同配置，牛熊双窗回测 + 随机零假设 2000 sims。
"""
import os
import sys
import subprocess
import time

ROOT = r"G:/ai_proj/quantified_decision"
PY = r"C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
os.chdir(ROOT)

TRAIN = os.path.join(ROOT, 'scripts', 'train_nam_model.py')
BT = os.path.join(ROOT, 'scripts', 'exp', 'run_backtest_batch.py')
COMMON = ['--stocks', '800', '--years', '3', '--end', '2022-09-05',
          '--disable-gate', '--target', 'returns', '--y-scale', '2',
          '--lambda-lb', '0', '--epochs', '60', '--min-epochs', '20',
          '--skip-baseline', '--folds', '0.8:1.0', '--seed', '42']

ARMS = {
    'A_rank':    ['--normalize-mode', 'rank'],
    'B_sigmoid': ['--normalize-mode', 'sigmoid'],
    'C_dual':    ['--normalize-mode', 'dual'],
}


def run(cmd, log):
    with open(log, 'w', encoding='utf-8') as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode


def main():
    t0 = time.time()
    for name, extra in ARMS.items():
        out = f'diagnose_output/norm3arm_{name}_s42'
        model = f'models/nam_gate/norm3arm_{name}_s42'
        train_cmd = [PY, TRAIN] + COMMON + extra + \
            ['--save-model-dir', model, '--output', out, '--plot-dir', out]
        print(f'=== 训练 {name} ===')
        rc = run(train_cmd, f'diagnose_output/norm3arm_{name}_train.log')
        if rc != 0:
            print(f'  训练失败 rc={rc}，跳过回测')
            continue
        for win, s, e in [('bear', '2022-09-05', '2024-08-05'),
                          ('bull', '2024-08-05', '2026-08-05')]:
            bt_cmd = [PY, BT, '--models', model, '--start', s, '--end', e,
                      '--tag', f'norm3arm_{name}', '--n-sims', '2000']
            print(f'=== 回测 {name} {win} ===')
            run(bt_cmd, f'diagnose_output/norm3arm_{name}_{win}_bt.log')
    print(f'\n=== 三臂对照完成，耗时 {(time.time()-t0)/60:.1f} min ===')


if __name__ == '__main__':
    main()
