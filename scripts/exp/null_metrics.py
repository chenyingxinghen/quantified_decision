"""随机零假设结果的效应量口径（共享）。

为什么不能只看 percentile
--------------------------
E1b 的 10 种子分布里，熊市分位中位是 **97.2**，到上界 100 只剩 2.8pp。
也就是说在主判窗口上，分位这个统计量**已经触顶**：它还能测出劣化，
但基本测不出改进（任何真实增益都会被压进 97~100 的窄缝里）。
E1b 里"熊市只有 3 个候选可测量且全是劣化"正是这个天花板造成的假象。

因此判定必须同时看三个口径：

| 口径 | 定义 | 性质 |
| --- | --- | --- |
| `pct`  | 实际收益在随机分布中的经验分位 | 有界 0~100，熊市已触顶，只保留作历史可比 |
| `z`    | (实际 − 随机均值) / 随机std | **无上界**，主效应量 |
| `excess` | 实际 − 随机均值（pp） | 无上界，经济口径，但没除掉窗口波动 |

10 种子零效应分布（E1b 样本）：

| 窗口 | pct 中位/std | z 中位/std | excess 中位/std |
| --- | --- | --- | --- |
| 牛 | 71.9 / 27.7 | 0.631 / 0.890 | +14.24 / 19.25 |
| 熊 | 97.2 / 16.3 | 1.951 / 1.264 | +23.53 / 15.71 |
"""

from __future__ import annotations

import json
import os

METRICS = ('pct', 'z', 'excess')
METRIC_LABEL = {'pct': '分位', 'z': 'z值', 'excess': '超额pp'}


def read_null(path):
    """读取 diag_random_null.py 的输出，返回三个口径的效应量。"""
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return {
        'pct': float(d['percentile_of_actual']),
        'z': float(d['z_score']),
        'excess': float(d['actual_return_pct']) - float(d['null_mean_pct']),
        'ret': float(d['actual_return_pct']),
        'null_mean': float(d['null_mean_pct']),
        'null_std': float(d['null_std_pct']),
        'p': d.get('p_value_one_sided'),
        'n_sims': d.get('n_sims'),
    }


# E1b 实测的种子零效应离散度（10 种子，nost 口径，真实成本）
SIGMA_SEED = {
    'bull': {'pct': 27.73, 'z': 0.890, 'excess': 19.25,
             'median': {'pct': 71.92, 'z': 0.631, 'excess': 14.24}},
    'bear': {'pct': 16.32, 'z': 1.264, 'excess': 15.71,
             'median': {'pct': 97.20, 'z': 1.951, 'excess': 23.53}},
}
