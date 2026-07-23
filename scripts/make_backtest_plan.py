"""
给定多目标模型 pkl，自动推导「按目标固定持仓」回测计划字符串。

用法:
  python scripts/make_backtest_plan.py <model.pkl>
输出:
  rank_y_ret_5d:5,rank_y_ret_20d:20,...   (stdout)
"""
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neural.nn_models import MultiObjectiveNeuralModel
from core.factors.ml_factor_model import MultiObjectiveFactorModel


def load_model(path):
    try:
        return MultiObjectiveNeuralModel.load_model(path)
    except Exception:
        return MultiObjectiveFactorModel.load_model(path)


def hold_days(obj: str) -> int:
    """从目标名解析持仓天数: rank_y_ret_5d -> 5, rank_y_mdd_20d -> 20。"""
    m = re.search(r'_(\d+)d$', obj)
    return int(m.group(1)) if m else 20


def main():
    if len(sys.argv) < 2:
        print("用法: make_backtest_plan.py <model.pkl>", file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    model = load_model(path)
    objs = list(model.models.keys())
    plan = ",".join(f"{o}:{hold_days(o)}" for o in objs)
    print(plan)


if __name__ == "__main__":
    main()
