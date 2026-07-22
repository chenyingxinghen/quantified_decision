"""
直接预测最佳股票组合（端到端入口）

与用户的诉求一致：不再只是"预测谁收益高就选谁"，而是加载已训练的多目标
神经网络模型，复用与回测完全一致的因子/选股逻辑，通过 PortfolioOptimizer
直接产出"平衡回撤/波动与最终收益"的带权组合（可直接用于资金分配）。

实现上复用 core.backtest.strategies.ml_factor_strategy 的 select_for_live，
它内部已经串好了：
    因子缓存 -> 横截面归一化 -> 多目标神经网络打分 -> PortfolioOptimizer 求解权重
因此本脚本与回测给出的组合完全一致，避免了"脚本里一套、回测里另一套"。

用法:
    python scripts/select_neural_portfolio.py \
        --model models/neural_multi_objective_20260722_222627 \
        --top-n 20 \
        --portfolio-method max_sharpe \
        --max-weight 0.2 \
        --candidate-pool 30 \
        --lookback-days 120 \
        --drawdown-penalty 0.3

输出: 按权重降序的组合清单（代码 / 权重 / 模型分数 / 现价 / 止损 / 止盈），
以及组合层面的预期收益、波动等诊断指标。

说明:
    --top-n          候选池上限（传给策略的 available_slots；优化器会在此范围内选股）
    --candidate-pool 进入优化器的候选股票数量（默认 30，越大越能发挥分散化）
    --portfolio-method  max_sharpe | mean_variance | risk_parity
    --drawdown-penalty  预测回撤惩罚系数（>0 时，模型预测回撤越大的票权重越低）
"""
import sys
import os
import argparse
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.jydb_config import DATABASE_PATH
from config import neural_config as nc
from core.backtest.strategies.ml_factor_strategy import MLFactorBacktestStrategy


def _print_portfolio(results, method, drawdown_penalty):
    """把组合结果以表格形式打印出来。"""
    if not results:
        print("\n[结果] 未选出任何股票（模型分数全部低于置信度阈值）。")
        return

    total_weight = sum(float(r.get("weight") or 0.0) for r in results)
    print("\n" + "=" * 78)
    print(f"  最佳组合（方法={method}, 回撤惩罚={drawdown_penalty}）  "
          f"持仓 {len(results)} 只, 权重和={total_weight:.4f}")
    print("=" * 78)
    header = f"{'排名':<4} {'代码':<10} {'权重':>8} {'模型分':>8} " \
             f"{'现价':>10} {'止损':>10} {'止盈':>10}"
    print(header)
    print("-" * 78)
    for i, r in enumerate(results, 1):
        w = float(r.get("weight") or 0.0)
        conf = float(r.get("confidence") or 0.0)
        price = float(r.get("current_price") or 0.0)
        sl = float(r.get("stop_loss") or 0.0)
        tp = float(r.get("take_profit") or 0.0)
        print(f"{i:<4} {r['stock_code']:<10} {w*100:>7.2f}% {conf:>8.2f} "
              f"{price:>10.2f} {sl:>10.2f} {tp:>10.2f}")
    print("-" * 78)
    print("提示: 权重为组合内相对占比（已归一化，和为 1）。可直接作为资金分配比例。")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="量化决策 - 神经网络直接预测最佳股票组合（端到端）"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="已训练的多目标神经网络模型路径（.pkl 文件或含模型的目录）",
    )
    parser.add_argument("--top-n", type=int, default=20,
                        help="候选池上限 / 目标持仓数量 (默认 20)")
    parser.add_argument("--portfolio-method", type=str, default="max_sharpe",
                        choices=["max_sharpe", "mean_variance", "risk_parity"],
                        help="组合优化目标 (默认 max_sharpe)")
    parser.add_argument("--max-weight", type=float, default=0.2,
                        help="单票权重上限，分散化用 (默认 0.2)")
    parser.add_argument("--min-weight", type=float, default=0.0,
                        help="单票权重下限 (默认 0.0)")
    parser.add_argument("--risk-aversion", type=float, default=1.0,
                        help="mean_variance 模式下的风险厌恶系数 (默认 1.0)")
    parser.add_argument("--drawdown-penalty", type=float, default=0.0,
                        help="预测回撤惩罚系数，>0 时回撤大的票权重降低 (默认 0.0)")
    parser.add_argument("--candidate-pool", type=int, default=30,
                        help="进入优化器的候选股票数 (默认 30)")
    parser.add_argument("--lookback-days", type=int, default=120,
                        help="估计协方差用的历史收益回看天数 (默认 120)")
    parser.add_argument("--score-to-return-scale", type=float, default=0.8,
                        help="模型分数映射到期望收益的尺度 (默认 0.8)")
    parser.add_argument("--shrinkage", type=float, default=0.1,
                        help="协方差收缩系数 Ledoit-Wolf 风格 (默认 0.1)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="模型分数(百分比)最低门槛 (默认 0.0)")
    parser.add_argument("--norm-stats", type=str, default=None,
                        help="归一化统计量路径（默认随模型目录查找 norm_stats.pkl）")
    parser.add_argument("--db-path", type=str, default=DATABASE_PATH,
                        help="行情数据库路径")
    parser.add_argument("--as-of", type=str, default=None,
                        help="选股基准日 YYYY-MM-DD（默认=数据库最新交易日，"
                             "适配静态/快照库）")
    args = parser.parse_args()

    # 组装组合优化配置，传给策略内部的 PortfolioOptimizer
    portfolio_config = {
        "method": args.portfolio_method,
        "max_weight": args.max_weight,
        "min_weight": args.min_weight,
        "risk_aversion": args.risk_aversion,
        "drawdown_penalty": args.drawdown_penalty,
        "candidate_pool": max(args.candidate_pool, args.top_n),
        "lookback_days": args.lookback_days,
        "score_to_return_scale": args.score_to_return_scale,
        "shrinkage": args.shrinkage,
    }

    print("=== 神经网络直接预测最佳组合 ===")
    print(f"模型: {args.model}")
    print(f"方法: {args.portfolio_method}  候选池: {portfolio_config['candidate_pool']}  "
          f"单票上限: {args.max_weight}  回撤惩罚: {args.drawdown_penalty}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 复用回测策略（含完全相同的因子归一化与打分逻辑）
    strategy = MLFactorBacktestStrategy(
        model_path=args.model,
        min_confidence=args.min_confidence,
        use_portfolio_optimizer=True,
        portfolio_config=portfolio_config,
        norm_stats_path=args.norm_stats,
        db_path=args.db_path,
        name="神经网络组合预测",
    )
    # initialize 会加载模型 + norm_stats + 预加载因子缓存（与回测一致）
    strategy.initialize()
    if strategy.model is None:
        print("[错误] 模型加载失败，请检查 --model 路径。")
        sys.exit(1)

    results = strategy.select_for_live(
        db_path=args.db_path,
        top_n=args.top_n,
        lookback_days=args.lookback_days,
        as_of=args.as_of,
    )

    _print_portfolio(results, args.portfolio_method, args.drawdown_penalty)
    strategy.cleanup()


if __name__ == "__main__":
    main()
