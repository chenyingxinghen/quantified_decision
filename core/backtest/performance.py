"""
性能分析器

计算回测性能指标和生成报告
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from dataclasses import asdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


class PerformanceAnalyzer:
    """性能分析器"""
    
    @staticmethod
    def calculate_metrics(trades: List,
                         initial_capital: float,
                         final_capital: float,
                         equity_curve: List = None) -> Dict:
        """
        计算性能指标
        
        参数:
            trades: 交易列表
            initial_capital: 初始资金
            final_capital: 最终资金
            equity_curve: 资金曲线
        
        返回:
            性能指标字典
        """
        if not trades:
            return PerformanceAnalyzer._empty_metrics()
        
        # 转换为DataFrame
        trades_df = pd.DataFrame([asdict(t) for t in trades])
        
        # 基础指标
        total_trades = len(trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        
        # 收益指标
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # 胜率
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 平均收益
        avg_return_pct = trades_df['pnl_pct'].mean() * 100
        avg_win_pct = trades_df[trades_df['pnl'] > 0]['pnl_pct'].mean() * 100 if winning_trades > 0 else 0
        avg_loss_pct = trades_df[trades_df['pnl'] <= 0]['pnl_pct'].mean() * 100 if losing_trades > 0 else 0
        
        # 盈亏比
        total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        total_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # 最大回撤
        if equity_curve and len(equity_curve) > 0:
            equity_series = pd.Series([e[1] for e in equity_curve])
            max_drawdown = PerformanceAnalyzer._calculate_max_drawdown(equity_series)
        else:
            cumulative_returns = (1 + trades_df['pnl_pct']).cumprod()
            max_drawdown = PerformanceAnalyzer._calculate_max_drawdown(cumulative_returns) * 100
        
        # 夏普比率
        returns = trades_df['pnl_pct']
        sharpe_ratio = PerformanceAnalyzer._calculate_sharpe_ratio(returns)
        
        # 持仓天数统计
        avg_holding_days = trades_df['holding_days'].mean()
        
        # 退出原因统计
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        # 最佳/最差交易
        best_trade_idx = trades_df['pnl_pct'].idxmax()
        worst_trade_idx = trades_df['pnl_pct'].idxmin()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'avg_return_pct': avg_return_pct,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_holding_days': avg_holding_days,
            'exit_reasons': exit_reasons,
            'best_trade': trades_df.loc[best_trade_idx].to_dict(),
            'worst_trade': trades_df.loc[worst_trade_idx].to_dict()
        }
    
    @staticmethod
    def _empty_metrics() -> Dict:
        """返回空指标"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'total_return_pct': 0,
            'avg_return_pct': 0,
            'avg_win_pct': 0,
            'avg_loss_pct': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'avg_holding_days': 0,
            'exit_reasons': {},
            'best_trade': None,
            'worst_trade': None
        }
    
    @staticmethod
    def _calculate_max_drawdown(equity_series: pd.Series) -> float:
        """计算最大回撤"""
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        return drawdown.min() * 100
    
    @staticmethod
    def _calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0) -> float:
        """计算夏普比率"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        excess_returns = returns - risk_free_rate
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)
    
    @staticmethod
    def print_summary(metrics: Dict, 
                     start_date: str,
                     end_date: str,
                     strategy_name: str = "策略"):
        """
        打印回测摘要
        
        参数:
            metrics: 性能指标
            start_date: 开始日期
            end_date: 结束日期
            strategy_name: 策略名称
        """
        print("\n" + "=" * 80)
        print(f"{strategy_name} - 回测结果")
        print("=" * 80)
        print(f"回测期间: {start_date} 至 {end_date}")
        
        print("\n【交易统计】")
        print(f"  总交易次数: {metrics['total_trades']}")
        print(f"  盈利交易: {metrics['winning_trades']} ({metrics['win_rate']:.2f}%)")
        print(f"  亏损交易: {metrics['losing_trades']}")
        print(f"  平均持仓天数: {metrics['avg_holding_days']:.1f}")
        
        print("\n【收益指标】")
        print(f"  总收益: {metrics['total_return']:.4f}")
        print(f"  总收益率: {metrics['total_return_pct']:.2f}%")
        print(f"  平均收益率: {metrics['avg_return_pct']:.2f}%")
        print(f"  平均盈利: {metrics['avg_win_pct']:.2f}%")
        print(f"  平均亏损: {metrics['avg_loss_pct']:.2f}%")
        
        print("\n【风险指标】")
        print(f"  盈亏比: {metrics['profit_factor']:.2f}")
        print(f"  最大回撤: {metrics['max_drawdown']:.2f}%")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
        
        if metrics.get('exit_reasons'):
            print("\n【退出原因】")
            for reason, count in metrics['exit_reasons'].items():
                pct = (count / metrics['total_trades']) * 100
                print(f"  {reason}: {count} ({pct:.1f}%)")
        
        if metrics.get('best_trade'):
            print("\n【最佳交易】")
            best = metrics['best_trade']
            print(f"  股票: {best['stock_code']}")
            print(f"  收益率: {best['pnl_pct']*100:.2f}%")
            print(f"  持仓: {best['buy_date']} -> {best['sell_date']}")
        
        if metrics.get('worst_trade'):
            print("\n【最差交易】")
            worst = metrics['worst_trade']
            print(f"  股票: {worst['stock_code']}")
            print(f"  收益率: {worst['pnl_pct']*100:.2f}%")
            print(f"  持仓: {worst['buy_date']} -> {worst['sell_date']}")
        
        print("\n" + "=" * 80)
    
    @staticmethod
    def save_trades_to_csv(trades: List, filepath: str):
        """保存交易记录到CSV"""
        if not trades:
            print("警告: 无交易记录")
            return
        
        df = pd.DataFrame([asdict(t) for t in trades])
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        # print(f"交易记录已保存: {filepath}")
    
    @staticmethod
    def save_equity_curve(equity_curve: List, filepath: str):
        """保存资金曲线到CSV"""
        if not equity_curve:
            print("警告: 无资金曲线数据")
            return
        
        df = pd.DataFrame(equity_curve, columns=['date', 'equity'])
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"资金曲线已保存: {filepath}")

    @staticmethod
    def plot_equity_curve(equity_curve: List, title: str = "Backtest Equity Curve", save_path: str = None):
        """绘制资金曲线"""

        
        if not equity_curve:
            print("警告: 无资金曲线数据，无法绘制")
            return
            
        # 提取数据
        dates = [pd.to_datetime(e[0]) for e in equity_curve]
        values = [e[1] for e in equity_curve]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, values, label='Portfolio Value', linewidth=2)
        
        # 格式化图表
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Equity', fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        
        # 格式化x轴日期
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            # print(f"资金曲线图已保存: {save_path}")
        else:
            plt.show()
        plt.close()

    @staticmethod
    def plot_confidence_performance(trades: List, title: str = "Confidence vs. Return", save_path: str = None):
        """
        绘制置信度与收益率的关系曲线
        
        参数:
            trades: 交易记录列表
            title: 图表标题
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if not trades:
            print("警告: 无交易记录，无法绘制置信度分析图")
            return
            
        # 提取数据
        data = []
        for t in trades:
            # Trade 是 dataclass，可以用 asdict 转换，或者直接访问其 metadata
            # 这里 trades 列表在 calculate_metrics 中已经被转换成 DataFrame 过了
            # 但传入这个函数的通常是原始的 Trade 对象列表
            try:
                # 尝试从 metadata 中获取置信度
                # 兼容 Trade 对象和字典
                if hasattr(t, 'metadata'):
                    confidence = t.metadata.get('confidence')
                    pnl_pct = t.pnl_pct * 100
                else:
                    confidence = t.get('metadata', {}).get('confidence')
                    pnl_pct = t.get('pnl_pct', 0) * 100
                    
                if confidence is not None:
                    data.append({
                        'confidence': confidence,
                        'pnl_pct': pnl_pct
                    })
            except Exception as e:
                continue
        
        if not data:
            print("警告: 交易记录中无置信度信息，无法绘制分析图")
            return
            
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(10, 6))
        
        # 散点图 + 回归趋势线
        sns.regplot(x='confidence', y='pnl_pct', data=df, 
                    scatter_kws={'alpha':0.5, 's': 30}, 
                    line_kws={'color':'red', 'label': 'Trend Line'})
        
        # 计算各种置信度区间的胜率
        correlation = df['confidence'].corr(df['pnl_pct'])
        
        plt.title(f'{title}\n(Correlation: {correlation:.2f})', fontsize=14)
        plt.xlabel('Prediction Confidence (%)', fontsize=12)
        plt.ylabel('Trade Return (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 添加一条 y=0 的水平线
        plt.axhline(0, color='black', linewidth=1, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            # print(f"置信度分析图已保存: {save_path}")
        else:
            plt.show()
        plt.close()

