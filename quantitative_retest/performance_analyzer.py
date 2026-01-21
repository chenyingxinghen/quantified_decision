# 性能分析器 - 计算回测性能指标和生成可视化

import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class PerformanceAnalyzer:
    """性能分析器"""
    
    def calculate_metrics(self, trades: List, initial_capital: float, final_capital: float) -> Dict:
        """
        计算性能指标
        
        返回:
            dict: 包含总收益率、胜率、交易次数等指标
        """
        if not trades:
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0,
                'avg_return': 0.0,
                'max_return': 0.0,
                'max_drawdown': 0.0
            }
        
        # 总收益率
        total_return = (final_capital - initial_capital) / initial_capital
        
        # 统计盈利和亏损交易
        winning_trades = [t for t in trades if t.profit > 0]
        losing_trades = [t for t in trades if t.profit <= 0]
        
        # 胜率
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        
        # 平均盈利和亏损
        avg_profit = sum(t.profit for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(t.profit for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
        
        # 最大盈利和亏损
        max_profit = max((t.profit for t in trades), default=0.0)
        max_loss = min((t.profit for t in trades), default=0.0)
        
        # 平均收益率
        avg_return = sum(t.return_rate for t in trades) / len(trades) if trades else 0.0
        
        # 最大收益率和最大亏损率
        max_return = max((t.return_rate for t in trades), default=0.0)
        max_drawdown = min((t.return_rate for t in trades), default=0.0)
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'avg_return': avg_return,
            'max_return': max_return,
            'max_drawdown': max_drawdown
        }
    
    def plot_equity_curve(self, equity_curve: List[Tuple[str, float]], save_path: str):
        """
        绘制收益率曲线
        
        参数:
            equity_curve: [(date, capital), ...] 资金曲线数据
            save_path: 图片保存路径
        """
        if not equity_curve:
            print("警告: 资金曲线数据为空，无法绘制")
            return
        
        # 提取日期和资金
        dates = [datetime.strptime(d, '%Y-%m-%d') for d, _ in equity_curve]
        capitals = [c for _, c in equity_curve]
        
        # 计算累计收益率
        initial_capital = capitals[0]
        returns = [(c - initial_capital) / initial_capital * 100 for c in capitals]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 子图1: 资金曲线
        ax1.plot(dates, capitals, linewidth=2, color='#2E86AB', label='资金曲线')
        ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='初始资金')
        ax1.set_xlabel('日期', fontsize=12)
        ax1.set_ylabel('资金', fontsize=12)
        ax1.set_title('资金曲线', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # 子图2: 累计收益率曲线
        ax2.plot(dates, returns, linewidth=2, color='#A23B72', label='累计收益率')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.fill_between(dates, returns, 0, alpha=0.3, color='#A23B72')
        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_ylabel('累计收益率 (%)', fontsize=12)
        ax2.set_title('累计收益率曲线', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 收益率曲线已保存到: {save_path}")
        
        plt.close()


class ReportGenerator:
    """报告生成器"""
    
    def print_summary(self, metrics: Dict, start_date: str, end_date: str, strategy_name: str):
        """在控制台输出回测摘要"""
        print("\n" + "=" * 80)
        print("回测摘要报告")
        print("=" * 80)
        print(f"回测时间范围: {start_date} 至 {end_date}")
        print(f"策略名称: {strategy_name}")
        print("-" * 80)
        print(f"总收益率: {metrics['total_return']*100:.2f}%")
        print(f"胜率: {metrics['win_rate']*100:.2f}%")
        print(f"总交易次数: {metrics['total_trades']}")
        print(f"  • 盈利交易: {metrics['winning_trades']} 笔")
        print(f"  • 亏损交易: {metrics['losing_trades']} 笔")
        print("-" * 80)
        print(f"平均盈利: {metrics['avg_profit']:.4f} ({metrics['avg_return']*100:.2f}%)")
        print(f"平均亏损: {metrics['avg_loss']:.4f}")
        print(f"最大盈利: {metrics['max_profit']:.4f} ({metrics['max_return']*100:.2f}%)")
        print(f"最大亏损: {metrics['max_loss']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
        print("=" * 80)
    
    def save_trades_to_csv(self, trades: List, file_path: str):
        """将交易记录保存为CSV文件"""
        if not trades:
            print("警告: 没有交易记录，无法保存CSV")
            return
        
        # 转换为DataFrame
        trade_data = []
        for trade in trades:
            trade_data.append({
                '股票代码': trade.stock_code,
                '买入日期': trade.buy_date,
                '买入价': f"{trade.buy_price:.2f}",
                '卖出日期': trade.sell_date,
                '卖出价': f"{trade.sell_price:.2f}",
                '股数': f"{trade.shares:.2f}",
                '买入手续费': f"{trade.buy_commission:.4f}",
                '卖出手续费': f"{trade.sell_commission:.4f}",
                '净利润': f"{trade.profit:.4f}",
                '收益率': f"{trade.return_rate*100:.2f}%",
                '卖出原因': trade.exit_reason
            })
        
        df = pd.DataFrame(trade_data)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"✓ 交易记录已保存到: {file_path}")
