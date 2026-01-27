# 回测可视化工具 - 绘制K线图、交易点和趋势线
# 支持三种使用方式：
# 1. 从回测结果中读取交易记录并可视化
# 2. 单独使用，指定股票代码、买入点、卖出点进行可视化
# 3. 从CSV文件批量生成图表

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import sqlite3
import sys
import os
from typing import List, Dict, Optional, Tuple

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATABASE_PATH
from trend_line_analyzer import TrendLineAnalyzer
from data_fetcher import DataFetcher

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class BacktestVisualizer:
    """回测可视化工具"""
    
    def __init__(self, db_path: str = None, commission_rate: float = 0.01):
        """
        初始化可视化工具
        
        参数:
            db_path: 数据库路径，默认使用配置文件中的路径
            commission_rate: 手续费率，默认0.01 (1%)
        """
        if db_path is None:
            db_path = DATABASE_PATH
        
        # 确保数据库路径为绝对路径
        if not os.path.isabs(db_path):
            project_root = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(project_root, db_path)
        
        self.db_path = db_path
        self.commission_rate = commission_rate
        self.trend_analyzer = TrendLineAnalyzer()
        self.data_fetcher = DataFetcher()
    
    def load_stock_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从数据库加载股票数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        
        返回:
            DataFrame: 股票数据
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT code, date, open, high, low, close, volume, amount, turnover_rate
            FROM daily_data
            WHERE code = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
        '''
        
        df = pd.read_sql_query(query, conn, params=(stock_code, start_date, end_date))
        conn.close()
        
        return df

    
    def simulate_trade(
        self,
        stock_code: str,
        buy_date: str,
        buy_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
        max_holding_days: int = 60
    ) -> Dict:
        """
        模拟交易过程，计算卖出点
        
        参数:
            stock_code: 股票代码
            buy_date: 买入日期 (YYYY-MM-DD)
            buy_price: 买入价格，如果为None则使用当天开盘价
            stop_loss: 止损价，如果为None则自动计算
            target: 目标价，如果为None则自动计算
            max_holding_days: 最大持仓天数，默认60天
        
        返回:
            Dict: 交易结果
        """
        # 加载数据
        buy_date_obj = datetime.strptime(buy_date, '%Y-%m-%d')
        start_date = (buy_date_obj - timedelta(days=120)).strftime('%Y-%m-%d')
        end_date = (buy_date_obj + timedelta(days=max_holding_days + 30)).strftime('%Y-%m-%d')
        
        data = self.load_stock_data(stock_code, start_date, end_date)
        
        if data.empty:
            raise ValueError(f"未找到股票 {stock_code} 的数据")
        
        buy_row = data[data['date'] == buy_date]
        if buy_row.empty:
            raise ValueError(f"未找到买入日期 {buy_date} 的数据")
        
        buy_idx = buy_row.index[0]
        
        if buy_price is None:
            buy_price = data.loc[buy_idx, 'open']
        
        # 分析趋势线
        trend_data = data.loc[:buy_idx]
        trend_analysis = self.trend_analyzer.analyze(trend_data)
        
        # 自动计算止损和目标价
        if stop_loss is None:
            if trend_analysis['short_uptrend_line']['valid']:
                support_price = (trend_analysis['short_uptrend_line']['slope'] * buy_idx + 
                               trend_analysis['short_uptrend_line']['intercept'])
                buffer = trend_analysis.get('suggested_stop_buffer', 0.03)
                stop_loss = support_price * (1 - buffer)
            else:
                stop_loss = buy_price * 0.97
        
        if target is None:
            if trend_analysis['trend_strength'] >= 70:
                target = buy_price * 1.12
            elif trend_analysis['trend_strength'] >= 50:
                target = buy_price * 1.10
            else:
                target = buy_price * 1.08
        
        # 模拟持仓
        holding_days = 0
        max_profit_rate = 0.0
        
        sell_date = None
        sell_price = None
        exit_reason = None
        
        for i in range(buy_idx + 1, len(data)):
            current_date = data.loc[i, 'date']
            open_price = data.loc[i, 'open']
            high = data.loc[i, 'high']
            low = data.loc[i, 'low']
            close = data.loc[i, 'close']
            
            holding_days += 1
            max_profit_rate = max(max_profit_rate, (high - buy_price) / buy_price)
            
            # 只保留跌破趋势线卖出条件
            current_trend_data = data.loc[:i]
            if len(current_trend_data) >= 30:
                current_trend_analysis = self.trend_analyzer.analyze(current_trend_data)
                if current_trend_analysis['broken_support']:
                    sell_price = close
                    sell_date = current_date
                    exit_reason = '跌破支撑'
                    break
        
        if sell_date is None:
            sell_date = data.iloc[-1]['date']
            sell_price = data.iloc[-1]['close']
            exit_reason = '数据结束'
        
        profit_rate = (sell_price - buy_price) / buy_price
        
        return {
            'stock_code': stock_code,
            'buy_date': buy_date,
            'buy_price': buy_price,
            'sell_date': sell_date,
            'sell_price': sell_price,
            'exit_reason': exit_reason,
            'stop_loss': stop_loss,
            'target': target,
            'holding_days': holding_days,
            'profit_rate': profit_rate,
            'max_profit_rate': max_profit_rate
        }

    
    def plot_candlestick_with_trades(
        self,
        stock_code: str,
        buy_date: str,
        sell_date: Optional[str] = None,
        days_before: int = 60,
        days_after: int = 30,
        buy_price: Optional[float] = None,
        sell_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
        exit_reason: Optional[str] = None,
        save_path: Optional[str] = None,
        show_trendlines: bool = True,
        show_sell_trendlines: bool = True  # 默认显示卖出时的趋势线
    ):
        """绘制K线图、交易点和趋势线"""
        buy_date_obj = datetime.strptime(buy_date, '%Y-%m-%d')
        start_date = (buy_date_obj - timedelta(days=days_before + 30)).strftime('%Y-%m-%d')
        
        if sell_date:
            sell_date_obj = datetime.strptime(sell_date, '%Y-%m-%d')
            end_date = (sell_date_obj + timedelta(days=days_after)).strftime('%Y-%m-%d')
        else:
            end_date = (buy_date_obj + timedelta(days=days_after)).strftime('%Y-%m-%d')
        
        data = self.load_stock_data(stock_code, start_date, end_date)
        
        if data.empty:
            print(f"未找到股票 {stock_code} 的数据")
            return
        
        buy_row = data[data['date'] == buy_date]
        if buy_row.empty:
            print(f"未找到买入日期 {buy_date} 的数据")
            return
        
        actual_buy_price = buy_price if buy_price is not None else buy_row['open'].iloc[0]
        
        actual_sell_price = None
        sell_idx = None
        if sell_date:
            sell_row = data[data['date'] == sell_date]
            if not sell_row.empty:
                actual_sell_price = sell_price if sell_price is not None else sell_row['open'].iloc[0]
                sell_idx = sell_row.index[0]
        
        buy_idx = data[data['date'] == buy_date].index[0]
        buy_trend_data = data.loc[:buy_idx]
        
        buy_trend_analysis = None
        if show_trendlines and len(buy_trend_data) >= 30:
            buy_trend_analysis = self.trend_analyzer.analyze(buy_trend_data)
        
        # 分析卖出时的趋势线（使用卖出前的数据）
        sell_trend_analysis = None
        if show_sell_trendlines and sell_idx is not None and len(data.loc[:sell_idx]) >= 30:
            sell_trend_data = data.loc[:sell_idx]
            sell_trend_analysis = self.trend_analyzer.analyze(sell_trend_data)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        self._plot_candlestick(ax1, data, stock_code)
        
        # 绘制买入时的支撑线
        if show_trendlines and buy_trend_analysis:
            self._plot_trendlines(ax1, data, buy_trend_analysis, buy_idx, 
                                label_prefix='买入时', line_style='--', alpha=0.7)
        
        # 绘制卖出时的支撑线
        if show_sell_trendlines and sell_trend_analysis and sell_idx is not None:
            self._plot_trendlines(ax1, data, sell_trend_analysis, sell_idx, 
                                label_prefix='卖出时', line_style='-', alpha=0.6)
        
        self._plot_trade_points(
            ax1, data, buy_date, sell_date, 
            actual_buy_price, actual_sell_price,
            stop_loss, target, exit_reason
        )
        
        self._plot_volume(ax2, data)
        
        title = f"{stock_code} 交易分析"
        if sell_date and actual_sell_price:
            profit_rate = (actual_sell_price - actual_buy_price) / actual_buy_price * 100
            title += f" | 收益率: {profit_rate:.2f}%"
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_candlestick(self, ax, data: pd.DataFrame, stock_code: str):
        """绘制K线图"""
        dates = pd.to_datetime(data['date'])
        opens = data['open'].values
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        
        x = np.arange(len(data))
        
        for i in range(len(data)):
            color = 'red' if closes[i] >= opens[i] else 'green'
            
            ax.plot([x[i], x[i]], [lows[i], highs[i]], color=color, linewidth=0.8)
            
            height = abs(closes[i] - opens[i])
            bottom = min(opens[i], closes[i])
            
            rect = Rectangle((x[i] - 0.3, bottom), 0.6, height,
                           facecolor=color, edgecolor=color, alpha=0.8)
            ax.add_patch(rect)
        
        step = max(1, len(data) // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([dates.iloc[i].strftime('%Y-%m-%d') for i in range(0, len(data), step)],
                          rotation=45, ha='right')
        
        ax.set_ylabel('价格', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, len(data))
    
    def _plot_trendlines(self, ax, data: pd.DataFrame, trend_analysis: Dict, 
                        reference_idx: int, sell_idx: Optional[int] = None,
                        label_prefix: str = '', line_style: str = '--', alpha: float = 0.7):
        """绘制趋势线（仅支撑线，不含延长线和阻力线）"""
        label_suffix = f" ({label_prefix})" if label_prefix else ""
        
        # 根据标签前缀选择颜色
        if '卖出时' in label_prefix:
            long_color = 'darkblue'
            short_color = 'darkcyan'
        else:
            long_color = 'blue'
            short_color = 'cyan'
        
        # 只绘制长期支撑线
        uptrend = trend_analysis['uptrend_line']
        if uptrend['valid']:
            start_idx = uptrend['start_idx']
            end_idx = min(uptrend['end_idx'], reference_idx)
            
            x_range = np.array([start_idx, end_idx])
            y_range = uptrend['slope'] * x_range + uptrend['intercept']
            ax.plot(x_range, y_range, color=long_color, linestyle=line_style, 
                   linewidth=2, label=f'长期支撑线{label_suffix}', alpha=alpha)
        
        # 只绘制短期支撑线
        short_uptrend = trend_analysis['short_uptrend_line']
        if short_uptrend['valid']:
            start_idx = short_uptrend['start_idx']
            end_idx = min(short_uptrend['end_idx'], reference_idx)
            
            x_range = np.array([start_idx, end_idx])
            y_range = short_uptrend['slope'] * x_range + short_uptrend['intercept']
            ax.plot(x_range, y_range, color=short_color, linestyle=line_style, 
                   linewidth=2, label=f'短期支撑线{label_suffix}', alpha=alpha)
        
        ax.legend(loc='upper left', fontsize=9)
    
    def _plot_trade_points(self, ax, data: pd.DataFrame, 
                          buy_date: str, sell_date: Optional[str],
                          buy_price: float, sell_price: Optional[float],
                          stop_loss: Optional[float], target: Optional[float],
                          exit_reason: Optional[str]):
        """绘制交易点（不含止损线和目标线）"""
        buy_idx = data[data['date'] == buy_date].index[0]
        
        ax.scatter(buy_idx, buy_price, color='red', s=200, marker='^', 
                  zorder=5, label='买入点', edgecolors='black', linewidths=2)
        
        ax.annotate(f'买入\n{buy_price:.2f}', 
                   xy=(buy_idx, buy_price),
                   xytext=(10, 20), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        if sell_date and sell_price:
            sell_idx = data[data['date'] == sell_date].index[0]
            
            profit = sell_price - buy_price
            color = 'green' if profit >= 0 else 'darkred'
            
            ax.scatter(sell_idx, sell_price, color=color, s=200, marker='v',
                      zorder=5, label='卖出点', edgecolors='black', linewidths=2)
            
            label_text = f'卖出\n{sell_price:.2f}'
            if exit_reason:
                label_text += f'\n({exit_reason})'
            
            ax.annotate(label_text,
                       xy=(sell_idx, sell_price),
                       xytext=(10, -30), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.legend(loc='upper right', fontsize=10)
    
    def _plot_volume(self, ax, data: pd.DataFrame):
        """绘制成交量"""
        x = np.arange(len(data))
        volumes = data['volume'].values
        closes = data['close'].values
        opens = data['open'].values
        
        colors = ['red' if closes[i] >= opens[i] else 'green' for i in range(len(data))]
        
        ax.bar(x, volumes, color=colors, alpha=0.6, width=0.8)
        
        dates = pd.to_datetime(data['date'])
        step = max(1, len(data) // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([dates.iloc[i].strftime('%Y-%m-%d') for i in range(0, len(data), step)],
                          rotation=45, ha='right')
        
        ax.set_ylabel('成交量', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, len(data))
    
    def visualize_backtest_results(self, 
                                   trades: List[Dict],
                                   output_dir: str = 'backtest_charts',
                                   max_charts: int = 20):
        """批量可视化回测结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"开始生成回测图表，共 {min(len(trades), max_charts)} 张...")
        
        for i, trade in enumerate(trades[:max_charts]):
            try:
                stock_code = trade['stock_code']
                buy_date = trade['buy_date']
                sell_date = trade.get('sell_date')
                
                filename = f"{stock_code}_{buy_date}"
                if sell_date:
                    filename += f"_{sell_date}"
                filename += ".png"
                
                save_path = os.path.join(output_dir, filename)
                
                self.plot_candlestick_with_trades(
                    stock_code=stock_code,
                    buy_date=buy_date,
                    sell_date=sell_date,
                    buy_price=trade.get('buy_price'),
                    sell_price=trade.get('sell_price'),
                    stop_loss=trade.get('stop_loss'),
                    target=trade.get('target'),
                    exit_reason=trade.get('exit_reason'),
                    save_path=save_path
                )
                
                print(f"[{i+1}/{min(len(trades), max_charts)}] 已生成: {filename}")
                
            except Exception as e:
                print(f"生成图表失败 ({trade.get('stock_code', 'unknown')}): {e}")
                continue
        
        print(f"\n所有图表已保存到: {output_dir}")
    
    def visualize_from_csv(self, 
                          csv_path: str,
                          output_dir: str = 'backtest_charts',
                          max_charts: int = 20):
        """从CSV回测报告文件批量生成图表（支持中英文列名）"""
        try:
            df = pd.read_csv(csv_path)
            print(f"成功读取CSV文件: {csv_path}")
            print(f"共 {len(df)} 条交易记录")
        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            return
        
        # 列名映射：支持中英文
        column_mapping = {
            # 股票代码
            'stock_code': ['stock_code', 'code', '股票代码', '代码'],
            # 买入日期
            'buy_date': ['buy_date', 'buy_time', '买入日期', '买入时间'],
            # 买入价格
            'buy_price': ['buy_price', '买入价', '买入价格'],
            # 卖出日期
            'sell_date': ['sell_date', 'sell_time', '卖出日期', '卖出时间'],
            # 卖出价格
            'sell_price': ['sell_price', '卖出价', '卖出价格'],
            # 卖出原因
            'exit_reason': ['exit_reason', 'reason', '卖出原因', '退出原因'],
            # 止损价
            'stop_loss': ['stop_loss', '止损价', '止损'],
            # 目标价
            'target': ['target', 'target_price', '目标价', '目标'],
        }
        
        # 查找实际的列名
        actual_columns = {}
        for key, possible_names in column_mapping.items():
            for name in possible_names:
                if name in df.columns:
                    actual_columns[key] = name
                    break
        
        # 检查必需的列
        required_keys = ['stock_code', 'buy_date', 'buy_price', 'sell_date', 'sell_price']
        missing_keys = [key for key in required_keys if key not in actual_columns]
        
        if missing_keys:
            print(f"错误: CSV文件缺少必需的列")
            print(f"缺少的列: {missing_keys}")
            print(f"当前CSV列名: {list(df.columns)}")
            return
        
        print(f"\n列名映射:")
        for key, col_name in actual_columns.items():
            print(f"  {key} -> {col_name}")
        
        # 解析交易记录
        trades = []
        for idx, row in df.iterrows():
            try:
                # 读取股票代码并格式化（补齐6位数字）
                stock_code = str(row[actual_columns['stock_code']])
                # 如果是纯数字，补齐到6位
                if stock_code.isdigit():
                    stock_code = stock_code.zfill(6)
                
                trade = {
                    'stock_code': stock_code,
                    'buy_date': str(row[actual_columns['buy_date']]),
                    'buy_price': float(row[actual_columns['buy_price']]),
                    'sell_date': str(row[actual_columns['sell_date']]),
                    'sell_price': float(row[actual_columns['sell_price']]),
                }
                
                # 添加可选字段
                if 'exit_reason' in actual_columns and pd.notna(row[actual_columns['exit_reason']]):
                    trade['exit_reason'] = str(row[actual_columns['exit_reason']])
                
                if 'stop_loss' in actual_columns and pd.notna(row[actual_columns['stop_loss']]):
                    trade['stop_loss'] = float(row[actual_columns['stop_loss']])
                
                if 'target' in actual_columns and pd.notna(row[actual_columns['target']]):
                    trade['target'] = float(row[actual_columns['target']])
                
                trades.append(trade)
            except Exception as e:
                print(f"警告: 第 {idx+1} 行数据解析失败: {e}")
                continue
        
        print(f"\n解析成功，共 {len(trades)} 条有效交易记录")
        
        self.visualize_backtest_results(
            trades=trades,
            output_dir=output_dir,
            max_charts=max_charts
        )


def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='回测可视化工具')
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 单个股票可视化
    single_parser = subparsers.add_parser('single', help='单个股票可视化')
    single_parser.add_argument('--stock', type=str, required=True, help='股票代码')
    single_parser.add_argument('--buy-date', type=str, required=True, help='买入日期')
    single_parser.add_argument('--sell-date', type=str, help='卖出日期')
    single_parser.add_argument('--buy-price', type=float, help='买入价格')
    single_parser.add_argument('--sell-price', type=float, help='卖出价格')
    single_parser.add_argument('--days-before', type=int, default=60, help='买入前显示天数')
    single_parser.add_argument('--days-after', type=int, default=60, help='卖出后显示天数')
    single_parser.add_argument('--exit-reason', type=str, help='卖出原因')
    single_parser.add_argument('--save', type=str, help='保存路径')
    single_parser.add_argument('--simulate', action='store_true', help='模拟回测')
    single_parser.add_argument('--max-holding-days', type=int, default=60, help='最大持仓天数')
    
    # CSV批量可视化
    csv_parser = subparsers.add_parser('csv', help='从CSV文件批量可视化')
    csv_parser.add_argument('--file', type=str, required=True, help='CSV文件路径')
    csv_parser.add_argument('--output-dir', type=str, default='backtest_charts', help='输出目录')
    csv_parser.add_argument('--max-charts', type=int, default=20, help='最多生成图表数量')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        print("\n示例用法:")
        print("  python visualize_backtest.py single --stock 600519 --buy-date 2024-01-15 --simulate")
        print("  python visualize_backtest.py csv --file backtest_report.csv")
        return
    
    visualizer = BacktestVisualizer()
    
    if args.command == 'single':
        if args.simulate:
            trade_result = visualizer.simulate_trade(
                stock_code=args.stock,
                buy_date=args.buy_date,
                buy_price=args.buy_price,
                max_holding_days=args.max_holding_days
            )
            
            print(f"\n模拟回测结果: 收益率 {trade_result['profit_rate']*100:.2f}%\n")
            
            visualizer.plot_candlestick_with_trades(
                stock_code=trade_result['stock_code'],
                buy_date=trade_result['buy_date'],
                sell_date=trade_result['sell_date'],
                days_before=args.days_before,
                days_after=args.days_after,
                buy_price=trade_result['buy_price'],
                sell_price=trade_result['sell_price'],
                exit_reason=trade_result['exit_reason'],
                save_path=args.save
            )
        else:
            visualizer.plot_candlestick_with_trades(
                stock_code=args.stock,
                buy_date=args.buy_date,
                sell_date=args.sell_date,
                days_before=args.days_before,
                days_after=args.days_after,
                buy_price=args.buy_price,
                sell_price=args.sell_price,
                exit_reason=args.exit_reason,
                save_path=args.save
            )
    
    elif args.command == 'csv':
        visualizer.visualize_from_csv(
            csv_path=args.file,
            output_dir=args.output_dir,
            max_charts=args.max_charts
        )


if __name__ == '__main__':
    main()
