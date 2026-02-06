"""
K线图和支撑线可视化工具
用于验证回测中支撑线计算的准确性
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import sqlite3
from datetime import datetime, timedelta
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATABASE_PATH
from core.analysis import TrendLineAnalyzer
from config.strategy_config import TREND_LINE_LONG_PERIOD, TREND_LINE_SHORT_PERIOD

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class SupportLineVisualizer:
    """支撑线可视化器"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = DATABASE_PATH
        
        # 确保数据库路径为绝对路径
        if not os.path.isabs(db_path):
            project_root = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(project_root, db_path)
        
        self.db_path = db_path
        self.analyzer = TrendLineAnalyzer()
    
    def load_stock_data(self, stock_code: str, days: int = 1500) -> pd.DataFrame:
        """加载股票数据"""
        conn = sqlite3.connect(self.db_path)
        
        # 获取最近N天的数据
        query = '''
            SELECT date, open, high, low, close, volume
            FROM daily_data
            WHERE code = ?
            ORDER BY date DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(stock_code, days))
        conn.close()
        
        # 按日期升序排列
        df = df.sort_values('date').reset_index(drop=True)
        
        if len(df) == 0:
            raise ValueError(f"未找到股票 {stock_code} 的数据")
        
        return df
    
    def visualize(self, stock_code: str, days: int = 1500, save_path: str = None):
        """
        可视化K线图和支撑线
        
        参数:
            stock_code: 股票代码
            days: 显示的天数
            save_path: 保存路径（可选）
        """
        # 加载数据
        print(f"正在加载 {stock_code} 的数据...")
        data = self.load_stock_data(stock_code, days)
        
        # # 去除最后10天的数据
        # if len(data) > 10:
        #     data = data[:-7].copy()
        # else:
        #     print(f"警告: 数据少于等于10天，无法去除后十天数据，当前数据长度: {len(data)}")

        if len(data) < 30:
            print(f"数据不足（只有{len(data)}天），至少需要30天")
            return
        
        print(f"成功加载 {len(data)} 天的数据")
        
        # 分析趋势线
        print("正在分析趋势线...")
        analysis = self.analyzer.analyze(data)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # 绘制K线图
        self._plot_candlestick(ax1, data)
        
        # 绘制趋势线
        self._plot_trendlines(ax1, data, analysis)
        
        # 绘制摆动点
        self._plot_swing_points(ax1, data, analysis)
        
        # 绘制成交量
        self._plot_volume(ax2, data)
        
        # 添加标题和信息
        self._add_info(ax1, stock_code, data, analysis)
        
        # 格式化x轴
        self._format_xaxis(ax1, ax2, data)
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_candlestick(self, ax, data):
        """绘制K线图"""
        for i in range(len(data)):
            row = data.iloc[i]
            
            # 确定颜色
            if row['close'] >= row['open']:
                color = 'red'
                body_color = 'red'
            else:
                color = 'green'
                body_color = 'green'
            
            # 绘制影线
            ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=0.8)
            
            # 绘制实体
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['open'], row['close'])
            
            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                           facecolor=body_color, edgecolor=color, linewidth=0.8)
            ax.add_patch(rect)
        
        ax.set_ylabel('价格', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    def _plot_trendlines(self, ax, data, analysis):
        """绘制趋势线"""
        # 长期上升趋势线（支撑线）
        if analysis['uptrend_line']['valid']:
            line = analysis['uptrend_line']
            x_values = range(len(data))
            y_values = [line['slope'] * x + line['intercept'] for x in x_values]
            
            ax.plot(x_values, y_values, 'b-', linewidth=2, 
                   label=f'长期支撑线 (斜率: {line["slope"]:.4f})', alpha=0.7)
            
            # 标注触点
            for i in range(len(data)):
                expected = line['slope'] * i + line['intercept']
                actual = data.iloc[i]['low']
                deviation = abs(actual - expected) / expected
                
                if deviation < 0.02:  # 2%容差
                    ax.plot(i, actual, 'bo', markersize=6, alpha=0.6)
        
        # 短期上升趋势线
        if analysis['short_uptrend_line']['valid']:
            line = analysis['short_uptrend_line']
            
            # 只绘制短期数据范围内的趋势线
            short_period = min(TREND_LINE_SHORT_PERIOD, len(data))
            start_idx = len(data) - short_period
            
            x_values = range(start_idx, len(data))
            y_values = [line['slope'] * x + line['intercept'] for x in x_values]
            
            ax.plot(x_values, y_values, 'c--', linewidth=2,
                   label=f'短期支撑线 (斜率: {line["slope"]:.4f})', alpha=0.7)
            
            # 标注触点
            for i in range(start_idx, len(data)):
                expected = line['slope'] * i + line['intercept']
                actual = data.iloc[i]['low']
                deviation = abs(actual - expected) / expected
                
                if deviation < 0.02:
                    ax.plot(i, actual, 'co', markersize=6, alpha=0.6)
        
        # 长期下降趋势线（阻力线）
        if analysis['downtrend_line']['valid']:
            line = analysis['downtrend_line']
            x_values = range(len(data))
            y_values = [line['slope'] * x + line['intercept'] for x in x_values]
            
            ax.plot(x_values, y_values, 'r-', linewidth=2,
                   label=f'长期阻力线 (斜率: {line["slope"]:.4f})', alpha=0.7)
        
        # 短期下降趋势线
        if analysis['short_downtrend_line']['valid']:
            line = analysis['short_downtrend_line']
            
            short_period = min(TREND_LINE_SHORT_PERIOD, len(data))
            start_idx = len(data) - short_period
            
            x_values = range(start_idx, len(data))
            y_values = [line['slope'] * x + line['intercept'] for x in x_values]
            
            ax.plot(x_values, y_values, 'm--', linewidth=2,
                   label=f'短期阻力线 (斜率: {line["slope"]:.4f})', alpha=0.7)
        
        ax.legend(loc='upper left', fontsize=10)
    
    def _plot_swing_points(self, ax, data, analysis):
        """绘制摆动点（用于验证趋势线计算）"""
        # 长期分段最低点
        if analysis['uptrend_line']['valid']:
            line = analysis['uptrend_line']
            if line.get('point_type') == 'segment_lows':
                # 对于分段方法，直接标注起点和终点
                ax.plot(line['start_idx'], line['start_price'], 'g^', markersize=10, alpha=0.7, label='长期支撑点')
                ax.plot(line['end_idx'], line['end_price'], 'g^', markersize=10, alpha=0.7)
        
        # 短期分段最低点
        if analysis['short_uptrend_line']['valid']:
            line = analysis['short_uptrend_line']
            if line.get('point_type') == 'segment_lows':
                ax.plot(line['start_idx'], line['start_price'], 'y^', markersize=8, alpha=0.7, label='短期支撑点')
                ax.plot(line['end_idx'], line['end_price'], 'y^', markersize=8, alpha=0.7)
    
    def _plot_volume(self, ax, data):
        """绘制成交量"""
        colors = ['red' if data.iloc[i]['close'] >= data.iloc[i]['open'] 
                 else 'green' for i in range(len(data))]
        
        ax.bar(range(len(data)), data['volume'], color=colors, alpha=0.6)
        ax.set_ylabel('成交量', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    def _add_info(self, ax, stock_code, data, analysis):
        """添加信息文本"""
        current_price = data['close'].iloc[-1]
        
        info_text = f"股票代码: {stock_code}\n"
        info_text += f"当前价格: {current_price:.2f}\n"
        info_text += f"价格位置: {analysis['current_position']}\n"
        info_text += f"入场质量: {analysis['entry_quality']:.1f}/100\n"
        info_text += f"趋势强度: {analysis['trend_strength']:.1f}/100\n"
        info_text += f"跌破支撑: {'是' if analysis['broken_support'] else '否'}\n"
        info_text += f"趋势反转: {'是' if analysis['trend_reversal'] else '否'}\n"
        
        if analysis['uptrend_line']['valid']:
            line = analysis['uptrend_line']
            current_idx = len(data) - 1
            support_price = line['slope'] * current_idx + line['intercept']
            distance = (current_price - support_price) / support_price * 100
            info_text += f"距离长期支撑: {distance:.2f}%\n"
        
        if analysis['short_uptrend_line']['valid']:
            line = analysis['short_uptrend_line']
            current_idx = len(data) - 1
            support_price = line['slope'] * current_idx + line['intercept']
            distance = (current_price - support_price) / support_price * 100
            info_text += f"距离短期支撑: {distance:.2f}%"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _format_xaxis(self, ax1, ax2, data):
        """格式化x轴"""
        # 设置x轴刻度
        step = max(1, len(data) // 10)
        x_ticks = range(0, len(data), step)
        x_labels = [data.iloc[i]['date'] for i in x_ticks]
        
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels([])
        
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_labels, rotation=45, ha='right')
        ax2.set_xlabel('日期', fontsize=12)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化K线图和支撑线')
    parser.add_argument('stock_code', type=str, help='股票代码（如：000001）')
    parser.add_argument('--days', type=int, default=90, help='显示天数（默认150）')
    parser.add_argument('--save', type=str, default=None, help='保存路径（可选）')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = SupportLineVisualizer()
    
    # 生成可视化
    try:
        visualizer.visualize(
            stock_code=args.stock_code,
            days=args.days,
            save_path=args.save
        )
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
