# 主程序入口
import time
from datetime import datetime
from data_fetcher import DataFetcher
from stock_screener import StockScreener
from advanced_strategies import AdvancedTradingStrategies
from price_action_analyzer import PriceActionAnalyzer
from config import UPDATE_INTERVAL, MARKET_OPEN_TIME, MARKET_CLOSE_TIME

class QuantStockSelector:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.screener = StockScreener()
        self.advanced_strategies = AdvancedTradingStrategies()
        self.price_action = PriceActionAnalyzer()
        self.is_running = False
    
    def update_data(self, incremental=True):
        """更新股票数据
        
        Args:
            incremental: 是否增量更新
        """
        print(f"[{datetime.now()}] 开始{'增量' if incremental else '全量'}更新数据...")
        
        # 获取股票列表并更新日线数据
        stock_list = self.data_fetcher.get_stock_list(markets=['sh', 'sz_main'])
        total_stocks = len(stock_list)
        
        print(f"开始更新 {total_stocks} 只股票的日线数据...")
        
        success_count = 0
        fail_count = 0
        
        for count, (idx, stock) in enumerate(stock_list.iterrows(), 1):
            try:
                self.data_fetcher.update_daily_data(stock['code'], incremental=incremental)
                success_count += 1
                
                # 进度显示
                if count % 100 == 0:
                    print(f"已更新 {count}/{total_stocks} 只股票 (成功: {success_count}, 失败: {fail_count})")
                
                time.sleep(0.1)  # 避免请求过于频繁
                
            except Exception as e:
                fail_count += 1
                print(f"更新股票 {stock['code']} 数据失败: {e}")
                continue
        
        print(f"[{datetime.now()}] 数据更新完成 (成功: {success_count}, 失败: {fail_count})")
    
    def run_screening(self, strategy="golden_cross"):
        """执行股票筛选"""
        print(f"[{datetime.now()}] 开始执行 {strategy} 策略筛选...")
        
        results = self.screener.screen_stocks_by_strategy(strategy)
        self.screener.generate_report(results, strategy)
        
        return results
    
    def run_custom_screening(self, conditions):
        """执行自定义筛选"""
        print(f"[{datetime.now()}] 开始执行自定义筛选...")
        
        results = self.screener.get_custom_screen_results(conditions)
        self.screener.generate_report(results, "custom")
        
        return results
    
    def is_market_time(self):
        """检查是否在交易时间"""
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        # 简单的交易时间检查（可以根据需要完善）
        return MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME
    
    def scheduled_update(self):
        """定时更新任务"""
        if self.is_market_time():
            self.update_data()
        else:
            print(f"[{datetime.now()}] 非交易时间，跳过数据更新")
    
    def start_scheduler(self):
        """启动定时任务"""
        print("启动定时数据更新...")
        
        # 每5分钟检查一次是否需要更新数据
        schedule.every(UPDATE_INTERVAL // 60).minutes.do(self.scheduled_update)
        
        self.is_running = True
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
    
    def stop_scheduler(self):
        """停止定时任务"""
        self.is_running = False
        print("定时任务已停止")
    
    def interactive_mode(self):
        """交互模式"""
        print("\n=== A股量化选股系统 ===")
        print("数据管理:")
        print("1. 初始化股票数据（首次使用）")
        print("2. 增量更新数据")
        print("3. 全量更新数据")
        print("4. 查看数据库状态")
        print("\n基础策略:")
        print("5. 黄金交叉策略筛选")
        print("6. 超卖反弹策略筛选")
        print("7. 突破策略筛选")
        print("8. 自定义筛选")
        print("\n高级策略 (基于K线结构和价格行为):")
        print("9. 威科夫积累策略")
        print("10. 黄金交叉动量策略")
        print("11. 均值回归超卖策略")
        print("12. 突破动量策略")
        print("13. 多周期共振策略")
        print("14. 综合策略分析")
        print("15. 价格行为深度分析")
        print("\n系统功能:")
        print("16. 启动定时更新")
        print("0. 退出")
        
        while True:
            try:
                choice = input("\n请选择操作 (0-16): ").strip()
                
                if choice == "0":
                    break
                elif choice == "1":
                    print("\n开始初始化股票数据（上证+深圳主板）...")
                    self.data_fetcher.init_all_stocks_data(markets=['sh', 'sz_main'], incremental=False)
                elif choice == "2":
                    self.update_data(incremental=True)
                elif choice == "3":
                    confirm = input("全量更新会重新下载所有数据，确认吗？(y/n): ").strip().lower()
                    if confirm == 'y':
                        self.update_data(incremental=False)
                elif choice == "4":
                    self.show_database_status()
                elif choice == "5":
                    self.run_screening("golden_cross")
                elif choice == "6":
                    self.run_screening("oversold_rebound")
                elif choice == "7":
                    self.run_screening("breakout")
                elif choice == "8":
                    self.custom_screening_interface()
                elif choice == "9":
                    self.run_advanced_strategy("wyckoff_accumulation")
                elif choice == "10":
                    self.run_advanced_strategy("golden_cross_momentum")
                elif choice == "11":
                    self.run_advanced_strategy("mean_reversion_oversold")
                elif choice == "12":
                    self.run_advanced_strategy("breakout_momentum")
                elif choice == "13":
                    self.run_advanced_strategy("multi_timeframe_confluence")
                elif choice == "14":
                    self.run_comprehensive_analysis()
                elif choice == "15":
                    self.run_price_action_analysis()
                elif choice == "16":
                    print("启动定时更新（按Ctrl+C停止）...")
                    try:
                        self.start_scheduler()
                    except KeyboardInterrupt:
                        self.stop_scheduler()
                else:
                    print("无效选择，请重新输入")
                    
            except KeyboardInterrupt:
                print("\n程序被用户中断")
                break
            except Exception as e:
                print(f"操作失败: {e}")
    
    def show_database_status(self):
        """显示数据库状态"""
        print("\n=== 数据库状态 ===")
        
        cursor = self.data_fetcher.conn.cursor()
        
        # 股票数量
        cursor.execute("SELECT COUNT(*) FROM stock_info")
        stock_count = cursor.fetchone()[0]
        print(f"股票总数: {stock_count}")
        
        # 日线数据
        cursor.execute("SELECT COUNT(*) FROM daily_data")
        daily_count = cursor.fetchone()[0]
        print(f"日线记录数: {daily_count}")
        
        cursor.execute("SELECT COUNT(DISTINCT code) FROM daily_data")
        stocks_with_data = cursor.fetchone()[0]
        print(f"有数据的股票: {stocks_with_data}")
        
        # 最新日期
        cursor.execute("SELECT MAX(date) FROM daily_data")
        latest_date = cursor.fetchone()[0]
        print(f"最新数据日期: {latest_date}")
        
        # 候选股票数量
        cursor.execute("""
            SELECT COUNT(DISTINCT s.code)
            FROM stock_info s
            INNER JOIN daily_data d ON s.code = d.code
            GROUP BY s.code
            HAVING COUNT(d.date) >= 30
        """)
        candidate_count = len(cursor.fetchall())
        print(f"候选股票数量（数据>=30天）: {candidate_count}")
        
        print("\n提示: 输入股票代码时只需输入数字部分，如 000001 或 600000")
    
    def custom_screening_interface(self):
        """自定义筛选界面"""
        print("\n=== 自定义筛选条件 ===")
        print("可用信号类型:")
        print("- ma_trend: bullish/bearish/neutral")
        print("- rsi_signal: overbought/oversold/neutral")
        print("- macd_signal: bullish/bearish/neutral")
        print("- bb_signal: overbought/oversold/neutral")
        print("- kdj_signal: overbought/oversold/neutral")
        
        conditions = []
        
        while True:
            signal_type = input("输入信号类型 (回车结束): ").strip()
            if not signal_type:
                break
            
            signal_value = input(f"输入 {signal_type} 的期望值: ").strip()
            
            conditions.append({
                'signal': signal_type,
                'value': signal_value
            })
        
        if conditions:
            self.run_custom_screening(conditions)
        else:
            print("未设置筛选条件")
    
    def run_advanced_strategy(self, strategy_name):
        """执行高级策略分析"""
        from config import SELECTION_CRITERIA
        
        print(f"\n{'=' * 80}")
        print(f"{strategy_name.upper()} 高级策略分析")
        print(f"{'=' * 80}")
        
        # 获取候选股票
        candidates = self.screener.get_candidate_stocks()
        if candidates.empty:
            print("❌ 没有找到符合基本条件的候选股票")
            return
        
        total_stocks = len(candidates)
        print(f"候选股票总数: {total_stocks}")
        print(f"基础筛选条件:")
        print(f"  • 价格区间: {SELECTION_CRITERIA['min_price']} - {SELECTION_CRITERIA['max_price']} 元")
        print(f"  • 最小换手率: {SELECTION_CRITERIA['min_turnover_rate']}%")
        print(f"  • 最小市值: {SELECTION_CRITERIA['min_market_cap']/100000000}亿")
        print(f"  • 最大市盈率: {SELECTION_CRITERIA['max_pe']}")
        
        # 询问是否限制分析数量
        limit_input = input(f"\n是否限制分析数量？(直接回车分析全部，或输入数字限制数量): ").strip()
        if limit_input.isdigit():
            analysis_limit = int(limit_input)
        else:
            analysis_limit = total_stocks
        
        analysis_limit = min(analysis_limit, total_stocks)
        
        results = []
        processed = 0
        passed_basic_filter = 0
        
        print(f"\n开始分析 {analysis_limit} 只候选股票...")
        print("=" * 60)
        
        for idx, stock in candidates.head(analysis_limit).iterrows():
            try:
                processed += 1
                
                # 应用完整的技术筛选条件（包括价格和换手率）
                passed, signals_or_reason = self.screener.apply_technical_filters(stock['code'])
                
                if not passed:
                    # 如果需要调试，可以取消下面的注释
                    # if processed <= 10:  # 只显示前10个未通过的原因
                    #     print(f"  {stock['code']} 未通过筛选: {signals_or_reason}")
                    continue
                
                passed_basic_filter += 1
                
                # 根据策略类型调用相应的分析函数
                if strategy_name == "wyckoff_accumulation":
                    result = self.advanced_strategies.wyckoff_accumulation_strategy(stock['code'])
                elif strategy_name == "golden_cross_momentum":
                    result = self.advanced_strategies.golden_cross_momentum_strategy(stock['code'])
                elif strategy_name == "mean_reversion_oversold":
                    result = self.advanced_strategies.mean_reversion_oversold_strategy(stock['code'])
                elif strategy_name == "breakout_momentum":
                    result = self.advanced_strategies.breakout_momentum_strategy(stock['code'])
                elif strategy_name == "multi_timeframe_confluence":
                    result = self.advanced_strategies.multi_timeframe_confluence_strategy(stock['code'])
                else:
                    continue
                
                # 只保留买入信号
                if result and result['signal'] in ['buy', 'strong_buy']:
                    results.append({
                        'code': stock['code'],
                        'name': stock['name'],
                        'strategy': result['strategy'],
                        'signal': result['signal'],
                        'confidence': result['confidence'],
                        'entry_price': result['entry_price'],
                        'stop_loss': result['stop_loss'],
                        'target': result['target'],
                        'conditions': result['conditions_met']
                    })
                
                # 进度显示
                if processed % 10 == 0 or processed == analysis_limit:
                    progress = processed / analysis_limit * 100
                    print(f"进度: {processed}/{analysis_limit} ({progress:.1f}%) | "
                          f"通过基础筛选: {passed_basic_filter} | 找到信号: {len(results)}")
                    
            except Exception as e:
                print(f"❌ 分析股票 {stock['code']} 时出错: {e}")
                continue
        
        print("=" * 60)
        print(f"✓ 分析完成！")
        print(f"  共处理: {processed} 只股票")
        print(f"  通过基础筛选: {passed_basic_filter} 只")
        print(f"  找到买入信号: {len(results)} 只")
        
        self.display_advanced_results(results, strategy_name)
    
    def run_comprehensive_analysis(self):
        """运行综合策略分析"""
        print("\n=== 综合策略分析 ===")
        
        stock_code = input("请输入股票代码（如 000001 或 600000）: ").strip()
        if not stock_code:
            print("股票代码不能为空")
            return
        
        # 移除可能的前缀
        stock_code = stock_code.replace('sh', '').replace('sz', '').replace('SH', '').replace('SZ', '')
        
        try:
            # 检查股票是否存在
            cursor = self.data_fetcher.conn.cursor()
            cursor.execute("SELECT name FROM stock_info WHERE code = ?", (stock_code,))
            result = cursor.fetchone()
            
            if not result:
                print(f"未找到股票代码 {stock_code}，请检查代码是否正确")
                return
            
            stock_name = result[0]
            print(f"正在分析: {stock_code} - {stock_name}")
            
            # 获取所有策略信号
            all_signals = self.advanced_strategies.get_all_strategy_signals(stock_code)
            
            if not all_signals:
                print(f"无法获取股票 {stock_code} 的分析数据")
                return
            
            # 显示综合分析结果
            self.display_comprehensive_analysis(stock_code, all_signals)
            
        except Exception as e:
            print(f"综合分析失败: {e}")
    
    def run_price_action_analysis(self):
        """运行价格行为分析"""
        print("\n=== 价格行为深度分析 ===")
        
        stock_code = input("请输入股票代码（如 000001 或 600000）: ").strip()
        if not stock_code:
            print("股票代码不能为空")
            return
        
        # 移除可能的前缀
        stock_code = stock_code.replace('sh', '').replace('sz', '').replace('SH', '').replace('SZ', '')
        
        try:
            # 检查股票是否存在
            cursor = self.data_fetcher.conn.cursor()
            cursor.execute("SELECT name FROM stock_info WHERE code = ?", (stock_code,))
            result = cursor.fetchone()
            
            if not result:
                print(f"未找到股票代码 {stock_code}，请检查代码是否正确")
                return
            
            stock_name = result[0]
            print(f"正在分析: {stock_code} - {stock_name}")
            
            # 获取股票数据
            stock_data = self.data_fetcher.get_stock_data(stock_code)
            
            if len(stock_data) < 50:
                print(f"股票 {stock_code} 数据不足（当前 {len(stock_data)} 条，需要至少 50 条），无法进行分析")
                return
            
            # 价格行为分析
            pa_analysis = self.price_action.get_comprehensive_analysis(stock_data)
            
            if not pa_analysis:
                print(f"无法完成股票 {stock_code} 的价格行为分析")
                return
            
            # 显示分析结果
            self.display_price_action_analysis(stock_code, pa_analysis, stock_data)
            
        except Exception as e:
            print(f"价格行为分析失败: {e}")
    
    def display_advanced_results(self, results, strategy_name):
        """显示高级策略结果"""
        import pandas as pd
        
        if not results:
            print(f"\n❌ {strategy_name} 策略未找到符合条件的股票")
            return
        
        print(f"\n{'=' * 80}")
        print(f"{strategy_name.upper()} 策略筛选结果")
        print(f"{'=' * 80}")
        print(f"找到 {len(results)} 只符合条件的股票\n")
        
        # 按置信度排序
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        for i, stock in enumerate(results, 1):
            risk_reward = ((stock['target'] - stock['entry_price']) / 
                          (stock['entry_price'] - stock['stop_loss'])) if stock['entry_price'] > stock['stop_loss'] else 0
            
            print(f"\n【{i}】 {stock['code']} - {stock['name']}")
            print(f"    信号强度: {stock['signal'].upper()} (置信度: {stock['confidence']:.1f}%)")
            print(f"    入场价格: ¥{stock['entry_price']:.2f}")
            print(f"    止损价格: ¥{stock['stop_loss']:.2f} (风险: {((stock['entry_price']-stock['stop_loss'])/stock['entry_price']*100):.2f}%)")
            print(f"    目标价格: ¥{stock['target']:.2f} (收益: {((stock['target']-stock['entry_price'])/stock['entry_price']*100):.2f}%)")
            print(f"    风险收益比: 1:{risk_reward:.2f}")
            print(f"    满足条件:")
            for condition in stock['conditions']:
                print(f"      ✓ {condition}")
            print("-" * 80)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"advanced_strategy_{strategy_name}_{timestamp}.csv"
        
        # 准备导出数据
        export_data = []
        for stock in results:
            risk_reward = ((stock['target'] - stock['entry_price']) / 
                          (stock['entry_price'] - stock['stop_loss'])) if stock['entry_price'] > stock['stop_loss'] else 0
            
            export_data.append({
                '股票代码': stock['code'],
                '股票名称': stock['name'],
                '策略': stock['strategy'],
                '信号': stock['signal'],
                '置信度': f"{stock['confidence']:.1f}%",
                '入场价': stock['entry_price'],
                '止损价': stock['stop_loss'],
                '目标价': stock['target'],
                '风险收益比': f"1:{risk_reward:.2f}",
                '满足条件': ' | '.join(stock['conditions'])
            })
        
        df_results = pd.DataFrame(export_data)
        df_results.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n✓ 结果已保存到: {filename}")
        print(f"{'=' * 80}\n")
    
    def display_comprehensive_analysis(self, stock_code, all_signals):
        """显示综合分析结果"""
        print(f"\n=== {stock_code} 综合策略分析报告 ===")
        
        composite = all_signals.get('composite_analysis', {})
        
        print(f"综合信号: {composite.get('composite_signal', 'N/A')}")
        print(f"平均置信度: {composite.get('average_confidence', 0):.1f}%")
        print(f"买入信号数量: {composite.get('buy_signals_count', 0)}")
        print(f"观察信号数量: {composite.get('watch_signals_count', 0)}")
        print(f"总策略数量: {composite.get('total_strategies', 0)}")
        
        print("\n各策略详细分析:")
        print("=" * 80)
        
        for strategy_name, result in all_signals.items():
            if strategy_name == 'composite_analysis':
                continue
                
            print(f"\n【{strategy_name.upper()}】")
            print(f"信号: {result.get('signal', 'N/A')} (置信度: {result.get('confidence', 0):.1f}%)")
            
            if 'entry_price' in result:
                print(f"入场价: ¥{result['entry_price']:.2f}")
                print(f"止损价: ¥{result['stop_loss']:.2f}")
                print(f"目标价: ¥{result['target']:.2f}")
            
            print("满足条件:")
            for condition in result.get('conditions_met', []):
                print(f"  • {condition}")
            
            print("-" * 50)
        
        # 生成投资建议
        self.generate_investment_advice(composite)
    
    def display_price_action_analysis(self, stock_code, pa_analysis, stock_data):
        """显示价格行为分析结果"""
        print(f"\n=== {stock_code} 价格行为分析报告 ===")
        
        current_price = stock_data['close'].iloc[-1]
        print(f"当前价格: ¥{current_price:.2f}")
        
        # 市场结构分析
        market_structure = pa_analysis.get('market_structure', {})
        print(f"\n【市场结构分析】")
        print(f"趋势方向: {market_structure.get('trend', 'N/A')}")
        print(f"趋势强度: {market_structure.get('strength', 0):.1f}")
        print(f"结构模式: {market_structure.get('pattern', 'N/A')}")
        
        key_levels = market_structure.get('key_levels', {})
        if key_levels.get('resistance'):
            print(f"关键阻力: ¥{key_levels['resistance']:.2f}")
        if key_levels.get('support'):
            print(f"关键支撑: ¥{key_levels['support']:.2f}")
        
        # K线形态分析
        patterns = pa_analysis.get('candlestick_patterns', [])
        if patterns:
            print(f"\n【K线形态识别】")
            for pattern in patterns:
                print(f"• {pattern['description']} (可靠性: {pattern['reliability']}%)")
        
        # 突破分析
        breakout = pa_analysis.get('breakout_analysis')
        if breakout:
            print(f"\n【突破分析】")
            print(f"突破类型: {breakout.get('type', 'N/A')}")
            print(f"突破强度: {breakout.get('strength', 0):.2f}%")
            print(f"成交量确认: {'是' if breakout.get('volume_confirmation') else '否'}")
        
        # 六根K线分析
        six_bar = pa_analysis.get('six_bar_analysis')
        if six_bar:
            print(f"\n【六根K线趋势分析】")
            print(f"趋势评分: {six_bar.get('trend_score', 0)}")
            print(f"趋势质量: {six_bar.get('trend_quality', 'N/A')}")
            print(f"一致性: {six_bar.get('consistency', 0):.2f}")
        
        # 综合信号
        composite_signal = pa_analysis.get('composite_signal', {})
        print(f"\n【综合价格行为信号】")
        print(f"方向: {composite_signal.get('direction', 'N/A')}")
        print(f"强度: {composite_signal.get('strength', 0):.1f}")
        print(f"置信度: {composite_signal.get('confidence', 0):.2f}")
    
    def generate_investment_advice(self, composite):
        """生成投资建议"""
        print(f"\n【投资建议】")
        
        signal = composite.get('composite_signal', 'no_signal')
        confidence = composite.get('average_confidence', 0)
        
        if signal == 'strong_buy':
            print("🟢 强烈推荐买入")
            print("多个策略同时发出买入信号，建议积极关注")
        elif signal == 'buy':
            print("🟡 建议买入")
            print("有买入信号，但需要控制仓位")
        elif signal == 'watch':
            print("🔵 密切观察")
            print("存在机会，但信号不够强烈，建议等待更好时机")
        else:
            print("🔴 暂不建议操作")
            print("当前信号不明确，建议观望")
        
        print(f"\n风险提示:")
        print(f"• 本分析基于技术指标，不构成投资建议")
        print(f"• 请结合基本面分析和市场环境")
        print(f"• 严格执行止损，控制风险")
        print(f"• 分散投资，不要集中持仓")
    
    def close(self):
        """关闭资源"""
        self.data_fetcher.close()
        self.screener.close()

def main():
    """主函数"""
    selector = QuantStockSelector()
    
    try:
        # 检查数据库是否已初始化
        cursor = selector.data_fetcher.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stock_info")
        stock_count = cursor.fetchone()[0]
        
        if stock_count == 0:
            print("=" * 60)
            print("检测到首次运行，需要初始化股票数据")
            print("默认初始化上证板块和深圳主板的股票")
            print("=" * 60)
            choice = input("\n是否现在初始化？(y/n): ").strip().lower()
            if choice == 'y':
                selector.data_fetcher.init_all_stocks_data(markets=['sh', 'sz_main'], incremental=False)
            else:
                print("跳过初始化，您可以稍后在菜单中选择初始化")
        else:
            print(f"数据库已有 {stock_count} 只股票数据")
        
        # 进入交互模式
        selector.interactive_mode()
        
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        selector.close()
        print("程序已退出")

if __name__ == "__main__":
    main()