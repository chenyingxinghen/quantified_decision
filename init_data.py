# 主程序入口
import time
from datetime import datetime
from data_fetcher import DataFetcher
from stock_screener import StockScreener
from wyckoff_strategy import WyckoffStrategy
from price_action_analyzer import PriceActionAnalyzer
from config import UPDATE_INTERVAL, MARKET_OPEN_TIME, MARKET_CLOSE_TIME, QUEST_INTERVAL, TEMP_ORDER


class QuantStockSelector:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.screener = StockScreener()
        self.wyckoff_strategy = WyckoffStrategy()
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
            if count>TEMP_ORDER:
                try:
                    self.data_fetcher.update_daily_data(stock['code'], incremental=incremental)
                    success_count += 1
                    if count%50==0:
                        print(f"已更新 {count}/{total_stocks} 只股票")
                    time.sleep(QUEST_INTERVAL)  # 避免请求过于频繁
                    
                except Exception as e:
                    fail_count += 1
                    print(f"更新股票 {stock['code']} 数据失败: {e}")
                    continue
            else:
                success_count += 1
                                # 进度显示
                if count==TEMP_ORDER:
                    print(f"已更新 {count}/{total_stocks} 只股票")
                
        print(f"[{datetime.now()}] 数据更新完成 (成功: {success_count}, 失败: {fail_count})")
    
    def run_smc_strategy(self):
        """执行SMC流动性猎取策略"""
        from smc_liquidity_strategy import SMCLiquidityStrategy

        print(f"\n{'=' * 80}")
        print("SMC流动性猎取策略")
        print(f"{'=' * 80}")
        
        # 获取候选股票
        candidates = self.screener.get_candidate_stocks()
        if candidates.empty:
            print("❌ 没有找到符合基本条件的候选股票")
            return
        
        total_stocks = len(candidates)
        print(f"候选股票总数: {total_stocks}")
        
        # 询问是否限制分析数量
        limit_input = input(f"\n是否限制分析数量？(直接回车分析全部，或输入数字限制数量): ").strip()
        if limit_input.isdigit():
            analysis_limit = int(limit_input)
        else:
            analysis_limit = total_stocks
        
        analysis_limit = min(analysis_limit, total_stocks)
        
        # 应用技术筛选
        filtered_codes = []
        for idx, stock in candidates.head(analysis_limit).iterrows():
            try:
                passed, _ = self.screener.apply_technical_filters(stock['code'])
                if passed:
                    filtered_codes.append(stock['code'])
            except:
                continue
        
        print(f"\n通过基础筛选: {len(filtered_codes)} 只股票")
        
        # 运行SMC策略
        smc = SMCLiquidityStrategy()
        results = smc.batch_screen(filtered_codes, max_results=100)
        
        if results:
            smc.generate_report(results)
        else:
            print("未找到符合SMC策略的股票")
    
    def run_integrated_screening(self):
        """运行整合选股"""
        from integrated_screener import IntegratedScreener
        
        screener = IntegratedScreener()
        
        print("\n请选择筛选模型:")
        print("1. SMC流动性猎取")
        print("2. 威科夫Spring")
        print("3. 综合筛选（所有模型）")
        
        choice = input("\n请选择 (1-3): ").strip()
        
        top_n = input("每个模型返回多少只股票？(默认100): ").strip()
        top_n = int(top_n) if top_n.isdigit() else 100
        
        if choice == "1":
            candidates = screener.get_filtered_candidates()
            results = screener.run_model_1_liquidity_grab(candidates, top_n)
            screener.smc_strategy.generate_report(results)
        elif choice == "2":
            candidates = screener.get_filtered_candidates()
            results = screener.run_model_3_wyckoff_spring(candidates, top_n)
            screener.smc_strategy.generate_report(results)
        elif choice == "3":
            screener.run_comprehensive_screening('all', top_n)
        else:
            print("无效选择")
        
        screener.close()
    
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
        print("\n选股策略:")
        print("4. 威科夫积累策略 (Wyckoff Spring)")
        print("5. SMC流动性猎取策略")
        print("6. 整合选股 (运行所有策略)")
        print("\n系统功能:")
        print("7. 启动定时更新")
        print("0. 退出")
        
        while True:
            try:
                choice = input("\n请选择操作 (0-8): ").strip()
                
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
                    self.run_advanced_strategy("wyckoff_accumulation")
                elif choice == "5":
                    self.run_smc_strategy()
                elif choice == "6":
                    self.run_integrated_screening()
                elif choice == "7":
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
                    result = self.wyckoff_strategy.wyckoff_accumulation_strategy(stock['code'])
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
        """显示策略结果"""
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
        filename = f"strategy_{strategy_name}_{timestamp}.csv"
        
        # 准备导出数据
        export_data = []
        for stock in results:
            risk_reward = ((stock['target'] - stock['entry_price']) / 
                          (stock['entry_price'] - stock['stop_loss'])) if stock['entry_price'] > stock['stop_loss'] else 0
            
            export_data.append({
                '股票代码': stock['code'],
                '股票名称': stock['name'],
                '策略': strategy_name,
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