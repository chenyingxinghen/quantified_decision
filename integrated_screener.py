# 整合选股器 - 结合SMC和威科夫策略
# 实现"主动筛选当前买点"的量化系统

import pandas as pd
from datetime import datetime, timedelta
from data_fetcher import DataFetcher
from stock_screener import StockScreener
from smc_liquidity_strategy import SMCLiquidityStrategy
from wyckoff_strategy import WyckoffStrategy
from config import TECHNICAL_PARAMS,SELECTION_CRITERIA
import argparse

MINCONFIDENCE = 50
TOP_N=100000
class IntegratedScreener:
    """
    整合选股器
    
    两大核心策略：
    1. SMC流动性猎取策略 - 识别机构建仓完成的瞬间
    2. 威科夫Spring策略 - 捕捉底部反转的最后机会
    """
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.basic_screener = StockScreener()
        self.smc_strategy = SMCLiquidityStrategy()
        self.wyckoff_strategy = WyckoffStrategy()
    
    def get_filtered_candidates(self):
        """
        获取经过基础筛选的候选股票
        
        使用 stock_screener 中统一的筛选逻辑，确保与其他模块筛选标准一致
        """
        print("=" * 60)
        print("第一步：基础筛选")
        print("=" * 60)
        print(f"筛选条件:")
        print(f"  • 价格区间: {SELECTION_CRITERIA['min_price']} - {SELECTION_CRITERIA['max_price']} 元")
        print(f"  • 最小换手率: {SELECTION_CRITERIA['min_turnover_rate']}%")
        print(f"  • 最小市值: {SELECTION_CRITERIA['min_market_cap']}亿")
        print(f"  • 最大市盈率: {SELECTION_CRITERIA['max_pe']}")
        print(f"  • 数据完整性: 至少30个交易日")
        
        candidates = self.basic_screener.get_candidate_stocks()
        
        if candidates.empty:
            print("❌ 未找到符合基础条件的股票")
            return []
        
        # 应用统一的技术筛选条件（价格、换手率等）
        filtered = []
        
        for idx, stock in candidates.iterrows():
            try:
                # 使用 stock_screener 中统一的筛选方法
                passed, reason = self.basic_screener.apply_technical_filters(stock['code'])
                
                if passed:
                    filtered.append(stock['code'])
                
            except Exception as e:
                continue
        
        print(f"✓ 基础筛选完成: {len(candidates)} → {len(filtered)} 只股票")
        print("=" * 60)
        
        return filtered
    
    def run_model_1_liquidity_grab(self, stock_codes, top_n=TOP_N):
        """
        策略一：SMC流动性猎取策略
        
        核心逻辑：
        - 寻找刚刚完成"诱空"或"扫止损"的股票
        - 价格已经收回，机构建仓完成
        - 这是起涨前的"折价区"
        
        适用场景：
        - 短期反转交易
        - 快速进出
        - 高胜率策略
        """
        print("\n" + "=" * 60)
        print("策略一：SMC流动性猎取策略")
        print("=" * 60)
        print("策略特点：捕捉机构'扫止损'后的快速反弹")
        print(f"分析股票数: {len(stock_codes)}")
        print("-" * 60)
        
        results = self.smc_strategy.batch_screen(
            stock_codes, 
            max_results=top_n
        )
        
        return results
    

    def run_model_2_wyckoff_spring(self, stock_codes, top_n=TOP_N):
        """
        策略二：威科夫Spring反转策略
        
        核心逻辑：
        - 识别Spring形态（底部最后一次洗盘）
        - 均线向上排列
        - 价格在均线上方或接近均线
        
        适用场景：
        - 底部反转交易
        - 中长期持有
        - 趋势跟随
        """
        print("\n" + "=" * 60)
        print("策略二：威科夫Spring反转策略")
        print("=" * 60)
        print("策略特点：捕捉底部反转的最后机会")
        print(f"分析股票数: {len(stock_codes)}")
        print("-" * 60)
        
        results = []
        
        for idx, stock_code in enumerate(stock_codes, 1):
            try:
                result = self.wyckoff_strategy.wyckoff_accumulation_strategy(stock_code)
                
                if result and result['signal'] in ['buy']:
                    results.append({
                        'stock_code': stock_code,
                        'signal': result['signal'],
                        'confidence': result['confidence'],
                        'current_price': result['entry_price'],
                        'entry_price': result['entry_price'],
                        'stop_loss': result['stop_loss'],
                        'target': result['target'],
                        'risk_reward_ratio': (result['target'] - result['entry_price']) / 
                                            (result['entry_price'] - result['stop_loss']),
                        'signals': result['conditions_met'],
                        'strategy': 'wyckoff_spring'
                    })
                    print(f"✓ {stock_code}: Spring形态 (置信度: {result['confidence']:.1f}%)")
                
                if idx % 50 == 0:
                    print(f"进度: {idx}/{len(stock_codes)} | 找到: {len(results)} 只")
                
                if len(results) >= top_n:
                    break
                    
            except Exception as e:
                continue
        
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        print("-" * 60)
        print(f"✓ 找到 {len(results)} 只符合威科夫Spring条件的股票")
        
        return results
    
    def run_comprehensive_screening(self, model='all', top_n=TOP_N):
        """
        运行综合筛选
        
        参数：
            model: 'all', 'liquidity', 'wyckoff'
            top_n: 每个策略返回的最大结果数
        """
        print("\n" + "=" * 80)
        print("整合选股系统 - 主动筛选当前买点")
        print("=" * 80)
        print(f"筛选时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # 第一步：基础筛选
        candidates = self.get_filtered_candidates()
        
        if not candidates:
            print("基础筛选未找到候选股票，请检查数据")
            return None
        
        all_results = {}
        
        # 第二步：应用策略
        if model in ['all', 'liquidity']:
            model1_results = self.run_model_1_liquidity_grab(candidates, top_n)
            all_results['liquidity_grab'] = model1_results
        

        if model in ['all', 'wyckoff']:
            model2_results = self.run_model_2_wyckoff_spring(candidates, top_n)
            all_results['wyckoff_spring'] = model2_results
        
        # 第三步：生成综合报告
        self.generate_comprehensive_report(all_results)
        
        return all_results
    
    def generate_comprehensive_report(self, all_results):
        """
        生成综合报告
        """
        print("\n" + "=" * 80)
        print("综合筛选报告")
        print("=" * 80)
        
        total_stocks = 0
        for strategy_name, results in all_results.items():
            count = len(results) if results else 0
            total_stocks += count
            print(f"{strategy_name}: {count} 只股票")
        
        print(f"\n总计找到: {total_stocks} 只符合条件的股票")
        
        # 找出在多个策略中都出现的股票（高置信度）
        if len(all_results) > 1:
            print("\n" + "-" * 80)
            print("多策略共振股票（最高置信度）:")
            print("-" * 80)
            
            stock_counts = {}
            for strategy_name, results in all_results.items():
                if results:
                    for result in results:
                        code = result['stock_code']
                        if code not in stock_counts:
                            stock_counts[code] = []
                        stock_counts[code].append(strategy_name)
            
            multi_strategy_stocks = {k: v for k, v in stock_counts.items() if len(v) > 1}
            
            if multi_strategy_stocks:
                for stock_code, strategies in multi_strategy_stocks.items():
                    print(f"  ★ {stock_code}: 出现在 {len(strategies)} 个策略中")
                    print(f"    策略: {', '.join(strategies)}")
            else:
                print("  未发现多策略共振股票")
        
        # 保存详细报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for strategy_name, results in all_results.items():
            if results:
                filename = f"screening_{strategy_name}_{timestamp}.csv"
                self._save_results_to_csv(results, filename)
                print(f"\n✓ {strategy_name} 结果已保存到: {filename}")
        
        print("=" * 80)

    def _save_results_to_csv(self, results, filename):
        """保存结果到CSV"""
        export_data = []
        
        for result in results:
            export_data.append({
                '股票代码': result.get('stock_code', ''),
                '信号': result.get('signal', ''),
                '置信度': f"{result.get('confidence', 0):.1f}%",
                '当前价格': f"{result.get('current_price', 0):.2f}",
                '入场价': f"{result.get('entry_price', 0):.2f}",
                '止损价': f"{result.get('stop_loss', 0):.2f}",
                '目标价': f"{result.get('target', 0):.2f}",
                '风险%': f"{((result.get('entry_price', 0)-result.get('stop_loss', 0))/result.get('entry_price', 1)*100):.2f}%" if result.get('entry_price', 0) > 0 else "0.00%",
                '收益%': f"{((result.get('target', 0)-result.get('entry_price', 0))/result.get('entry_price', 1)*100):.2f}%" if result.get('entry_price', 0) > 0 else "0.00%",
                '风险收益比': f"1:{result.get('risk_reward_ratio', 0):.2f}",
                '信号详情': ' | '.join(result.get('signals', [])[:3])  # 只保存前3个信号
            })
        
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    def run_historical_screening(self, days_back=5, model='all', top_n=TOP_N):
        """
        历史回测模式 - 遍历倒数n天进行策略筛选
        
        参数：
            days_back: 回测天数（倒数n天）
            model: 'all', 'liquidity', 'wyckoff'
            top_n: 每个策略每天返回的最大结果数
        
        返回：
            dict: {日期: {策略名: [结果列表]}}
        """
        print("\n" + "=" * 80)
        print("历史回测模式 - 遍历倒数N天筛选")
        print("=" * 80)
        print(f"回测天数: {days_back} 天")
        print(f"策略模式: {model}")
        print(f"每天每策略最多返回: {top_n} 只股票")
        print("=" * 80)
        
        # 获取交易日列表
        trading_days = self._get_trading_days(days_back)
        
        if not trading_days:
            print("❌ 无法获取交易日列表")
            return None
        
        print(f"\n找到 {len(trading_days)} 个交易日:")
        for day in trading_days:
            print(f"  • {day}")
        
        # 存储所有结果
        all_historical_results = {}
        
        # 遍历每个交易日
        for idx, target_date in enumerate(trading_days, 1):
            print("\n" + "=" * 80)
            print(f"[{idx}/{len(trading_days)}] 分析日期: {target_date}")
            print("=" * 80)
            
            # 获取该日期的候选股票
            candidates = self.get_filtered_candidates()
            
            if not candidates:
                print(f"⚠ {target_date}: 未找到候选股票")
                all_historical_results[target_date] = {}
                continue
            
            daily_results = {}
            
            # 运行策略
            if model in ['all', 'liquidity']:
                print(f"\n运行SMC流动性猎取策略...")
                model1_results = self.run_model_1_liquidity_grab(candidates, top_n)
                daily_results['liquidity_grab'] = model1_results
            
            if model in ['all', 'wyckoff']:
                print(f"\n运行威科夫Spring策略...")
                model2_results = self.run_model_2_wyckoff_spring(candidates, top_n)
                daily_results['wyckoff_spring'] = model2_results
            
            all_historical_results[target_date] = daily_results
            
            # 显示当日摘要
            total_signals = sum(len(results) for results in daily_results.values() if results)
            print(f"\n✓ {target_date}: 找到 {total_signals} 个信号")
        
        # 生成历史回测报告
        self.generate_historical_report(all_historical_results, days_back)
        
        return all_historical_results
    
    def _get_trading_days(self, days_back):
        """
        获取最近n个交易日
        
        参数：
            days_back: 需要的交易日数量
        
        返回：
            list: 交易日列表（从旧到新）
        """
        try:
            # 从数据库获取任意一只股票的交易日期
            query = """
            SELECT DISTINCT date 
            FROM stock_data 
            WHERE date >= date('now', '-{} days')
            ORDER BY date DESC
            LIMIT {}
            """.format(days_back * 2, days_back)  # 多取一些以确保有足够的交易日
            
            df = pd.read_sql_query(query, self.data_fetcher.conn)
            
            if df.empty:
                # 如果数据库查询失败，使用日期推算
                trading_days = []
                current_date = datetime.now()
                
                while len(trading_days) < days_back:
                    # 跳过周末
                    if current_date.weekday() < 5:  # 0-4 是周一到周五
                        trading_days.append(current_date.strftime('%Y-%m-%d'))
                    current_date -= timedelta(days=1)
                
                return list(reversed(trading_days))
            
            # 返回最近的n个交易日（从旧到新）
            return df['date'].tolist()[::-1]
            
        except Exception as e:
            print(f"⚠ 获取交易日失败: {e}")
            # 降级方案：使用简单的日期推算
            trading_days = []
            current_date = datetime.now()
            
            while len(trading_days) < days_back:
                if current_date.weekday() < 5:
                    trading_days.append(current_date.strftime('%Y-%m-%d'))
                current_date -= timedelta(days=1)
            
            return list(reversed(trading_days))
    
    def generate_historical_report(self, all_historical_results, days_back):
        """
        生成历史回测综合报告
        
        参数：
            all_historical_results: {日期: {策略名: [结果列表]}}
            days_back: 回测天数
        """
        print("\n" + "=" * 80)
        print(f"历史回测综合报告 (最近 {days_back} 天)")
        print("=" * 80)
        
        # 统计每个策略的总信号数
        strategy_totals = {}
        date_totals = {}
        
        for date, daily_results in all_historical_results.items():
            date_total = 0
            for strategy_name, results in daily_results.items():
                count = len(results) if results else 0
                date_total += count
                
                if strategy_name not in strategy_totals:
                    strategy_totals[strategy_name] = 0
                strategy_totals[strategy_name] += count
            
            date_totals[date] = date_total
        
        # 显示每日统计
        print("\n每日信号统计:")
        print("-" * 80)
        for date, total in date_totals.items():
            print(f"  {date}: {total} 个信号")
        
        # 显示策略统计
        print("\n策略总计:")
        print("-" * 80)
        total_signals = 0
        for strategy_name, count in strategy_totals.items():
            print(f"  {strategy_name}: {count} 个信号")
            total_signals += count
        
        print(f"\n总计: {total_signals} 个信号")
        
        # 找出高频出现的股票
        print("\n" + "-" * 80)
        print("高频信号股票（多次出现）:")
        print("-" * 80)
        
        stock_frequency = {}
        for date, daily_results in all_historical_results.items():
            for strategy_name, results in daily_results.items():
                if results:
                    for result in results:
                        code = result['stock_code']
                        if code not in stock_frequency:
                            stock_frequency[code] = {'count': 0, 'dates': [], 'strategies': set()}
                        stock_frequency[code]['count'] += 1
                        stock_frequency[code]['dates'].append(date)
                        stock_frequency[code]['strategies'].add(strategy_name)
        
        # 按出现次数排序
        high_freq_stocks = {k: v for k, v in stock_frequency.items() if v['count'] > 1}
        sorted_stocks = sorted(high_freq_stocks.items(), key=lambda x: x[1]['count'], reverse=True)
        
        if sorted_stocks:
            for stock_code, info in sorted_stocks[:20]:  # 显示前20只
                print(f"  ★ {stock_code}: 出现 {info['count']} 次")
                print(f"    日期: {', '.join(info['dates'])}")
                print(f"    策略: {', '.join(info['strategies'])}")
        else:
            print("  未发现高频信号股票")
        
        # 保存历史回测结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        
        # 保存汇总结果
        summary_filename = f"historical_summary_{days_back}days_{timestamp}.csv"
        self._save_historical_summary(all_historical_results, stock_frequency, summary_filename)
        
        print(f"\n✓ 历史回测结果已保存")
        print(f"  • 详细结果: historical_[策略]_[日期]_{timestamp}.csv")
        print(f"  • 汇总结果: {summary_filename}")
        print("=" * 80)
    
    def _save_historical_summary(self, all_historical_results, stock_frequency, filename):
        """保存历史回测汇总"""
        summary_data = []
        
        for stock_code, info in sorted(stock_frequency.items(), key=lambda x: x[1]['count'], reverse=True):
            summary_data.append({
                '股票代码': stock_code,
                '出现次数': info['count'],
                '出现日期': ', '.join(info['dates']),
                '涉及策略': ', '.join(info['strategies'])
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    def close(self):
        """关闭资源"""
        self.data_fetcher.close()
        self.basic_screener.close()


def main():
    """主函数 - 支持命令行参数和交互式选股"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='整合选股系统 - 主动筛选当前买点')
    parser.add_argument('-n', '--days', type=int, default=0,
                        help='历史回测模式：遍历倒数n天进行筛选（例如：-n 5 表示最近5天）')
    parser.add_argument('-m', '--model', type=str, default='all',
                        choices=['all', 'liquidity', 'wyckoff'],
                        help='选择策略模式：all(全部), liquidity(SMC流动性), wyckoff(威科夫Spring)')
    parser.add_argument('-t', '--top', type=int, default=TOP_N,
                        help=f'每个策略返回的最大股票数（默认：{TOP_N}）')
    
    args = parser.parse_args()
    
    screener = IntegratedScreener()
    
    try:
        # 历史回测模式
        if args.days > 0:
            print("\n" + "=" * 80)
            print("历史回测模式")
            print("=" * 80)
            print(f"回测天数: {args.days}")
            print(f"策略模式: {args.model}")
            print(f"每天最多返回: {args.top} 只股票")
            print("=" * 80)
            
            screener.run_historical_screening(
                days_back=args.days,
                model=args.model,
                top_n=args.top
            )
            return
        
        # 交互式模式
        print("\n" + "=" * 80)
        print("整合选股系统 - 主动筛选当前买点")
        print("=" * 80)
        print("\n请选择筛选策略:")
        print("1. 策略一：SMC流动性猎取（快速反弹，高胜率）")
        print("2. 策略二：威科夫Spring（底部反转，趋势跟随）")
        print("3. 综合筛选（运行所有策略）")
        print("4. 历史回测模式（遍历倒数n天）")
        print("0. 退出")
        
        choice = input("\n请选择 (0-4): ").strip()
        
        if choice == "0":
            print("退出程序")
            return
        
        top_n = input(f"每个策略返回多少只股票？(默认{TOP_N}): ").strip()
        top_n = int(top_n) if top_n.isdigit() else TOP_N
        
        if choice == "1":
            candidates = screener.get_filtered_candidates()
            results = screener.run_model_1_liquidity_grab(candidates, top_n)
            screener.smc_strategy.generate_report(results)
        
        elif choice == "2":
            candidates = screener.get_filtered_candidates()
            results = screener.run_model_2_wyckoff_spring(candidates, top_n)
            screener.smc_strategy.generate_report(results)
        
        elif choice == "3":
            screener.run_comprehensive_screening('all', top_n)
        
        elif choice == "4":
            days_back = input("回测多少天？(例如：5): ").strip()
            days_back = int(days_back) if days_back.isdigit() else 5
            
            print("\n选择策略模式:")
            print("1. SMC流动性猎取")
            print("2. 威科夫Spring")
            print("3. 全部策略")
            
            model_choice = input("请选择 (1-3): ").strip()
            model_map = {'1': 'liquidity', '2': 'wyckoff', '3': 'all'}
            model = model_map.get(model_choice, 'all')
            
            screener.run_historical_screening(
                days_back=days_back,
                model=model,
                top_n=top_n
            )
        
        else:
            print("无效选择")
    
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    finally:
        screener.close()
        print("\n程序已退出")


if __name__ == "__main__":
    main()
