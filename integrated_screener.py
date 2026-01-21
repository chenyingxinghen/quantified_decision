# 整合选股器 - 结合SMC和威科夫策略
# 实现"主动筛选当前买点"的量化系统

import pandas as pd
from datetime import datetime
from data_fetcher import DataFetcher
from stock_screener import StockScreener
from smc_liquidity_strategy import SMCLiquidityStrategy
from wyckoff_strategy import WyckoffStrategy
from config import TECHNICAL_PARAMS,SELECTION_CRITERIA

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
    
    def close(self):
        """关闭资源"""
        self.data_fetcher.close()
        self.basic_screener.close()


def main():
    """主函数 - 交互式选股"""
    screener = IntegratedScreener()
    
    print("\n" + "=" * 80)
    print("整合选股系统 - 主动筛选当前买点")
    print("=" * 80)
    print("\n请选择筛选策略:")
    print("1. 策略一：SMC流动性猎取（快速反弹，高胜率）")
    print("2. 策略二：威科夫Spring（底部反转，趋势跟随）")
    print("3. 综合筛选（运行所有策略）")
    print("0. 退出")
    
    try:
        choice = input("\n请选择 (0-3): ").strip()
        
        if choice == "0":
            print("退出程序")
            return
        
        top_n = input("每个策略返回多少只股票？(默认100000): ").strip()
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
        
        else:
            print("无效选择")
    
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    finally:
        screener.close()
        print("\n程序已退出")


if __name__ == "__main__":
    main()
