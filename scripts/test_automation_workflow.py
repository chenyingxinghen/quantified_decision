"""
自动化交易全流程真实测试脚本 (用于模拟盘实验)

主要用途：
1. 验证交易客户端 (同花顺/模拟盘) 的连接与 GUI 识别是否正常。
2. 验证资金、持仓数据的获取是否准确（含验证码识别测试）。
3. 验证买入、卖出指令的执行路径。
4. 验证本地状态追踪 (tracking.json) 的更新逻辑。

使用建议：
- 请先手动登录同花顺“模拟炒股”客户端。
- 确保窗口未被锁定，且可见。
- 本脚本会执行真实的买卖动作（模拟盘），请谨慎操作。
"""

import sys
import os
import time
import json
import logging
from datetime import datetime

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.automation.trader_interface import AutoTrader
from core.automation.execution_controller import ExecutionController, OperationStatus
from config.automation_config import DRY_RUN

# 配置专用的测试日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TestWorkflow")

def print_menu():
    print("\n" + "="*50)
    print("   自动化交易模块 - 全流程测试工具 (模拟盘专用)")
    print("="*50)
    print(" [1] 建立连接 & 检查资金 (Connection & Balance)")
    print(" [2] 同步并打印持仓 (Sync & Print Positions)")
    print(" [3] 模拟买入测试 (Manual Buy Test)")
    print(" [4] 模拟卖出测试 (Manual Sell Test)")
    print(" [5] 查看本地追踪状态 (View Tracking Data)")
    print(" [6] 重置今日处理记录 (Reset Today's Processed List)")
    print(" [7] 测试验证码识别 (Test Captcha ONLY)")
    print(" [8] 运行单循环完整流程 (Single Cycle: Sync->Signals->Buys->Sells)")
    print(" [0] 退出测试")
    print("="*50)

def main():
    if DRY_RUN:
        logger.warning("当前处于 DRY_RUN 模式，所有交易操作将仅模拟，不发往客户端。")
    
    trader = AutoTrader()
    controller = ExecutionController(trader)

    while True:
        print_menu()
        choice = input("请选择操作编号: ").strip()

        if choice == '1':
            logger.info("正在尝试连接客户端并获取资金...")
            if trader.connect():
                balance = trader.get_balance()
                if balance:
                    print(f"\n>>> 资金获取成功:")
                    for k, v in balance.items():
                        print(f"    {k}: {v}")
                else:
                    logger.error("资金获取失败！请检查客户端是否正常打开。")
            else:
                logger.error("连接失败！请检查 TRADER_TYPE 配置及客户端路径。")

        elif choice == '2':
            logger.info("正在同步持仓状态...")
            controller.sync_positions(cleanup=False)
            positions = trader.get_positions()
            if positions is not None:
                print(f"\n>>> 当前持仓 (共 {len(positions)} 只):")
                if not positions:
                    print("    (空仓)")
                else:
                    for p in positions:
                        code = p.get('证券代码', p.get('stock_code', ''))
                        name = p.get('证券名称', p.get('stock_name', ''))
                        amount = p.get('可用余额', p.get('可卖数量', 0))
                        price = p.get('当前价', p.get('现价', 0))
                        print(f"    代码: {code} | 名称: {name} | 可卖: {amount} | 现价: {price}")
            else:
                logger.error("获取持仓失败！")

        elif choice == '3':
            code = input("请输入要买入的股票代码 (例如 600000): ").strip()
            if len(code) != 6:
                print("代码无效，请输入6位数字。")
                continue
            
            try:
                price_str = input("请输入委托价格 (留空则使用 10.0): ").strip()
                price = float(price_str) if price_str else 10.0
                vol_str = input("请输入买入数量 (留空则使用 100): ").strip()
                volume = int(vol_str) if vol_str else 100
            except ValueError:
                print("输入数值无效。")
                continue

            print(f"准备执行模拟买入: {code} | 价格: {price} | 数量: {volume}")
            confirm = input("确定执行？(y/n): ").strip().lower()
            if confirm != 'y': continue

            # 构造临时信号
            test_signal = {
                'stock_code': code,
                'current_price': price,
                'stop_loss': price * 0.9,
                'take_profit': price * 1.1,
                'is_st': False
            }
            
            # 手动注入信号并运行 execute_buys (注意：execute_buys 内部会检查可用资金)
            # 为了绕过 ExecutionController 里的时间检查，我们直接调用它的内部逻辑或设置缓存
            controller.signals_cache = [test_signal]
            # 我们直接调用 controller.execute_buys()，它内部不检查系统时间，只检查 is_in_buy_window (主调度才查时间)
            # 但 execute_buys 会检查 processed_today。如果想多次测试同一只，需要选选项 [6] 重置。
            controller.execute_buys()

        elif choice == '4':
            code = input("请输入要全仓卖出的股票代码 (例如 600000): ").strip()
            if len(code) != 6:
                print("代码无效。")
                continue
            
            print(f"准备执行全仓卖出: {code} (将自动获取持仓数量并挂跌停价/现价)")
            confirm = input("确定执行？(y/n): ").strip().lower()
            if confirm != 'y': continue

            # 绕过 T+1 保护进行强制卖出测试
            # 手动移除 processed_today 中的买入记录
            base_code = code[:6]
            if base_code in controller.tracking_data["processed_today"]:
                logger.info(f"检测到 T+1 保护记录，正在为测试临时移除: {base_code}")
                del controller.tracking_data["processed_today"][base_code]

            # 执行卖出
            # 我们通过调用 _do_sell_robust 模拟 controller.execute_sells 的核心行为
            # 获取当前价作为参考
            balance = trader.get_balance() or {}
            pos = trader.get_positions() or []
            ref_price = 0
            for p in pos:
                if base_code in str(p.get('证券代码', '')):
                    ref_price = float(p.get('当前价', p.get('现价', 0)) or 0)
                    break
            
            if ref_price == 0:
                price_input = input("未在持仓找到该股或无法获取价格。请输入参考卖出价: ").strip()
                ref_price = float(price_input) if price_input else 10.0

            success, status = controller._do_sell_robust(code, ref_price=ref_price, is_st=False)
            print(f">>> 卖出执行结果: {status} (Success={success})")

        elif choice == '5':
            print(f"\n>>> 追踪数据 ({controller.tracking_file}):")
            print(json.dumps(controller.tracking_data, indent=4, ensure_ascii=False))

        elif choice == '6':
            controller.tracking_data["processed_today"] = {}
            controller._save_tracking()
            logger.info("今日处理记录已重置。您可以重新测试买入/卖出同一只股票。")

        elif choice == '7':
            logger.info("开始验证码识别专项测试...")
            success = trader.test_captcha()
            if success:
                logger.info("验证码测试通过！识别与填写流程正常。")
            else:
                logger.error("验证码测试失败，请检查 Tesseract 安装及配置。")

        elif choice == '8':
            logger.info(">>> 开始运行单循环完整流程测试 (与主程序逻辑一致)...")
            from scripts.main_auto_trade import get_latest_signals
            
            # 1. 同步持仓
            logger.info("步骤 1/4: 同步实盘持仓状态...")
            controller.sync_positions(cleanup=False)
            
            # 2. 获取最新信号
            logger.info("步骤 2/4: 获取选股信号 (select_stocks)...")
            signals = get_latest_signals()
            if not signals:
                logger.warning("未获取到任何信号，后续买入步骤将跳过。")
            else:
                controller.set_buy_signals(signals)
            
            # 3. 执行买入 (绕过时间窗检查直接运行核心逻辑)
            logger.info("步骤 3/4: 执行买入逻辑 (execute_buys)...")
            controller.execute_buys()
            
            # 4. 执行卖出 (绕过时间窗检查直接运行核心逻辑)
            logger.info("步骤 4/4: 执行卖出逻辑 (execute_sells)...")
            # 注意：execute_sells 会检查退出条件（止损/止盈/时间止损）
            controller.execute_sells()
            
            logger.info(">>> 单循环完整流程测试执行完毕。")

        elif choice == '0':
            print("退出测试。")
            break
        else:
            print("无效选择，请重新输入。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试终止。")
    except Exception as e:
        logger.error(f"发生未预期错误: {e}", exc_info=True)
